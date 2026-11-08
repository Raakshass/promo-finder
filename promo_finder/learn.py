"""Learn the four ranking weights from how posted promos perform, and pick what to post next. No model calls.

The ranking (rank.py) scores a clip as

    score = m x (w_hook*hook + w_cliff*cliff + w_standalone*standalone + w_density*density)

with the four terms in 0..1 and m the clip's multiplier (editor's-check penalties x turning-point lift x 0.9 for a
two-part join). The weights in config.yaml are hand-set. `promo-finder learn` refits them from traction.

Data: one row per posted (or reviewed) promo, its id and one metric: hook rate, hold rate, CTR (continuous) or
approved/kept 0/1 (binary). Each id is joined with the run's promos.json / candidates.jsonl for its four terms and its
multiplier, so the regressors are z_k = m x term_k and the run's own score is z . w.

Model    continuous:  metric = a + b x z.w + noise                    (hook rate, hold rate, CTR)
         binary:      P(metric = 1) = sigmoid(a + b x z.w)            (approved / kept)
         w are the weights (>= 0, sum 1); b > 0 is how strongly the score moves the metric; a and b are free.
Prior    w ~ Normal(w_old, (kappa x lam x S)^-1) on the plane where the weights sum to 1: the old weights, worth kappa
         posts. One post tells lam x S about w (lam = b^2 / noise variance; for a binary metric b^2 times the posts'
         mean p(1 - p)), so this prior counts exactly as much as kappa posts: after kappa
         posts the weights have moved about halfway from the old ones to what the posts alone say.
         S: covariance of the regressors over posts like the observed ones (plus POOL_PSEUDO clips of the run's
         candidate pool, so it is defined with 2 or 3 posts).
         lam: the signal level the fit finds, but never below the level at which the old score explains R2_REF of
         the metric's variance. The floor matters: when the metric is mostly noise, "halfway after kappa posts" would
         be halfway toward noise, so a noisy metric moves the weights less (simulated in tests/test_learn.py).
Fit      MAP over (a, b, w): b by a 1-D search, w in closed form given b (ridge regression toward the old weights) for
         a continuous metric; Newton's method for the logistic one. Weights that come out below 0 are set to 0 and
         the rest rescaled. Uncertainty: Laplace approximation (a Gaussian at the MAP, covariance = inverse Hessian).
Plan     Thompson sampling (a contextual bandit; each post is a one-step decision with an immediate reward): draw w
         from that posterior, re-score the run's candidate pool with it, and build the slate under the draw with the
         pipeline's own rules (near-duplicate removal, opening/middle/late quotas, at most 2 clips per episode in the
         top 10, same-episode and same-hook-type penalties). One draw's slate is what to post next; across many
         draws, a clip's share of slates says how sure the posterior is about it.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
TERMS = ("hook", "cliff", "standalone", "density")
WEIGHT_KEYS = tuple(f"w_{t}" for t in TERMS)
DEFAULT_PRIOR_STRENGTH = 30.0  # posts the old weights are worth: about 30 posts move them halfway
R2_REF = 0.25                  # ...posts of a metric the score explains at least 25% of (a noisier one moves them less)
R2_CAP = None                  # ...and at most this much (None: the prior's strength follows the fitted signal upward)
POOL_PSEUDO = 4.0              # pool clips mixed into the prior's covariance, so it is defined with 2-3 posts
TYPICAL_TERM_SD = 0.2          # regressor spread used when the pool is too small to measure (official run: 0.14-0.25)
NU0 = 2.0                      # prior degrees of freedom of the noise variance, centred on "the score explains nothing"
TWO_SEGMENT_FACTOR = 0.9       # rank.combine: an unproven two-part join
ID_COLUMNS = ("promo_id", "cand_id", "clip_id", "id", "rank")
META_COLUMNS = set(ID_COLUMNS) | {"segment", "episode", "in", "out", "start", "end", "duration_s", "score",
                                  "cold_viewer_comprehensibility", "spoiler_risk", "minutes_into_show"}
PREFERRED_METRICS = ("hook_rate", "hold_rate", "ctr", "tap_rate", "taps", "completion_rate", "watch_time",
                     "approved", "kept", "decision", "reward")
TRUE_WORDS = {"approved", "approve", "kept", "keep", "yes", "true", "posted"}
FALSE_WORDS = {"rejected", "reject", "dropped", "drop", "no", "false"}
SKIP_WORDS = {"", "trim", "na", "n/a", "nan", "none", "null", "-"}


class LearnError(Exception):
    """A readable reason why nothing can be learned (bad file, unknown ids, no variation, contradicting data)."""


# ---------------------------------------------------------------------------------------------------------------- pool
@dataclass
class Clip:
    cand_id: str
    terms: np.ndarray        # hook, cliff, standalone, density (0..1), as the run scored them
    mult: float              # editor's checks x turning point x two-part join
    score: float             # the run's own score (= mult x terms . w_old, up to rounding)
    segments: list           # [(episode, start_s, end_s)]
    kind: str = "single"
    block: str | None = None
    hook_type: str | None = None
    hook_beat: str | None = None
    cliff_beat: str | None = None
    promo_id: str | None = None
    rank: int | None = None
    judged: bool = False     # the first-time-viewer judge rated it (standalone is its rating, not the proxy)
    spoiler_risk: bool = False
    source: str = "promos"   # promos | candidates | replay

    @property
    def episode(self) -> int:
        return int(self.segments[0][0])

    @property
    def z(self) -> np.ndarray:
        return self.terms * self.mult


@dataclass
class Pool:
    run_dir: Path
    cfg: dict
    weights: np.ndarray      # the run's weights (normalised): the prior
    clips: list[Clip]        # promos.json in rank order, then candidates.jsonl best first; one entry per clip
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.by_id = {c.cand_id.lower(): c for c in self.clips}
        self.by_rank = {c.rank: c for c in self.clips if c.rank is not None}

    def find(self, ident: str) -> Clip | None:
        """'P03', 'p3' or '3' (a rank) -> that promo; anything else is a candidate id."""
        s = str(ident).strip()
        m = re.fullmatch(r"[Pp]?0*(\d+)", s)
        if m:
            return self.by_rank.get(int(m.group(1)))
        return self.by_id.get(s.lower())

    @property
    def Z(self) -> np.ndarray:
        return np.array([c.z for c in self.clips]) if self.clips else np.zeros((0, 4))


def _mult(sub: dict, kind: str, score: float) -> float:
    m = float(sub.get("rubric_mult", 1.0)) * float(sub.get("arc_mult", 1.0)) * (TWO_SEGMENT_FACTOR if kind == "two_segment" else 1.0)
    base = float(sub.get("base", 0.0))
    if base > 0 and abs(base * m - score) > 2e-3:  # a factor this module does not know about: use the effective one
        m = score / base
    return m


def _promo_clip(p: dict) -> Clip:
    sub, fl = p["subscores"], p.get("flags") or {}
    kind = str(fl.get("kind", "single"))
    return Clip(cand_id=str(fl.get("cand_id") or p["promo_id"]), terms=np.array([float(sub[t]) for t in TERMS]),
                mult=_mult(sub, kind, float(p["score"])), score=float(p["score"]),
                segments=[(int(s["episode"]), float(s["start_s"]), float(s["end_s"])) for s in p["segments"]], kind=kind,
                block=fl.get("block"), hook_type=fl.get("hook_type"), hook_beat=(p.get("hook") or {}).get("beat_id"),
                cliff_beat=(p.get("cliffhanger") or {}).get("beat_id"), promo_id=p["promo_id"], rank=int(p["rank"]),
                judged=p.get("cold_viewer") is not None, spoiler_risk=bool(fl.get("spoiler_risk")), source="promos")


def _cand_clip(c: dict, w=None) -> Clip | None:
    """A candidates.jsonl row. Runs from before this module stored no terms: then `w` (the replayed window) supplies them."""
    sub = c.get("subscores") or (w.subscores if w is not None else None)
    if not sub or any(t not in sub for t in TERMS):
        return None
    kind = str(c.get("kind") or (w.kind if w is not None else "single"))
    fl = w.flags if w is not None else {}
    return Clip(cand_id=str(c["cand_id"]), terms=np.array([float(sub[t]) for t in TERMS]), mult=_mult(sub, kind, float(c["score"])),
                score=float(c["score"]), segments=[(int(s["episode"]), float(s["start_s"]), float(s["end_s"])) for s in c["segments"]],
                kind=kind, block=c.get("block") or fl.get("block"), hook_type=c.get("hook_type") or fl.get("hook_type"),
                hook_beat=c.get("hook_beat") or (w.hook_beat if w is not None else None),
                cliff_beat=c.get("cliff_beat") or (w.cliff_beat if w is not None else None),
                judged=bool(c.get("judged", "standalone_proxy" in sub)),
                spoiler_risk=bool(c.get("spoiler_risk", fl.get("spoiler_risk", False))),
                source="candidates" if c.get("subscores") else "replay")


def _cached_judge(run_dir: Path) -> dict:
    """The first-time-viewer verdicts the run saved (llm_raw/cold_viewer), read back without calling anything. A batch
    whose answer does not validate is skipped as a whole, as the run itself did."""
    from .judge import ColdViewerBatchOut
    from .llm.base import _extract_json
    from .models import ColdViewer
    out: dict = {}
    for f in sorted((run_dir / "llm_raw" / "cold_viewer").glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            data = _extract_json(rec.get("response_text") or "")
            if isinstance(data, dict) and "clips" in data:
                for c in ColdViewerBatchOut.model_validate(data).clips:
                    out[c.clip_id.strip()] = ColdViewer(**{k: getattr(c, k) for k in ColdViewer.model_fields})
            else:
                out[str(rec.get("tag") or f.stem).strip()] = ColdViewer.model_validate(data)
        except Exception:  # noqa: BLE001 - an unreadable answer is an unjudged clip, as in the run
            continue
    return out


def _find_show(recorded: str | None, show_dir: Path | None) -> Path | None:
    for p in ([Path(show_dir)] if show_dir else []) + ([Path(recorded), ROOT / recorded] if recorded else []):
        if p.exists():
            return p
    return None


def _replayed_windows(run_dir: Path, cfg: dict, show_dir: Path | None, notes: list[str]) -> dict | None:
    """Rebuild every scored window of an older run (beats/*.json + the saved judge verdicts), zero model calls."""
    from .rank import apply_judge, flag_spoilers, generate_windows, score_window
    from .stability import replay
    try:
        args = json.loads((run_dir / "run_log.json").read_text())["args"]
    except (OSError, KeyError, ValueError):
        notes.append("run_log.json is missing: cannot rebuild the candidates' terms")
        return None
    src = _find_show(args.get("show_dir"), show_dir)
    if src is None:
        notes.append(f"the show folder {args.get('show_dir')!r} is not here (pass --show-dir): candidates without stored terms are left out")
        return None
    try:
        ctx = replay(run_dir, src)
        source = "llm" if ctx.has_llm else "heuristic"
        windows = generate_windows(ctx, cfg, source)
        for w in windows:
            score_window(w, ctx, cfg, source)
        flag_spoilers(windows, ctx, cfg)
    except Exception as e:  # noqa: BLE001 - a run that cannot be rebuilt still has its promos
        notes.append(f"could not rebuild the candidates from {src} ({type(e).__name__}: {e}); only promos.json is used")
        return None
    judged = _cached_judge(run_dir)
    apply_judge([w for w in windows if w.cand_id in judged], judged, cfg)
    return {w.cand_id: w for w in windows}


def load_pool(run_dir: Path, show_dir: Path | None = None) -> Pool:
    """The run's clips with their four terms and multiplier: promos.json (stored), candidates.jsonl (stored by newer
    runs; rebuilt from the run's own files for older ones, and kept only if the rebuilt score matches the stored one)."""
    run_dir = Path(run_dir)
    if not (run_dir / "promos.json").exists() or not (run_dir / "config.yaml").exists():
        raise LearnError(f"{run_dir} is not a finished run (promos.json and config.yaml are needed)")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    w = np.array([float(cfg["scoring"][k]) for k in WEIGHT_KEYS])
    if (w < 0).any() or w.sum() <= 0:
        raise LearnError(f"the run's weights {w.tolist()} are not usable as a prior")
    notes: list[str] = []
    clips = [_promo_clip(p) for p in json.loads((run_dir / "promos.json").read_text(encoding="utf-8"))]
    seen = {c.cand_id for c in clips}
    rows = []
    if (run_dir / "candidates.jsonl").exists():
        rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["cand_id"] not in seen]
    replayed = None
    if any(not r.get("subscores") for r in rows):
        replayed = _replayed_windows(run_dir, cfg, show_dir, notes)
    unverified = missing = 0
    for r in rows:
        if r.get("subscores"):
            clips.append(_cand_clip(r))
            continue
        win = (replayed or {}).get(r["cand_id"])
        if win is None:
            missing += 1
            continue
        if abs(win.score - float(r["score"])) > 2e-3:  # the code changed since the run: this rebuild cannot be trusted
            unverified += 1
            continue
        c = _cand_clip(r, win)
        if c is not None:
            clips.append(c)
    if replayed is not None:
        notes.append(f"rebuilt the terms of {sum(c.source == 'replay' for c in clips)} candidates from the run's own files "
                     f"(scores match the stored ones); {unverified} did not match and were left out")
    if missing and replayed is not None:
        notes.append(f"{missing} candidates could not be rebuilt and were left out")
    return Pool(run_dir=run_dir, cfg=cfg, weights=w / w.sum(), clips=[c for c in clips if c is not None], notes=notes)


# ------------------------------------------------------------------------------------------------------------ outcomes
@dataclass
class Outcomes:
    ids: list[str]
    y: np.ndarray
    metric: str
    kind: str                # continuous | binary
    id_column: str
    skipped: int = 0         # rows without a usable metric value
    notes: list[str] = field(default_factory=list)


def _value(s) -> float | None:
    """A metric cell -> number; None if empty or 'trim'. Accepts 35%, approved/rejected, yes/no, true/false."""
    if s is None:
        return None
    if isinstance(s, bool):
        return float(s)
    if isinstance(s, (int, float)):
        return float(s) if math.isfinite(float(s)) else None
    t = str(s).strip().lower()
    if t in SKIP_WORDS:
        return None
    if t in TRUE_WORDS:
        return 1.0
    if t in FALSE_WORDS:
        return 0.0
    v = float(t[:-1] if t.endswith("%") else t)  # ValueError: not a number
    return v if math.isfinite(v) else None


def _rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":  # the app's decisions.json ({promo_id: decision}) or a list of row objects
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [{"promo_id": k, "decision": v} for k, v in data.items()]
        if isinstance(data, list) and all(isinstance(r, dict) for r in data):
            return data
        raise LearnError(f"{path.name}: expected {{promo_id: decision}} or a list of rows")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_outcomes(path: Path, metric: str | None = None, kind: str = "auto") -> Outcomes:
    path = Path(path)
    if not path.exists():
        raise LearnError(f"no outcomes file at {path}")
    raw = _rows(path)
    if not raw:
        raise LearnError(f"{path.name} has no rows")
    rows = [{str(k).strip().lower(): v for k, v in r.items() if k is not None} for r in raw]
    cols = list(rows[0].keys())
    id_col = next((c for c in ID_COLUMNS if c in cols), None)
    if id_col is None:
        raise LearnError(f"{path.name} needs a promo_id column (or rank, or cand_id); it has {cols}")
    notes = []
    if "segment" in cols:  # promos.csv has one row per segment: count a two-part promo once
        before = len(rows)
        rows = [r for r in rows if str(r.get("segment") or "1").strip() in ("", "1")]
        if len(rows) < before:
            notes.append(f"kept the first segment row of each two-part promo ({before - len(rows)} rows dropped)")

    def numeric(c: str) -> bool:
        try:
            vals = [_value(r.get(c)) for r in rows]
        except ValueError:
            return False
        return any(v is not None for v in vals)

    if metric:
        metric = metric.strip().lower()
        if metric not in cols:
            raise LearnError(f"{path.name} has no column {metric!r}; it has {cols}")
        if not numeric(metric):
            raise LearnError(f"column {metric!r} is not numeric (or approved/rejected)")
    else:
        cands = [c for c in cols if c not in META_COLUMNS and numeric(c)]
        if not cands:
            raise LearnError(f"{path.name} has no metric column (numbers such as hook_rate, or approved 0/1); it has {cols}")
        pref = [c for c in PREFERRED_METRICS if c in cands]
        if len(cands) > 1 and not pref:
            raise LearnError(f"{path.name} has several metric columns {cands}: choose one with --metric")
        metric = pref[0] if pref else cands[0]
        if len(cands) > 1:
            notes.append(f"several metric columns {cands}; used {metric!r} (choose with --metric)")
    ids, ys, skipped = [], [], 0
    for r in rows:
        v = _value(r.get(metric))
        ident = str(r.get(id_col) or "").strip()
        if v is None or not ident:
            skipped += 1
            continue
        ids.append(ident)
        ys.append(v)
    if not ys:
        raise LearnError(f"no row of {path.name} has both an id and a {metric} value")
    y = np.array(ys, dtype=float)
    is01 = bool(np.isin(y, (0.0, 1.0)).all())
    if kind == "auto":
        kind = "binary" if is01 else "continuous"
    elif kind == "binary" and not is01:
        raise LearnError(f"--kind binary needs 0/1 values in {metric!r}")
    elif kind not in ("binary", "continuous"):
        raise LearnError("--kind is auto, continuous or binary")
    return Outcomes(ids=ids, y=y, metric=metric, kind=kind, id_column=id_col, skipped=skipped, notes=notes)


@dataclass
class Joined:
    clips: list[Clip]
    y: np.ndarray
    unknown: list[str]

    @property
    def Z(self) -> np.ndarray:
        return np.array([c.z for c in self.clips])


def join(obs: Outcomes, pool: Pool) -> Joined:
    clips, ys, unknown = [], [], []
    for ident, v in zip(obs.ids, obs.y):
        c = pool.find(ident)
        if c is None:
            unknown.append(ident)
        else:
            clips.append(c)
            ys.append(v)
    if not clips:
        raise LearnError(f"none of the {len(obs.ids)} ids is a promo or candidate of this run (first: {unknown[:5]}); "
                         "use promo_id as in promos.csv (P01...), the rank, or a cand_id from candidates.jsonl")
    return Joined(clips=clips, y=np.array(ys), unknown=unknown)


# ----------------------------------------------------------------------------------------------------------------- fit
def _plane_basis() -> np.ndarray:
    """Orthonormal basis (4 x 3) of the changes d with sum(d) = 0: the weights move inside the plane where they sum to 1."""
    u, _, _ = np.linalg.svd(np.eye(4) - 0.25)
    return u[:, :3]


BASIS = _plane_basis()


def _normalise(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def project(w: np.ndarray) -> np.ndarray:
    """Any weight vector -> usable weights: negative ones become 0, the rest are rescaled to sum to 1."""
    p = np.clip(np.asarray(w, dtype=float), 0.0, None)
    return p / p.sum()


@dataclass
class Fit:
    kind: str                # continuous | binary
    n: int
    prior_strength: float
    old: np.ndarray          # prior weights (sum 1)
    new: np.ndarray          # learned weights (>= 0, sum 1)
    sd: np.ndarray           # Laplace posterior sd of each weight
    prior_sd: np.ndarray     # the prior's sd of each weight, at the strength the fit used
    intercept: float
    scale: float             # b: metric units (log-odds if binary) per unit of score; 0 = no positive relation found
    r2: float                # share of the metric's variance the learned score explains in the fit (latent R2 if binary)
    floor: bool              # the signal was below R2_REF, so the prior was held at its reference strength
    data_share: float        # share of the information about the weights that comes from the posts (the prior has the rest)
    clipped: list[str]
    stats: dict
    warnings: list[str]
    mode: np.ndarray         # (b, u) at the MAP; the weights are w_old + BASIS @ u before clipping
    cov: np.ndarray | None   # Laplace covariance of (b, u); None when the MAP sits at b = 0
    prior_cov_u: np.ndarray  # prior covariance of u

    @property
    def moved(self) -> float:
        """Share of the total weight that changed terms (total variation distance, 0..1)."""
        return 0.5 * float(np.abs(self.new - self.old).sum())

    def sample(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, int]:
        """Posterior draws of the weights (Laplace approximation), each clipped at 0 and rescaled to sum 1. A draw with
        b <= 0 (a score that would not help at all) is dropped: returns (draws, number dropped)."""
        if self.cov is None:  # no evidence yet that the score helps: the weights are as uncertain as the prior says
            u = rng.standard_normal((size, 3)) @ np.linalg.cholesky(self.prior_cov_u + 1e-12 * np.eye(3)).T
            return np.array([project(self.old + BASIS @ x) for x in u]).reshape(-1, 4), 0
        x = self.mode + rng.standard_normal((size, 4)) @ np.linalg.cholesky(self.cov + 1e-12 * np.eye(4)).T
        ok = x[:, 0] > 0
        return np.array([project(self.old + BASIS @ xi[1:]) for xi in x[ok]]).reshape(-1, 4), int((~ok).sum())

    def as_dict(self) -> dict:
        r = lambda a: dict(zip(TERMS, [round(float(x), 4) for x in a]))  # noqa: E731
        return {"kind": self.kind, "n": self.n, "prior_strength": self.prior_strength, "old": r(self.old), "new": r(self.new),
                "sd": r(self.sd), "prior_sd": r(self.prior_sd), "moved": round(self.moved, 4), "data_share": round(self.data_share, 3),
                "r2": round(self.r2, 3), "floor": self.floor, "scale": round(self.scale, 4), "intercept": round(self.intercept, 4),
                "clipped": self.clipped, "stats": self.stats, "warnings": self.warnings}


def _reference_cov(pool_Z) -> np.ndarray:
    if pool_Z is not None and len(pool_Z) >= 10:
        ref = np.cov(np.asarray(pool_Z, dtype=float), rowvar=False, bias=True)
    else:
        ref = np.eye(4) * TYPICAL_TERM_SD ** 2
    return ref + np.eye(4) * 1e-4


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _newton_logistic(A, y, m0, P0, offset=None, start=None, iters: int = 100):
    """MAP of a logistic regression with a Gaussian prior N(m0, P0^-1): Newton's method with step halving."""
    off = np.zeros(len(y)) if offset is None else offset

    def obj(b):
        eta = A @ b + off
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * (b - m0) @ P0 @ (b - m0))
    beta = (m0 if start is None else start).astype(float).copy()
    f = obj(beta)
    for _ in range(iters):
        p = _sigmoid(A @ beta + off)
        g = A.T @ (p - y) + P0 @ (beta - m0)
        step = np.linalg.solve(A.T @ (A * (p * (1 - p))[:, None]) + P0, g)
        t = 1.0
        while True:
            nb = beta - t * step
            nf = obj(nb)
            if nf <= f + 1e-12 or t < 1e-10:
                break
            t *= 0.5
        if nf > f + 1e-12:  # no step improves the objective: this is the optimum to numerical precision
            break
        done = abs(f - nf) < 1e-12 or float(np.max(np.abs(t * step))) < 1e-10
        beta, f = nb, nf
        if done:
            break
    p = _sigmoid(A @ beta + off)
    return beta, A.T @ (A * (p * (1 - p))[:, None]) + P0, f


def _argmin_1d(f, hi: float, near: float | None = None) -> float:
    """Minimum of f on [0, hi]: a log grid, then golden-section search around the best grid point. With `near` (the
    previous answer, while an outer loop converges), only a short grid around it and 0 are searched."""
    if near is not None and near > 0:
        grid = np.concatenate([[0.0], np.geomspace(near / 2, min(2 * near, hi), 9)])
    else:
        grid = np.concatenate([[0.0], np.geomspace(hi * 1e-5, hi, 40)])
    vals = [f(x) for x in grid]
    i = int(np.argmin(vals))
    a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(50):
        if b - a <= 1e-7 * max(b, 1e-12):
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x if f(x) <= vals[i] else float(grid[i])


def _fit_continuous(Zc, yc, w0, P_delta, vy, b_hi, near=None):
    """MAP of (b, u) for metric = a + b x z.(w_old + BASIS u) + noise, with u ~ N(0, (BASIS' P_delta BASIS)^-1) and a flat
    prior on a and on b >= 0. The noise variance alternates with them (prior: NU0 posts' worth of the metric's variance)."""
    n = len(yc)
    ZB, q0 = Zc @ BASIS, Zc @ w0
    G, h0, h1 = ZB.T @ ZB, ZB.T @ yc, ZB.T @ q0
    yy, q0y, q0q0 = float(yc @ yc), float(q0 @ yc), float(q0 @ q0)
    Lu = BASIS.T @ P_delta @ BASIS

    def solve_u(b, s2):
        return np.linalg.solve((b * b / s2) * G + Lu, (b / s2) * (h0 - b * h1))

    def rss(b, u):
        return yy - 2 * b * (q0y + u @ h0) + b * b * (q0q0 + 2 * (u @ h1) + u @ G @ u)

    def objective(t, s2):
        u = solve_u(t, s2)
        return 0.5 * rss(t, u) / s2 + 0.5 * (u @ Lu @ u)

    s2, b = vy, near
    for _ in range(60):
        b = _argmin_1d(lambda t, s2=s2: objective(t, s2), b_hi, near=b)
        u = solve_u(b, s2)
        new = (max(rss(b, u), 0.0) + NU0 * vy) / (n - 1 + NU0)
        done = abs(new - s2) <= 1e-7 * s2
        s2 = new
        if done:
            break
    u = solve_u(b, s2)
    q = Zc @ (w0 + BASIS @ u)
    r = yc - b * q
    H = np.empty((4, 4))  # Hessian of the negative log posterior in (b, u), noise variance fixed
    H[0, 0] = q @ q / s2
    H[0, 1:] = H[1:, 0] = (b * (ZB.T @ q) - ZB.T @ r) / s2
    H[1:, 1:] = (b * b) * G / s2 + Lu
    return {"b": b, "u": u, "a": 0.0, "s2": s2, "Lu": Lu, "H": H, "info": (b * b) * G / s2, "lam": b * b / s2}


def _fit_binary(Zc, y, w0, P_delta, a0, b_hi, s_b, near=None):
    """MAP of (a, b, u) for P(metric = 1) = sigmoid(a + b x z.(w_old + BASIS u)), u ~ N(0, (BASIS' P_delta BASIS)^-1),
    a ~ N(logit(base rate), 2.5^2), b ~ N(0, s_b^2) on b >= 0 (weak priors that keep a separable sample finite)."""
    n = len(y)
    ZB, q0 = Zc @ BASIS, Zc @ w0
    Lu = BASIS.T @ P_delta @ BASIS
    P0 = np.zeros((4, 4))
    P0[0, 0], P0[1:, 1:] = 1 / 2.5 ** 2, Lu
    m0 = np.array([a0, 0.0, 0.0, 0.0])
    warm = {"x": m0}

    def inner(b):
        beta, _, f = _newton_logistic(np.column_stack([np.ones(n), b * ZB]), y, m0, P0, offset=b * q0, start=warm["x"])
        warm["x"] = beta
        return f + 0.5 * b * b / s_b ** 2, beta

    b = _argmin_1d(lambda t: inner(t)[0], b_hi, near=near)
    _, beta = inner(b)
    a, u = float(beta[0]), beta[1:]
    q = Zc @ (w0 + BASIS @ u)
    p = _sigmoid(a + b * q)
    W, e = p * (1 - p), p - y
    H = np.empty((5, 5))  # Hessian of the negative log posterior in (a, b, u)
    H[0, 0] = W.sum() + 1 / 2.5 ** 2
    H[0, 1] = H[1, 0] = W @ q
    H[0, 2:] = H[2:, 0] = b * (ZB.T @ W)
    H[1, 1] = q @ (W * q) + 1 / s_b ** 2
    H[1, 2:] = H[2:, 1] = b * (ZB.T @ (W * q)) + ZB.T @ e
    H[2:, 2:] = (b * b) * (ZB.T @ (ZB * W[:, None])) + Lu
    # the signal level: information about w per post per unit of S = b^2 x the mean p(1 - p) of the posts (near-certain
    # outcomes teach little, so a sharp model does not make the prior count for more than kappa real posts)
    return {"b": b, "u": u, "a": a, "s2": None, "Lu": Lu, "H": H, "info": (b * b) * (ZB.T @ (ZB * W[:, None])), "lam": b * b * float(W.mean())}


def _set_strength(at, lam_ref: float, lam_cap: float | None = None):
    """The prior is worth kappa posts at the signal level the fit finds (lam = b^2 / noise), but never at less than the
    reference level nor more than the cap: solve lam = clip(fitted lam, lam_ref, lam_cap) by bisection (the fitted signal
    falls as the prior tightens). `at(lam, near)` fits at prior level lam; `near` warm-starts the search for b.
    Returns (solution, "floor" | "cap" | "fitted")."""
    sol = at(lam_ref, None)
    if sol["lam"] <= lam_ref:
        return sol, "floor"
    if lam_cap is not None and lam_cap > lam_ref:
        top = at(lam_cap, sol["b"])
        if top["lam"] >= lam_cap:
            return at(lam_cap, None), "cap"
    lo, hi = lam_ref, sol["lam"] if lam_cap is None else min(sol["lam"], lam_cap)
    sol_hi = at(hi, sol["b"])
    for _ in range(30):
        if sol_hi["lam"] <= hi:
            break
        lo, hi = hi, hi * 4 if lam_cap is None else min(hi * 4, lam_cap)
        sol_hi = at(hi, sol_hi["b"])
    for _ in range(60):
        if hi / lo < 1 + 2e-3:
            break
        mid = math.sqrt(lo * hi)
        s = at(mid, sol_hi["b"])
        if s["lam"] > mid:
            lo = mid
        else:
            hi, sol_hi = mid, s
    return at(hi, None), "fitted"  # the final fit searches b over its whole range again


def _auc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(wins / (len(pos) * len(neg))), 3)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def fit(Z, y, w_old, kind: str = "continuous", prior_strength: float = DEFAULT_PRIOR_STRENGTH, pool_Z=None) -> Fit:
    """MAP weights under the prior 'the old weights are worth `prior_strength` posts'. Z: n x 4 regressors (m x terms)."""
    Z, y, w0 = np.asarray(Z, dtype=float), np.asarray(y, dtype=float), _normalise(w_old)
    n, kappa = len(y), float(prior_strength)
    if kind not in ("continuous", "binary"):
        raise LearnError(f"unknown kind {kind!r}")
    if kappa <= 0:
        raise LearnError("--prior-strength must be > 0 (it is how many posts the current weights are worth)")
    if n < 2:
        raise LearnError(f"{n} matched post: at least 2 posts with different outcomes are needed")
    if np.ptp(y) == 0:
        raise LearnError("every matched post has the same outcome: there is nothing to learn from")
    zbar = Z.mean(0)
    Zc = Z - zbar
    S = (Zc.T @ Zc + POOL_PSEUDO * _reference_cov(pool_Z)) / (n + POOL_PSEUDO)  # covariance of the pseudo-posts
    var_s0 = float(w0 @ S @ w0)
    s0c = Zc @ w0
    stats: dict = {}
    if kind == "continuous":
        yc = y - y.mean()
        vy = float(yc @ yc / (n - 1))
        lam_ref = R2_REF / (1 - R2_REF) / var_s0  # b^2/noise at which the old score explains R2_REF of the metric
        lam_cap = R2_CAP / (1 - R2_CAP) / var_s0 if R2_CAP else None
        b_hi = 20 * math.sqrt(vy / var_s0)  # b is searched on [0, b_hi]: 20 metric sd per score sd
        sol, level = _set_strength(lambda lam, near: _fit_continuous(Zc, yc, w0, kappa * lam * S, vy, b_hi, near), lam_ref, lam_cap)
        ss = float(s0c @ s0c)
        b1 = float(s0c @ yc / ss) if ss > 1e-12 else 0.0
        se1 = math.sqrt(max(float(((yc - b1 * s0c) ** 2).sum()), 1e-300) / max(n - 2, 1) / max(ss, 1e-12))
        stats.update({"slope_on_old_score": round(b1, 4), "slope_t": round(b1 / se1, 2) if n > 2 else None,
                      "residual_sd": round(math.sqrt(sol["s2"]), 4), "metric_sd": round(math.sqrt(vy), 4)})
        noise = sol["s2"]
    else:
        ybar = float(y.mean())
        v = min(max(ybar * (1 - ybar), 0.05), 0.25)  # Fisher information of one post at the base rate (the reference level)
        pc = min(max(ybar, 0.01), 0.99)
        a0 = math.log(pc / (1 - pc))
        lam_ref = v * R2_REF / (1 - R2_REF) * (math.pi ** 2 / 3) / var_s0  # latent-variable R2 at the base rate's information
        lam_cap = v * R2_CAP / (1 - R2_CAP) * (math.pi ** 2 / 3) / var_s0 if R2_CAP else None
        s_b = 5.0 / math.sqrt(var_s0)
        b_hi = 30.0 / math.sqrt(var_s0)  # 30 log-odds per score sd
        sol, level = _set_strength(lambda lam, near: _fit_binary(Zc, y, w0, kappa * lam * S, a0, b_hi, s_b, near), lam_ref, lam_cap)
        (_, b1), H1, _ = _newton_logistic(np.column_stack([np.ones(n), s0c]), y, np.array([a0, 0.0]), np.diag([1 / 2.5 ** 2, var_s0 / 2.5 ** 2]))
        z1 = b1 / math.sqrt(np.linalg.inv(H1)[1, 1])
        stats.update({"slope_on_old_score": round(float(b1), 4), "slope_z": round(float(z1), 2), "base_rate": round(ybar, 3)})
        noise = math.pi ** 2 / 3
    floor, ceiling = level == "floor", level == "cap"
    b, u, Lu = float(sol["b"]), sol["u"], sol["Lu"]
    boundary = b <= 1e-6 * b_hi  # the best scale is 0: the metric does not rise with any score near the old one
    t1 = stats.get("slope_t") if kind == "continuous" else stats.get("slope_z")
    if boundary and n >= 10 and t1 is not None and t1 < -2:
        raise LearnError(f"the metric goes down as the score goes up in these {n} posts (slope t = {t1}): they contradict the "
                         "formula, so its weights cannot be refit from them. Check the ids and the metric column.")
    prior_cov_u = np.linalg.inv(Lu)
    prior_sd = np.sqrt(np.clip(np.diag(BASIS @ prior_cov_u @ BASIS.T), 0.0, None))
    cov = None
    if not boundary:
        try:
            full = np.linalg.inv(sol["H"])
            cov = full[1:, 1:] if kind == "binary" else full
            if np.any(np.linalg.eigvalsh(cov) <= 0):
                cov = None
        except np.linalg.LinAlgError:
            cov = None
    raw = w0 + BASIS @ u
    new = project(raw)
    sd = np.sqrt(np.clip(np.diag(BASIS @ cov[1:, 1:] @ BASIS.T), 0.0, None)) if cov is not None else prior_sd
    vq = float(np.var(Zc @ raw)) if n > 1 else 0.0
    r2 = b * b * vq / (b * b * vq + noise) if noise > 0 else 0.0
    data_share = float(np.trace(sol["info"]) / np.trace(sol["info"] + Lu))
    intercept = (float(y.mean()) if kind == "continuous" else sol["a"]) - float(zbar @ (b * raw))
    clipped = [t for t, x in zip(TERMS, raw) if x < 0]
    if kind == "continuous":
        stats.update({"r_old_score": _corr(Z @ w0, y), "r_new_score_in_sample": _corr(Z @ new, y)})
    else:
        stats.update({"auc_old_score": _auc(Z @ w0, y), "auc_new_score_in_sample": _auc(Z @ new, y)})
    warnings: list[str] = []
    if n < 30:
        warnings.append(f"only {n} posts: the result is still mostly the prior (the old weights count as {kappa:g} posts); "
                        "treat it as a direction to watch, not a new formula")
    if boundary:
        warnings.append("the metric did not rise with the score in these posts, so there is nothing to refit: the weights stay as they are")
    elif floor:
        warnings.append(f"the score explains under {R2_REF:.0%} of the metric here (fit R2 {r2:.2f}), so the prior was held at its "
                        f"reference strength: the weights moved less than n/(n+{kappa:g}) of the way")
    if clipped:
        warnings.append(f"{', '.join(clipped)}: the fit pushed the weight below 0; set to 0 (the data say it does not help)")
    if not boundary and np.all(np.abs(new - w0) <= 2 * np.maximum(sd, 1e-9)):
        warnings.append("no weight moved by more than 2 posterior sd: the posts do not show a reliable change yet")
    return Fit(kind=kind, n=n, prior_strength=kappa, old=w0, new=new, sd=sd, prior_sd=prior_sd, intercept=intercept, scale=b,
               r2=float(r2), floor=bool(floor), data_share=data_share, clipped=clipped, stats=stats, warnings=warnings,
               mode=np.concatenate([[b], u]), cov=cov, prior_cov_u=prior_cov_u)


# ---------------------------------------------------------------------------------------------------------------- plan
def _window(c: Clip, block: str):
    from .models import Segment, Window
    flags = {"block": block, "episode": c.episode}
    if c.hook_type:
        flags["hook_type"] = c.hook_type
    return Window(cand_id=c.cand_id, segments=[Segment(episode=e, start_s=s, end_s=t) for e, s, t in c.segments],
                  hook_beat=c.hook_beat or f"?hook:{c.cand_id}", cliff_beat=c.cliff_beat or f"?cliff:{c.cand_id}",
                  beat_ids=[], duration_s=sum(t - s for _, s, t in c.segments), kind=c.kind, flags=flags)


def _same_promo(a, b, thr: float) -> bool:
    """rank.nms's rule: >= thr time overlap, or a shared opening/closing line."""
    from .rank import iou
    return a.cand_id == b.cand_id or iou(a, b) >= thr or bool({a.hook_beat, a.cliff_beat} & {b.hook_beat, b.cliff_beat})


def plan(res: Fit, pool: Pool, k: int, posted: list[Clip], draws: int = 1000, seed: int = 7) -> dict:
    """Thompson sampling at the slate level: post the slate built under ONE posterior draw of the weights; report, for
    each pick, the share of `draws` posterior draws whose slate also contains it and whose top clip it is."""
    from . import slate as sl
    from .rank import diversity_order
    cfg = pool.cfg
    eps = [c.episode for c in pool.clips]
    lo, hi = min(eps), max(eps)
    thr = float(cfg["scoring"]["nms_iou"])
    posted_w = [_window(c, c.block or "") for c in posted]
    elig = [c for c in pool.clips if not any(_same_promo(_window(c, ""), p, thr) for p in posted_w)]
    out = {"k": k, "draws": draws, "seed": seed, "pool": len(pool.clips), "eligible": len(elig),
           "excluded_as_posted": len(pool.clips) - len(elig), "rows": [],
           "rules": "near-duplicate removal (>= {:.0%} overlap or a shared opening/closing line), opening/middle/late quotas "
                    "{}, at most {} clips per episode in the top 10, x{:g} per clip already taken from an episode, x{:g} for the "
                    "same hook type twice in a row".format(thr, "/".join(f"{float(cfg['slate']['shares'][b]):.0%}" for b in ("opening", "middle", "late")),
                                                           cfg["scoring"]["max_per_episode_top10"], 1 - cfg["scoring"]["same_episode_penalty"],
                                                           1 - cfg["scoring"]["same_type_penalty"]),
           "out_of_scope": "the first-time-viewer judge (picks marked judged=no have only the standalone proxy: run the judge "
                           "on them before posting), the promise ledger, the spoiler flag (shown, not enforced)"}
    if not elig or k <= 0:
        return out
    wins = [_window(c, c.block or sl.block_of(c.episode, lo, hi, cfg)) for c in elig]
    idx = {w.cand_id: i for i, w in enumerate(wins)}
    N = len(elig)
    ov = np.array([[_same_promo(a, b, thr) for b in wins] for a in wins])
    X = np.array([c.terms for c in elig])
    m = np.array([c.mult for c in elig])

    def slate_under(w: np.ndarray) -> tuple[list[int], int]:
        sc = m * (X @ w)
        order = np.argsort(-sc, kind="stable")
        supp = np.zeros(N, dtype=bool)
        kept = []
        for i in order:
            if not supp[i]:
                kept.append(int(i))
                supp |= ov[i]
        for i in kept:
            wins[i].score = float(sc[i])
        return [idx[w.cand_id] for w in sl.build([wins[i] for i in kept], cfg, k, diversity_order)], int(order[0])

    map_slate, _ = slate_under(res.new)
    rng = np.random.default_rng(seed)
    ws, rejected = np.zeros((0, 4)), 0
    for _ in range(20):  # draws with b <= 0 (a score that would not help) are dropped; top up until there are enough
        got, rej = res.sample(rng, draws)
        ws, rejected = np.vstack([ws, got]), rejected + rej
        if len(ws) >= draws:
            break
    ws = ws[:draws]
    in_slate, top = np.zeros(N), np.zeros(N)
    ts_slate = None
    for w in ws:
        picks, best = slate_under(w)
        in_slate[picks] += 1
        top[best] += 1
        if ts_slate is None:  # Thompson sampling: post the slate of the first draw
            ts_slate = picks
    valid = len(ws)
    out["valid_draws"] = valid
    out["rejected_draws"] = rejected
    if ts_slate is None:  # no draw kept the score useful: fall back to the posterior mode
        ts_slate = map_slate
        out["note"] = "no posterior draw had a positive slope; the plan is the posterior mode's slate"
    map_sc = m * (X @ res.new)
    map_rank = {int(i): r for r, i in enumerate(np.argsort(-map_sc, kind="stable"), 1)}
    for slot, i in enumerate(ts_slate, 1):
        c = elig[i]
        out["rows"].append({"slot": slot, "cand_id": c.cand_id, "promo_id": c.promo_id, "episodes": [s[0] for s in c.segments],
                            "block": wins[i].flags.get("block"), "kind": c.kind, "score_new": round(float(map_sc[i]), 4),
                            "rank_new": map_rank[i], "score_old": round(float(m[i] * X[i] @ res.old), 4),
                            "in_slate_share": round(in_slate[i] / max(valid, 1), 3), "top_share": round(top[i] / max(valid, 1), 3),
                            "pick": "exploit" if i in map_slate else "explore", "judged": c.judged, "spoiler_risk": c.spoiler_risk})
    out["map_slate"] = [elig[i].cand_id for i in map_slate]
    return out


# -------------------------------------------------------------------------------------------------------------- config
def _rounded(w: np.ndarray, places: int = 4) -> list[float]:
    """Round to `places` decimals so the rounded weights still sum to exactly 1 (largest remainder)."""
    unit = 10 ** places
    raw = np.asarray(w) * unit
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q))[: unit - int(q.sum())]:
        q[i] += 1
    return [int(x) / unit for x in q]


def check_out_path(out: Path, base: Path) -> None:
    if out.name == "config.yaml" or (out.exists() and base.exists() and out.resolve() == base.resolve()):
        raise LearnError(f"--out {out} would overwrite a config.yaml: write the learned config to a new file")


def write_config(out: Path, base: Path, res: Fit, provenance: dict) -> list[float]:
    """Copy `base` (comments kept), change only scoring.w_*, and put a provenance header on top."""
    out, base = Path(out), Path(base)
    check_out_path(out, base)
    new = _rounded(res.new)
    text = base.read_text(encoding="utf-8")
    for key, val in zip(WEIGHT_KEYS, new):
        text, hits = re.subn(rf"^(\s*{key}\s*:\s*)[-+0-9.eE]+", lambda mo, v=val: f"{mo.group(1)}{v:.4f}", text, count=1, flags=re.M)
        if hits != 1:
            break
    cfg = yaml.safe_load(text) or {}
    if any(abs(float(cfg.get("scoring", {}).get(k, -1)) - v) > 1e-9 for k, v in zip(WEIGHT_KEYS, new)):
        cfg = yaml.safe_load(base.read_text(encoding="utf-8"))  # unusual layout: rewrite it (comments are lost)
        cfg.setdefault("scoring", {}).update({k: v for k, v in zip(WEIGHT_KEYS, new)})
        text = yaml.safe_dump(cfg, sort_keys=False)
    how = "linear model of the metric (ridge toward the prior)" if res.kind == "continuous" else "logistic model of the metric"
    head = [f"Learned ranking weights, written by `promo-finder learn` on {dt.datetime.now():%Y-%m-%d %H:%M}.",
            f"Only scoring.{', '.join(WEIGHT_KEYS)} differ from the base config ({base.name}).",
            f"Run: {provenance.get('run', '?')}",
            f"Outcomes: {provenance.get('outcomes', '?')} · metric {provenance.get('metric', '?')} ({res.kind}) · {res.n} posts used"
            + (f" · {provenance['unknown']} unknown id{'s' if provenance['unknown'] != 1 else ''} skipped" if provenance.get("unknown") else ""),
            f"Prior: the run's weights, worth {res.prior_strength:g} posts. Fit: MAP, {how}; Laplace posterior sd.",
            *[f"  {k:<13} {o:.3f} -> {v:.4f}  (sd {s:.3f})" for k, o, v, s in zip(WEIGHT_KEYS, res.old, new, res.sd)],
            *[f"Warning: {w}" for w in res.warnings],
            "Use: promo-finder run <show> --config <this file>"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"# {h}\n" for h in head) + "\n" + text, encoding="utf-8")
    return new
