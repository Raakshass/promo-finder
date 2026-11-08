"""Do the ranking coefficients match what professional editors pick? An offline check on DramaAD.

The ranking (promo_finder/rank.py) is
    score = (w_hook*hook + w_cliff*cliff + w_standalone*standalone + w_density*density)
            x a penalty per failed editor's check x 1.05 turning-point episode x 0.9 two-segment clip
with hand-set priors w = 0.35 / 0.30 / 0.25 / 0.10 (config.yaml). This script asks whether professional editors' ads
back those priors. Input: promo-finder runs on DramaAD episodes (made with --no-judge, so `standalone` here is the code
proxy, not the first-time viewer's rating) and the DramaAD ground truth (script.json: which seconds each ad uses).

  1. Weight sweep. Rebuild every candidate window offline (stability.replay -> generate_windows -> score_window ->
     flag_spoilers; no model calls), re-score under each weight vector (rank.combine with that vector; penalties and
     multipliers unchanged), suppress overlaps (rank.nms), and score the top-k against the ads exactly as
     scripts/benchmark_dramaad.py does (lift, within_lift, hit@k), straight from the candidates and through the slate
     (its --source candidates / --source slate). Vectors: official; each weight +-0.05 and +-0.10 (renormalized; the
     ranking does not depend on the weights' sum); leave one term out; each term alone; equal; Dirichlet(1,1,1,1).
     Ties are broken by generation order, as in the pipeline; named vectors are re-run with shuffled tie-breaks.
  2. Per-term evidence, per window: Spearman correlation of each term with editor usage (per drama), and a logistic
     regression of hit (>= 50% of the clip inside some professional ad) on the four standardized terms, pooled
     (with a per-drama intercept) and per drama, with bootstrap 95% CIs that resample whole episodes (numpy only).
  3. Overfitting: the vector that maximizes lift on one drama, tested on the other, and the reverse.
  4. Prior values: do the per-term intervals exclude the prior weights themselves (not only 0)? Uncorrected 95% and
     Bonferroni-corrected intervals (4 terms; a stricter 8-test version too), the joint likelihood-ratio test, and a
     seed x ridge grid. Plus an extra robustness view: a control for whether any ad uses the window's episode.

  uv run python scripts/weights_vs_editors.py --dramaad <DramaAD dir> --runs <run dir> <run dir> \
      --published <official run>/eval/external_benchmark_slate.json <official run>/eval/external_benchmark_moments.json \
      --out <results dir>

Writes <out>/editors.json and <out>/editors.md. Reads local files only: no model calls, no network.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # importing scripts/benchmark_dramaad.py must not leave a cache folder behind
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_dramaad as bd  # noqa: E402
from promo_finder import slate as sl  # noqa: E402
from promo_finder.cli import load_config  # noqa: E402
from promo_finder.rank import combine, diversity_order, flag_spoilers, generate_windows, nms, score_window  # noqa: E402
from promo_finder.stability import replay  # noqa: E402

TERMS = ("hook", "cliff", "standalone", "density")
SOURCES = ("candidates", "slate")
KS = (10, 20)
KEYS = tuple(f"{s}@{k}" for s in SOURCES for k in KS)
METRICS = ("lift", "within_lift", "hit_rate")
GENRE_EN = {"现代言情": "modern romance", "穿越": "time travel"}


# ---------------------------------------------------------------- weight vectors

def normalize(w) -> tuple[float, ...]:
    s = sum(w)
    return tuple(float(x) / s for x in w)


def weight_vectors(official: tuple[float, ...], n_dirichlet: int, seed: int) -> list[dict]:
    vecs = [{"name": "official", "family": "official", "w": official}]
    for j, t in enumerate(TERMS):
        for d in (-0.10, -0.05, 0.05, 0.10):
            w = list(official)
            w[j] = max(0.0, w[j] + d)
            vecs.append({"name": f"{t} {d:+.2f}", "family": "one_at_a_time", "w": normalize(w)})
    for j, t in enumerate(TERMS):
        w = list(official)
        w[j] = 0.0
        vecs.append({"name": f"without {t}", "family": "leave_one_out", "w": normalize(w)})
    for j, t in enumerate(TERMS):
        vecs.append({"name": f"only {t}", "family": "single_term", "w": tuple(1.0 if i == j else 0.0 for i in range(4))})
    vecs.append({"name": "equal", "family": "equal", "w": (0.25, 0.25, 0.25, 0.25)})
    rng = np.random.default_rng(seed)
    for i, w in enumerate(rng.dirichlet(np.ones(4), size=n_dirichlet)):
        vecs.append({"name": f"dirichlet {i:03d}", "family": "dirichlet", "w": tuple(float(x) for x in w)})
    return vecs


def with_weights(cfg: dict, w) -> dict:
    c = copy.deepcopy(cfg)
    for t, v in zip(TERMS, w):
        c["scoring"][f"w_{t}"] = float(v)
    return c


# the config sections that decide which windows exist and how they are scored and ranked (judge.top_n = finalist count)
RANKING_SECTIONS = ("promo", "candidates", "scoring", "rubric", "slate", "judge.top_n", "output")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_diff(run_cfg: dict, repo_cfg: dict) -> dict:
    """Which settings of a run's config.yaml differ from the repository's, split into ranking-relevant and the rest."""
    fr, fc, miss = _flat(repo_cfg), _flat(run_cfg), object()
    diff = sorted(k for k in set(fr) | set(fc) if fr.get(k, miss) != fc.get(k, miss))
    rank = [k for k in diff if any(k == s or k.startswith(s + ".") for s in RANKING_SECTIONS)]
    return {"ranking_sections_compared": list(RANKING_SECTIONS), "ranking_relevant_differing_keys": rank,
            "other_differing_keys": [k for k in diff if k not in rank]}


# ---------------------------------------------------------------- one benchmark drama, replayed offline

class Drama:
    """A promo-finder run on DramaAD episodes, replayed with zero model calls, with editor usage attached per window."""

    def __init__(self, run: Path, dramaad: Path, n_eps: int):
        self.run, self.id = run, run.parent.name
        self.cfg = load_config(run / "config.yaml")
        mnu, self.ads = bd.load(dramaad / self.id)
        self.info = mnu["drama_info"]
        self.total_episodes = len(mnu["episode_info"])
        self.dur: dict[int, float] = {}
        for kk, ep in mnu["episode_info"].items():  # as benchmark_dramaad.cmd_score
            dia, scenes = bd.episode_lines(ep)
            self.dur[int(kk)] = max([t1 for _, t1, _, _ in dia] + [t1 for _, t1, _, _ in scenes] + [0.0])
        self.n_eps = n_eps
        self.eps = set(range(1, n_eps + 1))
        self.u = bd.usage_map(self.ads, self.eps)
        secs = lambda e: range(int(self.dur.get(e, 0)) + 1)  # noqa: E731
        self.mean_all = st.mean(self.u.get(e, {}).get(t, 0) for e in self.eps for t in secs(e))
        self.ep_mean = {e: st.mean([self.u.get(e, {}).get(t, 0) for t in secs(e)] or [0]) for e in self.eps}
        self.partial = {e for e in self.eps if len({self.u.get(e, {}).get(t, 0) for t in secs(e)}) > 1}
        self.ctx = replay(run)
        if len(self.ctx.episodes) != n_eps:
            raise SystemExit(f"{self.id}: the run has {len(self.ctx.episodes)} episodes, --episodes says {n_eps}")
        self.windows = generate_windows(self.ctx, self.cfg, "llm")
        for w in self.windows:
            score_window(w, self.ctx, self.cfg, "llm")
        flag_spoilers(self.windows, self.ctx, self.cfg)
        if len({w.cand_id for w in self.windows}) != len(self.windows):
            raise SystemExit(f"{self.id}: duplicate candidate ids")
        self.label = {w.cand_id: self.clip_label(w) for w in self.windows}

    @staticmethod
    def segs(w) -> list[dict]:
        # rounded to 2 decimals, as candidates.jsonl and promos.json store them (the benchmark reads those files)
        return [{"episode": g.episode, "start_s": round(g.start_s, 2), "end_s": round(g.end_s, 2)} for g in w.segments]

    def clip_label(self, w) -> dict:
        """The benchmark's per-clip quantities (benchmark_dramaad.evaluate_run.stats_for), for one window."""
        secs = [(e, t) for e, t in bd.clip_seconds(self.segs(w)) if e in self.eps]
        vals = [self.u.get(e, {}).get(t, 0) for e, t in secs]
        mean_u = st.mean(vals)
        base = st.mean(self.ep_mean[e] for e, _ in secs)
        covered = sum(1 for v in vals if v > 0) / len(vals)
        ep = w.segments[0].episode
        used = any(self.ep_mean[e] > 0 for e in {e for e, _ in secs})  # some ad uses an episode this clip touches
        return {"usage": mean_u, "lift": mean_u / self.mean_all if self.mean_all else 0.0, "within": mean_u / base if base else 1.0,
                "covered": covered, "hit": covered >= 0.5, "episode": ep, "partial_episode": ep in self.partial, "episode_used": used}

    def metrics(self, clips: list) -> dict:
        labs = [self.label[w.cand_id] for w in clips]
        if not labs:
            return {"lift": None, "within_lift": None, "hit_rate": None, "n": 0}
        return {"lift": st.mean(l["lift"] for l in labs), "within_lift": st.mean(l["within"] for l in labs),
                "hit_rate": sum(1 for l in labs if l["hit"]) / len(labs), "n": len(labs)}

    def rank(self, w, order: list[int] | None = None) -> tuple[list, list]:
        """Re-score every window under weights w, then the pipeline's own NMS and (--no-judge) slate."""
        cfg = with_weights(self.cfg, w)
        for win in self.windows:
            combine(win, cfg)
        pool = self.windows if order is None else [self.windows[i] for i in order]
        kept = nms(pool, cfg)  # best first; ties keep their input order (stable sort), as in the pipeline
        finalists = sl.build(kept, cfg, cfg["judge"]["top_n"], diversity_order)
        slate = sl.build(finalists, cfg, cfg["output"]["final_top_n"], diversity_order)
        return kept, slate

    def evaluate(self, w, order: list[int] | None = None) -> dict:
        kept, slate = self.rank(w, order)
        out = {"kept": len(kept), "slate": len(slate)}
        lists = {"candidates": kept, "slate": slate}
        for s in SOURCES:
            for k in KS:
                out[f"{s}@{k}"] = self.metrics(lists[s][:k])
                out[f"ids {s}@{k}"] = [x.cand_id for x in lists[s][:k]]
        out["tie_at_k"] = {k: bool(len(kept) > k and kept[k - 1].score == kept[k].score) for k in KS}
        return out

    def rows(self, population: list) -> list[dict]:
        """One row per window: the four terms, the official base and score, the editors' usage."""
        out = []
        for w in population:
            lab = self.label[w.cand_id]
            out.append({"drama": self.id, "cand_id": w.cand_id, "episode": lab["episode"], "kind": w.kind,
                        "block": w.flags.get("block"), "position": (lab["episode"] - 1) / max(self.n_eps - 1, 1),
                        **{t: float(w.subscores[t]) for t in TERMS}, "base": float(w.subscores["base"]), "score": float(w.score),
                        "dialogue_share": float(w.subscores.get("dialogue_share", 0.0)), "duration_s": float(w.duration_s),
                        "usage": lab["usage"], "within": lab["within"], "covered": lab["covered"], "hit": int(lab["hit"]),
                        "partial_episode": lab["partial_episode"], "episode_used": int(lab["episode_used"])})
        return out

    def describe(self, official) -> dict:
        ads_in = {a: [s for s in segs if s["epid"] in self.eps] for a, segs in self.ads.items()}
        ad_len = [sum(s["end"] - s["start"] for s in segs) / 1000 for segs in self.ads.values()]
        seg_len = [(s["end"] - s["start"]) / 1000 for segs in self.ads.values() for s in segs]
        seg_in = [(s["end"] - s["start"]) / 1000 for segs in ads_in.values() for s in segs]
        secs = [self.u.get(e, {}).get(t, 0) for e in sorted(self.eps) for t in range(int(self.dur.get(e, 0)) + 1)]
        kept, slate = self.rank(official)
        dia = [ev.text for ep in self.ctx.episodes for ev in ep.events if ev.kind == "dialogue"]
        cjk = sum(1 for s in dia for ch in s if "一" <= ch <= "鿿")
        latin = sum(1 for s in dia for ch in s if ch.isascii() and ch.isalpha())
        log = json.loads((self.run / "run_log.json").read_text())
        kinds: dict[str, int] = {}
        for w in self.windows:
            kinds[w.kind] = kinds.get(w.kind, 0) + 1
        lo, hi = sl.episode_range(self.ctx)
        opening_eps = [e for e in range(lo, hi + 1) if sl.block_of(e, lo, hi, self.cfg) == "opening"]
        raw = lambda gid, key: str(int(round(getattr(self.ctx.score(gid, "llm"), key))))  # noqa: E731
        count = lambda vals: dict(sorted({v: vals.count(v) for v in set(vals)}.items()))  # noqa: E731
        return {
            "drama": self.id, "genre": GENRE_EN.get(self.info.get("category"), self.info.get("category")),
            "episodes_in_drama": self.total_episodes, "episodes_evaluated": self.n_eps,
            "evaluated_minutes": round(sum(self.dur[e] for e in self.eps) / 60, 1),
            "ads": len(self.ads), "ads_using_evaluated_episodes": sum(1 for v in ads_in.values() if v),
            "ad_segments": len(seg_len), "ad_segments_in_evaluated_episodes": len(seg_in),
            "ad_length_s": {"min": round(min(ad_len), 1), "median": round(st.median(ad_len), 1), "max": round(max(ad_len), 1)},
            "ad_segment_length_s": {"min": round(min(seg_len), 1), "median": round(st.median(seg_len), 1), "max": round(max(seg_len), 1)},
            "ad_segment_length_in_evaluated_episodes_s": {"median": round(st.median(seg_in), 1)} if seg_in else {},
            "share_of_evaluated_seconds_in_some_ad": round(sum(1 for v in secs if v > 0) / len(secs), 3),
            "mean_usage_per_second": round(st.mean(secs), 3), "max_usage_per_second": max(secs),
            "episodes_with_partial_use": len(self.partial),
            "episodes_fully_used_or_unused": sorted(e for e in self.eps if e not in self.partial),
            "episodes_never_used": sorted(e for e in self.eps if self.ep_mean[e] == 0),
            "candidate_windows": len(self.windows), "candidate_windows_by_kind": kinds,
            "episodes_with_candidates": len({w.segments[0].episode for w in self.windows}),
            "kept_after_nms_official": len(kept), "slate_size": len(slate),
            "kept_hit_share_official": round(st.mean(self.label[w.cand_id]["hit"] for w in kept), 3),
            "candidate_duration_s_median": round(st.median(w.duration_s for w in self.windows), 1),
            "dialogue_lines": len(dia), "dialogue_cjk_share_of_letters": round(cjk / max(cjk + latin, 1), 3),
            "model": log["args"].get("model"), "judge_used": not log["args"].get("no_judge", False),
            "episodes_on_heuristic_fallback": len(log.get("llm_failures") or {}),
            "opening_block_episodes": opening_eps,
            "slate_quotas": sl.quotas(self.cfg["output"]["final_top_n"], self.cfg),
            "physical_hook_bonus_windows": sum(1 for w in self.windows if w.flags.get("physical_hook")),
            "writer_cue_bonus_windows": sum(1 for w in self.windows if w.flags.get("writer_cue")),
            "romance_cap_windows": sum(1 for w in self.windows if w.flags.get("romance_cap")),
            "model_hook_score_of_opening_line": count([raw(w.hook_beat, "hook") for w in self.windows]),
            "model_cliff_score_of_closing_line": count([raw(w.cliff_beat, "cliff") for w in self.windows]),
            "official_slate_top10_blocks": {b: sum(1 for w in slate[:10] if w.flags.get("block") == b) for b in sl.BLOCKS},
            "official_slate_top20_blocks": {b: sum(1 for w in slate[:20] if w.flags.get("block") == b) for b in sl.BLOCKS},
            "official_candidates_top10_blocks": {b: sum(1 for w in kept[:10] if w.flags.get("block") == b) for b in sl.BLOCKS},
            "config_vs_repo": config_diff(self.cfg, load_config(ROOT / "config.yaml")),
        }


# ---------------------------------------------------------------- statistics (numpy only)

def rankdata(x) -> np.ndarray:
    """Average ranks (0-based); tied values share their mean rank, as in benchmark_dramaad.spearman."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    return ((end - counts + end - 1) / 2.0)[inv]


def spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def ci(samples, level: float = 0.95, nd: int | None = 3) -> list[float] | None:
    """Percentile interval; nd=None keeps full precision (used when a bound is compared with a prior)."""
    s = np.asarray([v for v in samples if v is not None and np.isfinite(v)], dtype=float)
    if len(s) < 10:
        return None
    lo, hi = np.percentile(s, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)] if nd is None else [round(float(lo), nd), round(float(hi), nd)]


class Clusters:
    """Bootstrap that resamples whole episodes (with replacement) within each drama: windows from one episode share
    that episode's coarse labels, so they are not independent observations."""

    def __init__(self, rows: list[dict]):
        groups: dict[str, dict[int, list[int]]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(r["drama"], {}).setdefault(r["episode"], []).append(i)
        self.groups = [[np.array(v) for _, v in sorted(g.items())] for _, g in sorted(groups.items())]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        out = []
        for eps in self.groups:
            for p in rng.integers(0, len(eps), size=len(eps)):
                out.append(eps[p])
        return np.concatenate(out)


def logit_fit(X: np.ndarray, y: np.ndarray, penalized: np.ndarray, lam: float, iters: int = 200) -> tuple[np.ndarray, bool, float]:
    """Logistic regression by Newton steps with an L2 penalty lam on the penalized columns (a tiny 1e-6 on the others
    keeps the Hessian invertible when a drama's labels are all one class). Returns (coefficients, converged, log-lik)."""
    beta = np.zeros(X.shape[1])
    pen = np.where(penalized, lam, 1e-6)
    converged = False
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (y - mu) - pen * beta
        H = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
    mu = np.clip(1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return beta, converged, ll


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution, closed forms for df 1-3 (all the tests here need)."""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    if df == 3:
        return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    raise ValueError(df)


def col_stats(rows: list[dict], cols: list[str]) -> dict:
    out: dict[str, dict] = {}
    for d in sorted({r["drama"] for r in rows}):
        sub = [r for r in rows if r["drama"] == d]
        out[d] = {c: (float(np.mean([r[c] for r in sub])), float(np.std([r[c] for r in sub]))) for c in cols}
    return out


def design(rows: list[dict], cols: list[str], dramas: list[str], stats: dict, position: bool = False):
    """Intercept per drama + within-drama z-scores of cols (+ optionally position and position^2, per drama).
    A column that is constant within a drama gets z = 0 there. Every column but the intercepts is ridge-penalized."""
    n = len(rows)
    names, X = [], []
    for d in dramas:
        names.append(f"intercept {d}")
        X.append(np.array([1.0 if r["drama"] == d else 0.0 for r in rows]))
    for c in cols:
        v = np.zeros(n)
        for i, r in enumerate(rows):
            m, s = stats[r["drama"]][c]
            v[i] = (r[c] - m) / s if s > 0 else 0.0
        names.append(c)
        X.append(v)
    if position:
        for d in dramas:
            for pc in ("position", "position2"):
                v = np.zeros(n)
                for i, r in enumerate(rows):
                    if r["drama"] == d:
                        m, s = stats[d][pc]
                        v[i] = (r[pc] - m) / s if s > 0 else 0.0
                names.append(f"{pc} {d}")
                X.append(v)
    penalized = np.array([not nm.startswith("intercept") for nm in names])
    return np.column_stack(X), names, penalized


def _rel(b) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    s = np.sum(np.abs(b))
    return b / s if s > 0 else np.full(len(b), np.nan)


def logistic_analysis(rows: list[dict], dramas: list[str], lam: float, boot: int, seed: int, position: bool = False,
                      extra: tuple[str, ...] = ()) -> dict:
    """hit ~ intercept per drama + the four within-drama standardized terms (+ position, + extra control columns such as
    episode_used), episode-cluster bootstrap CIs. The bootstrap draws are kept under '_draws' (stripped before saving)."""
    for r in rows:
        r["position2"] = r["position"] ** 2
    stats = col_stats(rows, list(TERMS) + ["position", "position2"] + list(extra))
    X, names, pen = design(rows, list(TERMS) + list(extra), dramas, stats, position)
    y = np.array([r["hit"] for r in rows], dtype=float)
    tc = [names.index(t) for t in TERMS]
    beta, conv, ll = logit_fit(X, y, pen, lam)
    beta_mle, conv_mle, ll_mle = logit_fit(X, y, pen, 0.0)
    separation = (not conv_mle) or bool(np.max(np.abs(beta_mle[tc])) > 15)
    # the weights in config.yaml act on the raw 0-1 terms: implied weight = coefficient / SD (SD pooled over the dramas)
    sd = np.array([math.sqrt(float(np.mean([stats[d][t][1] ** 2 for d in dramas]))) for t in TERMS])

    def raw_rel(b):
        return _rel(np.where(sd > 0, np.asarray(b) / np.where(sd > 0, sd, 1.0), 0.0))

    rng = np.random.default_rng(seed)
    cl = Clusters(rows)
    bs, fails = [], 0
    for _ in range(boot):
        idx = cl.draw(rng)
        b, ok, _ = logit_fit(X[idx], y[idx], pen, lam)
        fails += 0 if ok else 1
        bs.append(b[tc])
    bs = np.array(bs)
    rel_bs = np.array([_rel(b) for b in bs])
    raw_bs = np.array([raw_rel(b) for b in bs])
    rel0, raw0 = _rel(beta[tc]), raw_rel(beta[tc])
    terms = {}
    for j, t in enumerate(TERMS):
        terms[t] = {
            "coef_per_sd": round(float(beta[tc[j]]), 3), "ci95": ci(bs[:, j]),
            "share_of_bootstrap_above_0": round(float(np.mean(bs[:, j] > 0)), 3),
            "coef_per_sd_unpenalized": round(float(beta_mle[tc[j]]), 3),
            "odds_ratio_per_sd": round(math.exp(float(beta[tc[j]])), 3),
            "relative_importance": round(float(rel0[j]), 3), "relative_importance_ci95": ci(rel_bs[:, j]),
            "implied_weight_raw_scale": round(float(raw0[j]), 3), "implied_weight_raw_scale_ci95": ci(raw_bs[:, j]),
            "sd": round(float(sd[j]), 4), "constant": bool(sd[j] == 0),
        }
    pos = [i for i, nm in enumerate(names) if nm.startswith("position")]
    ex = [names.index(c) for c in extra]
    return {"n": len(rows), "hits": int(y.sum()), "dramas": dramas, "position_control": position, "extra_controls": list(extra),
            "ridge_lambda": lam, "seed": seed, "terms": terms, "position_coefs": {names[i]: round(float(beta[i]), 3) for i in pos},
            "extra_coefs": {names[i]: round(float(beta[i]), 3) for i in ex},
            "converged": conv, "unpenalized_converged": conv_mle, "unpenalized_separation": separation,
            "bootstrap": boot, "bootstrap_nonconverged": fails, "log_likelihood": round(ll, 3), "log_likelihood_unpenalized": round(ll_mle, 3),
            "_draws": {"coef_per_sd": bs, "relative_importance": rel_bs, "implied_weight_raw_scale": raw_bs,
                       "point": {"coef_per_sd": beta[tc], "relative_importance": rel0, "implied_weight_raw_scale": raw0}}}


# Bonferroni levels for the prior-value test: 4 terms; and a stricter 8 (4 terms x 2 scales, as an independent check did)
LEVELS = {"uncorrected": 0.95, "bonferroni_4": 1 - 0.05 / 4, "bonferroni_8": 1 - 0.05 / 8}
SCALES = (("weight_0_1", "implied_weight_raw_scale", "0-1 scale"), ("importance_per_sd", "relative_importance", "per SD"))


def value_robustness(fits: dict[str, dict], official: dict) -> dict:
    """The uncorrected value test again on each variant fit (position control, single-episode windows, episode-used
    control, ...). Per-SD priors use each fit's own term SDs."""
    out = {}
    for lab, fit in fits.items():
        sds = np.array([fit["terms"][t]["sd"] for t in TERMS])
        w = np.array([official[t] for t in TERMS])
        pi = w * sds / np.sum(w * sds)
        out[lab] = {"n": fit["n"], "hits": fit["hits"], "terms": {}}
        for j, t in enumerate(TERMS):
            out[lab]["terms"][t] = {}
            for scale, key, _ in SCALES:
                prior = official[t] if scale == "weight_0_1" else float(pi[j])
                c = ci(fit["_draws"][key][:, j], LEVELS["uncorrected"], None)
                out[lab]["terms"][t][scale] = {"prior": round(prior, 3), "point": round(float(fit["_draws"]["point"][key][j]), 3),
                                               "ci95": [round(x, 3) for x in c] if c else None, "excludes_prior_95": _excl(c, prior)}
    return out


def _excl(c, prior) -> str | None:
    return None if not c else ("below" if c[1] < prior else ("above" if c[0] > prior else None))


def prior_value_test(fit: dict, fit_corr: dict, official: dict, prior_imp: dict) -> dict:
    """Do the per-term intervals exclude the prior VALUES (not just 0)?  On the 0-1 scale the prior is the weight itself
    (the official weights sum to 1); per SD it is weight x SD, normalized.  Uncorrected 95% from the main bootstrap
    (the intervals shown in the tables); Bonferroni-corrected from a larger bootstrap (fit_corr), whose first draws are
    the main ones (same seed).  Bounds are compared unrounded."""
    out: dict = {"levels": LEVELS, "draws_uncorrected": fit["bootstrap"], "draws_corrected": fit_corr["bootstrap"], "terms": {}}
    for t in TERMS:
        j = TERMS.index(t)
        e = {}
        for scale, key, _ in SCALES:
            prior = official[t] if scale == "weight_0_1" else prior_imp[t]
            c95 = ci(fit["_draws"][key][:, j], LEVELS["uncorrected"], None)
            cc = {lv: ci(fit_corr["_draws"][key][:, j], LEVELS[lv], None) for lv in LEVELS}
            e[scale] = {"prior": round(prior, 4), "point": round(float(fit["_draws"]["point"][key][j]), 3),
                        "ci95": [round(x, 3) for x in c95] if c95 else None, "excludes_prior_95": _excl(c95, prior),
                        **{f"ci_{lv}_{fit_corr['bootstrap']}draws": [round(x, 3) for x in c] if c else None for lv, c in cc.items()},
                        **{f"excludes_prior_{lv}_{fit_corr['bootstrap']}draws": _excl(c, prior) for lv, c in cc.items()},
                        "share_of_bootstrap_below_prior": round(float(np.mean(fit["_draws"][key][:, j] < prior)), 3)}
        out["terms"][t] = e
    nb = fit_corr["bootstrap"]
    out["excluded_uncorrected"] = {t: {s: v[s]["excludes_prior_95"] for s, _, _ in SCALES} for t, v in out["terms"].items()
                                   if any(v[s]["excludes_prior_95"] for s, _, _ in SCALES)}
    for lv in ("bonferroni_4", "bonferroni_8"):
        out[f"excluded_{lv}"] = {t: {s: v[s][f"excludes_prior_{lv}_{nb}draws"] for s, _, _ in SCALES} for t, v in out["terms"].items()
                                 if any(v[s][f"excludes_prior_{lv}_{nb}draws"] for s, _, _ in SCALES)}
    return out


def prior_value_grid(rows: list[dict], dramas: list[str], seeds: list[int], lams: list[float], boot: int,
                     official: dict, prior_imp: dict) -> dict:
    """The uncorrected and 4-term-corrected exclusions again, for several bootstrap seeds and ridge penalties."""
    cells, count = [], {t: {s: {"uncorrected": 0, "bonferroni_4": 0} for s, _, _ in SCALES} for t in TERMS}
    for lam in lams:
        for sd in seeds:
            f = logistic_analysis(rows, dramas, lam, boot, sd)
            cell = {"ridge_lambda": lam, "seed": sd, "draws": boot, "terms": {}}
            for t in TERMS:
                j = TERMS.index(t)
                cell["terms"][t] = {}
                for scale, key, _ in SCALES:
                    prior = official[t] if scale == "weight_0_1" else prior_imp[t]
                    r_ = {}
                    for lv in ("uncorrected", "bonferroni_4"):
                        c = ci(f["_draws"][key][:, j], LEVELS[lv], None)
                        r_[lv] = {"ci": [round(x, 3) for x in c] if c else None, "excludes_prior": _excl(c, prior)}
                        count[t][scale][lv] += 1 if r_[lv]["excludes_prior"] else 0
                    cell["terms"][t][scale] = r_
            cells.append(cell)
    return {"seeds": seeds, "ridge_lambdas": lams, "draws": boot, "n_cells": len(cells), "cells_excluding_prior": count, "cells": cells}


def score_predicts_hit(rows: list[dict], dramas: list[str], col: str, lam: float, boot: int, seed: int) -> dict:
    """hit ~ intercept per drama + one standardized column (the official weighted sum, or the official full score)."""
    stats = col_stats(rows, [col])
    X, _, pen = design(rows, [col], dramas, stats)
    y = np.array([r["hit"] for r in rows], dtype=float)
    beta, _, _ = logit_fit(X, y, pen, lam)
    rng = np.random.default_rng(seed)
    cl = Clusters(rows)
    bs = []
    for _ in range(boot):
        idx = cl.draw(rng)
        bs.append(logit_fit(X[idx], y[idx], pen, lam)[0][-1])
    return {"coef_per_sd": round(float(beta[-1]), 3), "ci95": ci(bs), "share_of_bootstrap_above_0": round(float(np.mean(np.array(bs) > 0)), 3),
            "spearman_with_usage": {d: round(spearman([r[col] for r in rows if r["drama"] == d], [r["usage"] for r in rows if r["drama"] == d]), 3) for d in dramas},
            "n": len(rows)}


def proportionality_test(rows: list[dict], dramas: list[str], official) -> dict:
    """Likelihood-ratio test (unpenalized, raw 0-1 terms): are the four free coefficients proportional to the priors?
    Restricted: intercept per drama + c * (official weighted sum).  Full: intercept per drama + b . terms.  df = 3."""
    y = np.array([r["hit"] for r in rows], dtype=float)
    ints = [np.array([1.0 if r["drama"] == d else 0.0 for r in rows]) for d in dramas]
    T = np.array([[r[t] for t in TERMS] for r in rows])
    Xr = np.column_stack(ints + [T @ np.asarray(official)])
    Xf = np.column_stack(ints + [T])
    br, okr, llr = logit_fit(Xr, y, np.array([False] * len(dramas) + [True]), 0.0)
    bf, okf, llf = logit_fit(Xf, y, np.array([False] * len(dramas) + [True] * 4), 0.0)
    lr = max(0.0, 2 * (llf - llr))
    return {"lr_stat": round(lr, 3), "df": 3, "p_value_asymptotic": round(chi2_sf(lr, 3), 4), "converged": bool(okr and okf),
            "restricted_slope_on_official_sum": round(float(br[-1]), 3),
            "free_raw_coefs": {t: round(float(b), 3) for t, b in zip(TERMS, bf[len(dramas):])}, "n": len(rows)}


def spearman_terms(rows: list[dict], label: str, boot: int, seed: int) -> dict:
    """Spearman of each term (and the official weighted sum / full score) with a per-window label; episode-cluster CIs."""
    out: dict = {}
    cl = Clusters(rows)
    rng = np.random.default_rng(seed)
    draws = [cl.draw(rng) for _ in range(boot)]
    yv = np.array([r[label] for r in rows], dtype=float)
    for t in list(TERMS) + ["base", "score"]:
        xv = np.array([r[t] for r in rows], dtype=float)
        rho = spearman(xv, yv)
        out[t] = {"rho": round(rho, 3) if np.isfinite(rho) else None, "ci95": ci([spearman(xv[i], yv[i]) for i in draws])}
    out["n"] = len(rows)
    out["episodes"] = len({r["episode"] for r in rows})
    return out


def within_episode_spearman(rows: list[dict], boot: int, seed: int, min_n: int = 3) -> dict:
    """Inside one episode, do windows with a higher term cover more-used seconds? Per-episode Spearman (term vs usage)
    over the episodes editors used only partly, averaged with weights = windows per episode; CI resamples episodes."""
    by: dict[int, list[dict]] = {}
    for r in rows:
        if r["partial_episode"]:
            by.setdefault(r["episode"], []).append(r)
    rng = np.random.default_rng(seed)
    out = {}
    for t in TERMS:
        vals = []
        for _, rs in sorted(by.items()):
            if len(rs) >= min_n:
                rho = spearman([r[t] for r in rs], [r["usage"] for r in rs])
                if np.isfinite(rho):
                    vals.append((rho, len(rs)))
        if not vals:
            out[t] = {"rho": None, "ci95": None, "episodes": 0, "windows": 0, "episodes_positive": 0}
            continue
        rv = np.array([v for v, _ in vals])
        nv = np.array([n for _, n in vals], dtype=float)
        bs = []
        for _ in range(boot):
            p = rng.integers(0, len(vals), size=len(vals))
            bs.append(float(np.sum(rv[p] * nv[p]) / np.sum(nv[p])))
        out[t] = {"rho": round(float(np.sum(rv * nv) / np.sum(nv)), 3), "ci95": ci(bs), "episodes": len(vals),
                  "windows": int(nv.sum()), "episodes_positive": int(np.sum(rv > 0))}
    return out


BANDS = ((0.0, 0.5, "low (< 0.5)"), (0.5, 0.9, "middle (0.5-0.9)"), (0.9, 1.0001, "high (>= 0.9)"))


def hit_by_band(rows: list[dict], ids: list[str]) -> dict:
    """Hit rate of the windows whose term falls in a low / middle / high band of its 0-1 scale (pooled and per drama)."""
    out: dict = {}
    for t in TERMS:
        out[t] = {}
        for scope in ["pooled"] + ids:
            sub = rows if scope == "pooled" else [r for r in rows if r["drama"] == scope]
            cells = []
            for lo, hi, lab in BANDS:
                b = [r for r in sub if lo <= r[t] < hi]
                cells.append({"band": lab, "n": len(b), "hits": sum(r["hit"] for r in b),
                              "hit_rate": round(sum(r["hit"] for r in b) / len(b), 3) if b else None})
            out[t][scope] = cells
    return out


# ---------------------------------------------------------------- the sweep and its summaries

def percentile_of(value: float, others) -> float:
    """Share of `others` below value, ties counted half (mid-rank percentile)."""
    others = list(others)
    lo = sum(1 for o in others if o < value - 1e-12)
    eq = sum(1 for o in others if abs(o - value) <= 1e-12)
    return (lo + 0.5 * eq) / max(len(others), 1)


def r4(x):
    return None if x is None else round(float(x), 4)


def run_sweep(dramas: list[Drama], vecs: list[dict]) -> list[dict]:
    """Metrics per vector. The saved values are rounded to 4 decimals; every statistic computed over vectors (percentiles,
    Spearman of weight vs metric, best-vector search) uses the unrounded values kept under '_raw' (stripped before saving)."""
    out, ids_off = [], {}
    for v in vecs:
        row = {"name": v["name"], "family": v["family"], "weights": dict(zip(TERMS, (round(x, 4) for x in v["w"]))), "per_drama": {}, "mean": {},
               "_raw": {"weights": dict(zip(TERMS, (float(x) for x in v["w"])))}}
        for d in dramas:
            ev = d.evaluate(v["w"])
            if v["family"] == "official":
                ids_off[d.id] = ev
            row["_raw"][d.id] = {key: {m: ev[key][m] for m in METRICS} for key in KEYS}
            row["per_drama"][d.id] = {"kept": ev["kept"], "tie_at_k": ev["tie_at_k"],
                                      **{key: {m: r4(ev[key][m]) for m in METRICS} for key in KEYS},
                                      "overlap_with_official": {key: len(set(ev[f"ids {key}"]) & set(ids_off[d.id][f"ids {key}"])) for key in KEYS}}
        row["_raw"]["mean"] = {key: {m: st.mean(row["_raw"][d.id][key][m] for d in dramas) for m in METRICS} for key in KEYS}
        for key in KEYS:
            row["mean"][key] = {m: r4(row["_raw"]["mean"][key][m]) for m in METRICS}
        out.append(row)
    return out


def strip_private(x):
    """Drop the '_'-prefixed working fields (unrounded sweep values, bootstrap draws) before saving."""
    if isinstance(x, dict):
        return {k: strip_private(v) for k, v in x.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(x, list):
        return [strip_private(v) for v in x]
    return x


def tie_robustness(dramas: list[Drama], vecs: list[dict], shuffles: int, seed: int) -> dict:
    """Named vectors with the generation-order tie-break replaced by random orders: mean/min/max of each metric."""
    out = {}
    for v in vecs:
        if v["family"] == "dirichlet":
            continue
        res = {}
        for d in dramas:
            acc = {key: {m: [] for m in METRICS} for key in KEYS}
            for r in range(shuffles):
                order = list(range(len(d.windows)))
                random.Random(seed * 100003 + r).shuffle(order)
                ev = d.evaluate(v["w"], order)
                for key in KEYS:
                    for m in METRICS:
                        acc[key][m].append(ev[key][m])
            res[d.id] = {key: {m: {"mean": r4(st.mean(x)), "min": r4(min(x)), "max": r4(max(x))} for m, x in ms.items()} for key, ms in acc.items()}
        out[v["name"]] = res
    return out


def sweep_summary(sweep: list[dict], ids: list[str]) -> dict:
    off = sweep[0]
    dirs = [r for r in sweep if r["family"] == "dirichlet"]
    out = {}
    for key in KEYS:
        per: dict = {}
        for scope in ids + ["mean"]:
            get = (lambda r, m, s=scope: r["_raw"][s][key][m])  # unrounded
            per[scope] = {}
            for m in METRICS:
                vals = [get(r, m) for r in dirs]
                o = get(off, m)
                per[scope][m] = {
                    "official": r4(o), "dirichlet_p05": r4(np.percentile(vals, 5)), "dirichlet_median": r4(np.median(vals)),
                    "dirichlet_p95": r4(np.percentile(vals, 95)), "dirichlet_min": r4(min(vals)), "dirichlet_max": r4(max(vals)),
                    "official_percentile": round(percentile_of(o, vals), 3),
                    "share_dirichlet_better": round(sum(1 for x in vals if x > o + 1e-12) / len(vals), 3),
                    "share_dirichlet_identical": round(sum(1 for x in vals if abs(x - o) <= 1e-12) / len(vals), 3)}
        # which weights go with higher lift across the random vectors? Spearman of each weight with lift (unrounded values)
        per["weight_vs_metric_spearman"] = {
            scope: {m: {t: r4(spearman([r["_raw"]["weights"][t] for r in dirs], [r["_raw"][scope][key][m] for r in dirs]))
                        for t in TERMS} for m in METRICS} for scope in ids + ["mean"]}
        per["weight_vs_lift_spearman"] = {scope: per["weight_vs_metric_spearman"][scope]["lift"] for scope in ids + ["mean"]}
        per["dirichlet_median_overlap_with_official"] = {i: float(np.median([r["per_drama"][i]["overlap_with_official"][key] for r in dirs])) for i in ids}
        per["dirichlet_share_identical_topk"] = {i: round(float(np.mean([r["per_drama"][i]["overlap_with_official"][key] == int(key.split("@")[1]) for r in dirs])), 3) for i in ids}
        out[key] = per
    return out


def overfit_check(sweep: list[dict], ids: list[str]) -> dict:
    """Pick the weights that maximize lift on one drama; how do they do on the other?"""
    if len(ids) != 2:
        return {}
    a, b = ids
    is_dir = np.array([r["family"] == "dirichlet" for r in sweep])
    out = {}
    for key in KEYS:
        la = np.array([r["_raw"][a][key]["lift"] for r in sweep])  # unrounded
        lb = np.array([r["_raw"][b][key]["lift"] for r in sweep])
        res = {"cross_drama_spearman_all_vectors": r4(spearman(la, lb)), "cross_drama_spearman_dirichlet": r4(spearman(la[is_dir], lb[is_dir]))}
        for train, test, ltr, lte in ((a, b, la, lb), (b, a, lb, la)):
            best = float(ltr.max())
            tied = [i for i in range(len(sweep)) if abs(ltr[i] - best) <= 1e-12]
            c = tied[0]
            res[f"train_{train}_test_{test}"] = {
                "chosen": sweep[c]["name"], "weights": sweep[c]["weights"], "n_vectors_tied_at_best": len(tied),
                "train_lift": r4(best), "train_lift_official": r4(ltr[0]),
                "test_lift": r4(lte[c]), "test_lift_tied_mean": r4(np.mean(lte[tied])), "test_lift_tied_min": r4(np.min(lte[tied])),
                "test_lift_tied_max": r4(np.max(lte[tied])), "test_lift_official": r4(lte[0]),
                "test_lift_median_all_vectors": r4(np.median(lte)), "test_lift_best_possible": r4(lte.max()),
                "test_percentile_of_chosen": round(percentile_of(float(lte[c]), lte), 3),
                "test_percentile_of_tied_mean": round(percentile_of(float(np.mean(lte[tied])), lte), 3)}
        out[key] = res
    return out


def reproduction(dramas: list[Drama], official, published: list[Path]) -> dict:
    """Official weights, replayed offline, vs (a) benchmark_dramaad.evaluate_run on the run folders, (b) the published files."""
    out: dict = {"evaluate_run": {}, "published": {}}
    evs = {d.id: d.evaluate(official) for d in dramas}
    mine = lambda i, key: {m: round(evs[i][key][m], 3) for m in METRICS}  # noqa: E731
    for d in dramas:
        for s in SOURCES:
            for k in KS:
                ref = bd.evaluate_run(d.run, d.ads, d.n_eps, d.dur, k, source=s)
                m_ = mine(d.id, f"{s}@{k}")
                out["evaluate_run"][f"{d.id} {s}@{k}"] = {"replay": m_, "evaluate_run": ref["ours"], "match": m_ == ref["ours"],
                                                         "random": ref["random"], "episode_openings": ref["episode_openings"], "episode_rho": ref["episode_rho"]}
    for f in published:
        pub = json.loads(f.read_text(encoding="utf-8"))
        ids = [i for i in pub.get("per_drama", {}) if i in evs]
        for i in ids:
            r = pub["per_drama"][i]
            key = f"{r['source']}@{r['k']}"
            m_ = mine(i, key)
            out["published"][f"{f.name}: drama {i}, {key}"] = {"replay": m_, "published": r["ours"], "match": m_ == r["ours"],
                                                           "published_run": r.get("run"), "replayed_run": next(d.run.name for d in dramas if d.id == i)}
        summ = pub.get("summary", {})
        if summ and ids:
            r0 = pub["per_drama"][ids[0]]
            key = f"{r0['source']}@{r0['k']}"
            # the published mean is the mean of the per-drama values already rounded to 3 decimals
            m_ = {m: round(st.mean(mine(i, key)[m] for i in ids), 3) for m in METRICS}
            p_ = {"lift": summ.get("ours_lift"), "within_lift": summ.get("ours_within_lift"), "hit_rate": summ.get("ours_hit_rate")}
            out["published"][f"{f.name}: mean of {len(ids)} dramas, {key}"] = {"replay": m_, "published": p_, "match": m_ == p_}
    out["all_match"] = all(v["match"] for v in out["evaluate_run"].values()) and all(v["match"] for v in out["published"].values())
    return out


# ---------------------------------------------------------------- verdicts (rules fixed before the results were read)

def _sign(c) -> int:
    if not c:
        return 0
    return 1 if c[0] > 0 else (-1 if c[1] < 0 else 0)


SIGN_POS, SIGN_NEG, SIGN_MIX, SIGN_NONE = "sign: positive", "sign: negative", "sign: differs between dramas", "sign: cannot judge"


def term_verdict(pooled: dict, per_drama: list[dict], t: str) -> str:
    """The SIGN rule, from 95% CIs of the logistic coefficients. It tests whether a term's weight should be positive,
    never whether the prior's size is right (prior_value_test does that)."""
    signs = [_sign(r["terms"][t]["ci95"]) for r in per_drama]
    p = _sign(pooled["terms"][t]["ci95"])
    if 1 in signs and -1 in signs:
        return SIGN_MIX
    if p == 1 and -1 not in signs:
        return SIGN_POS
    if p == -1 and 1 not in signs:
        return SIGN_NEG
    return SIGN_NONE


def overall_verdict(v: dict[str, str]) -> str:
    parts = []
    for lab, txt in ((SIGN_NEG, "negative for"), (SIGN_MIX, "differs between the dramas for"), (SIGN_POS, "positive for"),
                     (SIGN_NONE, "cannot be judged for")):
        ts = [t for t in TERMS if v[t] == lab]
        if ts:
            parts.append(f"sign {txt} " + ", ".join(ts))
    return "; ".join(parts)


# ---------------------------------------------------------------- plain-English findings, built from the numbers

def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def fci(c, nd=2):
    return "n/a" if not c else f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def where(c) -> str:
    s = _sign(c)
    return "above 0" if s == 1 else ("below 0" if s == -1 else "includes 0")


def corroboration(res: dict, ids: list[str], t: str, sign: int) -> dict:
    """For a term the rule flags: does the same direction show up in the other views of the data?"""
    k = res["spearman_covered"]["kept"]
    pos = res["logistic_position"]["kept"]["pooled"]["terms"][t]
    sgl = res["logistic_single"]["kept"]["pooled"]["terms"][t]
    checks = {
        "position-controlled fit": _sign(pos["ci95"]) == sign,
        "single-episode windows only": _sign(sgl["ci95"]) == sign,
        "rank correlation with the covered share, both dramas": all((k[i][t]["rho"] or 0) * sign > 0 for i in ids),
        "inside episodes, both dramas": all((res["within_episode_spearman"][i][t]["rho"] or 0) * sign > 0 for i in ids),
    }
    # added after review, reported beside the pre-set checks but not counted in 'corroborated'
    usd, uon = res["logistic_used"]["kept"]["pooled"]["terms"][t], res["logistic_used_only"]["kept"]["pooled"]["terms"][t]
    extra = {"episode-used control": _sign(usd["ci95"]) == sign, "used episodes only": _sign(uon["ci95"]) == sign}
    return {"checks": checks, "corroborated": all(checks.values()), "extra_checks_added_after_review": extra,
            "used_coef": usd["coef_per_sd"], "used_ci95": usd["ci95"], "used_only_coef": uon["coef_per_sd"], "used_only_ci95": uon["ci95"],
            "position_coef": pos["coef_per_sd"], "position_ci95": pos["ci95"], "single_coef": sgl["coef_per_sd"], "single_ci95": sgl["ci95"],
            "spearman_usage": {i: res["spearman"]["kept"][i][t]["rho"] for i in ids}, "spearman_covered": {i: k[i][t]["rho"] for i in ids},
            "within_episode": {i: res["within_episode_spearman"][i][t]["rho"] for i in ids}}


def narrative(res: dict, ids: list[str]) -> dict:
    A, B = ids
    rep, ss, over = res["reproduction"], res["sweep_summary"], res["overfit"]
    sw = {r["name"]: r for r in res["sweep"]}
    desc = {d["drama"]: d for d in res["dramas"]}
    lk, lp = res["logistic"]["kept"]["pooled"], res["logistic"]["pool"]["pooled"]
    sph = res["score_predicts_hit"]["kept"]["pooled"]
    out: dict = {}

    cfgd = {d["drama"]: d["config_vs_repo"] for d in res["dramas"]}
    rank_diff = sorted({k for v in cfgd.values() for k in v["ranking_relevant_differing_keys"]})
    other_diff = sorted({k for v in cfgd.values() for k in v["other_differing_keys"]})
    secs_txt = ", ".join(RANKING_SECTIONS)
    if rank_diff:
        cfg_txt = (f"Note: the runs' config.yaml differs from the repository's in ranking-relevant settings ({', '.join(rank_diff)}); "
                   "the replay uses each run's own config, so the reproduction holds for the runs as made, not for today's config.yaml.")
    else:
        cfg_txt = (f"The ranking-relevant sections of the runs' config.yaml ({secs_txt}) equal the repository's"
                   + (f"; only settings for model calls and the evaluation checks differ ({', '.join(other_diff)}), and none of them "
                      "touches the ranking" if other_diff else "; nothing else differs either") + ".")
    if rep["all_match"]:
        out["reproduction_note"] = (
            "Every replayed number equals the published one and the output of `benchmark_dramaad.evaluate_run` on the run folders "
            "(both sources, k = 10 and 20). There is no difference to explain: the replay reuses the cached model answers and each run's "
            "own config.yaml, and rebuilds the same candidates in the same order and the same 20-clip slate. " + cfg_txt)
    else:
        bad = [k for k, v in {**rep["evaluate_run"], **rep["published"]}.items() if not v["match"]]
        out["reproduction_note"] = "Mismatches (" + "; ".join(bad) + "). See editors.json -> reproduction before trusting the rest."

    bl = []
    pub_sl = next((v for k, v in rep["published"].items() if "mean of" in k and "slate@10" in k), None)
    if rep["all_match"] and pub_sl:
        bl.append(f"**The offline replay reproduces the published benchmark exactly** (official weights, slate@10, mean of the two dramas: lift "
                  f"{pub_sl['replay']['lift']:.3f}, within_lift {pub_sl['replay']['within_lift']:.3f}, hit@10 {pub_sl['replay']['hit_rate']:.3f}), "
                  "so every other weight vector is scored on the same footing.")
    else:
        bl.append("**The offline replay does not reproduce the published numbers**; see section 1 before trusting anything else.")

    def rank_word(p):
        return f"high (percentile {p:.0%})" if p >= 0.8 else (f"low (percentile {p:.0%})" if p <= 0.2 else f"mid-pack (percentile {p:.0%})")

    c10, c20 = ss["candidates@10"]["mean"]["lift"], ss["candidates@20"]["mean"]["lift"]
    bl.append(f"**Against {res['n_dirichlet']} random weight vectors, the official weights rank {rank_word(c10['official_percentile'])} at k = 10 "
              f"and {rank_word(c20['official_percentile'])} at k = 20** on lift (candidates ranking, mean of the two dramas). Top 10: lift "
              f"{c10['official']:.2f} vs a random-vector median of {c10['dirichlet_median']:.2f} (5th-95th {c10['dirichlet_p05']:.2f}-{c10['dirichlet_p95']:.2f}); "
              f"top 20: {c20['official']:.2f} vs {c20['dirichlet_median']:.2f} ({c20['dirichlet_p05']:.2f}-{c20['dirichlet_p95']:.2f}).")
    s10 = ss["slate@10"]["mean"]["lift"]
    comp = {i: desc[i]["official_slate_top10_blocks"] for i in ids}
    open_kept = {i: sum(1 for r in res["rows_kept"] if r["drama"] == i and r["block"] == "opening") for i in ids}
    bl.append(f"**Through the slate the weights move lift {'less' if (s10['dirichlet_p95'] - s10['dirichlet_p05']) < (c10['dirichlet_p95'] - c10['dirichlet_p05']) else 'no less'}**: "
              f"slate@10 lift {s10['official']:.2f} for the official weights, random vectors 5th-95th {s10['dirichlet_p05']:.2f}-{s10['dirichlet_p95']:.2f} "
              f"(candidates@10: {c10['dirichlet_p05']:.2f}-{c10['dirichlet_p95']:.2f}). The opening quota reserves {desc[A]['slate_quotas']['opening']} of the "
              f"{sum(desc[A]['slate_quotas'].values())} slate places for episodes {desc[A]['opening_block_episodes'][0]}-{desc[A]['opening_block_episodes'][-1]}, "
              "so the weights mostly choose among those few clips: " + "; ".join(f"drama {nm} top 10 = {comp[i]['opening']} opening + {comp[i]['middle']} middle + {comp[i]['late']} late clips "
                                 f"({open_kept[i]} opening candidates survive overlap removal)" for nm, i in (("A", A), ("B", B))) + ".")

    pi, pa = res["prior_importance"]["kept"], res["prior_importance"]["pool"]
    raw_h = {i: desc[i]["model_hook_score_of_opening_line"] for i in ids}
    n_win = {i: desc[i]["candidate_windows"] for i in ids}
    top2 = {i: (raw_h[i].get("2", 0) + raw_h[i].get("3", 0)) / n_win[i] for i in ids}
    bl.append("**What the weights do in practice:** a term moves the ranking in proportion to its weight times how much it varies across candidates. "
              "Per standard deviation, the official formula's effective emphasis is "
              + ", ".join(f"{t} {pi[t]:.2f}" for t in TERMS) + f" on the kept windows (the {len(res['rows_kept'])} candidates left after overlap removal; "
              + ", ".join(f"{t} {pa[t]:.2f}" for t in TERMS) + " on all windows), against nominal "
              + ", ".join(f"{res['official_weights'][t]:.2f}" for t in TERMS) + f": {max(pi, key=pi.get)} does the most to order the candidates. "
              f"On this benchmark hook is close to a yes/no flag: the model scored the opening line 2 or 3 out of 3 in "
              + " and ".join(f"{top2[i]:.0%}" for i in ids) + " of the windows (A and B)"
              + (", and the physical-action and writers'-cue bonuses never fire (their keyword rules are English; the scripts are Chinese)."
                 if not any(desc[i]["physical_hook_bonus_windows"] or desc[i]["writer_cue_bonus_windows"] for i in ids) else
                 "; the physical-action / writers'-cue bonuses fire on " + " and ".join(
                     f"{desc[i]['physical_hook_bonus_windows']} / {desc[i]['writer_cue_bonus_windows']}" for i in ids) + " windows."))
    sig_k = [t for t in TERMS if _sign(lk["terms"][t]["ci95"]) != 0]
    term_txt = "; ".join(f"{t} {lk['terms'][t]['coef_per_sd']:+.2f} {fci(lk['terms'][t]['ci95'])}" for t in TERMS)
    if not sig_k:
        bl.append(f"**No term's link to editor picks is distinguishable from zero.** Pooled logistic coefficients per SD on the kept windows (95% CI): {term_txt}.")
    else:
        bl.append("**Only " + " and ".join(f"{t} ({where(lk['terms'][t]['ci95'])})" for t in sig_k)
                  + f" has a link to editor picks distinguishable from zero.** Pooled logistic coefficients per SD on the kept windows (95% CI): {term_txt}. "
                  + " ".join(f"Taken at face value, the fit would give {t} a weight of {lk['terms'][t]['implied_weight_raw_scale']:.2f} "
                             f"{fci(lk['terms'][t]['implied_weight_raw_scale_ci95'])} on the 0-1 scale (signed share of the four), against the prior "
                             f"{res['official_weights'][t]:.2f}"
                             + (f", about {lk['terms'][t]['implied_weight_raw_scale'] / res['official_weights'][t]:.1f} times the prior. "
                                "So this backs a positive weight, not the prior's size" if res["official_weights"][t] > 0 else "") + "."
                             for t in sig_k))
    corr = {}
    for t in sig_k:
        c = corroboration(res, ids, t, _sign(lk["terms"][t]["ci95"]))
        corr[t] = c
        failed = [k for k, ok in c["checks"].items() if not ok]
        ex = c["extra_checks_added_after_review"]
        bl.append(f"**Is the {t} signal robust?** " + ("Yes on every pre-set check: " if not failed else "Only partly: ")
                  + f"with position control {c['position_coef']:+.2f} {fci(c['position_ci95'])}; single-episode windows only {c['single_coef']:+.2f} {fci(c['single_ci95'])}; "
                  f"rank correlation with the covered share {c['spearman_covered'][A]:+.2f} (A) / {c['spearman_covered'][B]:+.2f} (B), with usage "
                  f"{c['spearman_usage'][A]:+.2f} / {c['spearman_usage'][B]:+.2f}; inside episodes {fmt(c['within_episode'][A])} / {fmt(c['within_episode'][B])}."
                  + (f" Fails: {', '.join(failed)}. Treat it as a lead, not a finding." if failed else "")
                  + f" Added after review: with a control for whether any ad uses the window's episode {c['used_coef']:+.2f} {fci(c['used_ci95'])}"
                  f" ({'holds' if ex['episode-used control'] else 'does not hold'}); on windows in used episodes only {c['used_only_coef']:+.2f} "
                  f"{fci(c['used_only_ci95'])} ({'holds' if ex['used episodes only'] else 'does not hold'}).")
        bands = res["hit_by_band"]["kept"][t]["pooled"]
        txt = (f"**Where the {t} signal comes from:** hit rate of the kept windows by {t} band: "
               + "; ".join(f"{c['band']} {c['hits']}/{c['n']}" + (f" ({c['hit_rate']:.0%})" if c["n"] else "") for c in bands) + ".")
        lo_rows = [r for r in res["rows_kept"] if r[t] < BANDS[0][1]]
        if lo_rows and t == "density":
            ds = sorted(r["dialogue_share"] for r in lo_rows)
            du = sorted(r["duration_s"] for r in lo_rows)
            txt += (f" The {len(lo_rows)} low-band windows have dialogue on {ds[0]:.0%}-{ds[-1]:.0%} of their running time and last {du[0]:.0f}-{du[-1]:.0f} s"
                    f" (density = dialogue share, halved below 35%, times a length prior that drops to 0.6 at 20 s and 60 s).")
        bl.append(txt)
    out["corroboration"] = corr
    sig_p = [t for t in TERMS if _sign(lp["terms"][t]["ci95"]) != 0]
    if sig_p != sig_k:
        bl.append("On all generated windows (overlapping, so less independent than the count suggests) the CI excludes 0 for: "
                  + (", ".join(f"{t} ({lp['terms'][t]['coef_per_sd']:+.2f} {fci(lp['terms'][t]['ci95'])})" for t in sig_p) or "no term") + ".")
    eu = res["episode_used_summary"]["kept"]
    lu, luo = res["logistic_used"]["kept"]["pooled"]["terms"], res["logistic_used_only"]["kept"]["pooled"]["terms"]
    survive = [t for t in TERMS if _sign(lk["terms"][t]["ci95"]) != 0 and _sign(lu[t]["ci95"]) == _sign(lk["terms"][t]["ci95"])]
    bl.append(f"**Whether any ad uses the episode explains much of the labels** (a control added after review). Of the kept windows, "
              f"{eu['never_used_hits']} of {eu['never_used']} in episodes no ad uses are hits, against {eu['used_hits']} of {eu['used']} in used episodes. "
              "With an episode-used indicator added to the pooled kept fit: " + "; ".join(f"{t} {lu[t]['coef_per_sd']:+.2f} {fci(lu[t]['ci95'])}" for t in TERMS)
              + ". On windows in used episodes only: " + "; ".join(f"{t} {luo[t]['coef_per_sd']:+.2f} {fci(luo[t]['ci95'])}" for t in TERMS) + ". "
              + "".join(f"{t.capitalize()} moves from {lk['terms'][t]['coef_per_sd']:+.2f} to {lu[t]['coef_per_sd']:+.2f} with the control. "
                        for t in TERMS if lk["terms"][t]["coef_per_sd"] * lu[t]["coef_per_sd"] < 0
                        and max(abs(lk["terms"][t]["coef_per_sd"]), abs(lu[t]["coef_per_sd"])) >= 0.1)
              + (f"The {', '.join(survive)} lead survives this control" if survive else "No term's lead survives this control")
              + (f"; {eu['low_density_nonhits_in_used']} of the {eu['low_density_nonhits']} low-density non-hits are in used episodes, so the "
                 "density effect is not just the unused episodes." if eu["low_density_nonhits"] else "."))
    side = []  # secondary fits whose CI excludes 0 for a term the sign rule leaves at 'cannot judge'
    pair = {"logistic_position": "logistic_position_used", "logistic_single": "logistic_single_used", "logistic": "logistic_used"}
    for t in TERMS:
        if t in sig_k:
            continue
        for lab, fam in (("position control", "logistic_position"), ("single-episode windows only", "logistic_single"), ("plain fit", "logistic")):
            for pop, plab in (("kept", "kept"), ("pool", "all")):
                if fam == "logistic" and pop == "kept":
                    continue
                c = res[fam][pop]["pooled"]["terms"][t]
                if _sign(c["ci95"]) != 0:
                    cu = res[pair[fam]][pop]["pooled"]["terms"][t]
                    side.append({"term": t, "fit": lab, "population": plab, "coef": c["coef_per_sd"], "ci95": c["ci95"],
                                 "with_episode_used_coef": cu["coef_per_sd"], "with_episode_used_ci95": cu["ci95"],
                                 "survives_episode_used_control": _sign(cu["ci95"]) == _sign(c["ci95"])})
    if side:
        kept_used = {t: res["logistic_used"]["kept"]["pooled"]["terms"][t] for t in TERMS}
        bl.append("**Secondary fits disagree with 'sign: cannot judge' in places:** "
                  + "; ".join(f"{s['term']}, {s['fit']}, {s['population']} windows: {s['coef']:+.2f} {fci(s['ci95'])}; with the episode-used control "
                              f"added, {'still ' if s['survives_episode_used_control'] else ''}{s['with_episode_used_coef']:+.2f} "
                              f"{fci(s['with_episode_used_ci95'])}" + ("" if s["survives_episode_used_control"] else ", which includes 0")
                              for s in side)
                  + ". These are robustness views, not the rule's primary fit. "
                  + " ".join(f"So the {s['term']} association in this view is not only an episode effect, but the primary kept-window fit with the "
                             f"same control gives {s['term']} {kept_used[s['term']]['coef_per_sd']:+.2f} {fci(kept_used[s['term']]['ci95'])}: the "
                             "evidence on it is mixed." for s in side if s["survives_episode_used_control"])
                  + ("With the control these associations go away, so they look driven by which episodes the editors used."
                     if not any(s["survives_episode_used_control"] for s in side) else ""))
    out["secondary_fits_excluding_0"] = side
    sc = sph["score"]
    verb = {1: "predicts", -1: "anti-predicts", 0: "does not detectably predict"}[_sign(sc["ci95"])]
    bl.append(f"**The official full score {verb} hits** on the kept windows: {sc['coef_per_sd']:+.2f} {fci(sc['ci95'])} log-odds per SD "
              f"(the weighted sum alone: {sph['base']['coef_per_sd']:+.2f} {fci(sph['base']['ci95'])}).")

    dirn = {}
    for t in TERMS:
        for m in ("lift", "within_lift"):
            vals = [ss[key]["weight_vs_metric_spearman"][i][m][t] for key in KEYS for i in ids]
            dirn.setdefault(t, {})[m] = {"negative": sum(1 for v in vals if v is not None and v < 0), "positive": sum(1 for v in vals if v is not None and v > 0),
                                         "min": min(v for v in vals if v is not None), "max": max(v for v in vals if v is not None), "cases": len(vals)}

    def lean(t):
        d, w = dirn[t]["lift"], dirn[t]["within_lift"]
        return f"{t} higher in {d['positive']}, lower in {d['negative']} (within_lift: higher in {w['positive']}, lower in {w['negative']})"

    n_cases = dirn[TERMS[0]]["lift"]["cases"]
    bl.append(f"**Which way the sweep leans:** across the {res['n_dirichlet']} random vectors, does giving a term more weight raise or lower the lift of the "
              f"resulting top-k? Sign of the Spearman correlation (on unrounded lifts), counted over {n_cases} cases (2 dramas x 2 rankings x k = 10, 20): "
              + "; ".join(lean(t) for t in TERMS)
              + ". This describes how the two dramas' top 10-20 clips move when weight shifts; it is not a significance test.")
    out["sweep_direction"] = dirn
    tr = {}
    for key in KEYS:
        o = over[key]
        ab, ba = o[f"train_{A}_test_{B}"], o[f"train_{B}_test_{A}"]
        tr[key] = bool(ab["test_percentile_of_tied_mean"] >= 0.6 and ba["test_percentile_of_tied_mean"] >= 0.6 and (o["cross_drama_spearman_all_vectors"] or 0) > 0.3)
    o = over["candidates@10"]
    ab, ba = o[f"train_{A}_test_{B}"], o[f"train_{B}_test_{A}"]
    yes = [k for k in KEYS if tr[k]]
    no = [k for k in KEYS if not tr[k]]
    bl.append(f"**Weights tuned on one drama transfer to the other for {', '.join(yes) if yes else 'none of the four rankings'}"
              + (f"; not for {', '.join(no)}" if no and yes else "") + ".** "
              f"Candidates@10: the best vectors for A (lift {ab['train_lift']:.2f}, official {ab['train_lift_official']:.2f}) average {ab['test_lift_tied_mean']:.2f} on B "
              f"(official {ab['test_lift_official']:.2f}; percentile {ab['test_percentile_of_tied_mean']:.0%}); the best for B ({ba['train_lift']:.2f}, official "
              f"{ba['train_lift_official']:.2f}) average {ba['test_lift_tied_mean']:.2f} on A (official {ba['test_lift_official']:.2f}; percentile "
              f"{ba['test_percentile_of_tied_mean']:.0%}). Spearman of a vector's lift on A vs on B, over all vectors: "
              + ", ".join(f"{k} {fmt(over[k]['cross_drama_spearman_all_vectors'])}" for k in KEYS) + ".")
    out["transfers"] = tr

    vk, vp = res["verdicts"]["kept"], res["verdicts"]["pool"]
    prop = res["proportionality"]["kept"]["pooled"]
    ov = overall_verdict(vk)
    if corr and not all(c["corroborated"] for c in corr.values()):
        ov += " (the flagged term" + ("s are" if len(corr) > 1 else " is") + " not robust on every pre-set check)"

    # ---- the prior VALUES (added after review): does any interval exclude the prior itself, and does it survive correction?
    pv, grid, vr = res["prior_test"]["kept"], res["prior_grid"], res["prior_test_robustness"]["kept"]
    nb = pv["draws_corrected"]
    off_w = res["official_weights"]
    unc, c4, c8 = pv["excluded_uncorrected"], pv["excluded_bonferroni_4"], pv["excluded_bonferroni_8"]
    lv4, lv8 = LEVELS["bonferroni_4"], LEVELS["bonferroni_8"]
    lab_of = {s: lab for s, _, lab in SCALES}
    on_of = {"weight_0_1": "on the 0-1 scale", "importance_per_sd": "per SD"}
    variants = [v for v in vr if v != "plain fit"]

    def ci_at(t: str, s: str, lv: str):
        e = pv["terms"][t][s]
        return e["ci95"] if lv == "uncorrected" else e[f"ci_{lv}_{nb}draws"]

    def scales_txt(t: str, lv: str, nd: int = 2) -> str:
        return "; ".join(f"{lab} {pv['terms'][t][s]['point']:+.{nd}f} {fci(ci_at(t, s, lv), nd)} vs {pv['terms'][t][s]['prior']:.{nd}f}"
                         for s, _, lab in SCALES)

    term_value: dict = {}
    for t, sc in unc.items():
        side = sorted({v for v in sc.values() if v})[0]
        ex4 = [s for s in sc if (c4.get(t) or {}).get(s)]
        ex8 = [s for s in sc if (c8.get(t) or {}).get(s)]
        g = {s: grid["cells_excluding_prior"][t][s] for s, _, _ in SCALES}
        hold = {s: [v for v in variants if vr[v]["terms"][t][s]["excludes_prior_95"] == side] for s, _, _ in SCALES}
        eu_ = vr["episode-used control"]["terms"][t]
        term_value[t] = {"side": side, "uncorrected_scales": [s for s, v in sc.items() if v], "bonferroni_4_scales": ex4,
                         "bonferroni_8_scales": ex8, "grid": g, "variants_holding": hold}
        on = [s for s, v in sc.items() if v]
        txt = (f"{t.capitalize()} is {side} its prior, uncorrected, " + ("on both scales" if len(on) == len(SCALES) else "on the " + lab_of[on[0]])
               + f" ({scales_txt(t, 'uncorrected')})")
        if ex4:
            txt += (f". After the four-term correction it stays {side} " + " and ".join(on_of[s] for s in ex4)
                    + " (" + "; ".join(f"{lab_of[s]} {fci(ci_at(t, s, 'bonferroni_4'), 3)} vs {pv['terms'][t][s]['prior']:.3f}, "
                                        f"in {g[s]['bonferroni_4']} of {grid['n_cells']} seed x ridge settings" for s in ex4) + ")")
            rest = [s for s in sc if s not in ex4]
            if rest:
                txt += ", but not " + " and ".join(f"{on_of[s]} ({fci(ci_at(t, s, 'bonferroni_4'), 3)} vs {pv['terms'][t][s]['prior']:.3f}; "
                                                      f"{g[s]['bonferroni_4']} of {grid['n_cells']} settings)" for s in rest)
        else:
            txt += (". After the four-term correction it is excluded on neither scale (" + scales_txt(t, "bonferroni_4") + "; "
                    + ", ".join(f"{lab_of[s]} in {g[s]['bonferroni_4']} of {grid['n_cells']} settings" for s, _, _ in SCALES) + ")")
        txt += (". The stricter 8-test correction excludes it on " + (" and ".join(lab_of[s] for s in ex8) if ex8 else "neither scale") + ". "
                f"Uncorrected, the exclusion holds in {len(hold['weight_0_1'])} (0-1 scale) and {len(hold['importance_per_sd'])} (per SD) of the "
                f"{len(variants)} variant fits; with the episode-used control it is "
                + "; ".join(f"{lab_of[s]} {eu_[s]['point']:+.2f} {fci(eu_[s]['ci95'])} vs {eu_[s]['prior']:.2f}"
                            + (f" ({eu_[s]['excludes_prior_95']})" if eu_[s]["excludes_prior_95"] else " (not excluded)") for s, _, _ in SCALES))
        term_value[t]["text"] = txt
    # terms the primary fit does not exclude, but some variant fit does (uncorrected)
    var_only: dict = {}
    for t in TERMS:
        if t in unc:
            continue
        vs = {v: {s: vr[v]["terms"][t][s] for s, _, _ in SCALES if vr[v]["terms"][t][s]["excludes_prior_95"]} for v in variants}
        vs = {v: d for v, d in vs.items() if d}
        if vs:
            sides = sorted({e["excludes_prior_95"] for d in vs.values() for e in d.values()})
            var_only[t] = {"side": " / ".join(sides), "variants": list(vs),
                           "text": f"{t.capitalize()} is not excluded in the primary fit, but its share is {' / '.join(sides)} its prior (uncorrected) in "
                                   f"{len(vs)} of {len(variants)} variant fits: " + "; ".join(
                                       f"{v}, " + ", ".join(f"{lab_of[s]} {e['point']:+.2f} {fci(e['ci95'])} vs {e['prior']:.2f}" for s, e in d.items())
                                       for v, d in vs.items())}
    out["value_test_summary"] = term_value
    out["value_test_variant_only"] = var_only
    unexcl = [t for t in TERMS if t not in unc and t not in var_only]
    if unc:
        bl.append("**Do the intervals exclude the prior values themselves, not just 0?** (Added after review.) "
                  + " ".join(term_value[t]["text"] + "." for t in unc)
                  + "".join(f" {v['text']}." for v in var_only.values())
                  + (f" {' and '.join(unexcl).capitalize()} {'is' if len(unexcl) == 1 else 'are'} not excluded in any fit, "
                     "even uncorrected." if unexcl else "") + f" Corrections: Bonferroni for four terms = {lv4:.2%} intervals, for 4 terms x 2 scales = "
                  f"{lv8:.3%}, from {nb} draws; the grid is bootstrap seeds {', '.join(map(str, grid['seeds']))} x ridge "
                  f"{', '.join(str(x) for x in grid['ridge_lambdas'])}, {grid['draws']} draws each. The joint test "
                  f"{'does not reject' if prop['p_value_asymptotic'] >= 0.05 else 'rejects'} the priors' proportions "
                  f"(likelihood ratio {prop['lr_stat']:.2f}, df 3, p = {prop['p_value_asymptotic']:.2f}).")
    else:
        bl.append("**Do the intervals exclude the prior values themselves, not just 0?** No term's uncorrected 95% interval excludes its prior, "
                  f"on either scale. Joint test: likelihood ratio {prop['lr_stat']:.2f}, df 3, p = {prop['p_value_asymptotic']:.2f}.")

    # ---- the decision on config.yaml (rule written after the results were seen, in response to review)
    reasons, change = [], bool(c4)
    if not unc:
        reasons.append("no term's interval excludes its prior")
    for t, v in term_value.items():
        if not v["bonferroni_4_scales"]:
            reasons.append(f"{t}'s share is {v['side']} its {off_w[t]:.2f} prior only before correcting for testing four terms")
        else:
            s4 = v["bonferroni_4_scales"]
            reasons.append(f"{t}'s share stays {v['side']} its prior after the four-term correction only "
                           + " and ".join(on_of[s] for s in s4) + ", narrowly and in "
                           + " / ".join(f"{v['grid'][s]['bonferroni_4']}" for s in s4) + f" of {grid['n_cells']} seed x ridge settings"
                           + ("" if v["bonferroni_8_scales"] else ", and not under the stricter 8-test correction")
                           + f"; even uncorrected it holds in only {min(len(v['variants_holding'][s]) for s in s4)} of {len(variants)} variant fits"
                           + ("" if all(res["prior_test_robustness"]["kept"]["episode-used control"]["terms"][t][s]["excludes_prior_95"] for s in s4)
                              else ", and not once the episode-used control is added"))
    for t, v in var_only.items():
        reasons.append(f"{t}'s share is {v['side']} its prior only in {len(v['variants'])} of {len(variants)} variant fits added after review "
                       f"({', '.join(v['variants'])}), uncorrected")
    if prop["p_value_asymptotic"] >= 0.05:
        reasons.append(f"the joint test does not reject the priors' proportions (LR {prop['lr_stat']:.2f}, df 3, p = {prop['p_value_asymptotic']:.2f})")
        change = False
    for t, c in corr.items():
        failed = [k for k, ok in c["checks"].items() if not ok]
        if failed:
            reasons.append(f"{t} fails the pre-set robustness check{'s' if len(failed) > 1 else ''} ({', '.join(failed)})")
            change = False
    if not any(tr.values()):
        reasons.append("tuned weights do not transfer between the dramas")
        change = False
    out["config_decision"] = {"change_config": change, "reasons": reasons,
                              "rule": "change a weight only when its interval excludes the prior after a correction for testing four terms, the "
                                      "joint test rejects the priors' proportions, the term passes every pre-set robustness check, and tuned "
                                      "weights transfer between the dramas (written after the results were seen, in response to review)"}
    head = "no change to config.yaml" if not change else "the data point to a change, to be confirmed on held-out shows"
    bl.append(f"**Verdict on 0.35 / 0.30 / 0.25 / 0.10: {head}.** " + "; ".join(r[0].upper() + r[1:] if i == 0 else r for i, r in enumerate(reasons)) + ". "
              f"Sign rule (kept windows, set before the results by the author's account): {ov}; all windows: {overall_verdict(vp)}. "
              "The sign rule only says whether a weight should be positive. It says nothing about whether 0.35 / 0.30 / 0.25 / 0.10 are the right sizes.")
    out["bottom_line"] = bl

    ties = res["tie_robustness"]
    single_gen = {t: sw[f"only {t}"]["mean"]["candidates@10"]["lift"] for t in TERMS}
    single_rnd = {t: st.mean(ties[f"only {t}"][i]["candidates@10"]["lift"]["mean"] for i in ids) for t in TERMS}
    off_rng = {nm: (ties["official"][i]["candidates@10"]["lift"]["min"], ties["official"][i]["candidates@10"]["lift"]["max"]) for nm, i in (("A", A), ("B", B))}
    out["tie_note"] = (
        "**Ties.** The pipeline breaks equal scores by generation order (earlier episodes first). Editors use early episodes most, so a vector with "
        "many ties can look better or worse for reasons unrelated to its weights. Hook and cliff take only 3-4 distinct values, so single-term "
        "vectors tie a lot. With random tie-breaks "
        f"({res['shuffles']} shuffled orders per named vector), candidates@10 lift for each term alone (mean of the dramas): "
        + "; ".join(f"{t} {single_rnd[t]:.2f} (generation order {single_gen[t]:.2f})" for t in TERMS)
        + ". The official vector across shuffles: " + "; ".join(f"drama {nm} {lo:.3f}-{hi:.3f}" for nm, (lo, hi) in off_rng.items()) + ".")

    no_transfer = not any(tr.values())
    sweep_txt = ("**Secondary evidence, outside the rule.** In the weight sweep (section 2), counted over the "
                 f"{n_cases} cases (2 dramas x 2 rankings x k = 10, 20), more weight on a term gave: " + "; ".join(lean(t) for t in TERMS) + ". "
                 + ("Tuned weights do not transfer between the two dramas in any ranking (section 4), so these directions are something to test on "
                    "traction data, not a reason to change config.yaml now." if no_transfer else
                    "Tuned weights transfer between the dramas only for " + ", ".join(k for k, v in tr.items() if v) + " (section 4)."))
    fails = ([] if prop["p_value_asymptotic"] < 0.05 else ["the joint test"]) + (["the transfer check"] if not any(tr.values()) else [])
    read = ("**How to read this.** Neither test says the priors are right. 'Sign: cannot judge' means two dramas with minute-long, "
            "position-driven labels are not enough evidence to say even whether a weight should be positive. The value test is the one that "
            "speaks to 0.35 / 0.30 / 0.25 / 0.10.")
    if unc:
        surv = [t for t in unc if term_value[t]["bonferroni_4_scales"]]
        read += (" Its uncorrected intervals point at " + " and ".join(f"{'less' if term_value[t]['side'] == 'below' else 'more'} weight on {t}" for t in unc)
                 + "".join(f" (and, only in {len(v['variants'])} of the {len(variants)} variant fits, {'less' if v['side'] == 'below' else 'more'} on {t})"
                           for t, v in var_only.items())
                 + ". " + (f"After correcting for four terms only {', '.join(surv)} keeps a narrow, setting-dependent exclusion; "
                           if surv else "None of this survives correcting for four terms; ")
                 + ("and " + " and ".join(fails) + (" fails" if len(fails) == 1 else " fail") + ". " if fails else "")
                 + "These are leads to test on traction data, not grounds to move a weight now.")
    read += (" The priors stay what they are, a reasoned starting point from the research, and should be refitted on the outcome they are "
             "meant to predict.")
    out["verdict_text"] = [sweep_txt, read,
        "**What would settle it.** Per-promo outcomes (hook rate, hold rate, taps) on a few hundred posted promos, or editor labels for 20-60 s "
        "clips rather than 2-4 minute ads. The logistic fit and the cross-drama check in this script are the template: fit on some shows, "
        "test on held-out shows, and move a weight only when its interval excludes the prior after correcting for testing several terms, "
        "and only when the refitted weights also hold up on held-out shows.",
        "**What this does not test.** The first-time-viewer judge's standalone rating (the benchmark ran without the judge), the editor's-check "
        "penalties, the turning-point lift, the slate quotas, and English-language shows like the brief's.",
    ]
    out["method"] = [
        "Windows: `stability.replay(run)` rebuilds each benchmark run from its cached beat scores; `generate_windows`, `score_window`, `flag_spoilers` "
        "as in the pipeline. For each weight vector, `rank.combine` re-scores every window with that vector (rubric penalties, turning-point lift and the "
        "two-segment discount unchanged), then `rank.nms`, then the `--no-judge` slate (`slate.build` to the 40 finalists, then to the 20-clip slate).",
        "Metrics: the per-clip arithmetic of `benchmark_dramaad.evaluate_run` (usage map, lift against the mean second of the evaluated episodes, "
        "within_lift against the clip's own episode, hit = at least 50% of seconds inside some ad), with segment times rounded to 2 decimals as the "
        "run files store them. `candidates` = top-k of the de-duplicated candidates by score; `slate` = the first k clips of the posted slate.",
        f"Weight vectors ({len(res['sweep'])}): official; each weight +-0.05 and +-0.10, then renormalized to sum 1 (scaling every weight changes no ranking); "
        f"each term left out (renormalized); each term alone; equal; {res['n_dirichlet']} Dirichlet(1,1,1,1) draws (numpy default_rng, seed {res['seeds']['dirichlet']}).",
        "Per-window statistics: Spearman with average ranks; logistic regression by Newton steps with a ridge penalty "
        f"(lambda = {res['ridge_lambda']}) on the standardized coefficients (terms z-scored within each drama; one intercept per drama when pooled); "
        "position control = episode position and its square, per drama. Bootstrap: "
        f"{res['bootstrap']} draws resampling whole episodes within each drama (seed {res['seeds']['bootstrap']}); percentile 95% CIs. "
        "Implied weights on the 0-1 scale = coefficient / term SD, divided by the sum of absolute values (signs kept). "
        "Proportionality: unpenalized likelihood ratio, free coefficients vs one coefficient on the official weighted sum (df = 3).",
        "Overfitting: the vector with the highest lift on one drama (the first, when several tie), evaluated on the other; percentiles are among all vectors. "
        "A ranking counts as transferring when the tied best vectors land at or above the 60th percentile on the other drama in both directions and "
        "the cross-drama Spearman of lift exceeds 0.3.",
        "A term the sign rule flags counts as robust only when the same direction appears in every other pre-set view: the position-controlled fit, "
        "single-episode windows only, the rank correlation with the covered share in both dramas, and inside episodes in both dramas.",
        f"Prior-value test: a term's interval 'excludes the prior' when the prior lies outside it, on the 0-1 scale (prior = the weight; the official "
        f"weights sum to 1) or per SD (prior = weight x SD, normalized). Uncorrected: the 95% intervals of the main {res['bootstrap']}-draw bootstrap. "
        f"Corrected: Bonferroni for the four terms ({LEVELS['bonferroni_4']:.2%} intervals) and a stricter version for 4 terms x 2 scales "
        f"({LEVELS['bonferroni_8']:.3%}), from a {res['bootstrap_corrected']}-draw bootstrap with the same seed (its first {res['bootstrap']} draws are the main ones). "
        "Bounds are compared unrounded. The same test is repeated over a grid of bootstrap seeds and ridge penalties, and (uncorrected) on each "
        "variant fit: position control, single-episode windows, episode-used control, position + episode-used control, used episodes only.",
        "Episode-used control: an indicator for whether any professional ad uses an episode the window touches, z-scored within each drama and "
        "added to the fit (ridge-penalized like the terms); and, separately, the fit on windows in used episodes only.",
        "Sweep statistics (percentiles, Spearman of each weight with a metric, the best-vector search in section 4) use unrounded metric values; "
        "the tables show them rounded.",
        "When each rule was set, by the author's own account (no record of the trial run was kept, so this cannot be checked from the files): "
        "the sign rule before any result was seen; the robustness checks, the hit-rate bands and the transfer thresholds after a small trial run "
        "(20 vectors, 50 bootstrap draws) and before the full run. These can only make a claim harder to pass. The prior-value test, the "
        "multiple-testing correction, the seed x ridge grid, the episode-used control and the decision rule for config.yaml were added after "
        "the full results had been seen, in response to review.",
        f"Ties: broken by generation order, as in the pipeline; the named vectors are also re-run with {res['shuffles']} random tie-break orders (seed {res['seeds']['shuffle']}).",
    ]
    return out


def markdown(res: dict) -> str:
    D = res["dramas"]
    ids = [d["drama"] for d in D]
    A, B = ids
    off = res["official_weights"]
    sw = {r["name"]: r for r in res["sweep"]}
    ss = res["sweep_summary"]
    L: list[str] = []
    add = L.append
    add("# Do the ranking weights match what professional editors pick?")
    add("")
    add(f"Generated by `scripts/weights_vs_editors.py` ({res['generated']}). Offline: no model calls, no network. "
        f"Fixed seeds: Dirichlet {res['seeds']['dirichlet']}, bootstrap {res['seeds']['bootstrap']}, tie shuffles {res['seeds']['shuffle']}. "
        "Every number is in `editors.json`.")
    add("")
    add(f"Drama **A** = DramaAD {A} ({D[0]['genre']}); drama **B** = DramaAD {B} ({D[1]['genre']}).")
    add("")
    add("The formula under test (promo_finder/rank.py):")
    add("")
    add("    score = (0.35*hook + 0.30*cliff + 0.25*standalone + 0.10*density)")
    add("            x penalty per failed editor's check x 1.05 turning-point episode x 0.9 two-segment clip")
    add("")
    add("Only the four weights are varied. The penalties, multipliers, overlap removal and slate rules stay as configured.")
    add("")
    add("## Bottom line")
    add("")
    for s in res["bottom_line"]:
        add(f"- {s}")
    add("")

    add("## The data, and why it can only say so much")
    add("")
    add("| | drama A | drama B |")
    add("|---|---|---|")
    rows = [
        ("DramaAD id", lambda d: d["drama"]),
        ("genre (DramaAD label)", lambda d: d["genre"]),
        ("episodes in the drama / evaluated", lambda d: f"{d['episodes_in_drama']} / the first {d['episodes_evaluated']}"),
        ("minutes evaluated", lambda d: f"{d['evaluated_minutes']}"),
        ("language of the dialogue", lambda d: f"Chinese ({d['dialogue_cjk_share_of_letters']:.0%} of letters are CJK, {d['dialogue_lines']} lines)"),
        ("professional ads / ads that use the evaluated episodes", lambda d: f"{d['ads']} / {d['ads_using_evaluated_episodes']}"),
        ("ad segments inside the evaluated episodes", lambda d: f"{d['ad_segments_in_evaluated_episodes']} (of {d['ad_segments']})"),
        ("ad length, s (min / median / max)", lambda d: f"{d['ad_length_s']['min']:.0f} / {d['ad_length_s']['median']:.0f} / {d['ad_length_s']['max']:.0f}"),
        ("ad segment length, s (median)", lambda d: f"{d['ad_segment_length_s']['median']:.0f}"),
        ("share of evaluated seconds inside some ad", lambda d: f"{d['share_of_evaluated_seconds_in_some_ad']:.0%}"),
        ("episodes used only partly (within-episode choice visible)", lambda d: f"{d['episodes_with_partial_use']} of {d['episodes_evaluated']}"),
        ("episodes no ad uses", lambda d: ", ".join(map(str, d["episodes_never_used"])) or "none"),
        ("candidate windows (before overlap removal)", lambda d: f"{d['candidate_windows']} from {d['episodes_with_candidates']} episodes"),
        ("kept after overlap removal (official weights) / share that are hits", lambda d: f"{d['kept_after_nms_official']} / {d['kept_hit_share_official']:.0%}"),
        ("candidate length, s (median)", lambda d: f"{d['candidate_duration_s_median']:.0f}"),
        ("model's hook score of the opening line (0-3): windows", lambda d: ", ".join(f"{k}: {v}" for k, v in d["model_hook_score_of_opening_line"].items())),
        ("model's cliff score of the closing line (0-3): windows", lambda d: ", ".join(f"{k}: {v}" for k, v in d["model_cliff_score_of_closing_line"].items())),
        ("windows given the physical-action / writers'-cue bonus / romance cap", lambda d: f"{d['physical_hook_bonus_windows']} / {d['writer_cue_bonus_windows']} / {d['romance_cap_windows']}"),
        ("beat-scoring model / first-time-viewer judge", lambda d: f"{d['model']} / {'yes' if d['judge_used'] else 'no (--no-judge)'}"),
    ]
    for label, f in rows:
        add(f"| {label} | " + " | ".join(str(f(d)) for d in D) + " |")
    add("")
    tot = res["totals"]
    add(f"In all: **{tot['dramas']} dramas** (of the 28 in DramaAD), **{tot['episodes']} episodes**, **{tot['candidate_windows']} candidate windows** "
        f"({tot['kept']} after overlap removal), **{tot['ads']} professional ads** ({tot['ads_using_evaluated']} of which use the evaluated episodes).")
    add("")
    add("Limits that apply to every number below:")
    add("")
    add(f"- **Coarse labels.** A professional ad runs {tot['ad_length_text']} (A and B) and is cut from segments of about a minute (median {tot['segment_text']}); "
        "our clips are 20-60 s. A clip is a hit when half its seconds sit inside *some* ad, so a hit says the editors used that stretch of the episode, "
        "not that they singled out our moment. Editors also reuse whole early episodes.")
    add("- **Position dominates the labels.** Which episodes the editors drew from explains most of the variation: no ad uses episodes "
        + " and ".join(f"{', '.join(map(str, d['episodes_never_used'])) or 'none'} of drama {nm}" for nm, d in zip(("A", "B"), D))
        + ". A term that merely drifts with episode position looks predictive (or anti-predictive) for that reason alone, so within-episode and "
          "position-controlled versions are reported too.")
    add("- **A different product and market.** The dramas are in Chinese and were scored by an English prompt; the ads are 2-4 minute edits, not "
        "20-60 s feed promos. Agreement or disagreement with these editors is indirect evidence about the brief's promos.")
    share23 = lambda d, key: (d[key].get("2", 0) + d[key].get("3", 0)) / max(d["candidate_windows"], 1)  # noqa: E731
    fired = [(d["physical_hook_bonus_windows"], d["writer_cue_bonus_windows"]) for d in D]
    add("- **Hook and cliff are coarser here than in the product.** The physical-action bonus and the writers'-cue bonus are English keyword rules; "
        "on these Chinese scripts they fire on " + " and ".join(f"{a_} / {b_}" for a_, b_ in fired) + " windows (A and B), so hook and cliff are "
        "essentially the model's raw 0-3 scores: 2 or 3 for the opening line in "
        + " / ".join(f"{share23(d, 'model_hook_score_of_opening_line'):.0%}" for d in D) + " of windows, and for the closing line in "
        + " / ".join(f"{share23(d, 'model_cliff_score_of_closing_line'):.0%}" for d in D) + ".")
    add("- **`standalone` is the code proxy here** (share of lines with no unexplained names) because the benchmark runs used `--no-judge`. "
        "In the product, the finalists' standalone becomes the first-time viewer's rating, which this data cannot test.")
    add(f"- **Small samples.** {tot['dramas']} dramas and {tot['kept']} nearly non-overlapping windows; only large effects could show up.")
    add("")

    rep = res["reproduction"]
    add("## 1. Reproduction check")
    add("")
    add("The official weights, replayed offline, must give the published benchmark numbers before any other vector is trusted.")
    add("")
    add("| published file: drama, ranking | replayed: lift, within_lift, hit@k | published | match |")
    add("|---|---|---|---|")
    for k, v in rep["published"].items():
        a_, p_ = v["replay"], v["published"]
        add(f"| {k} | {fmt(a_['lift'], 3)}, {fmt(a_['within_lift'], 3)}, {fmt(a_['hit_rate'], 3)} | "
            f"{fmt(p_['lift'], 3)}, {fmt(p_['within_lift'], 3)}, {fmt(p_['hit_rate'], 3)} | {'yes' if v['match'] else 'NO'} |")
    n_ev = len(rep["evaluate_run"])
    add(f"| `benchmark_dramaad.evaluate_run` on the run folders: both dramas, both sources, k = 10 and 20 | | | "
        f"{sum(v['match'] for v in rep['evaluate_run'].values())} of {n_ev} |")
    add("")
    add(res["reproduction_note"])
    add("")

    add("## 2. Weight sweep: which weights pick what editors pick?")
    add("")
    add("lift = how much more often editors used our clips' seconds than an average second of the evaluated episodes (1.0 = random). "
        "within_lift = the same against each clip's own episode (removes episode position). hit@k = share of the top-k clips that are hits. "
        "`candidates` ranks the de-duplicated candidates by score; `slate` goes through the opening/middle/late quotas (55/33/12%), as the product does. "
        "Lift is shown as the mean of the two dramas, then A / B.")
    add("")
    bs_ = res["baselines"]
    add(f"Baselines from `benchmark_dramaad.py` (candidates@10, mean of the two dramas): random clips lift {bs_['random_lift_mean']:.2f}, "
        f"the opening seconds of each episode lift {bs_['openings_lift_mean']:.2f}.")
    add("")
    for s in SOURCES:
        add(f"### Ranked through `{s}`")
        add("")
        add("| weights | h / c / s / d | lift@10 (A / B) | lift@20 (A / B) | within_lift@10 | hit@10 | hit@20 | top-10 overlap with official (A / B) |")
        add("|---|---|---|---|---|---|---|---|")
        for name in res["named_order"]:
            r = sw[name]
            w = r["weights"]
            m10, m20 = r["mean"][f"{s}@10"], r["mean"][f"{s}@20"]
            add(f"| {name} | {w['hook']:.2f} / {w['cliff']:.2f} / {w['standalone']:.2f} / {w['density']:.2f} | "
                f"{fmt(m10['lift'])} ({fmt(r['per_drama'][A][f'{s}@10']['lift'])} / {fmt(r['per_drama'][B][f'{s}@10']['lift'])}) | "
                f"{fmt(m20['lift'])} ({fmt(r['per_drama'][A][f'{s}@20']['lift'])} / {fmt(r['per_drama'][B][f'{s}@20']['lift'])}) | "
                f"{fmt(m10['within_lift'])} | {fmt(m10['hit_rate'])} | {fmt(m20['hit_rate'])} | "
                f"{r['per_drama'][A]['overlap_with_official'][f'{s}@10']} / {r['per_drama'][B]['overlap_with_official'][f'{s}@10']} |")
        q10, q20 = ss[f"{s}@10"], ss[f"{s}@20"]
        add(f"| {res['n_dirichlet']} random vectors: median (5th-95th pct) | | "
            f"{fmt(q10['mean']['lift']['dirichlet_median'])} ({fmt(q10['mean']['lift']['dirichlet_p05'])}-{fmt(q10['mean']['lift']['dirichlet_p95'])}) | "
            f"{fmt(q20['mean']['lift']['dirichlet_median'])} ({fmt(q20['mean']['lift']['dirichlet_p05'])}-{fmt(q20['mean']['lift']['dirichlet_p95'])}) | "
            f"{fmt(q10['mean']['within_lift']['dirichlet_median'])} | {fmt(q10['mean']['hit_rate']['dirichlet_median'])} | {fmt(q20['mean']['hit_rate']['dirichlet_median'])} | "
            f"{q10['dirichlet_median_overlap_with_official'][A]:.0f} / {q10['dirichlet_median_overlap_with_official'][B]:.0f} |")
        add("")
        for k in KS:
            q = ss[f"{s}@{k}"]
            wl = q["weight_vs_lift_spearman"]
            add(f"- `{s}@{k}`: official lift at percentile {q['mean']['lift']['official_percentile']:.0%} of the random vectors on the two-drama mean "
                f"(A {q[A]['lift']['official_percentile']:.0%}, B {q[B]['lift']['official_percentile']:.0%}); {q['mean']['lift']['share_dirichlet_better']:.0%} of them do better. "
                f"within_lift percentile {q['mean']['within_lift']['official_percentile']:.0%}, hit@{k} percentile {q['mean']['hit_rate']['official_percentile']:.0%}. "
                f"Spearman of each weight with lift over the random vectors, on unrounded lifts (A / B): " + ", ".join(f"{t} {fmt(wl[A][t])} / {fmt(wl[B][t])}" for t in TERMS) + ".")
        add("")
    add(res["tie_note"])
    add("")

    add("## 3. How well does each term predict editor usage?")
    add("")
    add("Per window: `usage` = mean number of professional ads containing each of its seconds; `hit` = at least half its seconds inside some ad. "
        "Primary population (`kept`): the candidates left after overlap removal under the official weights, nearly non-overlapping. "
        "Robustness population (`all`): every generated window; many more rows, but heavily overlapping, so far fewer independent ones. "
        f"Every CI resamples whole episodes ({res['bootstrap']} bootstrap draws), because windows from one episode share its labels.")
    add("")
    for pop, title in (("kept", "kept"), ("pool", "all")):
        sp, sc_ = res["spearman"][pop], res["spearman_covered"][pop]
        add(f"### Spearman correlation with usage, `{title}` windows (A: {sp[A]['n']} windows in {sp[A]['episodes']} episodes; B: {sp[B]['n']} in {sp[B]['episodes']})")
        add("")
        add("| term | A: rho with usage [95% CI] | B: rho with usage [95% CI] | rho with covered share (A / B) |")
        add("|---|---|---|---|")
        for t in list(TERMS) + ["base", "score"]:
            label = {"base": "official weighted sum", "score": "official full score"}.get(t, t)
            add(f"| {label} | {fmt(sp[A][t]['rho'])} {fci(sp[A][t]['ci95'])} | {fmt(sp[B][t]['rho'])} {fci(sp[B][t]['ci95'])} | "
                f"{fmt(sc_[A][t]['rho'])} / {fmt(sc_[B][t]['rho'])} |")
        add("")
    we = res["within_episode_spearman"]
    add("### Inside an episode (position removed): Spearman of term vs usage per episode, all windows, partly-used episodes only")
    add("")
    add("| term | A: weighted mean rho [95% CI] (episodes; with rho > 0) | B: weighted mean rho [95% CI] (episodes; with rho > 0) |")
    add("|---|---|---|")
    for t in TERMS:
        add(f"| {t} | " + " | ".join(f"{fmt(we[i][t]['rho'])} {fci(we[i][t]['ci95'])} ({we[i][t]['episodes']}; {we[i][t]['episodes_positive']})" for i in ids) + " |")
    add("")
    hb = res["hit_by_band"]["kept"]
    add("### Hit rate by band of each term, `kept` windows: hits / windows, pooled (A; B)")
    add("")
    add("| term | " + " | ".join(lab for _, _, lab in BANDS) + " |")
    add("|---|" + "---|" * len(BANDS))
    for t in TERMS:
        cells = []
        for j in range(len(BANDS)):
            pc, ca, cb = hb[t]["pooled"][j], hb[t][A][j], hb[t][B][j]
            cells.append(f"{pc['hits']}/{pc['n']}" + (f" ({pc['hit_rate']:.0%})" if pc["n"] else "") + f" ({ca['hits']}/{ca['n']}; {cb['hits']}/{cb['n']})")
        add(f"| {t} | " + " | ".join(cells) + " |")
    add("")
    for pop, title in (("kept", "kept"), ("pool", "all")):
        lg, lgp, lgs = res["logistic"][pop], res["logistic_position"][pop], res["logistic_single"][pop]
        add(f"### Logistic regression of hit on the four standardized terms, `{title}` windows")
        add("")
        add("Coefficient = change in the log-odds of a hit per standard deviation of the term (within drama). "
            f"A weak ridge penalty (lambda = {res['ridge_lambda']}) keeps fits finite when a resample separates hits from misses; unpenalized estimates are in the JSON.")
        add("")
        add("| term | pooled [95% CI] | A [95% CI] | B [95% CI] | pooled, position control | pooled, single-episode windows only |")
        add("|---|---|---|---|---|---|")
        for t in TERMS:
            p = lg["pooled"]["terms"][t]
            add(f"| {t} | {fmt(p['coef_per_sd'])} {fci(p['ci95'])} | {fmt(lg[A]['terms'][t]['coef_per_sd'])} {fci(lg[A]['terms'][t]['ci95'])} | "
                f"{fmt(lg[B]['terms'][t]['coef_per_sd'])} {fci(lg[B]['terms'][t]['ci95'])} | "
                f"{fmt(lgp['pooled']['terms'][t]['coef_per_sd'])} {fci(lgp['pooled']['terms'][t]['ci95'])} | "
                f"{fmt(lgs['pooled']['terms'][t]['coef_per_sd'])} {fci(lgs['pooled']['terms'][t]['ci95'])} |")
        add(f"| windows (hits) | {lg['pooled']['n']} ({lg['pooled']['hits']}) | {lg[A]['n']} ({lg[A]['hits']}) | {lg[B]['n']} ({lg[B]['hits']}) | "
            f"{lgp['pooled']['n']} ({lgp['pooled']['hits']}) | {lgs['pooled']['n']} ({lgs['pooled']['hits']}) |")
        add("")
        add("Relative weights implied by the pooled fit (each coefficient divided by the sum of absolute coefficients; a negative one stays negative), against the priors:")
        add("")
        add("| term | prior weight | prior, as importance per SD | data: importance per SD [95% CI] | data: implied weight on the 0-1 scale [95% CI] |")
        add("|---|---|---|---|---|")
        for t in TERMS:
            p = lg["pooled"]["terms"][t]
            add(f"| {t} | {off[t]:.2f} | {fmt(res['prior_importance'][pop][t])} | {fmt(p['relative_importance'])} {fci(p['relative_importance_ci95'])} | "
                f"{fmt(p['implied_weight_raw_scale'])} {fci(p['implied_weight_raw_scale_ci95'])} |")
        add("")
        pt = res["proportionality"][pop]
        add("Are the four coefficients proportional to 0.35 / 0.30 / 0.25 / 0.10? Likelihood-ratio test (unpenalized, df = 3): "
            f"pooled LR = {pt['pooled']['lr_stat']:.2f}, p = {pt['pooled']['p_value_asymptotic']:.3f}; "
            f"A LR = {pt[A]['lr_stat']:.2f}, p = {pt[A]['p_value_asymptotic']:.3f}; B LR = {pt[B]['lr_stat']:.2f}, p = {pt[B]['p_value_asymptotic']:.3f}. "
            "The p-values assume independent windows; with windows clustered in episodes they are, if anything, too small.")
        add("")
        sph = res["score_predicts_hit"][pop]
        add("Does the official formula itself predict hits? Logistic of hit on the standardized official weighted sum / full score: " + "; ".join(
            f"{lab}: weighted sum {fmt(sph[scope]['base']['coef_per_sd'])} {fci(sph[scope]['base']['ci95'])}, full score {fmt(sph[scope]['score']['coef_per_sd'])} {fci(sph[scope]['score']['ci95'])}"
            for lab, scope in (("pooled", "pooled"), ("A", A), ("B", B))) + ".")
        add("")
        eu = res["episode_used_summary"][pop]
        fams = (("plain fit", "logistic"), ("+ episode-used indicator", "logistic_used"), ("+ position + episode-used", "logistic_position_used"),
                ("used episodes only", "logistic_used_only"), ("single-episode windows + episode-used", "logistic_single_used"))
        add(f"Episode-used control (added after review), `{title}` windows. In episodes no ad uses, {eu['never_used_hits']} of {eu['never_used']} windows are hits; "
            f"in used episodes, {eu['used_hits']} of {eu['used']}. Pooled coefficients per SD [95% CI]:")
        add("")
        add("| term | " + " | ".join(lab for lab, _ in fams) + " |")
        add("|---|" + "---|" * len(fams))
        for t in TERMS:
            add(f"| {t} | " + " | ".join(f"{fmt(res[f][pop]['pooled']['terms'][t]['coef_per_sd'])} {fci(res[f][pop]['pooled']['terms'][t]['ci95'])}" for _, f in fams) + " |")
        add("| episode-used indicator | " + " | ".join(
            (f"{fmt(res[f][pop]['pooled']['extra_coefs']['episode_used'])}" if "episode_used" in res[f][pop]["pooled"]["extra_coefs"] else "") for _, f in fams) + " |")
        add("| windows (hits) | " + " | ".join(f"{res[f][pop]['pooled']['n']} ({res[f][pop]['pooled']['hits']})" for _, f in fams) + " |")
        add("")

    add("## 4. Overfitting check: tune on one drama, test on the other")
    add("")
    add(f"Search space: all {len(res['sweep'])} vectors. When several vectors tie for the best training lift, the first is reported, and the tied set's mean test lift too.")
    add("")
    add("| ranking | tuned on | best training lift (official) | chosen h / c / s / d | its test lift | official test lift | median / best test lift, all vectors | test percentile of chosen | tied at best: n, mean test lift (percentile) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for key in KEYS:
        o = res["overfit"][key]
        for trn, tst, ltr in ((A, B, "A"), (B, A, "B")):
            r = o[f"train_{trn}_test_{tst}"]
            w = r["weights"]
            add(f"| {key} | {ltr} | {fmt(r['train_lift'])} ({fmt(r['train_lift_official'])}) | {w['hook']:.2f} / {w['cliff']:.2f} / {w['standalone']:.2f} / {w['density']:.2f} | "
                f"{fmt(r['test_lift'])} | {fmt(r['test_lift_official'])} | {fmt(r['test_lift_median_all_vectors'])} / {fmt(r['test_lift_best_possible'])} | "
                f"{r['test_percentile_of_chosen']:.0%} | {r['n_vectors_tied_at_best']}, {fmt(r['test_lift_tied_mean'])} ({r['test_percentile_of_tied_mean']:.0%}) |")
    add("")
    add("Do weights that work on one drama work on the other? Spearman, across all vectors, of a vector's lift on A vs on B: " + "; ".join(
        f"{key} {fmt(res['overfit'][key]['cross_drama_spearman_all_vectors'])}" for key in KEYS)
        + ". Transfers (rule in 'How this was computed'): " + "; ".join(f"{k} {'yes' if v else 'no'}" for k, v in res["transfers"].items()) + ".")
    add("")

    add("## 5. Verdict on the 0.35 / 0.30 / 0.25 / 0.10 priors")
    add("")
    cd = res["config_decision"]
    add(f"**Decision: {'no change to config.yaml' if not cd['change_config'] else 'the data point to a change, to be confirmed on held-out shows'}.** "
        "Reasons: " + "; ".join(cd["reasons"]) + ".")
    add("")
    add(f"Decision rule (written after the results were seen, in response to review): {cd['rule'].split(' (written')[0]}.")
    add("")
    add("Two separate questions are asked here. The **sign rule** asks whether a term's weight should be positive at all. The **value test** "
        "asks whether the data rule out the prior's size. Only the value test speaks to 0.35 / 0.30 / 0.25 / 0.10.")
    add("")
    add("### Sign rule")
    add("")
    add("Set before the results were read, by the author's account (not checkable from the files). A term gets **sign: positive** when its "
        "pooled logistic coefficient has a 95% CI above 0 and neither drama's CI is below 0, and **sign: negative** when the pooled CI is below 0 "
        "and neither drama's CI is above 0. It gets **sign: differs between dramas** when one drama's CI is above 0 and the other's below. "
        "Otherwise it is **sign: cannot judge**.")
    add("")
    add("| term | prior | sign, kept windows (primary) | sign, all windows | robust on every pre-set check? | added after review: episode-used control / used episodes only |")
    add("|---|---|---|---|---|---|")
    for t in TERMS:
        c = res["corroboration"].get(t)
        rob = "n/a" if c is None else ("yes" if c["corroborated"] else "no: " + ", ".join(k for k, ok in c["checks"].items() if not ok))
        ex = "n/a" if c is None else " / ".join("holds" if v else "does not hold" for v in c["extra_checks_added_after_review"].values())
        add(f"| {t} | {off[t]:.2f} | {res['verdicts']['kept'][t]} | {res['verdicts']['pool'][t]} | {rob} | {ex} |")
    add("")
    add(f"Overall: **{res['overall_verdict']['kept']}** (kept windows); {res['overall_verdict']['pool']} (all windows).")
    add("")
    add("### Value test: do the intervals exclude the prior values?")
    add("")
    pv = res["prior_test"]
    nb = pv["kept"]["draws_corrected"]
    add(f"Added after review. On the 0-1 scale the prior is the weight itself; per SD it is weight x SD, normalized (section 3). Uncorrected = "
        f"the 95% intervals from the main {pv['kept']['draws_uncorrected']}-draw bootstrap (as in section 3). Corrected = Bonferroni for the four "
        f"terms ({LEVELS['bonferroni_4']:.2%}) and, stricter, for 4 terms x 2 scales ({LEVELS['bonferroni_8']:.3%}), from {nb} draws. "
        "'below' / 'above' = the interval lies entirely below / above the prior.")
    add("")
    for pop, title in (("kept", "kept windows (primary)"), ("pool", "all windows")):
        add(f"**{title}**")
        add("")
        add("| term | scale | prior | data [95% CI] | excludes? | 4-term corrected CI | excludes? | 8-test corrected CI | excludes? |")
        add("|---|---|---|---|---|---|---|---|---|")
        for t in TERMS:
            for s, _, lab in SCALES:
                e = pv[pop]["terms"][t][s]
                add(f"| {t} | {lab} | {e['prior']:.3f} | {fmt(e['point'], 3)} {fci(e['ci95'], 3)} | {e['excludes_prior_95'] or 'no'} | "
                    f"{fci(e[f'ci_bonferroni_4_{nb}draws'], 3)} | {e[f'excludes_prior_bonferroni_4_{nb}draws'] or 'no'} | "
                    f"{fci(e[f'ci_bonferroni_8_{nb}draws'], 3)} | {e[f'excludes_prior_bonferroni_8_{nb}draws'] or 'no'} |")
        prop_ = res["proportionality"][pop]["pooled"]
        add("")
        add(f"Joint test (are the four coefficients proportional to the priors?): likelihood ratio {prop_['lr_stat']:.2f}, df 3, "
            f"p = {prop_['p_value_asymptotic']:.3f}" + (" (windows overlap heavily here, so this p-value is too small)" if pop == "pool" else "") + ".")
        add("")
    g = res["prior_grid"]
    add(f"Seed x ridge grid, kept windows ({g['n_cells']} cells: bootstrap seeds {', '.join(map(str, g['seeds']))} x ridge "
        f"{', '.join(str(x) for x in g['ridge_lambdas'])}; {g['draws']} draws each). Number of cells whose interval excludes the prior:")
    add("")
    add("| term | 0-1 scale: uncorrected | 0-1 scale: 4-term corrected | per SD: uncorrected | per SD: 4-term corrected |")
    add("|---|---|---|---|---|")
    for t in TERMS:
        cnt = g["cells_excluding_prior"][t]
        add(f"| {t} | {cnt['weight_0_1']['uncorrected']} | {cnt['weight_0_1']['bonferroni_4']} | "
            f"{cnt['importance_per_sd']['uncorrected']} | {cnt['importance_per_sd']['bonferroni_4']} |")
    add("")
    vr = res["prior_test_robustness"]["kept"]
    add("The uncorrected value test on each variant fit, kept windows: share [95% CI] against the prior. "
        "'below' / 'above' marks an interval that excludes the prior. Per-SD priors use each fit's own term SDs.")
    add("")
    add("| term | scale | " + " | ".join(f"{v} ({vr[v]['n']} windows)" for v in vr) + " |")
    add("|---|---|" + "---|" * len(vr))
    for t in TERMS:
        for s, _, lab in SCALES:
            add(f"| {t} | {lab} | " + " | ".join(
                f"{fmt(vr[v]['terms'][t][s]['point'])} {fci(vr[v]['terms'][t][s]['ci95'])} vs {vr[v]['terms'][t][s]['prior']:.2f}"
                + (f" **{vr[v]['terms'][t][s]['excludes_prior_95']}**" if vr[v]["terms"][t][s]["excludes_prior_95"] else "") for v in vr) + " |")
    add("")
    for s in res["verdict_text"]:
        add(s)
        add("")
    add("## How this was computed")
    add("")
    for s in res["method"]:
        add(f"- {s}")
    add("")
    add("Reproduce: `uv run python scripts/weights_vs_editors.py --dramaad <DramaAD dir> --runs <run A> <run B> --published <files> --out <dir>` "
        "(the exact paths are in `editors.json` -> inputs).")
    add("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dramaad", type=Path, required=True, help="DramaAD folder: <id>/{multimodal_narration_understanding.json,script.json}")
    ap.add_argument("--runs", type=Path, nargs=2, required=True, help="the two promo-finder run folders (<runs>/<drama id>/<run id>)")
    ap.add_argument("--published", type=Path, nargs="*", default=[], help="published benchmark JSON files to reproduce")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--dirichlet", type=int, default=500)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--shuffles", type=int, default=20, help="random tie-break orders per named vector")
    ap.add_argument("--ridge", type=float, default=1.0, help="L2 penalty on the standardized logistic coefficients")
    ap.add_argument("--boot-corrected", type=int, default=4000, help="bootstrap draws for the Bonferroni-corrected prior-value intervals")
    ap.add_argument("--grid-seeds", type=int, default=5, help="bootstrap seeds in the prior-value grid (bootstrap seed, +1, ...)")
    ap.add_argument("--grid-ridges", type=float, nargs="+", default=[0.0, 1.0, 3.0], help="ridge penalties in the prior-value grid")
    ap.add_argument("--grid-boot", type=int, default=4000, help="bootstrap draws per grid cell")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    dramas = [Drama(r, a.dramaad, a.episodes) for r in a.runs]
    ids = [d.id for d in dramas]
    official = tuple(float(dramas[0].cfg["scoring"][f"w_{t}"]) for t in TERMS)
    if any(tuple(float(d.cfg["scoring"][f"w_{t}"]) for t in TERMS) != official for d in dramas):
        raise SystemExit("the runs were made with different weights")
    seeds = {"dirichlet": a.seed, "bootstrap": a.seed + 1, "shuffle": a.seed + 2}
    print(f"replayed {len(dramas)} runs ({time.time() - t0:.0f}s): " + ", ".join(f"{d.id} {len(d.windows)} windows" for d in dramas))

    rep = reproduction(dramas, official, a.published)
    print(f"reproduction: all match = {rep['all_match']}")
    vecs = weight_vectors(official, a.dirichlet, seeds["dirichlet"])
    sweep = run_sweep(dramas, vecs)
    print(f"sweep: {len(vecs)} vectors ({time.time() - t0:.0f}s)")
    ties = tie_robustness(dramas, vecs, a.shuffles, seeds["shuffle"])
    print(f"tie-break robustness ({time.time() - t0:.0f}s)")
    summ = sweep_summary(sweep, ids)
    over = overfit_check(sweep, ids)

    kept_rows, pool_rows = [], []
    for d in dramas:
        kept, _ = d.rank(official)  # puts the official scores back on every window
        kept_rows += d.rows(kept)
        pool_rows += d.rows(d.windows)
    pops = {"kept": kept_rows, "pool": pool_rows}
    B_ = a.boot
    spear = {p: {i: spearman_terms([r for r in rows if r["drama"] == i], "usage", B_, seeds["bootstrap"]) for i in ids} for p, rows in pops.items()}
    spear_w = {p: {i: spearman_terms([r for r in rows if r["drama"] == i], "within", B_, seeds["bootstrap"]) for i in ids} for p, rows in pops.items()}
    spear_c = {p: {i: spearman_terms([r for r in rows if r["drama"] == i], "covered", B_, seeds["bootstrap"]) for i in ids} for p, rows in pops.items()}
    within_ep = {i: within_episode_spearman([r for r in pool_rows if r["drama"] == i], B_, seeds["bootstrap"]) for i in ids}
    logi, logi_pos, logi_one, sph, prop, prior_imp = {}, {}, {}, {}, {}, {}
    logi_used, logi_pos_used, logi_one_used, logi_used_only, ep_used = {}, {}, {}, {}, {}
    for p, rows in pops.items():
        logi[p] = {"pooled": logistic_analysis(rows, ids, a.ridge, B_, seeds["bootstrap"])}
        logi_pos[p] = {"pooled": logistic_analysis(rows, ids, a.ridge, B_, seeds["bootstrap"], position=True)}
        single = [r for r in rows if r["kind"] == "single"]  # no episode-boundary or two-part clips
        logi_one[p] = {"pooled": logistic_analysis(single, ids, a.ridge, B_, seeds["bootstrap"])}
        # control for whether any ad uses the window's episode (added after review)
        used_rows = [r for r in rows if r["episode_used"]]
        logi_used[p] = {"pooled": logistic_analysis(rows, ids, a.ridge, B_, seeds["bootstrap"], extra=("episode_used",))}
        logi_pos_used[p] = {"pooled": logistic_analysis(rows, ids, a.ridge, B_, seeds["bootstrap"], position=True, extra=("episode_used",))}
        logi_one_used[p] = {"pooled": logistic_analysis(single, ids, a.ridge, B_, seeds["bootstrap"], extra=("episode_used",))}
        logi_used_only[p] = {"pooled": logistic_analysis(used_rows, ids, a.ridge, B_, seeds["bootstrap"])}
        low_nonhit = [r for r in rows if r["density"] < BANDS[0][1] and not r["hit"]]
        ep_used[p] = {"never_used": sum(1 for r in rows if not r["episode_used"]), "never_used_hits": sum(r["hit"] for r in rows if not r["episode_used"]),
                      "used": len(used_rows), "used_hits": sum(r["hit"] for r in used_rows),
                      "low_density_nonhits": len(low_nonhit), "low_density_nonhits_in_used": sum(r["episode_used"] for r in low_nonhit),
                      "per_drama": {i: {"never_used": sum(1 for r in rows if r["drama"] == i and not r["episode_used"]),
                                        "never_used_hits": sum(r["hit"] for r in rows if r["drama"] == i and not r["episode_used"]),
                                        "used": sum(1 for r in used_rows if r["drama"] == i),
                                        "used_hits": sum(r["hit"] for r in used_rows if r["drama"] == i)} for i in ids}}
        sph[p] = {"pooled": {c: score_predicts_hit(rows, ids, c, a.ridge, B_, seeds["bootstrap"]) for c in ("base", "score")}}
        prop[p] = {"pooled": proportionality_test(rows, ids, official)}
        for i in ids:
            sub = [r for r in rows if r["drama"] == i]
            logi[p][i] = logistic_analysis(sub, [i], a.ridge, B_, seeds["bootstrap"])
            logi_pos[p][i] = logistic_analysis(sub, [i], a.ridge, B_, seeds["bootstrap"], position=True)
            logi_one[p][i] = logistic_analysis([r for r in single if r["drama"] == i], [i], a.ridge, B_, seeds["bootstrap"])
            sph[p][i] = {c: score_predicts_hit(sub, [i], c, a.ridge, B_, seeds["bootstrap"]) for c in ("base", "score")}
            prop[p][i] = proportionality_test(sub, [i], official)
        sds = [logi[p]["pooled"]["terms"][t]["sd"] for t in TERMS]
        tot = sum(w * s for w, s in zip(official, sds))
        prior_imp[p] = {t: round(w * s / tot, 3) for t, w, s in zip(TERMS, official, sds)}
    print(f"per-term statistics ({time.time() - t0:.0f}s)")
    off_d = dict(zip(TERMS, official))
    if abs(sum(official) - 1.0) > 1e-9:
        raise SystemExit("the official weights do not sum to 1, so the 0-1-scale prior shares need normalizing first")
    prior_test = {}
    for p, rows in pops.items():
        corr_fit = logistic_analysis(rows, ids, a.ridge, a.boot_corrected, seeds["bootstrap"])
        prior_test[p] = prior_value_test(logi[p]["pooled"], corr_fit, off_d, prior_imp[p])
    grid = prior_value_grid(kept_rows, ids, [seeds["bootstrap"] + i for i in range(a.grid_seeds)], a.grid_ridges, a.grid_boot, off_d, prior_imp["kept"])
    value_rob = {p: value_robustness({"plain fit": logi[p]["pooled"], "position control": logi_pos[p]["pooled"],
                                      "single-episode windows only": logi_one[p]["pooled"], "episode-used control": logi_used[p]["pooled"],
                                      "position + episode-used control": logi_pos_used[p]["pooled"],
                                      "used episodes only": logi_used_only[p]["pooled"]}, off_d) for p in pops}
    print(f"prior-value tests ({time.time() - t0:.0f}s)")
    verdicts = {p: {t: term_verdict(logi[p]["pooled"], [logi[p][i] for i in ids], t) for t in TERMS} for p in pops}

    desc = [d.describe(official) for d in dramas]
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "question": "Do the ranking coefficients (0.35 hook, 0.30 cliff, 0.25 standalone, 0.10 density) match what professional editors pick?",
        "inputs": {"runs": [str(r) for r in a.runs], "dramaad": str(a.dramaad), "published": [str(p) for p in a.published], "episodes": a.episodes},
        "official_weights": dict(zip(TERMS, official)), "seeds": seeds, "n_dirichlet": a.dirichlet, "bootstrap": a.boot, "shuffles": a.shuffles,
        "bootstrap_corrected": a.boot_corrected, "ridge_lambda": a.ridge, "dramas": desc,
        "totals": {"dramas": len(dramas), "episodes": sum(d.n_eps for d in dramas), "candidate_windows": sum(len(d.windows) for d in dramas),
                   "kept": len(kept_rows), "ads": sum(x["ads"] for x in desc), "ads_using_evaluated": sum(x["ads_using_evaluated_episodes"] for x in desc),
                   "ad_length_text": " and ".join(f"{x['ad_length_s']['min']:.0f}-{x['ad_length_s']['max']:.0f} s" for x in desc),
                   "segment_text": " and ".join(f"{x['ad_segment_length_s']['median']:.0f} s" for x in desc)},
        "reproduction": rep,
        "baselines": {
            "random_lift_mean": round(st.mean(rep["evaluate_run"][f"{i} candidates@10"]["random"]["lift"] for i in ids), 3),
            "openings_lift_mean": round(st.mean(rep["evaluate_run"][f"{i} candidates@10"]["episode_openings"]["lift"] for i in ids), 3),
            "per_case": {k: {"random": v["random"], "episode_openings": v["episode_openings"], "episode_rho": v["episode_rho"]} for k, v in rep["evaluate_run"].items()}},
        "named_order": [v["name"] for v in vecs if v["family"] != "dirichlet"],
        "sweep": sweep, "sweep_summary": summ, "tie_robustness": ties, "overfit": over,
        "spearman": spear, "spearman_covered": spear_c, "spearman_within_episode_ratio": spear_w, "within_episode_spearman": within_ep,
        "logistic": logi, "logistic_position": logi_pos, "logistic_single": logi_one, "score_predicts_hit": sph, "proportionality": prop,
        "logistic_used": logi_used, "logistic_position_used": logi_pos_used, "logistic_single_used": logi_one_used,
        "logistic_used_only": logi_used_only, "episode_used_summary": ep_used,
        "prior_importance": prior_imp, "prior_test": prior_test, "prior_grid": grid, "prior_test_robustness": value_rob,
        "hit_by_band": {p: hit_by_band(rows, ids) for p, rows in pops.items()},
        "verdicts": verdicts, "overall_verdict": {p: overall_verdict(verdicts[p]) for p in pops},
        "rows_kept": kept_rows,
    }
    res.update(narrative(res, ids))
    md = markdown(res)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "editors.json").write_text(json.dumps(strip_private(res), indent=1, ensure_ascii=False), encoding="utf-8")
    (a.out / "editors.md").write_text(md, encoding="utf-8")
    print(f"wrote {a.out / 'editors.json'} and {a.out / 'editors.md'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
