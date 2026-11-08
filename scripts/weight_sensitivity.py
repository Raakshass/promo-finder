"""How much does the posted slate depend on the four hand-set weights? Coefficient sensitivity on a finished run.

  score = (w_hook*hook + w_cliff*cliff + w_standalone*standalone + w_density*density)
          x a penalty per failed editor's check x 1.05 turning-point episode x 0.9 two-segment join   (rank.combine)

Everything is rebuilt offline from the run folder, with zero model calls: stability.replay() for the scored show, the
run's own config.yaml for every other knob, and the first-time-viewer answers stored in <run>/llm_raw/cold_viewer/,
re-attached to the clips each call was given and applied as cli.py does (apply_judge, then the final slate). The script
first checks that this rebuild reproduces promos.json, then re-ranks under other weight vectors:
  - each weight +-0.05 and +-0.10, the other three rescaled so the four still sum to 1
  - leave one term out (renormalised), equal weights, each term alone
  - --draws vectors from Dirichlet(20 x official weights) (near the official weights) and as many from
    Dirichlet(1,1,1,1) (anywhere on the simplex)
Only the four weights change; penalties, quotas and diversity rules stay as configured.

The judge only ever saw the official shortlist (judge.top_n clips, 40 in a default run), and judging a new clip would
need a model call. So each weight vector is re-ranked three ways, which bracket the answer:
  judged_pool   the judged shortlist only, re-scored and re-slated: full information, but no new clip can come in
                (understates change)
  full_imputed  the whole pipeline (re-score, de-duplicate, shortlist, judge, final slate); a shortlisted clip
                the official run never judged gets the median judge effect (score after / before) measured on the
                judged clips of the same re-rank
  full_raw      the whole pipeline; a never-judged clip keeps its code-only score. The judge lowers most scores, so
                this favours never-judged clips (overstates change)
This measures stability, not correctness: no traction data is used, so it cannot say which weights are right.

  uv run python scripts/weight_sensitivity.py <run_dir> --out sensitivity.json [--md sensitivity.md] [--seed 7]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics as st
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promo_finder import rubric as rb  # noqa: E402
from promo_finder import slate as sl  # noqa: E402
from promo_finder.judge import ColdViewerBatchOut  # noqa: E402
from promo_finder.llm.base import _extract_json  # noqa: E402  (JSON parsing only; nothing here builds a model client)
from promo_finder.models import ColdViewer, Window  # noqa: E402
from promo_finder.rank import apply_judge, combine, diversity_order, flag_spoilers, generate_windows, iou, nms, score_window  # noqa: E402
from promo_finder.stability import replay  # noqa: E402

TERMS = ("hook", "cliff", "standalone", "density")
MODES = ("judged_pool", "full_imputed", "full_raw")
JUDGE_SHUFFLE_SEED = 7  # judge.cold_viewer() shuffles the shortlist with this seed before batching it


# ---------------------------------------------------------------- loading -------------------------------------------
def load_cfg(run: Path, override: Path | None) -> tuple[dict, str]:
    if override:
        return yaml.safe_load(override.read_text()), str(override)
    if (run / "config.yaml").exists():
        return yaml.safe_load((run / "config.yaml").read_text()), "the run's own config.yaml"
    return yaml.safe_load((ROOT / "config.yaml").read_text()), "repo config.yaml (the run folder has none)"


def show_dir_of(run: Path, override: Path | None) -> Path:
    if override:
        return override
    p = Path(json.loads((run / "run_log.json").read_text())["args"]["show_dir"])
    return p if p.is_absolute() or p.exists() else ROOT / p  # run_log keeps the path as typed, usually repo-relative


def stored_answers(run: Path) -> tuple[dict[str, dict[str, ColdViewer]], list[str]]:
    """file name -> {clip id -> judgment}, parsed from the raw first-time-viewer answers the run kept."""
    folder = run / "llm_raw" / "cold_viewer"
    out: dict[str, dict[str, ColdViewer]] = {}
    errors: list[str] = []
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            data = _extract_json(json.loads(f.read_text())["response_text"])
            if isinstance(data, dict) and "clips" in data:
                clips = ColdViewerBatchOut.model_validate(data).clips
                out[f.name] = {c.clip_id.strip(): ColdViewer(**{k: getattr(c, k) for k in ColdViewer.model_fields}) for c in clips}
            else:  # one clip per call: the file is named after the clip
                out[f.name] = {f.stem: ColdViewer.model_validate(data)}
        except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
            errors.append(f"{f.name}: {type(e).__name__}")
    return out, errors


def judge_calls(shortlist: list[Window], cfg: dict) -> dict[str, list[str]]:
    """Which clips each first-time-viewer call was given, rebuilt the way judge.cold_viewer() batches them."""
    order = list(shortlist)
    random.Random(JUDGE_SHUFFLE_SEED).shuffle(order)
    per = int(cfg["judge"].get("clips_per_call", 1))
    if per <= 1:
        return {f"{w.cand_id}.json": [w.cand_id] for w in order}
    return {f"batch-{i // per + 1:02d}.json": [w.cand_id for w in order[i:i + per]] for i in range(0, len(order), per)}


def attach_judgments(answers: dict[str, dict[str, ColdViewer]], calls: dict[str, list[str]]) -> tuple[dict[str, ColdViewer], dict]:
    """cli.py keeps, from each call, only the answers for the clips that call was given. Do the same when the stored
    files line up with the rebuilt calls; otherwise fall back to matching every stored answer by clip id."""
    foreign = sum(1 for f, ids in calls.items() if f in answers for i in answers[f] if i not in ids)
    if all(f in answers for f in calls) and foreign == 0:
        judged = {i: answers[f][i] for f, ids in calls.items() for i in ids if i in answers[f]}
        how = "re-attached call by call, as cli.py applied them"
    else:
        judged = {i: cv for f in sorted(answers) for i, cv in answers[f].items()}
        how = "stored files do not line up with the rebuilt calls: every stored answer matched by clip id"
    return judged, {"calls_rebuilt": len(calls), "files_found": sum(1 for f in calls if f in answers),
                    "clips_without_answer": sum(1 for f, ids in calls.items() for i in ids if i not in answers.get(f, {})),
                    "answers_for_clips_not_in_that_call": foreign, "method": how}


# ---------------------------------------------------------------- re-ranking ----------------------------------------
def with_weights(cfg: dict, w: dict[str, float]) -> dict:
    c = copy.deepcopy(cfg)
    c["scoring"].update({f"w_{t}": float(w[t]) for t in TERMS})
    return c


@dataclass
class Result:
    shortlist: list[str]
    final: list[str]


@dataclass
class FullRun:
    raw: Result
    imputed: Result
    pre: np.ndarray            # every candidate's score before the judge, in generation order
    judge_ratio: float | None  # median score after / before the judge over the judged shortlisted clips
    never_judged: int          # shortlisted clips without a stored judgment


class Reranker:
    """Holds every scored candidate once; each re-rank restores the pristine sub-scores and checks, then runs the
    repo's own combine / nms / slate / apply_judge, so the arithmetic is the pipeline's, not a copy of it."""

    def __init__(self, windows: list[Window], judged: dict[str, ColdViewer], cfg: dict):
        self.windows, self.judged, self.cfg = windows, judged, cfg
        self.by_id = {w.cand_id: w for w in windows}
        self.idx = {w.cand_id: i for i, w in enumerate(windows)}
        self.pristine = [(dict(w.subscores), dict(w.rubric)) for w in windows]

    def reset(self) -> None:
        for w, (sub, rub) in zip(self.windows, self.pristine):
            w.subscores, w.rubric = dict(sub), dict(rub)

    def full(self, weights: dict[str, float]) -> FullRun:
        cfg = with_weights(self.cfg, weights)
        n_final = cfg["output"]["final_top_n"]
        self.reset()
        for w in self.windows:
            combine(w, cfg)
        pre = np.array([w.score for w in self.windows])
        shortlist = sl.build(nms(self.windows, cfg), cfg, cfg["judge"]["top_n"], diversity_order)
        apply_judge(shortlist, self.judged, cfg)
        ids = [w.cand_id for w in shortlist]
        raw = Result(ids, [w.cand_id for w in sl.build(shortlist, cfg, n_final, diversity_order)])
        ratios = [w.score / pre[self.idx[w.cand_id]] for w in shortlist if w.cand_id in self.judged and pre[self.idx[w.cand_id]] > 0]
        ratio = float(np.median(ratios)) if ratios else None
        unjudged = [w for w in shortlist if w.cand_id not in self.judged]
        if ratio is not None:
            for w in unjudged:
                w.score = round(w.score * ratio, 4)
        imputed = Result(ids, [w.cand_id for w in sl.build(shortlist, cfg, n_final, diversity_order)])
        return FullRun(raw, imputed, pre, ratio, len(unjudged))

    def pool(self, weights: dict[str, float], pool_ids: list[str]) -> Result:
        cfg = with_weights(self.cfg, weights)
        self.reset()
        pool = [self.by_id[c] for c in pool_ids]
        for w in pool:
            combine(w, cfg)
        shortlist = sl.build(pool, cfg, len(pool), diversity_order)  # the same clips, in the order the pipeline hands on
        apply_judge(shortlist, self.judged, cfg)
        final = sl.build(shortlist, cfg, cfg["output"]["final_top_n"], diversity_order)
        return Result([w.cand_id for w in shortlist], [w.cand_id for w in final])


# ---------------------------------------------------------------- statistics ----------------------------------------
def kendall_tau(x: list[float], y: list[float]) -> float | None:
    """Kendall tau-b (ties allowed); None if fewer than two items or one side is constant."""
    x, y = [float(v) for v in x], [float(v) for v in y]
    n = len(x)
    if n < 2:
        return None
    s = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            s += dx * dy
            tx += dx == 0
            ty += dy == 0
    n0 = n * (n - 1) / 2
    den = ((n0 - tx) * (n0 - ty)) ** 0.5
    return s / den if den else None


class PairSigns:
    """Kendall tau-b against a fixed reference vector, vectorised for the ~1,400 candidates of a show."""

    def __init__(self, ref: np.ndarray):
        self.i, self.j = np.triu_indices(len(ref), 1)
        self.ref = np.sign(ref[self.i] - ref[self.j]).astype(np.int64)
        self.n0, self.t_ref = len(self.i), int((self.ref == 0).sum())

    def tau(self, y: np.ndarray) -> float | None:
        sy = np.sign(y[self.i] - y[self.j]).astype(np.int64)
        den = ((self.n0 - self.t_ref) * (self.n0 - int((sy == 0).sum()))) ** 0.5
        return float((self.ref * sy).sum() / den) if den else None


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    ra, rb_ = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    if len(set(ra)) < 2 or len(set(rb_)) < 2:
        return None
    return round(float(np.corrcoef(ra, rb_)[0, 1]), 3)


def dist(vals: list) -> dict | None:
    v = [x for x in vals if x is not None]
    if not v:
        return None
    a = np.array(v, dtype=float)
    return {"n": len(v), "median": round(float(np.median(a)), 3), "p10": round(float(np.percentile(a, 10)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "mean": round(float(a.mean()), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def dirichlet(rng: random.Random, alpha: list[float]) -> dict[str, float]:
    g = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(g)
    return {t: x / s for t, x in zip(TERMS, g)}


# ---------------------------------------------------------------- scenarios -----------------------------------------
def shifted(prior: dict[str, float], term: str, delta: float) -> dict[str, float] | None:
    new = prior[term] + delta
    if not 0.0 <= new <= 1.0:
        return None
    rest = 1.0 - prior[term]
    return {t: (new if t == term else prior[t] * (1.0 - new) / rest) for t in TERMS}


def named_scenarios(prior: dict[str, float]) -> list[tuple[str, str, dict[str, float]]]:
    out = []
    for t in TERMS:
        for d in (+0.05, -0.05, +0.10, -0.10):
            w = shifted(prior, t, d)
            if w:
                out.append((f"{t} {d:+.2f}", "plus_minus_0.05" if abs(d) < 0.075 else "plus_minus_0.10", w))
    for t in TERMS:
        rest = 1.0 - prior[t]
        out.append((f"without {t}", "leave_one_out", {u: (0.0 if u == t else prior[u] / rest) for u in TERMS}))
    out.append(("equal weights", "equal", {t: 0.25 for t in TERMS}))
    for t in TERMS:
        out.append((f"{t} only", "single_term", {u: float(u == t) for u in TERMS}))
    return out


# ---------------------------------------------------------------- metrics -------------------------------------------
class Scorer:
    def __init__(self, rr: Reranker, base: Result, judged_rubric: dict[str, dict[str, bool]]):
        self.rr, self.base, self.jrub = rr, base, judged_rubric
        self.thr = float(rr.cfg["scoring"]["nms_iou"])
        self.b10, self.b20 = base.final[:10], base.final[:20]

    def moment(self, a: Window, b: Window) -> bool:
        """The same moment: the same opening line, the same closing line, or time overlap (intersection over union) at
        or above scoring.nms_iou."""
        return a.hook_beat == b.hook_beat or a.cliff_beat == b.cliff_beat or iou(a, b) >= self.thr

    def duplicate(self, a: Window, b: Window) -> bool:
        """rank.nms()'s wider rule: also when one clip's closing line is the other's opening line. De-duplication keeps
        only one of two such clips, but they are neighbouring moments, so they do not count as the same moment."""
        return bool({a.hook_beat, a.cliff_beat} & {b.hook_beat, b.cliff_beat}) or iou(a, b) >= self.thr

    def same_moment(self, a: list[str], b: list[str]) -> int:
        bw = [self.rr.by_id[y] for y in b]
        return sum(1 for x in a if any(self.moment(self.rr.by_id[x], w) for w in bw))

    def quality(self, ids: list[str]) -> dict:
        j = [c for c in ids if c in self.rr.judged]
        if not j:
            return {"judged": 0, "of": len(ids), "check_pass_rate": None, "all_checks_pass": None, "follows_mean": None, "would_tap_mean": None}
        rubs = [self.jrub[c] for c in j]
        return {"judged": len(j), "of": len(ids),
                "check_pass_rate": round(st.mean(sum(r.values()) / len(r) for r in rubs), 3),
                "all_checks_pass": sum(1 for r in rubs if all(r.values())),
                "follows_mean": round(st.mean(self.rr.judged[c].comprehensibility for c in j), 2),
                "would_tap_mean": round(st.mean(self.rr.judged[c].would_tap for c in j), 2)}

    def shape(self, ids: list[str]) -> dict:
        def blocks(xs):
            c = Counter(self.rr.by_id[x].flags.get("block") for x in xs)
            return {b: c.get(b, 0) for b in sl.BLOCKS}
        return {"top20_blocks": blocks(ids[:20]), "top10_blocks": blocks(ids[:10]),
                "distinct_episodes_top20": len({int(self.rr.by_id[x].flags["episode"]) for x in ids[:20]}),
                "kinds_top20": dict(sorted(Counter(self.rr.by_id[x].kind for x in ids[:20]).items()))}

    def compare(self, res: Result, detail: bool = False) -> dict:
        s10, s20 = res.final[:10], res.final[:20]
        shared = [c for c in self.b20 if c in s20]
        pos = {c: i for i, c in enumerate(s20)}
        tau = kendall_tau(list(range(len(shared))), [pos[c] for c in shared])
        k10, k20 = len(set(self.b10) & set(s10)), len(set(self.b20) & set(s20))
        m10, m20 = self.same_moment(self.b10, s10), self.same_moment(self.b20, s20)
        out = {
            "top10_overlap": round(k10 / len(self.b10), 3),
            "top20_overlap": round(k20 / len(self.b20), 3),
            "top10_overlap_same_moment": round(m10 / len(self.b10), 3),
            "top20_overlap_same_moment": round(m20 / len(self.b20), 3),
            "kendall_tau_shared_top20": round(tau, 3) if tau is not None else None,
            "shared_top20": len(shared),
            "baseline_top10_kept_in_top10": k10,
            "baseline_top10_kept_in_top20": len(set(self.b10) & set(s20)),
            "baseline_top20_kept_in_top20": k20,
            "baseline_top10_same_moment_in_top10": m10,
            "baseline_top20_same_moment_in_top20": m20,
            "top10_identical_order": s10 == self.b10,
            "shape": self.shape(res.final),
            "shortlist_judged": sum(1 for c in res.shortlist if c in self.rr.judged),
            "shortlist_size": len(res.shortlist),
            "top10_quality": self.quality(s10),
            "top20_quality": self.quality(s20),
        }
        if detail:
            out["entered_top10"] = [self.describe(c) for c in s10 if c not in self.b10]
            out["left_top10"] = [self.label(c) for c in self.b10 if c not in s10]
            out["top10"], out["top20"] = [self.label(c) for c in s10], [self.label(c) for c in s20]
        return out

    def label(self, c: str) -> str:
        return f"P{self.b20.index(c) + 1:02d}" if c in self.b20 else c

    def describe(self, c: str) -> dict:
        w = self.rr.by_id[c]
        return {"id": self.label(c), "cand_id": c, "episode": int(w.flags["episode"]), "block": w.flags.get("block"),
                "kind": w.kind, "judged": c in self.rr.judged, "on_baseline_slate": c in self.b20,
                "same_moment_as": [self.label(b) for b in self.b20 if b != c and self.moment(w, self.rr.by_id[b])],
                "dedup_rival_of": [self.label(b) for b in self.b20 if b != c and self.duplicate(w, self.rr.by_id[b])
                                   and not self.moment(w, self.rr.by_id[b])]}


ROW_FIELDS = {"w": list(TERMS), "tau_all": "score_order_tau_all_candidates", "sj": "shortlist_judged (of 40)",
              "ratio": "median judge effect used for never-judged clips",
              "per mode": ["top10_kept", "top20_kept", "top10_same_moment", "top20_same_moment", "kendall_tau_shared_top20",
                           "top10_judged", "top10_follows_mean", "top10_would_tap_mean", "top10_check_pass_rate"]}


def compact(weights: dict[str, float], fr: FullRun, tau_all: float | None, cmp: dict[str, dict]) -> dict:
    row = {"w": [round(weights[t], 4) for t in TERMS], "tau_all": round(tau_all, 4) if tau_all is not None else None,
           "sj": cmp["full_raw"]["shortlist_judged"], "ratio": round(fr.judge_ratio, 4) if fr.judge_ratio is not None else None}
    for m in MODES:
        c, q = cmp[m], cmp[m]["top10_quality"]
        row[m] = [c["baseline_top10_kept_in_top10"], c["baseline_top20_kept_in_top20"], c["baseline_top10_same_moment_in_top10"],
                  c["baseline_top20_same_moment_in_top20"], c["kendall_tau_shared_top20"], q["judged"], q["follows_mean"],
                  q["would_tap_mean"], q["check_pass_rate"]]
    return row


def summarise(cmps: list[dict[str, dict]], base_shape: dict) -> dict:
    out = {}
    for m in MODES:
        xs = [c[m] for c in cmps]
        out[m] = {
            "top10_kept": dist([x["baseline_top10_kept_in_top10"] for x in xs]),
            "top20_kept": dist([x["baseline_top20_kept_in_top20"] for x in xs]),
            "top10_kept_same_moment": dist([x["baseline_top10_same_moment_in_top10"] for x in xs]),
            "top20_kept_same_moment": dist([x["baseline_top20_same_moment_in_top20"] for x in xs]),
            "kendall_tau_shared_top20": dist([x["kendall_tau_shared_top20"] for x in xs]),
            "share_top10_set_unchanged": round(sum(1 for x in xs if x["baseline_top10_kept_in_top10"] == 10) / len(xs), 3),
            "share_top10_order_unchanged": round(sum(1 for x in xs if x["top10_identical_order"]) / len(xs), 3),
            "share_slate_shape_unchanged": round(sum(1 for x in xs if x["shape"]["top20_blocks"] == base_shape["top20_blocks"]) / len(xs), 3),
            "top10_judged": dist([x["top10_quality"]["judged"] for x in xs]),
            "top10_follows_mean": dist([x["top10_quality"]["follows_mean"] for x in xs]),
            "top10_would_tap_mean": dist([x["top10_quality"]["would_tap_mean"] for x in xs]),
            "top10_check_pass_rate": dist([x["top10_quality"]["check_pass_rate"] for x in xs]),
        }
    return out


# ---------------------------------------------------------------- main ----------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a finished run folder (outputs/<show>/<run_id>)")
    ap.add_argument("--out", type=Path, required=True, help="where to write the JSON result")
    ap.add_argument("--md", type=Path, default=None, help="also write a plain-English summary here")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=500, help="random weight vectors per Dirichlet family")
    ap.add_argument("--show-dir", type=Path, default=None, help="the show folder, if run_log.json's path does not resolve")
    ap.add_argument("--config", type=Path, default=None, help="default: the run folder's own config.yaml")
    a = ap.parse_args()
    t0 = time.time()
    run = a.run.resolve()
    cfg, cfg_src = load_cfg(run, a.config)
    prior = {t: float(cfg["scoring"][f"w_{t}"]) for t in TERMS}

    # 1. rebuild the scored show and every candidate once (sub-scores and checks do not depend on the weights)
    ctx = replay(run, show_dir_of(run, a.show_dir))
    windows = generate_windows(ctx, cfg, "llm")
    for w in windows:
        score_window(w, ctx, cfg, "llm")
    flag_spoilers(windows, ctx, cfg)
    assert len({w.cand_id for w in windows}) == len(windows), "candidate ids are not unique"

    # 2. the official ranking, with the stored first-time-viewer answers re-attached
    rr = Reranker(windows, {}, cfg)
    pre_judge = rr.full(prior).raw  # the 40-clip shortlist the judge saw is fixed before any judgment is applied
    answers, parse_errors = stored_answers(run)
    calls = judge_calls([rr.by_id[c] for c in pre_judge.shortlist], cfg)
    judged, attach = attach_judgments(answers, calls)
    rr.judged = judged
    base_run = rr.full(prior)
    base = base_run.raw
    pristine_rub = {w.cand_id: dict(r) for w, (_, r) in zip(windows, rr.pristine)}
    judged_rubric = {c: rb.judge_rubric(pristine_rub[c], cv) for c, cv in judged.items() if c in pristine_rub}

    official = json.loads((run / "promos.json").read_text())
    off_ids = [p["flags"]["cand_id"] for p in official]
    rr.full(prior)  # windows now hold the official state: compare sub-scores and checks with promos.json
    sub_mism, rub_mism, score_diff = [], [], 0.0
    for p in official:
        w = rr.by_id.get(p["flags"]["cand_id"])
        if w is None or w.cand_id not in base.final:
            continue
        score_diff = max(score_diff, abs(float(p["score"]) - w.score))
        sub_mism += [f"{p['promo_id']}.{k}" for k in ("hook", "cliff", "standalone", "density", "base", "rubric_mult", "arc_mult")
                     if abs(float(p["subscores"].get(k, -1)) - float(w.subscores.get(k, -2))) > 1e-9]
        if p.get("rubric") != w.rubric:
            rub_mism.append(p["promo_id"])
    stored_ids = {i for d in answers.values() for i in d}
    exact = off_ids == base.final
    reasons = []
    if not exact:
        if set(pre_judge.shortlist) != stored_ids:
            reasons.append("the rebuilt 40-clip shortlist is not the set of clips the judge saw: the code, config or show "
                           "data changed after the run, so later steps start from different clips")
        if sub_mism or rub_mism:
            reasons.append("sub-scores or checks differ for promos on both lists: the scoring code changed after the run")
        if parse_errors or attach["clips_without_answer"]:
            reasons.append("some stored judgments could not be read or matched")
        reasons = reasons or ["unexplained: same inputs, different order (look for ties or a changed slate rule)"]
    reproduction = {
        "exact": exact, "same_order": exact,
        "top20_overlap": len(set(off_ids) & set(base.final)), "same_position": sum(1 for x, y in zip(off_ids, base.final) if x == y),
        "max_abs_score_diff": round(score_diff, 6), "subscore_mismatches": sub_mism, "check_mismatches": rub_mism,
        "official_order": off_ids, "rebuilt_order": base.final,
        "judgments": {"source": "llm_raw/cold_viewer/*.json (raw answers of the run's first-time-viewer calls)",
                      "files": len(answers), "parse_errors": parse_errors, "applied": len(judged),
                      "shortlist_equals_judged_clips": set(pre_judge.shortlist) == stored_ids, **attach,
                      "not_used": "llm_raw/check_experiment/cold_viewer holds answers from a separate experiment (it may use "
                                  "another judge model); cli.py never applies them"},
        "reason_if_not_exact": reasons,
        "baseline": "the rebuilt list" + ("" if exact else " (used as the baseline because the rebuild is not exact)"),
    }
    sc = Scorer(rr, base, judged_rubric)
    pairs = PairSigns(base_run.pre)
    # the judged pool in generation order: ties then break as in the pipeline, not in the baseline's favour
    pool_ids = sorted(base.shortlist, key=lambda c: rr.idx[c])
    pool_base = rr.pool(prior, pool_ids)

    # 3. how the terms spread, and what the judge does to a clip's score (the direction of the never-judged bias)
    def spread(ws: list[Window]) -> dict:
        sds = {t: float(np.std([w.subscores[t] for w in ws])) for t in TERMS}
        eff = {t: prior[t] * sds[t] for t in TERMS}
        tot = sum(eff.values()) or 1.0
        lb = np.log([max(w.subscores["base"], 1e-6) for w in ws])
        lm = np.log([max(w.score, 1e-6) for w in ws]) - lb
        return {"n": len(ws), "mean": {t: round(float(np.mean([w.subscores[t] for w in ws])), 3) for t in TERMS},
                "distinct_values": {t: len({round(w.subscores[t], 3) for w in ws}) for t in TERMS},
                "sd": {t: round(sds[t], 3) for t in TERMS},
                "weight_x_sd_share": {t: round(eff[t] / tot, 3) for t in TERMS},
                "sd_log_weighted_sum": round(float(np.std(lb)), 3), "sd_log_multiplier": round(float(np.std(lm)), 3),
                "correlations": {f"{x}~{y}": round(float(np.corrcoef([w.subscores[x] for w in ws], [w.subscores[y] for w in ws])[0, 1]), 3)
                                 for i, x in enumerate(TERMS) for y in TERMS[i + 1:]}}
    cfg_p = with_weights(cfg, prior)
    rr.reset()
    for w in windows:  # pre-judge state: de-duplication runs before the judge in the pipeline
        combine(w, cfg_p)
    kept = nms(windows, cfg_p)
    kept_spread = spread(kept)
    pre_score = {w.cand_id: w.score for w in windows}
    rr.full(prior)  # the official state: the judge's values applied to the shortlist
    shortlist_spread = spread([rr.by_id[c] for c in base.shortlist])
    ratios = [rr.by_id[c].score / pre_score[c] for c in base.shortlist if c in judged and pre_score[c] > 0]
    term_spread = {
        "deduplicated_candidates_before_judge": kept_spread,
        "shortlist_after_judge": shortlist_spread,
        "judge_effect_on_shortlist": {
            "n": len(ratios), "median_score_ratio_after_vs_before": round(float(np.median(ratios)), 3) if ratios else None,
            "share_lower_after": round(sum(1 for r in ratios if r < 1 - 1e-9) / len(ratios), 3) if ratios else None,
            "standalone_proxy_mean": round(st.mean(rr.by_id[c].subscores.get("standalone_proxy", 0) for c in base.shortlist if c in judged), 3) if judged else None,
            "standalone_judged_mean": round(st.mean(rr.by_id[c].subscores["standalone"] for c in base.shortlist if c in judged), 3) if judged else None},
        "blocks_after_dedup": {b: sum(1 for w in kept if w.flags.get("block") == b) for b in sl.BLOCKS},
        "shortlist_quotas": sl.quotas(cfg["judge"]["top_n"], cfg), "slate_quotas": sl.quotas(cfg["output"]["final_top_n"], cfg),
    }

    def rerank(w: dict[str, float], detail: bool = False) -> tuple[FullRun, float | None, dict[str, dict]]:
        fr = rr.full(w)
        tau_all = pairs.tau(fr.pre)
        cmp = {"full_raw": sc.compare(fr.raw, detail), "full_imputed": sc.compare(fr.imputed, detail)}
        cmp["judged_pool"] = sc.compare(rr.pool(w, pool_ids), detail)
        return fr, tau_all, {m: cmp[m] for m in MODES}

    # 4. named scenarios
    named = []
    for name, family, w in named_scenarios(prior):
        fr, tau_all, cmp = rerank(w, detail=True)
        named.append({"name": name, "family": family, "weights": {t: round(w[t], 4) for t in TERMS},
                      "score_order_tau_all_candidates": round(tau_all, 4) if tau_all is not None else None,
                      "never_judged_on_shortlist": fr.never_judged,
                      "judge_ratio_median": round(fr.judge_ratio, 4) if fr.judge_ratio is not None else None, **cmp})
    fam = {}
    for f in ("plus_minus_0.05", "plus_minus_0.10"):
        xs = [s for s in named if s["family"] == f]
        fam[f] = {m: {"n": len(xs),
                      "top10_kept_mean": round(st.mean(s[m]["baseline_top10_kept_in_top10"] for s in xs), 2),
                      "top10_kept_min": min(s[m]["baseline_top10_kept_in_top10"] for s in xs),
                      "worst": [s["name"] for s in xs if s[m]["baseline_top10_kept_in_top10"] == min(x[m]["baseline_top10_kept_in_top10"] for x in xs)],
                      "top10_same_moment_mean": round(st.mean(s[m]["baseline_top10_same_moment_in_top10"] for s in xs), 2),
                      "top20_kept_mean": round(st.mean(s[m]["baseline_top20_kept_in_top20"] for s in xs), 2),
                      "top20_kept_min": min(s[m]["baseline_top20_kept_in_top20"] for s in xs),
                      "cases_top10_unchanged": sum(1 for s in xs if s[m]["baseline_top10_kept_in_top10"] == 10)}
                  for m in MODES}

    # 5. random weight vectors
    random_out, all_rows = {}, []
    families = (("dirichlet_20x_official", [20 * prior[t] for t in TERMS], a.seed),
                ("dirichlet_flat", [1.0] * len(TERMS), a.seed + 1))
    for fname, alpha, seed in families:
        rng = random.Random(seed)
        rows, cmps = [], []
        keep10 = {m: Counter() for m in MODES}
        keep20 = {m: Counter() for m in MODES}
        newcomers = {m: Counter() for m in ("full_imputed", "full_raw")}
        for _ in range(a.draws):
            w = dirichlet(rng, alpha)
            fr, tau_all, cmp = rerank(w, detail=True)
            for m in MODES:
                top10 = set(cmp[m]["top10"])
                top20 = set(cmp[m]["top20"])
                keep10[m].update(sc.label(c) for c in sc.b10 if sc.label(c) in top10)
                keep20[m].update(sc.label(c) for c in sc.b20 if sc.label(c) in top20)
                if m in newcomers:
                    newcomers[m].update(e["cand_id"] for e in cmp[m]["entered_top10"])
                for k in ("entered_top10", "left_top10", "top10", "top20"):
                    cmp[m].pop(k)
            cmps.append(cmp)
            rows.append(compact(w, fr, tau_all, cmp))
        wts = {t: [r["w"][i] for r in rows] for i, t in enumerate(TERMS)}
        random_out[fname] = {
            "alpha": [round(x, 3) for x in alpha], "seed": seed, "n": a.draws,
            "weight_spread": {t: dist(wts[t]) for t in TERMS},
            "l1_from_official": dist([sum(abs(r["w"][i] - prior[t]) for i, t in enumerate(TERMS)) for r in rows]),
            "summary": summarise(cmps, sc.shape(base.final)),
            "score_order_tau_all_candidates": dist([r["tau_all"] for r in rows]),
            "shortlist_judged": dist([r["sj"] for r in rows]),
            "judge_ratio_median": dist([r["ratio"] for r in rows]),
            "keep_rate_top10": {m: {sc.label(c): round(keep10[m][sc.label(c)] / a.draws, 3) for c in sc.b10} for m in MODES},
            "keep_rate_top20": {m: {sc.label(c): round(keep20[m][sc.label(c)] / a.draws, 3) for c in sc.b20} for m in MODES},
            "most_frequent_newcomers_top10": {m: [{**sc.describe(c), "share_of_draws": round(n / a.draws, 3)} for c, n in newcomers[m].most_common(8)]
                                              for m in newcomers},
            "spearman_weight_vs": {
                **{f"top10_kept_{m}": {t: spearman(wts[t], [r[m][0] for r in rows]) for t in TERMS} for m in MODES},
                "top10_follows_mean_judged_pool": {t: spearman(wts[t], [r["judged_pool"][6] for r in rows]) for t in TERMS},
                "top10_would_tap_mean_judged_pool": {t: spearman(wts[t], [r["judged_pool"][7] for r in rows]) for t in TERMS},
                "top10_check_pass_rate_judged_pool": {t: spearman(wts[t], [r["judged_pool"][8] for r in rows]) for t in TERMS}},
            "draws": rows,
        }
        all_rows += rows

    # what tells a frequent newcomer from the baseline promo it pushes out? pre-judge sub-scores, as de-duplication sees them
    pre_sub = {w.cand_id: rr.pristine[rr.idx[w.cand_id]][0] for w in windows}
    swaps, unrelated = [], []
    for c in dict.fromkeys(n["cand_id"] for f in random_out for n in random_out[f]["most_frequent_newcomers_top10"]["full_imputed"]):
        if c in sc.b20:
            continue  # a promo from lower on the official slate moving up, not a new clip
        w = rr.by_id[c]
        rel = [(sc.moment(w, rr.by_id[b]), sc.duplicate(w, rr.by_id[b]), iou(w, rr.by_id[b]), b) for b in sc.b20]
        rel = [r for r in rel if r[1] or r[2] > 0]
        if not rel:
            unrelated.append(c)
            continue
        same, dup, ov, b = max(rel, key=lambda r: (r[0], r[1], r[2]))
        swaps.append({"newcomer": c, "baseline": sc.label(b), "same_moment": same, "duplicate_by_dedup_rule": dup,
                      "iou": round(ov, 2), "newcomer_judged": c in judged,
                      "newcomer_minus_baseline": {t: round(pre_sub[c][t] - pre_sub[b][t], 3) for t in TERMS}})
    swap_anatomy = {
        "what": "most frequent newcomers to the top 10 that are not on the official slate (full_imputed, both random sets), each "
                "paired with the official promo it is tied to: the same moment first, then a de-duplication rival, then the "
                "largest time overlap",
        "n": len(swaps), "unrelated_newcomers": unrelated, "pairs": swaps,
        "by_term": {t: {"differs": sum(1 for p in swaps if abs(p["newcomer_minus_baseline"][t]) > 1e-9),
                        "newcomer_higher": sum(1 for p in swaps if p["newcomer_minus_baseline"][t] > 1e-9),
                        "mean_abs_diff": round(st.mean(abs(p["newcomer_minus_baseline"][t]) for p in swaps), 3) if swaps else None}
                    for t in TERMS}}

    # how far from the official weights can one go? L1 distance bins over both families
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 2.01)]
    dr = []
    for lo, hi in bins:
        xs = [r for r in all_rows if lo <= sum(abs(r["w"][i] - prior[t]) for i, t in enumerate(TERMS)) < hi]
        dr.append({"l1_from_official": [lo, round(min(hi, 2.0), 2)], "n": len(xs),
                   **{m: {"top10_kept": dist([r[m][0] for r in xs]), "top20_kept": dist([r[m][1] for r in xs]),
                          "top10_same_moment": dist([r[m][2] for r in xs])} for m in MODES}})

    # 6. determinism: the official weights still give the official slate after every other re-rank
    again = rr.full(prior)
    checks = {"rebuild_reproduces_promos_json": exact,
              "official_weights_again_after_all_scenarios": again.raw.final == base.final and again.raw.shortlist == base.shortlist,
              "judged_pool_with_official_weights_reproduces": pool_base.final == base.final,
              "imputation_is_a_no_op_at_official_weights": base_run.imputed.final == base.final and base_run.never_judged == 0,
              "weight_vectors": len(named) + 2 * a.draws}

    near, flat = random_out["dirichlet_20x_official"]["summary"], random_out["dirichlet_flat"]["summary"]
    headline = {
        "pm005_top10_kept_mean": {m: fam["plus_minus_0.05"][m]["top10_kept_mean"] for m in MODES},
        "pm010_top10_kept_mean": {m: fam["plus_minus_0.10"][m]["top10_kept_mean"] for m in MODES},
        "near_official_top10_kept_median": {m: near[m]["top10_kept"]["median"] for m in MODES},
        "any_weights_top10_kept_median": {m: flat[m]["top10_kept"]["median"] for m in MODES},
        "slate_shape_unchanged_everywhere": all(s[m]["shape"]["top20_blocks"] == sc.shape(base.final)["top20_blocks"] for s in named for m in MODES)
                                            and all(x[m]["share_slate_shape_unchanged"] == 1.0 for x in (near, flat) for m in MODES),
    }
    out = {
        "run": run.name, "config_source": cfg_src, "seed": a.seed, "draws_per_family": a.draws,
        "official_weights": prior,
        "formula": "score = (w_hook*hook + w_cliff*cliff + w_standalone*standalone + w_density*density) x penalty per failed "
                   "editor's check x turning_point_bonus x 0.9 if two-segment; then de-duplication, 40-clip shortlist by slate "
                   "quotas, first-time-viewer judge on the shortlist (standalone := follows/5, judge checks), final slate",
        "modes": {"judged_pool": "only the official run's judged 40-clip shortlist, re-scored and re-slated (no new clip can enter)",
                  "full_imputed": "whole pipeline; a never-judged shortlisted clip's score x the median judge effect of that re-rank",
                  "full_raw": "whole pipeline; a never-judged clip keeps its code-only score (favoured: the judge lowers most scores)"},
        "metric_definitions": {
            "kept": "same clip id in the baseline's and the scenario's top 10 (or top 20); top10_overlap = kept / 10",
            "same_moment": "a clip also counts when it is the same moment as the baseline clip: the same opening line, the same "
                           "closing line, or time overlap (IoU) >= scoring.nms_iou",
            "dedup_rival_of": "one clip's closing line is the other's opening line: rank.nms() keeps only one of the two",
            "kendall_tau_shared_top20": "Kendall tau-b between baseline and scenario positions of the clips both top 20s share",
            "score_order_tau_all_candidates": "Kendall tau-b between baseline and scenario scores of every candidate before de-duplication and judge",
            "top10_quality": "over the scenario's top-10 clips that have a stored judgment: editor's-check pass rate (9 checks), clips "
                             "passing all checks, mean follows-it (comprehensibility 1-5) and would-tap (1-5); 'judged' is the coverage",
            "draw rows": ROW_FIELDS},
        "reproduction": reproduction,
        "baseline": {"top10": [sc.label(c) for c in sc.b10], "top20_cand_ids": sc.b20,
                     "promos": [sc.describe(c) for c in sc.b20], "shape": sc.shape(base.final),
                     "shortlist_size": len(base.shortlist), "shortlist_judged": sum(1 for c in base.shortlist if c in judged),
                     "top10_quality": sc.quality(sc.b10), "top20_quality": sc.quality(sc.b20)},
        "headline": headline,
        "term_spread": term_spread,
        "swap_anatomy": swap_anatomy,
        "named_scenarios": named,
        "plus_minus_summary": fam,
        "random": random_out,
        "distance_response": dr,
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False))
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(markdown(out))
    print(f"rebuild exact: {exact} · {checks['weight_vectors']} weight vectors in {out['seconds']} s · one weight +-0.05 keeps "
          + " / ".join(f"{headline['pm005_top10_kept_mean'][m]}" for m in MODES) + f" of the top 10 ({' / '.join(MODES)}) → {a.out}")


# ---------------------------------------------------------------- summary -------------------------------------------
def _w(w: dict) -> str:
    return " / ".join(f"{w[t]:.2f}" for t in TERMS)


def _q(q: dict) -> str:
    if not q["judged"]:
        return "none judged"
    return f"{q['follows_mean']:.2f} / {q['would_tap_mean']:.2f} / {q['check_pass_rate']:.2f} ({q['judged']} of {q['of']})"


def _d(d: dict | None, digits: int = 0) -> str:
    if not d:
        return "n/a"
    f = f"{{:.{digits}f}}"
    if d["p10"] == d["p90"]:
        return f"{f.format(d['median'])} (10th-90th percentile: all {f.format(d['p10'])})"
    return f"{f.format(d['median'])} (10th-90th percentile {f.format(d['p10'])}-{f.format(d['p90'])})"


def _order(d: dict | None) -> str:
    if not d:
        return "Order of the promos that stay: n/a"
    m = d["median"]
    return ("The promos that stay keep their order" if m >= 0.9 else "The promos that stay mostly keep their order" if m >= 0.7
            else "The promos that stay are re-ordered")


def _three(vals: dict, digits: int = 1) -> str:
    f = f"{{:.{digits}f}}"
    return f"{f.format(vals['judged_pool'])} / {f.format(vals['full_imputed'])} / {f.format(vals['full_raw'])}"


def in_short(o: dict) -> list[str]:
    """One paragraph with the answer, built from the same numbers as the findings."""
    hl, pw, sw = o["headline"], o["official_weights"], o["swap_anatomy"]
    near, flat = o["random"]["dirichlet_20x_official"], o["random"]["dirichlet_flat"]
    p5, p10 = hl["pm005_top10_kept_mean"], hl["pm010_top10_kept_mean"]
    rn = near["spearman_weight_vs"]["top10_kept_full_imputed"]
    drv = min((t for t in TERMS if rn[t] is not None), key=lambda t: rn[t])
    shape = o["baseline"]["shape"]["top20_blocks"]
    n_short = o["baseline"]["shortlist_size"]
    txt = (f"**In short.** Through the whole pipeline (B below), moving any one weight by 0.05 keeps {p5['full_imputed']:.1f} of the official "
           f"top 10 on average and moving one by 0.10 keeps {p10['full_imputed']:.1f}; among the {n_short} clips the judge saw (A) the top 10 "
           + ("barely moves" if p5["judged_pool"] >= 9.5 else "moves less") + f" ({p5['judged_pool']:.1f} and {p10['judged_pool']:.1f}). "
           f"Random weights near the official ones keep a median {near['summary']['full_imputed']['top10_kept']['median']:.0f} of 10, and weights "
           f"drawn anywhere keep {flat['summary']['full_imputed']['top10_kept']['median']:.0f}. ")
    if hl["slate_shape_unchanged_everywhere"]:
        txt += f"The slate shape ({shape['opening']} opening / {shape['middle']} middle / {shape['late']} late) never changes. "
    txt += (f"The {'smallest ' if drv == min(pw, key=pw.get) else ''}weight, {drv}, moves the top 10 most (Spearman {rn[drv]:+.2f} with the "
            "number kept, near-official draws)")
    if sw["n"]:
        nj = sum(1 for q in sw["pairs"] if not q["newcomer_judged"])
        total = sw["n"] + len(sw["unrelated_newcomers"])
        txt += (f"; the clips that come in are cuts or neighbours of official promos ({sw['n']} of the {total} most frequent), and {nj} of "
                f"those {sw['n']} were never judged")
    txt += ". This is stability, not correctness: only traction data can say which weights are right."
    return [txt]


def markdown(o: dict) -> str:
    rep, base, ts, hl = o["reproduction"], o["baseline"], o["term_spread"], o["headline"]
    pm, R = o["plus_minus_summary"], o["random"]
    named = {s["name"]: s for s in o["named_scenarios"]}
    near, flat = R["dirichlet_20x_official"], R["dirichlet_flat"]
    ns, fs = near["summary"], flat["summary"]
    T = ("hook", "cliff", "standalone", "density")
    N = base["shortlist_size"]
    L = ["# Weight sensitivity on the official run", "",
         f"Run `{o['run']}`, offline (no model calls). Official weights hook / cliff / standalone / density = "
         f"{_w(o['official_weights'])}. {o['checks']['weight_vectors']} weight vectors, each re-ranked three ways; seed {o['seed']}, {o['seconds']} s.", "",
         *in_short(o), "",
         "## How to read the numbers", "",
         "Only the four weights change; penalties, quotas and diversity rules stay as configured. \"Kept\" = the same clip (by id) is "
         f"still in the top 10 (the first 10 promos in posting order). The first-time-viewer judge only saw the official {N}-clip "
         "shortlist, and judging a new clip would need a model call, so every weight vector is re-ranked three ways, written "
         "**A / B / C** below:", "",
         f"- **A, judged pool**: only the {N} judged clips, re-scored and re-ordered. No new clip can enter, so this understates change.",
         "- **B, full pipeline, judge effect imputed**: everything re-run (re-score, de-duplicate, shortlist, judge, final slate); a "
         "never-judged clip gets the median score change the judge caused on the judged clips of the same re-rank. The middle estimate.",
         "- **C, full pipeline, raw**: a never-judged clip keeps its code-only score. "
         f"The judge lowered {ts['judge_effect_on_shortlist']['share_lower_after']:.0%} of the official finalists' scores (median factor "
         f"{ts['judge_effect_on_shortlist']['median_score_ratio_after_vs_before']}), so this favours never-judged clips and overstates change.",
         "", "## Reproduction", ""]
    if rep["exact"]:
        L.append(f"The offline rebuild reproduces promos.json exactly: the same 20 promos in the same order, largest score difference "
                 f"{rep['max_abs_score_diff']}, no sub-score or check mismatch. The {rep['judgments']['applied']} stored first-time-viewer "
                 f"answers (`llm_raw/cold_viewer/`, {rep['judgments']['files']} files) were {rep['judgments']['method']}; they cover "
                 f"exactly the rebuilt {N}-clip shortlist. The baseline below is therefore the official slate.")
    else:
        L.append(f"The rebuild is NOT exact: {rep['top20_overlap']} of 20 promos shared, {rep['same_position']} in the same position. "
                 f"Reason: {'; '.join(rep['reason_if_not_exact'])}. The rebuilt list is used as the baseline.")
    L += ["", "## Findings (top 10 kept, A / B / C)", ""]
    p5, p10 = pm["plus_minus_0.05"], pm["plus_minus_0.10"]
    L.append(f"- **One weight moved by 0.05** (the other three rescaled; 8 cases): on average {_three(hl['pm005_top10_kept_mean'])} of the "
             f"top 10 stay. The top 10 is unchanged in {p5['judged_pool']['cases_top10_unchanged']} / {p5['full_imputed']['cases_top10_unchanged']} / "
             f"{p5['full_raw']['cases_top10_unchanged']} of the 8 cases; the worst case in B keeps {p5['full_imputed']['top10_kept_min']} "
             f"({', '.join(p5['full_imputed']['worst'])}). Top 20: {p5['judged_pool']['top20_kept_mean']:.1f} / {p5['full_imputed']['top20_kept_mean']:.1f} / "
             f"{p5['full_raw']['top20_kept_mean']:.1f} of 20.")
    L.append(f"- **One weight moved by 0.10** (8 cases): on average {_three(hl['pm010_top10_kept_mean'])} of the top 10 stay; the worst case in B "
             f"keeps {p10['full_imputed']['top10_kept_min']} ({', '.join(p10['full_imputed']['worst'])}). Top 20: {p10['judged_pool']['top20_kept_mean']:.1f} / "
             f"{p10['full_imputed']['top20_kept_mean']:.1f} / {p10['full_raw']['top20_kept_mean']:.1f} of 20.")
    def recovered(f: dict) -> float | None:  # share of the official top-10 promos that drop out but keep their moment
        lost = 10 - f["top10_kept_mean"]
        return (f["top10_same_moment_mean"] - f["top10_kept_mean"]) / lost if lost > 0 else None
    r5, r10 = recovered(p5["full_imputed"]), recovered(p10["full_imputed"])
    L.append(f"- **Counting another cut of the same moment as kept** (the same opening line, the same closing line, or half or more of "
             f"the time in common), B keeps {p5['full_imputed']['top10_same_moment_mean']:.2f} of 10 at +-0.05 (vs "
             f"{p5['full_imputed']['top10_kept_mean']:.2f} by exact clip) and {p10['full_imputed']['top10_same_moment_mean']:.2f} at +-0.10 (vs "
             f"{p10['full_imputed']['top10_kept_mean']:.2f})"
             + (f": of the official top-10 promos that drop out, {r5:.0%} (+-0.05) and {r10:.0%} (+-0.10) stay in the top 10 as another "
                "cut of the same moment." if r5 is not None and r10 is not None else "."))
    L.append("- **Leaving one term out** keeps " + ", ".join(
        f"{named[f'without {t}']['judged_pool']['baseline_top10_kept_in_top10']} / {named[f'without {t}']['full_imputed']['baseline_top10_kept_in_top10']} / "
        f"{named[f'without {t}']['full_raw']['baseline_top10_kept_in_top10']} without {t}" for t in T) + ".")
    eq = named["equal weights"]
    L.append(f"- **Equal weights** (0.25 each) keep {eq['judged_pool']['baseline_top10_kept_in_top10']} / {eq['full_imputed']['baseline_top10_kept_in_top10']} / "
             f"{eq['full_raw']['baseline_top10_kept_in_top10']}. **One term alone** keeps " + ", ".join(
                 f"{named[f'{t} only']['judged_pool']['baseline_top10_kept_in_top10']} / {named[f'{t} only']['full_imputed']['baseline_top10_kept_in_top10']} / "
                 f"{named[f'{t} only']['full_raw']['baseline_top10_kept_in_top10']} ({t})" for t in T) + ".")
    hs, nl1, fl1 = near["weight_spread"]["hook"], near["l1_from_official"], flat["l1_from_official"]
    L.append(f"- **Random weights near the official ones** ({near['n']} draws from Dirichlet({', '.join(f'{x:g}' for x in near['alpha'])}); "
             f"the hook weight's 10th-90th percentile is {hs['p10']:.2f}-{hs['p90']:.2f}, and the four weights change by a total of "
             f"{nl1['median']:.2f} in the median draw, {nl1['median'] / 0.10:.1f} times the 0.10 total of a single 0.05 move): "
             f"median kept {ns['judged_pool']['top10_kept']['median']:.0f} / {ns['full_imputed']['top10_kept']['median']:.0f} / "
             f"{ns['full_raw']['top10_kept']['median']:.0f}; for B the 10th-90th percentile is {ns['full_imputed']['top10_kept']['p10']:.0f}-"
             f"{ns['full_imputed']['top10_kept']['p90']:.0f}. The top-10 set is unchanged in {ns['judged_pool']['share_top10_set_unchanged']:.0%} / "
             f"{ns['full_imputed']['share_top10_set_unchanged']:.0%} / {ns['full_raw']['share_top10_set_unchanged']:.0%} of draws. Top 20: median "
             f"{ns['judged_pool']['top20_kept']['median']:.0f} / {ns['full_imputed']['top20_kept']['median']:.0f} / {ns['full_raw']['top20_kept']['median']:.0f} "
             f"kept. {_order(ns['full_imputed']['kendall_tau_shared_top20'])}: Kendall tau on the shared top-20 promos, B, "
             f"{_d(ns['full_imputed']['kendall_tau_shared_top20'], 2)}.")
    L.append(f"- **Any weights at all** ({flat['n']} draws from Dirichlet(1,1,1,1); total change {fl1['median']:.2f} in the median draw): "
             f"median kept {fs['judged_pool']['top10_kept']['median']:.0f} / "
             f"{fs['full_imputed']['top10_kept']['median']:.0f} / {fs['full_raw']['top10_kept']['median']:.0f}; for B the 10th-90th percentile is "
             f"{fs['full_imputed']['top10_kept']['p10']:.0f}-{fs['full_imputed']['top10_kept']['p90']:.0f}. Top 20: median "
             f"{fs['judged_pool']['top20_kept']['median']:.0f} / {fs['full_imputed']['top20_kept']['median']:.0f} / {fs['full_raw']['top20_kept']['median']:.0f} "
             f"kept; Kendall tau on the shared top-20 promos, B, {_d(fs['full_imputed']['kendall_tau_shared_top20'], 2)}.")
    rows = [d for d in o["distance_response"] if d["n"]]
    L.append("- **How far the weights can move** (B, both random sets, by total change |w - official| summed over the four weights): median top 10 kept "
             + "; ".join(f"{d['l1_from_official'][0]:.1f}-{d['l1_from_official'][1]:.1f}: {d['full_imputed']['top10_kept']['median']:.0f} (n={d['n']})" for d in rows) + ".")
    rn, rf = near["spearman_weight_vs"]["top10_kept_full_imputed"], flat["spearman_weight_vs"]["top10_kept_full_imputed"]
    drv = min((t for t in T if rn[t] is not None), key=lambda t: rn[t])
    L.append(f"- **Which weight drives the change**: across the near-official draws (B), the number of the top 10 kept falls most as the "
             f"{drv} weight rises (Spearman {rn[drv]:+.2f}; any weights {rf[drv]:+.2f}). Correlation with the other weights, near-official: "
             + ", ".join(f"{t} {rn[t]:+.2f}" for t in T if t != drv) + ".")
    sw = o["swap_anatomy"]
    if sw["n"]:
        bt, kd = sw["by_term"], ts["deduplicated_candidates_before_judge"]
        n, sm = sw["n"], sum(1 for p in sw["pairs"] if p["same_moment"])
        cross = sum(1 for p in sw["pairs"] if p["duplicate_by_dedup_rule"] and not p["same_moment"])
        nj = sum(1 for p in sw["pairs"] if not p["newcomer_judged"])
        L.append(f"- **Why**: each of the {n} most frequent newcomers (B, both random sets) is tied to an official promo: {sm} are the same "
                 f"moment, {cross} share a line the other way round (one's closing line is the other's opening line, so de-duplication keeps "
                 f"only one) and {n - sm - cross} partly overlap one"
                 + (f" ({len(sw['unrelated_newcomers'])} more overlap no official promo)" if sw["unrelated_newcomers"] else "")
                 + f"; {nj} of {n} were never judged. Before the judge, newcomer and official promo differ on density in {bt['density']['differs']} "
                 f"of {n} pairs (the newcomer higher in {bt['density']['newcomer_higher']}), on hook in {bt['hook']['differs']}, on cliff in "
                 f"{bt['cliff']['differs']}, and on the standalone proxy by {bt['standalone']['mean_abs_diff']} on average. Hook and cliff take few "
                 f"values ({kd['distinct_values']['hook']} and {kd['distinct_values']['cliff']} distinct values among the {kd['n']} de-duplicated "
                 f"candidates) because they come from the model's 0-3 scale, so they often tie between two cuts of one moment; density takes "
                 f"{kd['distinct_values']['density']}, so its weight often decides which cut survives de-duplication.")
    shape = base["shape"]["top20_blocks"]
    L.append(f"- **Slate shape**: {shape['opening']} opening / {shape['middle']} middle / {shape['late']} late "
             + ("in every re-rank, in all three modes. " if hl["slate_shape_unchanged_everywhere"] else "in most re-ranks (see JSON). ")
             + "The quotas set it, not the weights. The opening block is posted first, so the top 10 is always opening-block clips.")
    kr = near["keep_rate_top10"]["full_imputed"]
    steady = [k for k, v in kr.items() if v >= 0.95]
    shaky = sorted(kr.items(), key=lambda kv: kv[1])[:3]
    L.append(f"- **Robust and fragile promos** (B, near-official draws): {len(steady)} of the top 10 stay in the top 10 in at least 95% of draws "
             f"({', '.join(steady) or 'none'}); the least robust are " + ", ".join(f"{k} ({v:.0%})" for k, v in shaky) + ".")
    nc = near["most_frequent_newcomers_top10"]["full_imputed"][:3]
    if nc:
        L.append("- **What comes in** (B, near-official draws): " + "; ".join(
            f"{n['id']} (episode {n['episode']}, {'judged' if n['judged'] else 'never judged'}"
            + (f", same moment as {', '.join(n['same_moment_as'])}" if n["same_moment_as"] else "")
            + (f", shares a line with {', '.join(n['dedup_rival_of'])} so only one can survive de-duplication" if n.get("dedup_rival_of") else "")
            + f", {n['share_of_draws']:.0%} of draws)" for n in nc) + ".")
    sp, kp = ts["shortlist_after_judge"], ts["deduplicated_candidates_before_judge"]
    L.append(f"- **What each term really contributes**: a weight matters in proportion to how much its term varies between clips. Among the {N} judged "
             f"finalists the spread (sd) is hook {sp['sd']['hook']}, cliff {sp['sd']['cliff']}, standalone {sp['sd']['standalone']}, density "
             f"{sp['sd']['density']}; weight x spread gives hook {sp['weight_x_sd_share']['hook']:.0%}, cliff {sp['weight_x_sd_share']['cliff']:.0%}, "
             f"standalone {sp['weight_x_sd_share']['standalone']:.0%}, density {sp['weight_x_sd_share']['density']:.0%} of the weighted sum's pull "
             f"(over all {kp['n']} de-duplicated candidates before the judge: {kp['weight_x_sd_share']['hook']:.0%} / {kp['weight_x_sd_share']['cliff']:.0%} / "
             f"{kp['weight_x_sd_share']['standalone']:.0%} / {kp['weight_x_sd_share']['density']:.0%}). On the {N} finalists the editor's-check multiplier "
             f"spreads the log-score {'more' if sp['sd_log_multiplier'] > sp['sd_log_weighted_sum'] else 'less'} than the weighted sum does "
             f"(sd {sp['sd_log_multiplier']} vs {sp['sd_log_weighted_sum']}; over the {kp['n']} candidates {kp['sd_log_multiplier']} vs {kp['sd_log_weighted_sum']})."
             + (f" By this measure {drv} pulls least, yet its weight moves the top 10 most: what matters is the margin between two cuts of "
                "one moment, where hook and cliff often tie (see Why)." if drv == min(T, key=lambda t: sp['weight_x_sd_share'][t]) else ""))
    bd, sq = ts["blocks_after_dedup"], ts["shortlist_quotas"]
    a5, b5 = hl["pm005_top10_kept_mean"]["judged_pool"], hl["pm005_top10_kept_mean"]["full_imputed"]
    sjn, sjf = near["shortlist_judged"], flat["shortlist_judged"]
    L.append(f"- **Where the weights act**: at the official weights de-duplication leaves {bd['opening']} opening / {bd['middle']} middle / {bd['late']} late "
             f"candidates for {sq['opening']} / {sq['middle']} / {sq['late']} shortlist places. "
             + ("In the opening block every surviving candidate is shortlisted, so there the weights only decide which cut of a moment survives "
                "de-duplication, and the order. " if bd["opening"] <= sq["opening"] else "")
             + (f"A 0.05 move changes the judged pool's top 10 less than the whole pipeline's (A keeps {a5:.1f}, B {b5:.1f}): most of the change "
                "comes from de-duplication and the shortlist cut, not from re-ordering clips that were judged. " if a5 - b5 >= 0.5 else
                f"A 0.05 move changes the judged pool's top 10 about as much as the whole pipeline's (A keeps {a5:.1f}, B {b5:.1f}). ")
             + f"A near-official draw keeps a median {sjn['median']:.0f} of the {N} judged clips on its shortlist ({sjf['median']:.0f} for any weights).")
    bq = base["top10_quality"]
    L.append(f"- **First-time-viewer quality of the top 10** (follows / would-tap / check pass rate, over judged clips): official {_q(bq)}. "
             f"In A (all judged), near-official draws give follows {_d(ns['judged_pool']['top10_follows_mean'], 2)} and would-tap "
             f"{_d(ns['judged_pool']['top10_would_tap_mean'], 2)}; any weights give follows {_d(fs['judged_pool']['top10_follows_mean'], 2)} and "
             f"would-tap {_d(fs['judged_pool']['top10_would_tap_mean'], 2)}. In B and C only part of the top 10 was ever judged (median "
             f"{ns['full_imputed']['top10_judged']['median']:.0f} of 10 near the official weights, {fs['full_imputed']['top10_judged']['median']:.0f} "
             "of 10 for any weights), so their quality numbers are partial.")
    tn, tf = near["spearman_weight_vs"]["top10_would_tap_mean_judged_pool"], flat["spearman_weight_vs"]["top10_would_tap_mean_judged_pool"]
    if all(tn[t] is not None for t in T):
        up, down = max(T, key=lambda t: tn[t]), min(T, key=lambda t: tn[t])
        wt = ns["judged_pool"]["top10_would_tap_mean"]
        L.append(f"- **A weak hint for the learning loop**: within A, the top 10's mean would-tap (a judge rating that is not part of the score) "
                 f"rises with the {up} weight (Spearman {tn[up]:+.2f} near-official, {tf[up]:+.2f} any weights) and falls with the {down} weight "
                 f"({tn[down]:+.2f}, {tf[down]:+.2f}). The whole 10th-90th percentile range is {wt['p10']:.2f}-{wt['p90']:.2f} out of 5, about one "
                 "clip's rating moving by one point, from one model's opinion: a direction to test with real traction, not a reason to change the weights.")
    L += ["", "## Caveats", "",
          "- This measures stability, not correctness. No traction data was used, so it cannot say the official weights are right, only how "
          "much the slate depends on them.",
          "- The follows score is not an independent outcome: after the judge, standalone = follows / 5, so more standalone weight "
          "mechanically favours clips with high follows scores. Would-tap is not in the score, but it comes from the same judge call.",
          f"- B assumes a never-judged clip would be judged like the median judged clip; the truth for a given clip could be anywhere "
          f"between A and C. On the {N} judged clips the standalone term fell from {ts['judge_effect_on_shortlist']['standalone_proxy_mean']} "
          f"(code proxy) to {ts['judge_effect_on_shortlist']['standalone_judged_mean']} (judge) on average.",
          "- One show (50 episodes), one scoring model and one free judge model; 10 clips per top 10, so differences of one clip are "
          "within noise. Random draws use fixed seeds.",
          "", "## Named scenarios (top 10 and top 20 kept, A / B / C)", "",
          "| scenario | weights h / c / s / d | top 10 kept | top 20 kept | B: same moment kept, top 10 | B: tau, shared top 20 | B: top-10 follows / tap / checks (judged of 10) |",
          "|---|---|---|---|---|---|---|"]
    for s in o["named_scenarios"]:
        b = s["full_imputed"]
        tau = "n/a" if b["kendall_tau_shared_top20"] is None else f"{b['kendall_tau_shared_top20']:.2f}"
        k10 = " / ".join(str(s[m]["baseline_top10_kept_in_top10"]) for m in MODES)
        k20 = " / ".join(str(s[m]["baseline_top20_kept_in_top20"]) for m in MODES)
        L.append(f"| {s['name']} | {_w(s['weights'])} | {k10} | {k20} | {b['baseline_top10_same_moment_in_top10']} | {tau} | {_q(b['top10_quality'])} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
