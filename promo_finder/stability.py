"""Consistency, measured instead of assumed.

1. replay(): rebuild a finished run from its own files (beats/*.json, aliases.yaml, show_arc.json) with zero model
   calls, so any run can be re-ranked under the current code.
2. noise_model(): from two independent runs of the same model, the empirical chance that a beat scored s comes back
   as s' (separately for hook and cliff). Flat lapses (beat_scoring.degenerate) are excluded: they are repaired, not noise.
3. simulate(): perturb every model score with that noise and re-run the (deterministic) ranking R times. For each promo
   on the slate: the share of simulated re-runs that keep it (same opening line, or >= 50% time overlap), and for
   the whole slate: the expected overlap between two runs. Code-only, so 200 simulated re-runs cost nothing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from .beat_scoring import ScoredShow
from .beats import build_beats
from .features import heuristic_scores
from .models import BeatScore, Window
from .parse import parse_show


def replay(run: Path, show_dir: Path | None = None) -> ScoredShow:
    args = json.loads((run / "run_log.json").read_text())["args"]
    show = Path(show_dir or args["show_dir"])
    eps = parse_show(show)
    if args.get("limit"):
        eps = eps[: int(args["limit"])]
    aliases = (yaml.safe_load((run / "aliases.yaml").read_text()) or {}).get("aliases", {}) if (run / "aliases.yaml").exists() else {}
    beats_by_ep = {ep.number: build_beats(ep, aliases) for ep in eps}
    heur_by_ep = {ep.number: heuristic_scores(ep, beats_by_ep[ep.number]) for ep in eps}
    llm_by_ep: dict[int, dict[str, BeatScore]] = {}
    for ep in eps:
        f = run / "beats" / f"ep-{ep.number:03d}.json"
        rows = {r["beat_id"]: r for r in json.loads(f.read_text())["beats"]} if f.exists() else {}
        sc = {}
        for b in beats_by_ep[ep.number]:
            r = rows.get(b.global_id)
            if r is None or r.get("source") != "llm":
                continue
            sc[b.beat_id] = BeatScore(beat_id=b.beat_id, hook=r["hook"], cliff=r["cliff"], beat_type=r.get("beat_type", "other"),
                                      unresolved_refs=r.get("unresolved_refs", []), role=r.get("role", "none"),
                                      intensity=r.get("intensity", 0), source="llm")
        llm_by_ep[ep.number] = {**heur_by_ep[ep.number], **sc} if sc else {}
    ctx = ScoredShow(eps, beats_by_ep, llm_by_ep, heur_by_ep, aliases)
    if (run / "show_arc.json").exists():
        from .show_arc import ShowArc
        ctx.arc = ShowArc.model_validate_json((run / "show_arc.json").read_text())
    return ctx


def _rows(run: Path) -> dict[str, dict]:
    out = {}
    for f in (run / "beats").glob("ep-*.json"):
        rows = {r["beat_id"]: r for r in json.loads(f.read_text())["beats"]}
        llm = [r for r in rows.values() if r.get("source") == "llm"]
        if llm and (max(r["hook"] for r in llm) <= 1 or max(r["cliff"] for r in llm) <= 1):
            continue  # a flat lapse, not noise
        out.update(rows)
    return out


def noise_model(run_a: Path, run_b: Path) -> dict:
    """P(score in run b | score in run a), for hook and cliff, from beats both runs scored with the model."""
    a, b = _rows(run_a), _rows(run_b)
    mats = {"hook": [[0] * 4 for _ in range(4)], "cliff": [[0] * 4 for _ in range(4)]}
    n = 0
    for k, ra in a.items():
        rb = b.get(k)
        if not rb or ra.get("source") != "llm" or rb.get("source") != "llm":
            continue
        n += 1
        for key in ("hook", "cliff"):
            mats[key][int(round(ra[key]))][int(round(rb[key]))] += 1
    probs = {key: [[c / max(sum(row), 1) for c in row] for row in m] for key, m in mats.items()}
    return {"beats": n, "counts": mats, "p": probs}


def _perturb(ctx: ScoredShow, noise: dict, rng: random.Random, votes: int = 1) -> ScoredShow:
    """One simulated re-run: every model score redrawn from the measured noise; with votes > 1, the mean of that many
    independent readings (what --votes does for real)."""
    def draw(key: str, v: int) -> float:
        w = noise["p"][key][int(round(v))]
        if not sum(w):
            return float(v)
        return sum(rng.choices(range(4), weights=w)[0] for _ in range(votes)) / votes
    llm2 = {}
    for n, scores in ctx.llm_by_ep.items():
        new = {}
        for bid, s in scores.items():
            if s.source != "llm":
                new[bid] = s
                continue
            new[bid] = s.model_copy(update={"hook": draw("hook", s.hook), "cliff": draw("cliff", s.cliff)})
        llm2[n] = new
    out = ScoredShow(ctx.episodes, ctx.beats_by_ep, llm2, ctx.heur_by_ep, ctx.aliases, ctx.alias_notes)
    out.arc = ctx.arc
    return out


def same_promo(a: Window, b: Window) -> bool:
    from .rank import iou
    return a.hook_beat == b.hook_beat or iou(a, b) >= 0.5


def slate_of(ctx: ScoredShow, cfg: dict, n: int, policy: str = "slate") -> list[Window]:
    from . import slate as sl
    from .rank import diversity_order, nms, score_window, generate_windows, flag_spoilers
    windows = generate_windows(ctx, cfg, "llm")
    for w in windows:
        score_window(w, ctx, cfg, "llm")
    flag_spoilers(windows, ctx, cfg)
    kept = nms(windows, cfg)
    if policy == "score_only":  # the v1/v2 policy: best scores anywhere in the show
        return diversity_order(kept, cfg, n)
    return sl.build(kept, cfg, n, diversity_order)


def overlap(x: list[Window], y: list[Window]) -> float:
    return sum(1 for a in x if any(same_promo(a, b) for b in y)) / max(len(x), 1)


def simulate(ctx: ScoredShow, cfg: dict, noise: dict, reps: int = 100, n: int = 20, seed: int = 7, policy: str = "slate", votes: int = 1) -> dict:
    rng = random.Random(seed)
    base = slate_of(ctx, cfg, n, policy)
    sims = [slate_of(_perturb(ctx, noise, rng, votes), cfg, n, policy) for _ in range(reps)]
    keep = {w.cand_id: round(sum(1 for s in sims if any(same_promo(w, x) for x in s)) / reps, 3) for w in base}
    pair = [overlap(sims[i], sims[i + 1]) for i in range(0, reps - 1, 2)]
    top10 = [overlap(sims[i][:10], sims[i + 1][:10]) for i in range(0, reps - 1, 2)]
    return {"policy": policy, "votes": votes, "reps": reps, "n": n, "base": base, "keep_rate": keep, "sims": sims,
            "expected_overlap_top_n": round(sum(pair) / max(len(pair), 1), 3),
            "expected_overlap_top10": round(sum(top10) / max(len(top10), 1), 3)}
