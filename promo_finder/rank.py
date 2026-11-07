"""How promos are ranked. Plain arithmetic; every number lives in config.yaml and every intermediate value is
written to promos.json, so any rank can be recomputed by hand.

  1. candidates.py      every (opening line, later closing line) pair -> a 20-60 s clip on camera cuts, hook <= 3 s
  2. score_window()     sub-scores in 0..1:
                          hook        = AI hook score of the opening line / 3   (+0.5/3 if a physical action in the first 3 s)
                          cliff       = AI cliff score of the closing line / 3  (+0.5/3 near the writers' end cue; love <= 2/3)
                          standalone  = share of lines with no unexplained names (later: first-time viewer's rating / 5)
                          density     = dialogue share (x0.5 below 35%) x length prior (1.0 for 25-45 s, 0.6 at 20/60 s)
  3. combine()          base  = 0.35 hook + 0.30 cliff + 0.25 standalone + 0.10 density
                        score = base x penalty per failed editor's check (rubric.py) x 1.05 turning-point episode
                                x 0.9 if it is a two-part join
  4. nms()              drop a clip that overlaps a better one by >= 50% of its time or shares its opening/closing line
  5. slate.build()      opening / middle / late quotas (55/33/12%), best-first inside each part via diversity_order():
                        max 2 per episode in the top 10, x0.85 per clip already taken from that episode, x0.90 if the same
                        hook type as the previous pick
  6. apply_judge()      the shortlist's standalone is replaced by the blind first-time viewer's rating and its checks are
                        applied; everything is re-scored and the final slate is built with the same quotas
The weights are hand-set priors from the research (docs/research.md), not learned; the approve/reject log is there to refit them.
"""
from __future__ import annotations

from .beat_scoring import ScoredShow
from .candidates import boundary_windows, windows_for_episode
from .features import physical_in_first_seconds, writer_cue_near
from .models import Window
from . import rubric as rb
from . import slate as sl


def generate_windows(ctx: ScoredShow, cfg: dict, source: str = "llm") -> list[Window]:
    out: list[Window] = []
    eps = sorted(ctx.episodes, key=lambda e: e.number)
    for ep in eps:
        out += windows_for_episode(ep, ctx.beats_by_ep[ep.number], ctx.scores(ep.number, source), cfg)
    for a, b in zip(eps, eps[1:]):
        if b.number == a.number + 1:
            out += boundary_windows(a, ctx.beats_by_ep[a.number], ctx.scores(a.number, source),
                                    b, ctx.beats_by_ep[b.number], ctx.scores(b.number, source), cfg)
    return out


def _duration_prior(d: float, cfg: dict) -> float:
    p = cfg["promo"]
    if p["preferred_min_s"] <= d <= p["preferred_max_s"]:
        return 1.0
    if d < p["preferred_min_s"]:
        span = p["preferred_min_s"] - p["min_duration_s"]
        return 1.0 - 0.4 * (p["preferred_min_s"] - d) / max(span, 1e-6)
    span = p["max_duration_s"] - p["preferred_max_s"]
    return 1.0 - 0.4 * (d - p["preferred_max_s"]) / max(span, 1e-6)


def _dialogue_seconds(w: Window, ctx: ScoredShow) -> float:
    total = 0.0
    for seg in w.segments:
        for e in ctx.episode(seg.episode).events:
            if e.kind == "dialogue":
                total += max(0.0, min(e.t1, seg.end_s) - max(e.t0, seg.start_s))
    return total


def combine(w: Window, cfg: dict) -> None:
    """score = weighted sub-scores × editor's-rubric penalties × turning-point lift (× 0.9 for an unproven two-segment
    join). Where in the show a clip comes from is not a score: the slate quotas (slate.py) decide that."""
    s = cfg["scoring"]
    base = (s["w_hook"] * w.subscores["hook"] + s["w_cliff"] * w.subscores["cliff"]
            + s["w_standalone"] * w.subscores["standalone"] + s["w_density"] * w.subscores["density"])
    mult = rb.multiplier(w.rubric, cfg) if cfg.get("rubric", {}).get("enabled", True) else 1.0
    w.subscores["base"] = round(base, 4)
    w.subscores["rubric_mult"] = round(mult, 3)
    w.subscores["rubric_pass"] = round(sum(w.rubric.values()) / max(len(w.rubric), 1), 3)
    # plot relevance from the show-arc pass: a small lift for turning-point episodes (not the climax: that is a spoiler)
    tp_m = float(cfg["rubric"].get("turning_point_bonus", 1.0)) if w.flags.get("turning_point") not in (None, "climax") else 1.0
    w.subscores["arc_mult"] = round(tp_m, 3)
    w.score = round(base * mult * tp_m * (0.9 if w.kind == "two_segment" else 1.0), 4)


def score_window(w: Window, ctx: ScoredShow, cfg: dict, source: str = "llm") -> None:
    s = cfg["scoring"]
    hook_beat, cliff_beat = ctx.beat(w.hook_beat), ctx.beat(w.cliff_beat)
    hs, cs = ctx.score(w.hook_beat, source), ctx.score(w.cliff_beat, source)
    ep_first, ep_last = ctx.episode(w.segments[0].episode), ctx.episode(w.segments[-1].episode)
    start, end = w.segments[0].start_s, w.segments[-1].end_s
    physical = physical_in_first_seconds(ep_first, start, cfg["promo"]["hook_window_s"])
    cue = writer_cue_near(ep_last, end, 2.0)
    hook_raw = min(3.0, hs.hook + (s["physical_hook_bonus"] if physical else 0.0))
    cliff_raw = min(3.0, cs.cliff + (s["writer_cue_bonus"] if cue else 0.0))
    # The brief defines a cliffhanger as an unresolved moment. Models rate romantic peaks (a tender declaration) as
    # 3/3 cliffs; a declaration is a resolution unless it is cut off or left hanging.
    if cs.beat_type == "romance" and not cliff_beat.text.rstrip('"”').rstrip().endswith(("?", "…", "...", "—", "-")):
        cliff_raw = min(cliff_raw, 2.0)
        w.flags["romance_cap"] = True
    dialogue_beats = [ctx.beat(g) for g in w.beat_ids if ctx.beat(g).kind == "dialogue"]
    if dialogue_beats:
        stand = sum(1.0 - min(len(ctx.score(b.global_id, source).unresolved_refs), 4) / 4.0 for b in dialogue_beats) / len(dialogue_beats)
    else:
        stand = 0.5
    density = _dialogue_seconds(w, ctx) / max(w.duration_s, 1e-6)
    dn = min(1.0, density / 0.5)
    if density < s["density_floor"]:
        dn *= 0.5
    prior = _duration_prior(w.duration_s, cfg)
    w.subscores = {
        "hook": round(hook_raw / 3.0, 3),
        "cliff": round(cliff_raw / 3.0, 3),
        "standalone": round(stand, 3),
        "density": round(dn * prior, 3),
        "dialogue_share": round(density, 3),
        "duration_prior": round(prior, 3),
    }
    w.flags.update({
        "hook_at_s": round(hook_beat.t0 - start, 2),
        "physical_hook": physical,
        "writer_cue": cue,
        "hook_type": hs.beat_type,
        "cliff_type": cs.beat_type,
        "source": "llm" if (hs.source == "llm" and cs.source == "llm") else "heuristic",
        "episode": w.segments[0].episode,
    })
    lo, hi = sl.episode_range(ctx)
    ep0 = w.segments[0].episode
    w.flags["block"] = sl.block_of(ep0, lo, hi, cfg)
    w.flags["position"] = round(sl.position(ep0, lo, hi), 3)
    w.flags["minutes_into_show"] = round((ctx.starts[ep0] + w.segments[0].start_s) / 60.0, 1)
    in_beats = [ctx.beat(g) for g in w.beat_ids]
    w.rubric = rb.code_rubric(w, in_beats, [ctx.score(g, source) for g in w.beat_ids], hs, cs, cfg, ctx.leads)
    arc = ctx.arc
    if arc is not None:
        tp = arc.tp_for(w.segments[0].episode)  # type: ignore[attr-defined]
        if tp:
            w.flags["turning_point"] = tp
        rev = arc.reveal_for(w.segments[-1].episode)  # type: ignore[attr-defined]
        if rev and w.flags["block"] != "opening":
            w.flags["major_reveal"] = rev
    combine(w, cfg)  # the two-segment join is unproven: combine() applies a slight discount, and it is flagged


def iou(a: Window, b: Window) -> float:
    inter = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                inter += max(0.0, min(sa.end_s, sb.end_s) - max(sa.start_s, sb.start_s))
    union = sum(s.duration_s for s in a.segments) + sum(s.duration_s for s in b.segments) - inter
    return inter / union if union > 0 else 0.0


def nms(windows: list[Window], cfg: dict) -> list[Window]:
    thr = cfg["scoring"]["nms_iou"]
    kept: list[Window] = []
    for w in sorted(windows, key=lambda x: -x.score):
        # suppress overlaps, and any clip whose opening or closing line is already the opening or closing
        # line of a kept clip: two promos built on the same key beat are one promo to a viewer
        if any(iou(w, k) >= thr or {w.hook_beat, w.cliff_beat} & {k.hook_beat, k.cliff_beat} for k in kept):
            continue
        kept.append(w)
    return kept


def diversity_order(windows: list[Window], cfg: dict, n: int) -> list[Window]:
    s = cfg["scoring"]
    remaining = sorted(windows, key=lambda x: -x.score)
    chosen: list[Window] = []
    while remaining and len(chosen) < n:
        per_ep: dict[int, int] = {}
        for c in chosen:
            per_ep[int(c.flags["episode"])] = per_ep.get(int(c.flags["episode"]), 0) + 1
        prev_type = chosen[-1].flags.get("hook_type") if chosen else None
        best, best_eff = None, -1.0
        for w in remaining:
            ep = int(w.flags["episode"])
            if len(chosen) < 10 and per_ep.get(ep, 0) >= s["max_per_episode_top10"]:
                continue
            eff = w.score * (1 - s["same_episode_penalty"]) ** per_ep.get(ep, 0)
            if prev_type and w.flags.get("hook_type") == prev_type:
                eff *= 1 - s["same_type_penalty"]
            if eff > best_eff:
                best, best_eff = w, eff
        if best is None:  # every remaining window is capped out; relax the cap
            best = remaining[0]
        chosen.append(best)
        remaining.remove(best)
    return chosen


def flag_spoilers(windows: list[Window], ctx: ScoredShow, cfg: dict) -> None:
    nums = [e.number for e in ctx.episodes]
    lo, hi = min(nums), max(nums)
    cutoff = hi - cfg["scoring"]["spoiler_last_fraction"] * (hi - lo)
    for w in windows:
        w.flags["spoiler_risk"] = int(w.flags["episode"]) > cutoff


def rank_windows(ctx: ScoredShow, cfg: dict, source: str = "llm", top_n: int | None = None) -> tuple[list[Window], list[Window]]:
    """Returns (ordered finalists, all scored windows)."""
    windows = generate_windows(ctx, cfg, source)
    for w in windows:
        score_window(w, ctx, cfg, source)
    flag_spoilers(windows, ctx, cfg)
    kept = nms(windows, cfg)
    n = top_n or cfg["judge"]["top_n"]
    return sl.build(kept, cfg, n, diversity_order), windows


def apply_judge(windows: list[Window], judged: dict[str, "ColdViewer"], cfg: dict) -> None:
    """Finalists: replace the standalone proxy with the cold viewer's comprehensibility and re-score."""
    s = cfg["scoring"]
    for w in windows:
        cv = judged.get(w.cand_id)
        if not cv:
            continue
        w.subscores["standalone_proxy"] = w.subscores["standalone"]
        w.subscores["standalone"] = round(cv.comprehensibility / 5.0, 3)
        w.subscores["would_tap"] = round(cv.would_tap / 5.0, 3)
        w.rubric = rb.judge_rubric(w.rubric, cv)
        combine(w, cfg)
