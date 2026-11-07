"""Window generation. Pure code: every start/end comes from parsed timestamps and the cut set.

Rules (config.yaml → promo.*):
  start = start of the camera shot containing the hook beat, if that shot started ≤ cut_snap_s before
          the hook; otherwise the hook beat's own start. So the hook always lands within 3 s of the clip start.
  end   = end of the shot containing the cliff beat's last word, capped at cliff.t1 + tail_pad_s, the
          scene end and the episode end.
  20 ≤ duration ≤ 60; no dialogue line may overhang either boundary by more than straddle_tolerance_s.
"""
from __future__ import annotations

from .beats import cut_after, cut_before, cut_set
from .models import Beat, BeatScore, Episode, Segment, Window


def _continuation(text: str) -> bool:
    t = text.strip().lstrip('"“')
    return t.startswith(("...", "…", "-", "—")) or (t[:1].islower() and t[:1].isalpha())


def _scene_change(ep: Episode, start: float, end: float) -> bool:
    return any(start + 0.5 < sc.t0 < end - 0.5 for sc in ep.scenes if sc.t0)


def _straddles(ep: Episode, t: float, tol: float) -> bool:
    for e in ep.events:
        if e.kind == "dialogue" and e.t0 < t - tol and e.t1 > t + tol:
            return True
    return False


def _scene_end(ep: Episode, scene_idx: int) -> float:
    for s in ep.scenes:
        if s.idx == scene_idx and s.t1:
            return s.t1
    return ep.duration


def _hook_candidates(beats: list[Beat], scores: dict[str, BeatScore], cfg: dict) -> list[Beat]:
    c = cfg["candidates"]
    # a sound effect is never the first thing we cut to, and a line that starts mid-sentence ("...not a dull
    # creature") or in lower case is the tail of something the viewer did not hear
    pool = [b for b in beats if b.kind != "sfx" and not _continuation(b.text)]
    hooks = [b for b in pool if scores[b.beat_id].hook >= c["hook_min_score"]]
    if len(hooks) < 3:
        ranked = sorted(pool, key=lambda b: (-scores[b.beat_id].hook, b.t0))
        hooks = ranked[: c["fallback_top_hooks"]]
    return hooks


def _cliff_candidates(beats: list[Beat], scores: dict[str, BeatScore], cfg: dict) -> list[Beat]:
    c = cfg["candidates"]
    pool = [b for b in beats if b.kind != "sfx"]  # the clip ends on a line or an action, never on a sound cue
    cliffs = [b for b in pool if scores[b.beat_id].cliff >= c["cliff_min_score"]]
    if pool and pool[-1] not in cliffs:
        cliffs.append(pool[-1])  # forced: the writers' own episode ending
    return cliffs


def _window_start(ep: Episode, cuts: list[float], h: Beat, cfg: dict) -> float:
    p = cfg["promo"]
    start = cut_before(cuts, h.t0)
    if h.t0 - start > p["cut_snap_s"]:
        start = h.t0
    return start


def _window_end(ep: Episode, cuts: list[float], c: Beat, cfg: dict) -> float:
    p = cfg["promo"]
    end = cut_after(cuts, c.t1)
    end = min(end, c.t1 + p["tail_pad_s"], _scene_end(ep, c.scene_idx), ep.duration)
    return max(end, c.t1)


def windows_for_episode(ep: Episode, beats: list[Beat], scores: dict[str, BeatScore], cfg: dict) -> list[Window]:
    p = cfg["promo"]
    cuts = cut_set(ep)
    out: list[Window] = []
    hooks = _hook_candidates(beats, scores, cfg)
    cliffs = _cliff_candidates(beats, scores, cfg)
    seen: set[tuple[float, float]] = set()
    for h in hooks:
        start = _window_start(ep, cuts, h, cfg)
        if _straddles(ep, start, p["straddle_tolerance_s"]):
            continue
        for c in cliffs:
            if c.idx <= h.idx or c.t1 <= h.t1:
                continue
            end = _window_end(ep, cuts, c, cfg)
            dur = end - start
            if dur < p["min_duration_s"]:
                continue
            if dur > p["max_duration_s"]:
                if cfg["candidates"].get("two_segment_enabled") and scores[h.beat_id].hook >= 3:
                    w = _two_segment(ep, beats, cuts, h, c, cfg)
                    if w and (w.segments[0].start_s, w.segments[1].start_s) not in seen:
                        seen.add((w.segments[0].start_s, w.segments[1].start_s))
                        out.append(w)
                continue
            if _straddles(ep, end, p["straddle_tolerance_s"]):
                continue
            key = (round(start, 2), round(end, 2))
            if key in seen:
                continue
            seen.add(key)
            ids = [b.global_id for b in beats if b.t0 >= start - 0.01 and b.t1 <= end + 0.01]
            out.append(
                Window(
                    cand_id=f"e{ep.number:03d}-{len(out) + 1:03d}",
                    segments=[Segment(episode=ep.number, start_s=start, end_s=end)],
                    hook_beat=h.global_id,
                    cliff_beat=c.global_id,
                    beat_ids=ids,
                    duration_s=round(dur, 2),
                    kind="single",
                    flags={"scene_change": _scene_change(ep, start, end)},
                )
            )
    return out


def _two_segment(ep: Episode, beats: list[Beat], cuts: list[float], h: Beat, c: Beat, cfg: dict) -> Window | None:
    """Hook shot (3–8 s) + hard cut + a 15–50 s run ending on the cliff. Only when the contiguous
    span would exceed the maximum. Flagged; the editor decides whether the join plays."""
    p = cfg["promo"]
    s1 = _window_start(ep, cuts, h, cfg)
    e1 = min(cut_after(cuts, h.t1), h.t1 + p["tail_pad_s"], ep.duration)
    if not (3.0 <= e1 - s1 <= 8.0) or _straddles(ep, e1, p["straddle_tolerance_s"]):
        return None
    e2 = _window_end(ep, cuts, c, cfg)
    if _straddles(ep, e2, p["straddle_tolerance_s"]):
        return None
    best = None
    for k in beats:
        if k.idx <= h.idx or k.t0 < e1 or k.idx > c.idx:
            continue
        s2 = _window_start(ep, cuts, k, cfg)
        if s2 < e1:
            continue
        if _straddles(ep, s2, p["straddle_tolerance_s"]):
            continue
        seg2 = e2 - s2
        total = (e1 - s1) + seg2
        if 15.0 <= seg2 <= 50.0 and p["min_duration_s"] <= total <= p["max_duration_s"]:
            if best is None or seg2 > best[1]:
                best = (s2, seg2)
    if not best:
        return None
    s2 = best[0]
    ids = [b.global_id for b in beats if (s1 - 0.01 <= b.t0 and b.t1 <= e1 + 0.01) or (s2 - 0.01 <= b.t0 and b.t1 <= e2 + 0.01)]
    return Window(
        cand_id=f"e{ep.number:03d}-2seg-{h.beat_id}-{c.beat_id}",
        segments=[Segment(episode=ep.number, start_s=s1, end_s=e1), Segment(episode=ep.number, start_s=s2, end_s=e2)],
        hook_beat=h.global_id,
        cliff_beat=c.global_id,
        beat_ids=ids,
        duration_s=round((e1 - s1) + (e2 - s2), 2),
        kind="two_segment",
        flags={"segments": 2},
    )


def boundary_windows(
    ep_a: Episode, beats_a: list[Beat], scores_a: dict[str, BeatScore],
    ep_b: Episode, beats_b: list[Beat], scores_b: dict[str, BeatScore], cfg: dict,
) -> list[Window]:
    """Tail of episode A + head of episode B on a virtual timeline. Episodes in this format continue
    directly (no recap), so the join is a real continuous moment split across two files."""
    p = cfg["promo"]
    span = cfg["candidates"]["boundary_span_s"]
    cuts_a, cuts_b = cut_set(ep_a), cut_set(ep_b)
    hooks = [b for b in _hook_candidates(beats_a, scores_a, cfg) if b.t0 >= ep_a.duration - span]
    cliffs = [b for b in _cliff_candidates(beats_b, scores_b, cfg) if b.t1 <= span and b is not beats_b[-1]]
    out: list[Window] = []
    for h in hooks:
        start = _window_start(ep_a, cuts_a, h, cfg)
        if _straddles(ep_a, start, p["straddle_tolerance_s"]):
            continue
        for c in cliffs:
            end_b = _window_end(ep_b, cuts_b, c, cfg)
            dur = (ep_a.duration - start) + end_b
            if not (p["min_duration_s"] <= dur <= p["max_duration_s"]):
                continue
            if _straddles(ep_b, end_b, p["straddle_tolerance_s"]):
                continue
            ids = [b.global_id for b in beats_a if b.t0 >= start - 0.01] + [b.global_id for b in beats_b if b.t1 <= end_b + 0.01]
            # some episodes re-speak the last line at the top of the next one; flag it so the editor trims the echo
            last_a = next((b for b in reversed(beats_a) if b.kind == "dialogue"), None)
            first_b = next((b for b in beats_b if b.kind == "dialogue" and b.t1 <= end_b + 0.01), None)
            echo = bool(last_a and first_b and last_a.text.strip().lower() == first_b.text.strip().lower())
            out.append(
                Window(
                    cand_id=f"e{ep_a.number:03d}-e{ep_b.number:03d}-{h.beat_id}-{c.beat_id}",
                    segments=[
                        Segment(episode=ep_a.number, start_s=start, end_s=ep_a.duration),
                        Segment(episode=ep_b.number, start_s=0.0, end_s=end_b),
                    ],
                    hook_beat=h.global_id,
                    cliff_beat=c.global_id,
                    beat_ids=ids,
                    duration_s=round(dur, 2),
                    kind="boundary",
                    flags={"segments": 2, "boundary": True, "boundary_echo": echo},
                )
            )
    return out
