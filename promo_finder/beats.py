"""Beats: the atomic units the LLM scores. A beat is a dialogue line, an ACTION block or an SFX block.
CAMERA blocks are not beats; they define the cut set (shot boundaries) and give each beat its shot context."""
from __future__ import annotations

import bisect

from .models import Beat, Episode, Event


def cut_set(ep: Episode) -> list[float]:
    cuts = {0.0, ep.duration}
    for e in ep.events:
        if e.kind == "camera":
            cuts.add(round(e.t0, 2))
            cuts.add(round(e.t1, 2))
    for s in ep.scenes:
        cuts.add(round(s.t0, 2))
        if s.t1:
            cuts.add(round(s.t1, 2))
    return sorted(cuts)


def cut_before(cuts: list[float], t: float) -> float:
    i = bisect.bisect_right(cuts, t + 1e-6) - 1
    return cuts[i] if i >= 0 else 0.0


def cut_after(cuts: list[float], t: float) -> float:
    i = bisect.bisect_left(cuts, t - 1e-6)
    return cuts[i] if i < len(cuts) else t


def _enclosing_camera(ep: Episode, t: float) -> Event | None:
    best = None
    for e in ep.events:
        if e.kind == "camera" and e.t0 - 1e-6 <= t <= e.t1 + 1e-6:
            if best is None or e.t0 > best.t0:
                best = e
    return best


def build_beats(ep: Episode, aliases: dict[str, str] | None = None) -> list[Beat]:
    aliases = aliases or {}
    cuts = cut_set(ep)
    beats: list[Beat] = []
    for e in ep.events:
        if e.kind not in ("dialogue", "action", "sfx"):
            continue
        cam = _enclosing_camera(ep, e.t0)
        idx = len(beats) + 1
        speaker = aliases.get(e.speaker or "", e.speaker) if e.speaker else None
        beats.append(
            Beat(
                beat_id=f"b{idx:02d}",
                global_id=f"e{ep.number:03d}-b{idx:02d}",
                episode=ep.number,
                idx=idx - 1,
                kind=e.kind,  # type: ignore[arg-type]
                t0=e.t0,
                t1=e.t1,
                speaker=speaker,
                emotions=e.emotions,
                text=e.text,
                shot=(cam.text if cam else "")[:160],
                scene_idx=e.scene_idx,
                cut_start=cam.t0 if cam else cut_before(cuts, e.t0),
                cut_end=cam.t1 if cam else cut_after(cuts, e.t1),
            )
        )
    return beats


def beat_line(b: Beat, with_shot: bool = False) -> str:
    span = f"{b.t0:05.1f}-{b.t1:05.1f}"
    if b.kind == "dialogue":
        emo = f" ({', '.join(b.emotions)})" if b.emotions else ""
        s = f"[{b.beat_id} {span}] {b.speaker}{emo}: \"{b.text}\""
    elif b.kind == "action":
        s = f"[{b.beat_id} {span}] ACTION: {b.text}"
    else:
        s = f"[{b.beat_id} {span}] SFX: {b.text}"
    if with_shot and b.shot:
        s += f"  // shot: {b.shot[:90]}"
    return s


def beat_sheet(ep: Episode, beats: list[Beat], strip_cues: bool = False) -> str:
    """Compact text the beat-scoring model sees. Scene headings are kept for orientation."""
    import re

    lines = [f"EPISODE {ep.number} (duration {ep.duration:.1f}s)"]
    if ep.logline:
        lines.append(f"LOGLINE: {ep.logline}")
    scene_heads = {s.idx: s.heading for s in ep.scenes}
    last_scene = None
    for b in beats:
        if b.scene_idx != last_scene:
            lines.append(f"--- SCENE {b.scene_idx}: {scene_heads.get(b.scene_idx, '')}")
            last_scene = b.scene_idx
        txt = beat_line(b)
        if strip_cues:
            txt = re.sub(r"(?i)cliff-?hanger|cut to black|freeze[- ]frame|freezes?|sting\b|fade to black", "[…]", txt)
        lines.append(txt)
    return "\n".join(lines)
