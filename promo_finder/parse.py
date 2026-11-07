"""Tolerant screenplay parser.

Grammar observed in the input pack (and the drift we tolerate):
  EPISODE NUMBER: Episode 12            header lines (any case)
  SCENE 01 / Scene heading: ... / Scene start: MM:SS.ff / Scene end: ...   (episode 6 uses UPPERCASE)
  **LABEL:**                             CAMERA | ACTION | SOUND FX | CHARACTER (Emotion) (O.S.)
  Start: (MM:SS.ff) | End: (MM:SS.ff)
  one content line
Blocks that do not fit are counted and skipped, never guessed. A file whose block parse rate
falls below `min_parse_rate` raises, because confident garbage is worse than a loud failure.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .models import Episode, Event, Scene

HEADER_RE = re.compile(r"^\s*(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.I)
SCENE_META_RE = re.compile(r"^\s*scene\s+(heading|start|end)\s*:\s*(.*)$", re.I)
LABEL_RE = re.compile(r"^\s*\*\*(.+?)\s*:?\s*\*\*\s*:?\s*$")
_T = r"(?:(\d+):)?(\d+):(\d+(?:[.,]\d+)?)"  # [HH:]MM:SS[.ff]
TS_RE = re.compile(r"start\s*:?\s*\(?\s*" + _T + r"\s*\)?\s*\|\s*end\s*:?\s*\(?\s*" + _T + r"\s*\)?", re.I)
TS_SIMPLE_RE = re.compile(_T)
MODE_RE = re.compile(r"^(O\.?S\.?|V\.?O\.?|OFF-?SCREEN|CONT'?D|CONTINUED)$", re.I)
NON_DIALOGUE = {
    "camera": {"CAMERA", "SHOT", "CAM"},
    "action": {"ACTION", "ACTIONS"},
    "sfx": {"SOUND FX", "SFX", "SOUND", "SOUND EFFECT", "SOUND EFFECTS", "FX"},
    "other": {"MUSIC", "TITLE", "TEXT", "VFX", "TRANSITION", "INSERT", "NOTE", "SUPER", "CAPTION"},
}
NON_SPEAKER_RE = re.compile(r"\b(ON[- ]?SCREEN|TITLE CARD|TEXT|CAPTION|CHYRON|LOWER THIRD|SUPER|SUBTITLE|GRAPHIC|MONTAGE|"
                            r"FLASHBACK|INTERCUT|TRANSITION|CUT TO|FADE (IN|OUT|TO))\b")
MIN_PARSE_RATE = 0.95


def _ts(h: str | None, m: str, s: str) -> float:
    return (int(h) if h else 0) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def parse_timestamp(value: str) -> float | None:
    m = TS_SIMPLE_RE.search(value or "")
    return _ts(m.group(1), m.group(2), m.group(3)) if m else None


def parse_label(label: str) -> tuple[str, str | None, str | None, list[str]]:
    """'LADY ELEANOR (O.S.) (Trembling, Pleading)' -> (kind, speaker, mode, emotions)."""
    raw = label.strip()
    head = raw.split("(", 1)[0].strip()
    key = head.upper().rstrip(":").strip()
    for kind, names in NON_DIALOGUE.items():
        if key in names:
            return kind, None, None, []
    if NON_SPEAKER_RE.search(key):  # "TEXT ON SCREEN", "TITLE CARD", "ON-SCREEN TEXT", "SUPER", "CHYRON": not a person
        return "other", None, None, []
    mode, emotions = None, []
    for group in re.findall(r"\(([^)]*)\)", raw):
        for tok in re.split(r"[/,;]", group):
            tok = tok.strip()
            if not tok:
                continue
            if MODE_RE.match(tok):
                mode = tok.upper().replace(" ", "")
            else:
                emotions.append(tok)
    return "dialogue", key, mode, emotions


def episode_number_from(file: Path, header_value: str | None) -> int:
    if header_value:
        m = re.search(r"(\d+)", header_value)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d+)", file.stem)
    return int(m.group(1)) if m else 0


SCRIPT_SUFFIXES = (".txt", ".md", ".fountain", ".text")
EPISODE_HEADER_RE = re.compile(r"^\s*EPISODE NUMBER\s*:", re.I | re.M)


def read_text(path: Path) -> str:
    """UTF-8 (with or without BOM) first; then Windows-1252, which is what Word exports on Indian/US Windows machines."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_episodes(text: str) -> list[str]:
    """One file may hold several episodes (a writer pasted the whole show into one document): split at each
    'EPISODE NUMBER:' header. A file with zero or one header is one episode."""
    starts = [m.start() for m in EPISODE_HEADER_RE.finditer(text)]
    if len(starts) < 2:
        return [text]
    if text[: starts[0]].strip():
        starts[0] = 0  # keep any preamble with the first episode
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


class NotAScreenplay(ValueError):
    """The file contains no timed blocks at all (a README, notes, an empty file): skipped, not fatal."""


def parse_episode(path: Path, min_parse_rate: float = MIN_PARSE_RATE, text: str | None = None, name: str | None = None) -> Episode:
    lines = (text if text is not None else read_text(path)).splitlines()
    shown = name or path.name
    header: dict[str, str] = {}
    scenes: list[Scene] = []
    events: list[Event] = []
    warnings: list[str] = []
    blocks_seen = blocks_parsed = 0
    uppercase_meta = False
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if (m := HEADER_RE.match(line)):
            header[m.group(1).upper()] = m.group(2).strip()
            i += 1
            continue
        if (m := SCENE_RE.match(line)):
            scenes.append(Scene(idx=len(scenes) + 1))
            i += 1
            continue
        if (m := SCENE_META_RE.match(line)):
            if line.lstrip().startswith("SCENE"):
                uppercase_meta = True
            if not scenes:
                scenes.append(Scene(idx=1))
            field, value = m.group(1).lower(), m.group(2).strip()
            if field == "heading":
                scenes[-1].heading = re.sub(r"\s*[–—-]\s*", " - ", value)
            elif field == "start":
                scenes[-1].t0 = parse_timestamp(value) or 0.0
            else:
                scenes[-1].t1 = parse_timestamp(value) or 0.0
            i += 1
            continue
        if (m := LABEL_RE.match(line)):
            blocks_seen += 1
            label = m.group(1).strip()
            # timestamp: next non-empty line
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            tm = TS_RE.search(lines[j]) if j < n else None
            if not tm:
                warnings.append(f"line {i + 1}: block '{label[:30]}' has no timestamp line; skipped")
                i += 1
                continue
            t0, t1 = _ts(tm.group(1), tm.group(2), tm.group(3)), _ts(tm.group(4), tm.group(5), tm.group(6))
            k = j + 1
            content: list[str] = []
            while k < n and lines[k].strip() and not LABEL_RE.match(lines[k]) and not SCENE_RE.match(lines[k]):
                content.append(lines[k].strip())
                k += 1
            text = " ".join(content).strip()
            kind, speaker, mode, emotions = parse_label(label)
            if kind == "dialogue":
                text = text.strip('"“”').strip()
            if not scenes:
                scenes.append(Scene(idx=1))
            if t1 < t0:
                warnings.append(f"line {j + 1}: end before start ({t0:.2f}>{t1:.2f}); swapped")
                t0, t1 = t1, t0
            events.append(
                Event(
                    block_id=f"k{len(events) + 1:04d}",
                    idx=len(events),
                    kind=kind,
                    label=label,
                    speaker=speaker,
                    mode=mode,
                    emotions=emotions,
                    t0=t0,
                    t1=t1,
                    text=text,
                    scene_idx=scenes[-1].idx,
                )
            )
            blocks_parsed += 1
            i = k
            continue
        i += 1

    if uppercase_meta:
        warnings.append("scene metadata labels are UPPERCASE (format drift; handled)")
    # scene-end clamping and ordering checks
    for s in scenes:
        if s.t1 and s.t0 and s.t1 < s.t0:
            warnings.append(f"scene {s.idx}: end before start; swapped")
            s.t0, s.t1 = s.t1, s.t0
    scene_end = {s.idx: s.t1 for s in scenes if s.t1}
    last = -1.0
    nonmono = 0
    for e in events:
        se = scene_end.get(e.scene_idx)
        if se and e.t1 > se + 0.05:
            warnings.append(f"{e.block_id} {e.label[:20]} overruns scene {e.scene_idx} end by {e.t1 - se:.2f}s; clamped")
            e.t1 = se
            if e.t0 > e.t1:
                e.t0 = e.t1
        if e.t0 < last - 0.001:
            nonmono += 1
        last = max(last, e.t0)
    if nonmono:
        warnings.append(f"{nonmono} blocks start earlier than a previous block (sorted by start)")
    events.sort(key=lambda e: (e.t0, e.idx))
    for k, e in enumerate(events):
        e.idx = k
    duration = max([s.t1 for s in scenes if s.t1] + [e.t1 for e in events] + [0.0])
    rate = blocks_parsed / blocks_seen if blocks_seen else 0.0
    if blocks_seen and rate < min_parse_rate:
        raise ValueError(
            f"{shown}: only {blocks_parsed}/{blocks_seen} blocks parsed ({rate:.0%}). "
            "The format differs from what this parser understands; refusing to guess."
        )
    if not events:
        raise NotAScreenplay(f"{shown}: no timed blocks found")
    number = episode_number_from(path, header.get("EPISODE NUMBER"))
    return Episode(
        number=number,
        file=shown,
        logline=header.get("EPISODE LOGLINE", ""),
        continuity=header.get("CONTINUITY NOTES", ""),
        scenes=scenes,
        events=events,
        duration=duration,
        warnings=warnings,
        blocks_seen=blocks_seen,
        blocks_parsed=blocks_parsed,
    )


def find_script_files(folder: Path) -> list[Path]:
    """A folder of screenplays (.txt/.md/.fountain), a single file, or a .zip of either."""
    if folder.is_file() and folder.suffix.lower() in SCRIPT_SUFFIXES:
        return [folder]
    if folder.is_file() and folder.suffix.lower() == ".zip":
        import tempfile
        import zipfile

        tmp = Path(tempfile.mkdtemp(prefix="promo-finder-"))
        with zipfile.ZipFile(folder) as z:
            z.extractall(tmp)
        folder = tmp
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SCRIPT_SUFFIXES
                   and not p.name.startswith(".") and "__MACOSX" not in p.parts)
    return files


FORMAT_HINT = ("Expected one screenplay per episode with blocks like:\n  **CHARACTER (Emotion):**\n  Start: (MM:SS.ff) | End: (MM:SS.ff)\n  "
               "the line\n(.txt/.md/.fountain files, a folder, or a .zip). Run `promo-finder parse <folder>` to see what was understood.")


def parse_show(folder: Path, skipped: list[dict] | None = None) -> list[Episode]:
    """Parse every episode. Files with no timed blocks at all are skipped with a note (a README in the folder must not
    kill the run); a screenplay that only partly parses is still fatal (format drift: refuse to guess)."""
    files = find_script_files(folder)
    if not files:
        raise FileNotFoundError(f"no screenplay files (.txt/.md/.fountain) under {folder}.\n{FORMAT_HINT}")
    episodes: list[Episode] = []
    skip: list[dict] = []
    for p in files:
        chunks = split_episodes(read_text(p))
        for k, chunk in enumerate(chunks, 1):
            name = p.name if len(chunks) == 1 else f"{p.name}#{k}"
            try:
                ep = parse_episode(p, text=chunk, name=name)
            except NotAScreenplay as e:
                skip.append({"file": name, "reason": str(e)})
                continue
            if len(chunks) > 1:
                ep.warnings.append(f"split from {p.name}: it holds {len(chunks)} episodes")
            episodes.append(ep)
    if skipped is not None:
        skipped.extend(skip)
    if not episodes:
        raise ValueError(f"no screenplay found under {folder} ({len(skip)} file(s) had no timed blocks).\n{FORMAT_HINT}")
    # numbering: no number anywhere (no header, no digits in the file name) -> number every episode by file order
    if any(e.number == 0 for e in episodes):
        for k, e in enumerate(episodes, 1):
            e.warnings.append(f"no episode number in header or file name; numbered by file order: {e.file} -> {k}")
            e.number = k
    # two files claiming the same episode (e.g. a revision left in the folder): keep the first, skip the rest, loudly
    kept: dict[int, Episode] = {}
    for e in episodes:
        if e.number in kept:
            skip.append({"file": e.file, "reason": f"duplicate of episode {e.number} ({kept[e.number].file}); skipped — remove one of them"})
            if skipped is not None:
                skipped.append(skip[-1])
            continue
        kept[e.number] = e
    return sorted(kept.values(), key=lambda e: e.number)


def name_drift_report(episodes: list[Episode]) -> dict:
    """Generic drift detectors: speaker-label inventory and proper-noun spelling clusters.
    A proper noun is a capitalised word that appears mid-sentence (after a lowercase word), so
    sentence-initial words such as 'Every' do not count."""
    speakers = Counter(e.speaker for ep in episodes for e in ep.events if e.kind == "dialogue" and e.speaker)
    words: Counter[str] = Counter()
    for ep in episodes:
        for e in ep.events:
            for w in re.findall(r"(?<=[a-z,;:'] )([A-Z][a-z]{4,})", e.text):
                words[w] += 1
    clusters: dict[str, Counter] = {}
    for w, c in words.items():
        clusters.setdefault(w[:3], Counter())[w] = c
    drift = {}
    for k, v in clusters.items():
        variants = {w: c for w, c in v.items() if c >= 2}
        if len(variants) >= 3:
            drift[k] = dict(sorted(variants.items(), key=lambda kv: -kv[1]))
    return {"speakers": dict(speakers.most_common()), "spelling_clusters": drift}


def parse_report(episodes: list[Episode]) -> dict:
    per_ep = [
        {
            "episode": ep.number,
            "file": ep.file,
            "duration_s": round(ep.duration, 2),
            "scenes": len(ep.scenes),
            "blocks": ep.blocks_parsed,
            "dialogue_lines": len(ep.dialogue),
            "parse_rate": round(ep.blocks_parsed / ep.blocks_seen, 3) if ep.blocks_seen else 0,
            "warnings": ep.warnings,
        }
        for ep in episodes
    ]
    total_blocks = sum(ep.blocks_parsed for ep in episodes)
    kinds = Counter(e.kind for ep in episodes for e in ep.events)
    return {
        "episodes": len(episodes),
        "total_blocks": total_blocks,
        "block_kinds": dict(kinds),
        "duration_min_s": round(min(ep.duration for ep in episodes), 1),
        "duration_max_s": round(max(ep.duration for ep in episodes), 1),
        "episodes_with_warnings": sum(1 for ep in episodes if ep.warnings),
        "per_episode": per_ep,
        "drift": name_drift_report(episodes),
    }
