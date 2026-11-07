"""Stage A orchestration: aliases (one call per show) and per-episode beat scoring.
The LLM returns scores keyed by beat id; code validates every id against the sheet it sent."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from .beats import beat_sheet, build_beats
from .features import heuristic_scores
from .llm import LLM, LLMError
from .models import Beat, BeatRole, BeatScore, BeatType, Episode

PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSION = "v2"  # v2: + role (compress/release) and intensity per beat; editor's rubric on clips
BEAT_TYPES = set(BeatType.__args__)  # type: ignore[attr-defined]
BEAT_ROLES = set(BeatRole.__args__)  # type: ignore[attr-defined]


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


class BeatScoreOut(BaseModel):
    beat_id: str
    hook: int = 0
    cliff: int = 0
    beat_type: str = "other"
    role: str = "none"
    intensity: int = 0
    unresolved_refs: list[str] = Field(default_factory=list)


class BeatScoresOut(BaseModel):
    beats: list[BeatScoreOut]


class EpisodeScoresOut(BaseModel):
    episode: int
    beats: list[BeatScoreOut]


class BatchScoresOut(BaseModel):
    episodes: list[EpisodeScoresOut]


BATCH_NOTE = (
    "\n\nYou will receive {n} episodes in one message, each headed '=== EPISODE N ==='. Score every beat of every "
    "episode. Return JSON only, shaped exactly like: {{\"episodes\": [{{\"episode\": N, \"beats\": [{{\"beat_id\": \"b01\", "
    "\"hook\": 0, \"cliff\": 0, \"beat_type\": \"line\", \"role\": \"setup\", \"intensity\": 0, \"unresolved_refs\": []}}, ...]}}, ...]}} — one entry per episode, "
    "beat ids exactly as given."
)


class AliasesOut(BaseModel):
    aliases: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


@dataclass
class ScoredShow:
    episodes: list[Episode]
    beats_by_ep: dict[int, list[Beat]]
    llm_by_ep: dict[int, dict[str, BeatScore]]       # empty dict per episode when no LLM
    heur_by_ep: dict[int, dict[str, BeatScore]]
    aliases: dict[str, str] = field(default_factory=dict)
    alias_notes: str = ""
    llm_failures: dict[int, str] = field(default_factory=dict)
    id_issues: dict[int, dict] = field(default_factory=dict)
    _beat_index: dict[str, Beat] = field(default_factory=dict)
    arc: object = None  # show_arc.ShowArc, attached after scoring

    @property
    def starts(self) -> dict[int, float]:
        """Seconds of watching before each episode starts (cached)."""
        if not hasattr(self, "_starts"):
            t, out = 0.0, {}
            for ep in sorted(self.episodes, key=lambda e: e.number):
                out[ep.number] = t
                t += ep.duration
            object.__setattr__(self, "_starts", out)
        return self._starts  # type: ignore[attr-defined]

    @property
    def leads(self) -> list[str]:
        if self.arc is not None and getattr(self.arc, "leads", None):
            return list(self.arc.leads)  # type: ignore[attr-defined]
        from collections import Counter
        c = Counter(b.speaker for bs in self.beats_by_ep.values() for b in bs if b.kind == "dialogue" and b.speaker)
        return [s for s, _ in c.most_common(2)]

    def __post_init__(self):
        self._beat_index = {b.global_id: b for bs in self.beats_by_ep.values() for b in bs}

    def episode(self, n: int) -> Episode:
        return next(e for e in self.episodes if e.number == n)

    def beat(self, gid: str) -> Beat:
        return self._beat_index[gid]

    def scores(self, ep_num: int, source: str = "llm") -> dict[str, BeatScore]:
        if source == "llm" and self.llm_by_ep.get(ep_num):
            return self.llm_by_ep[ep_num]
        return self.heur_by_ep[ep_num]

    def score(self, gid: str, source: str = "llm") -> BeatScore:
        b = self.beat(gid)
        return self.scores(b.episode, source)[b.beat_id]

    @property
    def has_llm(self) -> bool:
        return any(self.llm_by_ep.values())


def build_aliases(episodes: list[Episode], llm: LLM | None, raw_dir: Path | None) -> tuple[dict[str, str], str]:
    if llm is None:
        return {}, "no model: labels left as written"
    from collections import Counter

    counts = Counter(e.speaker for ep in episodes for e in ep.events if e.kind == "dialogue" and e.speaker)
    labels = "\n".join(f"{c:4d}  {name}" for name, c in counts.most_common())
    loglines = "\n".join(f"Ep {ep.number}: {ep.logline}" for ep in episodes if ep.logline)
    user = f"SPEAKER LABELS (count, label):\n{labels}\n\nLOGLINES:\n{loglines}"
    try:
        out = llm.complete_json(load_prompt("aliases"), user, AliasesOut, "aliases", "show",
                                raw_path=(raw_dir / "aliases" / "show.json") if raw_dir else None)
    except LLMError as e:
        if e.fatal:
            raise
        return {}, f"alias call failed ({e}); labels left as written"
    proposed = {k.strip().upper(): v.strip().upper() for k, v in out.aliases.items() if k.strip() and v.strip() and k.strip().upper() in counts}
    proposed = {k: v for k, v in proposed.items() if k != v}
    # Conservative rule: a merge is applied only when the canonical name is literally inside the variant label
    # ("LADY ELEANOR" -> "ELEANOR"). Anything else ("HOODED STRANGER" -> "EDMUND") is a guess about
    # identity that a wrong model answer would turn into a wrong name on a promo card, so it is kept as a
    # suggestion for a human in aliases.yaml instead of being applied.
    valid = {k: v for k, v in proposed.items() if re.search(r"\b" + re.escape(v) + r"\b", k)}
    rejected = {k: v for k, v in proposed.items() if k not in valid}
    for k, v in list(valid.items()):  # no chains
        if v in valid:
            valid[k] = valid[v]
    notes = out.notes + (f" | not applied (needs a human): {rejected}" if rejected else "")
    return valid, notes


def _merge(ep: Episode, beats: list[Beat], out: BeatScoresOut, heur: dict[str, BeatScore]) -> tuple[dict[str, BeatScore], dict]:
    by_id = {b.beat_id: b for b in beats}
    got = {o.beat_id.strip(): o for o in out.beats}
    unknown = sorted(set(got) - set(by_id))
    missing = sorted(set(by_id) - set(got))
    merged: dict[str, BeatScore] = {}
    for bid in by_id:
        o = got.get(bid)
        if o is None:
            merged[bid] = heur[bid]
            continue
        bt = o.beat_type.strip().lower() if o.beat_type else "other"
        role = o.role.strip().lower() if isinstance(o.role, str) else "none"
        try:
            inten = max(0, min(3, int(o.intensity)))
        except (TypeError, ValueError):
            inten = heur[bid].intensity
        merged[bid] = BeatScore(
            beat_id=bid,
            hook=max(0, min(3, int(o.hook))),
            cliff=max(0, min(3, int(o.cliff))),
            beat_type=bt if bt in BEAT_TYPES else "other",  # type: ignore[arg-type]
            unresolved_refs=[r for r in o.unresolved_refs if isinstance(r, str)][:4],
            role=role if role in BEAT_ROLES else heur[bid].role,  # type: ignore[arg-type]
            intensity=inten,
            source="llm",
        )
    return merged, {"unknown_ids": unknown, "missing_ids_filled_from_heuristic": missing}


RETRY_NOTE = (
    "\n\nNOTE: a previous answer for this episode gave nearly every beat the same score. Read each beat and score it "
    "on its own merits; do not give every beat the same value, and do not inflate a beat that is genuinely flat."
)


def degenerate(scores: dict[str, BeatScore]) -> str | None:
    """A 'flat' answer: no beat above 1 for hook or for cliff. Long batched answers from small models sometimes fill
    a whole episode with zeros; a microdrama episode always contains some conflict and ends on a cliff."""
    llm = [s for s in scores.values() if s.source == "llm"]
    if not llm:
        return None
    mh, mc = max(s.hook for s in llm), max(s.cliff for s in llm)
    if mh <= 1 or mc <= 1:
        return f"flat answer (max hook {mh}, max cliff {mc})"
    return None


def score_show(
    episodes: list[Episode], llm: LLM | None, cfg: dict, raw_dir: Path | None,
    strip_cues: bool = False, progress: Callable[[int], None] | None = None,
    aliases: dict[str, str] | None = None, alias_notes: str = "", votes: int = 1,
) -> ScoredShow:
    """Score every beat. With votes > 1, each episode is read that many times independently (readings after the first
    carry a note that changes the prompt, so neither the cache nor the provider returns the first answer again) and the
    readings are combined: mean hook / cliff / intensity, majority beat type and role. Simulated on this show's measured
    noise, 2 readings raise the expected top-10 overlap between two runs from 40% to 52% (docs/research.md)."""
    first = _score_once(episodes, llm, cfg, raw_dir, strip_cues, progress, aliases, alias_notes, "")
    if llm is None or votes <= 1:
        return first
    readings = [first] + [_score_once(episodes, llm, cfg, (raw_dir / f"vote-{v}") if raw_dir else None, strip_cues, None, aliases,
                                      alias_notes, f"\n\n(Independent reading {v} of {votes}: score from scratch.)") for v in range(2, votes + 1)]
    from collections import Counter
    for ep in episodes:
        merged = {}
        for bid, s0 in first.llm_by_ep.get(ep.number, {}).items():
            got = [r.llm_by_ep.get(ep.number, {}).get(bid) for r in readings]
            got = [g for g in got if g is not None and g.source == "llm"]
            if not got:
                merged[bid] = s0
                continue
            mode = lambda xs: Counter(xs).most_common(1)[0][0]  # noqa: E731
            merged[bid] = s0.model_copy(update={
                "hook": round(sum(g.hook for g in got) / len(got), 3), "cliff": round(sum(g.cliff for g in got) / len(got), 3),
                "intensity": round(sum(g.intensity for g in got) / len(got), 3),
                "beat_type": mode([g.beat_type for g in got]), "role": mode([g.role for g in got])})
        first.llm_by_ep[ep.number] = merged
    first.alias_notes = (first.alias_notes + f" | {votes} independent readings per episode, averaged").strip(" |")
    return first


def _score_once(
    episodes: list[Episode], llm: LLM | None, cfg: dict, raw_dir: Path | None,
    strip_cues: bool, progress: Callable[[int], None] | None,
    aliases: dict[str, str] | None, alias_notes: str, system_suffix: str,
) -> ScoredShow:
    beats_by_ep = {ep.number: build_beats(ep, aliases) for ep in episodes}
    heur_by_ep = {ep.number: heuristic_scores(ep, beats_by_ep[ep.number]) for ep in episodes}
    llm_by_ep: dict[int, dict[str, BeatScore]] = {ep.number: {} for ep in episodes}
    failures: dict[int, str] = {}
    issues: dict[int, dict] = {}
    if llm is not None:
        system = load_prompt("score_beats") + system_suffix
        per_call = max(1, int(cfg["llm"].get("episodes_per_call", 1)))

        def _fix_flat(ep: Episode, merged: dict[str, BeatScore], info: dict) -> tuple[dict[str, BeatScore], dict]:
            """Re-ask a flat episode alone, once. Keep the second answer either way: if it is flat too, the model
            means it (a genuinely quiet episode); if not, the first answer was a lapse. Both cases are logged."""
            why = degenerate(merged)
            if not why:
                return merged, info
            try:
                again = llm.complete_json(system + RETRY_NOTE, beat_sheet(ep, beats_by_ep[ep.number], strip_cues=strip_cues), BeatScoresOut,
                                          "score_beats", f"ep-{ep.number:03d}-flat-retry",
                                          raw_path=(raw_dir / "score_beats" / f"ep-{ep.number:03d}-flat-retry.json") if raw_dir else None)
            except LLMError as e:
                if e.fatal:
                    raise
                return merged, {**info, "note": f"{why}; retry failed ({e}); kept the flat answer"}
            m2, i2 = _merge(ep, beats_by_ep[ep.number], again, heur_by_ep[ep.number])
            still = degenerate(m2)
            return m2, {**i2, "note": f"{why}; re-scored alone" + (f", still {still} (kept: the model means it)" if still else ", fixed")}

        def work(chunk: list[Episode]) -> None:
            if len(chunk) == 1:
                ep = chunk[0]
                beats = beats_by_ep[ep.number]
                sheet = beat_sheet(ep, beats, strip_cues=strip_cues)
                try:
                    out = llm.complete_json(system, sheet, BeatScoresOut, "score_beats", f"ep-{ep.number:03d}",
                                            raw_path=(raw_dir / "score_beats" / f"ep-{ep.number:03d}.json") if raw_dir else None)
                    merged, info = _merge(ep, beats, out, heur_by_ep[ep.number])
                    merged, info = _fix_flat(ep, merged, info)
                    llm_by_ep[ep.number] = merged
                    if info["unknown_ids"] or info["missing_ids_filled_from_heuristic"] or info.get("note"):
                        issues[ep.number] = info
                except LLMError as e:
                    if e.fatal:
                        raise
                    failures[ep.number] = str(e)
                if progress:
                    progress(ep.number)
                return
            tag = f"batch-{chunk[0].number:03d}-{chunk[-1].number:03d}"
            user = "\n\n".join(f"=== EPISODE {ep.number} ===\n{beat_sheet(ep, beats_by_ep[ep.number], strip_cues=strip_cues)}" for ep in chunk)
            got: dict[int, EpisodeScoresOut] = {}
            err = None
            try:
                out = llm.complete_json(system + BATCH_NOTE.format(n=len(chunk)), user, BatchScoresOut, "score_beats", tag,
                                        raw_path=(raw_dir / "score_beats" / f"{tag}.json") if raw_dir else None)
                got = {e.episode: e for e in out.episodes}
            except LLMError as e:
                if e.fatal:
                    raise
                err = str(e)
            for ep in chunk:
                entry = got.get(ep.number)
                if entry is None and err is None:
                    # the model dropped this episode from its batch answer: ask for it alone (one extra call)
                    beats = beats_by_ep[ep.number]
                    try:
                        single = llm.complete_json(system, beat_sheet(ep, beats, strip_cues=strip_cues), BeatScoresOut, "score_beats", f"ep-{ep.number:03d}-retry",
                                                   raw_path=(raw_dir / "score_beats" / f"ep-{ep.number:03d}-retry.json") if raw_dir else None)
                        entry = EpisodeScoresOut(episode=ep.number, beats=single.beats)
                        issues[ep.number] = {"unknown_ids": [], "missing_ids_filled_from_heuristic": [], "note": f"missing from batch {tag}; re-scored alone"}
                    except LLMError as e:
                        if e.fatal:
                            raise
                        err = str(e)
                if entry is None:
                    failures[ep.number] = err or f"episode {ep.number} missing from batch response {tag}"
                else:
                    merged, info = _merge(ep, beats_by_ep[ep.number], BeatScoresOut(beats=entry.beats), heur_by_ep[ep.number])
                    if len(info["missing_ids_filled_from_heuristic"]) == len(beats_by_ep[ep.number]) and err is None:
                        got.pop(ep.number, None)  # empty entry: retry alone via the branch above on next pass
                        try:
                            single = llm.complete_json(system, beat_sheet(ep, beats_by_ep[ep.number], strip_cues=strip_cues), BeatScoresOut, "score_beats", f"ep-{ep.number:03d}-retry",
                                                       raw_path=(raw_dir / "score_beats" / f"ep-{ep.number:03d}-retry.json") if raw_dir else None)
                            merged, info = _merge(ep, beats_by_ep[ep.number], single, heur_by_ep[ep.number])
                            info = {**info, "note": f"empty in batch {tag}; re-scored alone"}
                        except LLMError as e:
                            if e.fatal:
                                raise
                    merged, info = _fix_flat(ep, merged, info)
                    llm_by_ep[ep.number] = merged
                    if info["unknown_ids"] or info["missing_ids_filled_from_heuristic"] or info.get("note"):
                        issues[ep.number] = info
                if progress:
                    progress(ep.number)

        chunks = [episodes[i:i + per_call] for i in range(0, len(episodes), per_call)]
        # the first chunk runs alone so that a bad key / unavailable model aborts in one call
        work(chunks[0])
        with ThreadPoolExecutor(max_workers=int(cfg["llm"].get("concurrency", 2))) as pool:
            list(pool.map(work, chunks[1:]))
    elif progress:
        for ep in episodes:
            progress(ep.number)
    return ScoredShow(episodes, beats_by_ep, llm_by_ep, heur_by_ep, aliases or {}, alias_notes, failures, issues)
