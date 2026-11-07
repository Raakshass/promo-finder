"""Whole-show understanding: one model call over every episode's logline and continuity note (the writers' own
summaries, ~3K tokens for 50 episodes) gives the premise, the leads, the turning points and the major reveals.

Why: per-episode scoring sees one episode at a time, so it cannot tell a plot-turning moment from an intense side
scene, or a premise-level hook from an endgame spoiler. Screenplay-trailer research (Papalampidi et al. 2019/2021)
found that intensity alone picks off-plot moments; structure plus intensity works. Used in ranking as:
  - the editor's check "features a lead" (a promo sells the people the show is about),
  - a small bonus for clips from turning-point episodes (except the climax),
  - a flag (not a penalty) on mid/late clips from an episode with a major reveal.
Without a model, leads fall back to the two speakers with the most lines; there are no turning points or reveals.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from .beat_scoring import load_prompt
from .llm import LLM, LLMError
from .models import Episode

TP_LABELS = {
    "opportunity": "opportunity", "change_of_plans": "change of plans", "point_of_no_return": "point of no return",
    "major_setback": "major setback", "climax": "climax",
}


class TurningPoint(BaseModel):
    episode: int
    tp: str
    event: str = ""


class Reveal(BaseModel):
    episode: int
    reveal: str = ""


class ShowArc(BaseModel):
    premise: str = ""
    leads: list[str] = Field(default_factory=list)
    antagonists: list[str] = Field(default_factory=list)
    turning_points: list[TurningPoint] = Field(default_factory=list)
    major_reveals: list[Reveal] = Field(default_factory=list)
    source: str = "heuristic"

    def tp_for(self, episode: int) -> str | None:
        for t in self.turning_points:
            if t.episode == episode:
                return t.tp
        return None

    def reveal_for(self, episode: int) -> str | None:
        for r in self.major_reveals:
            if r.episode == episode:
                return r.reveal
        return None


def speaker_counts(episodes: list[Episode], aliases: dict[str, str]) -> Counter:
    return Counter(aliases.get(e.speaker, e.speaker) for ep in episodes for e in ep.events if e.kind == "dialogue" and e.speaker)


def heuristic_arc(episodes: list[Episode], aliases: dict[str, str]) -> ShowArc:
    counts = speaker_counts(episodes, aliases)
    return ShowArc(leads=[s for s, _ in counts.most_common(2)], source="heuristic")


def validate(arc: ShowArc, episodes: list[Episode], counts: Counter) -> ShowArc:
    """Keep only labels that exist and episodes that exist; a model's guess never becomes a wrong name or number."""
    labels = {k.upper(): k for k in counts}
    nums = {e.number for e in episodes}
    fix = lambda xs: [labels[x.strip().upper()] for x in xs if x and x.strip().upper() in labels][:3]  # noqa: E731
    arc.leads = fix(arc.leads) or [s for s, _ in counts.most_common(2)]
    arc.antagonists = [a for a in fix(arc.antagonists) if a not in arc.leads]
    seen: set[str] = set()
    tps = []
    for t in arc.turning_points:
        key = t.tp.strip().lower().replace(" ", "_").replace("-", "_")
        if key in TP_LABELS and key not in seen and t.episode in nums:
            seen.add(key)
            tps.append(TurningPoint(episode=t.episode, tp=key, event=t.event[:200]))
    arc.turning_points = sorted(tps, key=lambda t: t.episode)
    arc.major_reveals = [Reveal(episode=r.episode, reveal=r.reveal[:200]) for r in arc.major_reveals if r.episode in nums][:6]
    return arc


def build_show_arc(episodes: list[Episode], llm: LLM | None, raw_dir: Path | None, aliases: dict[str, str]) -> tuple[ShowArc, str]:
    counts = speaker_counts(episodes, aliases)
    if llm is None:
        return heuristic_arc(episodes, aliases), "no model: leads = the two speakers with the most lines"
    eps = sorted(episodes, key=lambda e: e.number)
    user = ("SPEAKER LABELS (lines, label):\n" + "\n".join(f"{c:4d}  {s}" for s, c in counts.most_common(25))
            + "\n\nEPISODES:\n" + "\n".join(f"Ep {e.number}: {e.logline}" + (f" | continuity: {e.continuity}" if e.continuity else "") for e in eps))
    try:
        arc = llm.complete_json(load_prompt("show_arc"), user, ShowArc, "show_arc", "show",
                                raw_path=(raw_dir / "show_arc" / "show.json") if raw_dir else None)
    except LLMError as e:
        if e.fatal:
            raise
        return heuristic_arc(episodes, aliases), f"show-arc call failed ({e}); leads = top two speakers"
    arc.source = "llm"
    return validate(arc, episodes, counts), "ok"
