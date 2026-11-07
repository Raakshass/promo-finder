"""Typed data model. Everything downstream of the parser speaks these types."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EventKind = Literal["dialogue", "camera", "action", "sfx", "other"]
BeatKind = Literal["dialogue", "action", "sfx"]
BeatType = Literal[
    "slap", "reveal", "threat", "line", "romance", "humiliation", "exposition", "other"
]
# Where a beat sits in the compress-and-release rhythm micro-drama writers use (压弹簧 / 放弹簧):
# setup (context, calm), escalation (pressure builds: insult, accusation, demand), turn (reversal or reveal that
# changes who has power), payoff (the release: the slap lands back, the apology, the rescue, the question answered).
BeatRole = Literal["setup", "escalation", "turn", "payoff", "none"]


def fmt_ts(seconds: float) -> str:
    """Seconds -> MM:SS.ff, the input pack's own format."""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


class Event(BaseModel):
    block_id: str
    idx: int
    kind: EventKind
    label: str
    speaker: Optional[str] = None
    mode: Optional[str] = None  # O.S. / V.O.
    emotions: list[str] = Field(default_factory=list)
    t0: float
    t1: float
    text: str
    scene_idx: int


class Scene(BaseModel):
    idx: int
    heading: str = ""
    t0: float = 0.0
    t1: float = 0.0


class Episode(BaseModel):
    number: int
    file: str
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    duration: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    blocks_seen: int = 0
    blocks_parsed: int = 0

    @property
    def dialogue(self) -> list[Event]:
        return [e for e in self.events if e.kind == "dialogue"]


class Beat(BaseModel):
    beat_id: str          # "b03" (episode-local, what the LLM sees)
    global_id: str        # "e012-b03"
    episode: int
    idx: int
    kind: BeatKind
    t0: float
    t1: float
    speaker: Optional[str] = None
    emotions: list[str] = Field(default_factory=list)
    text: str
    shot: str = ""        # enclosing camera description (context only)
    scene_idx: int = 0
    cut_start: float = 0.0  # start of the camera block containing t0 (or t0)
    cut_end: float = 0.0    # end of the camera block containing t1 (or t1)


class BeatScore(BaseModel):
    beat_id: str
    hook: float = Field(ge=0, le=3)    # 0-3; a mean of several readings with --votes
    cliff: float = Field(ge=0, le=3)
    beat_type: BeatType = "other"
    unresolved_refs: list[str] = Field(default_factory=list)
    role: BeatRole = "none"
    intensity: float = Field(default=0, ge=0, le=3)
    source: Literal["llm", "heuristic"] = "heuristic"


class Segment(BaseModel):
    episode: int
    start_s: float
    end_s: float

    @property
    def start(self) -> str:
        return fmt_ts(self.start_s)

    @property
    def end(self) -> str:
        return fmt_ts(self.end_s)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Window(BaseModel):
    cand_id: str
    segments: list[Segment]
    hook_beat: str       # global beat id
    cliff_beat: str
    beat_ids: list[str]  # global ids of beats inside the window, in order
    duration_s: float
    kind: Literal["single", "boundary", "two_segment"] = "single"
    subscores: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0
    flags: dict[str, bool | int | float | str] = Field(default_factory=dict)
    rubric: dict[str, bool] = Field(default_factory=dict)  # named editor checks, True = pass


class ColdViewer(BaseModel):
    comprehensibility: int = Field(ge=1, le=5)
    what_happened: str
    want_next: str
    unresolved_refs: list[str] = Field(default_factory=list)
    why_it_works: str
    would_tap: int = Field(ge=1, le=5)
    # editor's checklist (v2): answered from the clip alone
    stakes: str = ""             # "X wants Y, but Z ..." in one line; empty if the clip does not make it clear
    root_for: str = ""           # who the viewer sides with; empty if nobody
    resolved_inside: bool = False  # the clip answers its own question before it ends (the payoff is spent)
    opening_conflict: bool = True  # conflict is visible in the first lines


class PromoSegment(BaseModel):
    episode: int
    start: str
    end: str
    start_s: float
    end_s: float


class PromoLine(BaseModel):
    beat_id: str
    at_s: float
    speaker: Optional[str] = None
    text: str
    score: int


class Promo(BaseModel):
    promo_id: str
    rank: int
    score: float
    segments: list[PromoSegment]
    duration_s: float
    hook: PromoLine
    cliffhanger: PromoLine
    writer_cue: bool = False
    subscores: dict[str, float]
    rubric: dict[str, bool] = Field(default_factory=dict)
    lines: list[dict] = Field(default_factory=list)  # [{t, speaker, text, kind}] for the preview
    cold_viewer: Optional[ColdViewer] = None
    why_it_works: str = ""
    post_first_reason: str = ""
    promise: dict = Field(default_factory=dict)  # promise & payoff ledger (ledger.py)
    flags: dict[str, bool | int | float | str] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
