"""Heuristic twin: lexicon-based hook/cliff scoring. Always runs. It is the zero-key fallback and the
ablation baseline the LLM must beat. Everything here is explainable in one sentence."""
from __future__ import annotations

import re

from .models import Beat, BeatScore, BeatType, Episode

HOOK_STRONG = r"\b(slap|slaps|slapped|strike[sd]?|struck|punch|hits? (him|her)|shove[sd]?|arrest|arrested|hang|hanged|gallows|execut|dead|die[sd]?|kill|murder|poison|pregnant|married|marrying|marry|affair|betray|liar|lie[sd]?|you knew|how dare|explosion|explode|blood|burn|fire|whip|chains?|dungeon|witch|ultimatum|divorce|mistress|bastard|worthless|nobody)\b"
HOOK_MED = r"\b(no!|why\?|stop|enough|never|secret|truth|proof|ring|watch|husband|wife|sister|father|mother|duke|duchess|crawl|beg|kneel|shame|disgrace)\b"
CLIFF_THREAT = r"\b(or (else|I|we|you)|if you|you will|you'll|tell me|choose|decide|swear|promise|unless|before (dawn|morning|sunrise)|by (dawn|morning|tonight)|one (chance|condition)|last (chance|warning))\b"
CLIFF_REVEAL = r"\b(recogni[sz]e|realis|realiz|sees?|notices?|discover|finds?|pulls? out|reveals?|unmask|identif|the (ring|watch|letter|seal|crest))\b"
PHYSICAL = r"\b(slap|strike|struck|punch|shove|grab|throws?|hurl|kiss|explosion|blast|crash|shatter|stab|blade|sword|dagger|gun|shot|whip|crack|collapse|falls?|faint)\b"
WRITER_CUE = r"(?i)cliff-?hanger|cut to black|freeze[- ]frame|freezes? on|sting\b|fade to black|smash cut|hard cut|hold on (her|his|the)"
INTENSE = {
    "furious": 1.0, "gloating": 1.0, "shocked": 1.0, "horrified": 1.0, "mocking": 0.8, "cruel": 1.0,
    "menacing": 1.0, "screaming": 1.0, "gasping": 0.8, "devastated": 0.8, "venomous": 1.0, "vicious": 1.0,
    "cold": 0.6, "commanding": 0.5, "threatening": 1.0, "triumphant": 0.8, "smug": 0.6, "sadistic": 1.0,
    "desperate": 0.6, "betrayed": 0.8, "outraged": 0.9, "defiant": 0.7, "fierce": 0.7, "icy": 0.6,
    "stunned": 0.8, "terrified": 0.9, "hysterical": 0.9, "whispering": 0.3, "pleading": 0.5,
}
TYPE_RULES: list[tuple[BeatType, str]] = [
    ("slap", PHYSICAL),
    ("threat", CLIFF_THREAT + r"|\b(hang|gallows|kill|burn|execut|dungeon|arrest)\b"),
    ("reveal", CLIFF_REVEAL),
    ("humiliation", r"\b(worthless|nobody|disgrace|shame|beg|crawl|kneel|tolerate|plain|low value|discarded)\b"),
    ("romance", r"\b(love|kiss|embrace|hold me|my wife|my husband|heart|tender)\b"),
    ("exposition", r"\b(years ago|remember when|as you know|my late|used to)\b"),
]


def emotion_intensity(emotions: list[str]) -> float:
    best = 0.0
    for e in emotions:
        for k, v in INTENSE.items():
            if k in e.lower():
                best = max(best, v)
    return best


def guess_type(text: str) -> BeatType:
    low = text.lower()
    for t, pat in TYPE_RULES:
        if re.search(pat, low):
            return t
    return "line"


def heuristic_hook(b: Beat) -> int:
    low = b.text.lower()
    s = 0.0
    if re.search(HOOK_STRONG, low):
        s += 1.6
    if re.search(HOOK_MED, low):
        s += 0.6
    if b.kind == "action" and re.search(PHYSICAL, low):
        s += 1.4
    s += emotion_intensity(b.emotions)
    if "!" in b.text:
        s += 0.4
    if b.kind == "dialogue" and (b.t1 - b.t0) <= 3.0:
        s += 0.3
    if b.kind == "sfx":
        s = 0.5 if re.search(PHYSICAL, low) else 0.0
    return int(max(0, min(3, round(s))))


def heuristic_cliff(b: Beat, ep: Episode, cue_text_near_end: bool) -> int:
    low = b.text.lower().strip()
    s = 0.0
    if low.endswith("?"):
        s += 1.0
    if low.endswith("...") or low.endswith("…"):
        s += 0.8
    if low.endswith(("—", "-", "–")):  # an interrupted line is a cut-away by construction
        s += 1.0
    if re.search(CLIFF_THREAT, low):
        s += 1.2
    if re.search(CLIFF_REVEAL, low):
        s += 1.0
    if re.search(HOOK_STRONG, low):
        s += 0.5
    s += 0.5 * emotion_intensity(b.emotions)
    if ep.duration - b.t1 <= 5.0:
        s += 1.5
    if cue_text_near_end:
        s += 1.0
    return int(max(0, min(3, round(s))))


def writer_cue_near(ep: Episode, t: float, window: float = 2.0) -> bool:
    for e in ep.events:
        if e.kind in ("camera", "action", "sfx") and abs(e.t1 - t) <= window or (e.t0 <= t <= e.t1):
            if re.search(WRITER_CUE, e.text):
                return True
    return False


def physical_in_first_seconds(ep: Episode, start: float, secs: float = 3.0) -> bool:
    for e in ep.events:
        if e.kind in ("action", "sfx") and start - 0.01 <= e.t0 <= start + secs:
            if re.search(PHYSICAL, e.text.lower()) or re.search(r"(?i)sting|gasp|slap", e.text):
                return True
    return False


def unresolved_refs_proxy(b: Beat) -> list[str]:
    """Names mentioned in the line other than the speaker: a weak proxy for 'needs backstory'."""
    names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", b.text))
    names -= {"The", "You", "Your", "And", "But", "What", "Why", "How", "Then", "Not", "She", "He", "They", "I"}
    if b.speaker:
        names -= {w.capitalize() for w in b.speaker.split()}
    return sorted(names)[:4]


# the release half of compress-and-release: tension spent on screen
PAYOFF = r"\b(i'?m sorry|forgive (me|you)|thank you|it'?s over|you'?re safe|i love you|i accept|i agree|very well|so be it|we won|you came|you saved|saved me|at last)\b"
SETUP = r"^(good (morning|evening|day)|hello|welcome|my lord|my lady|yes[,.]|come in|sit down|please,)"


def heuristic_role(b: Beat, hook: int, cliff: int) -> str:
    low = b.text.lower().strip()
    if b.kind == "sfx":
        return "none"
    if re.search(PAYOFF, low):
        return "payoff"
    if re.search(CLIFF_REVEAL, low) or (b.kind == "action" and re.search(PHYSICAL, low)):
        return "turn"
    if hook >= 2 or cliff >= 2 or re.search(CLIFF_THREAT, low) or re.search(HOOK_STRONG, low):
        return "escalation"
    if re.search(SETUP, low):
        return "setup"
    return "none"


def heuristic_intensity(b: Beat, hook: int, cliff: int) -> int:
    s = 3.0 * emotion_intensity(b.emotions) * 0.7 + 0.3 * max(hook, cliff) + (0.5 if "!" in b.text else 0.0)
    return int(max(0, min(3, round(s))))


def heuristic_scores(ep: Episode, beats: list[Beat]) -> dict[str, BeatScore]:
    out: dict[str, BeatScore] = {}
    for b in beats:
        cue = writer_cue_near(ep, b.t1)
        h, c = heuristic_hook(b), heuristic_cliff(b, ep, cue)
        out[b.beat_id] = BeatScore(
            beat_id=b.beat_id,
            hook=h,
            cliff=c,
            role=heuristic_role(b, h, c),  # type: ignore[arg-type]
            intensity=heuristic_intensity(b, h, c),
            beat_type=guess_type(b.text) if b.kind != "sfx" else "other",
            unresolved_refs=unresolved_refs_proxy(b) if b.kind == "dialogue" else [],
            source="heuristic",
        )
    return out
