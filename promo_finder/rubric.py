"""The editor's rubric: named, explainable checks on a whole clip. Each check comes from a published standard or
from how micro-drama ad editors work (docs/research.md has the sources). A failed check multiplies the clip's score by
a penalty from config.yaml (rubric.penalties); nothing is a silent hard filter, and every verdict is written out.

Code checks (computed for every candidate from beat scores, no extra model calls):
  opens_on_conflict  the opening beat is pressure or a turn, not setup        (golden 3 seconds; ABCD "start big")
  context_free_open  the opening beat names nobody/nothing a stranger lacks   (Loewenstein; Rowney: context in 10 s)
  small_cast         at most 3 speaking characters                            (Netflix: >3 faces underperform; Cowan 3-5 chunks)
  ends_open          no payoff after the last escalation/turn; cliff is not a payoff  (compress-and-release; "pose the question")
  builds             the last third is at least as intense as the middle third (rising action; MrBeast "stakes get higher")
  no_dead_air        no gap longer than rubric.max_gap_s between beats         (MrBeast "no dull moments")
  features_lead      a lead speaks or acts in the clip (leads from the show-arc pass) (trailers sell the leads; Netflix: the
                     right faces; GraphTrailer: intensity without plot relevance picks side scenes)
Judge checks (finalists only, from the cold-viewer pass):
  stakes_clear       a stranger can say "X wants Y, but Z" from the clip alone (Zillmann; ABCD "connection")
  rootable           a stranger knows whom to side with                       (affective disposition theory)
  ends_open          AND the stranger says the clip did not answer its own question
  opens_on_conflict  replaced by the stranger's own read of the first lines
"""
from __future__ import annotations

from .models import Beat, BeatScore, ColdViewer, Window

CONFLICT_TYPES = {"slap", "threat", "reveal", "humiliation"}
CODE_CHECKS = ("opens_on_conflict", "context_free_open", "small_cast", "ends_open", "builds", "no_dead_air", "features_lead")
JUDGE_CHECKS = ("stakes_clear", "rootable")

LABELS = {
    "opens_on_conflict": "opens on conflict",
    "context_free_open": "opening needs no backstory",
    "small_cast": "≤3 speakers",
    "ends_open": "ends before the payoff",
    "builds": "builds to the end",
    "no_dead_air": "no dead air",
    "features_lead": "features a lead",
    "stakes_clear": "stakes clear to a stranger",
    "rootable": "someone to root for",
}


def _thirds(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    if n < 3:
        m = sum(vals) / max(n, 1)
        return m, m, m
    a, b = n // 3, 2 * n // 3
    avg = lambda xs: sum(xs) / max(len(xs), 1)  # noqa: E731
    return avg(vals[:a]), avg(vals[a:b]), avg(vals[b:])


def _features_lead(beats: list[Beat], leads: list[str]) -> bool:
    if not leads:
        return True
    names = {l.upper() for l in leads}
    words = {w.capitalize() for l in leads for w in l.split() if len(w) > 2 and w not in {"LADY", "LORD", "THE", "SIR", "DUKE"}}
    for b in beats:
        if b.kind == "dialogue" and b.speaker and b.speaker.upper() in names:
            return True
        if b.kind == "action" and any(w in b.text for w in words):
            return True
    return False


def code_rubric(w: Window, beats: list[Beat], scores: list[BeatScore], hook: BeatScore, cliff: BeatScore, cfg: dict,
                leads: list[str] | None = None) -> dict[str, bool]:
    r = cfg["rubric"]
    speakers = {b.speaker for b in beats if b.kind == "dialogue" and b.speaker}
    roles = [s.role for s in scores]
    last_push = max((i for i, ro in enumerate(roles) if ro in ("escalation", "turn")), default=-1)
    payoff_after = any(ro == "payoff" for ro in roles[last_push + 1:]) if last_push >= 0 else "payoff" in roles
    _, mid, last = _thirds([float(s.intensity) for s in scores])
    gaps = [b2.t0 - b1.t1 for b1, b2 in zip(beats, beats[1:]) if b1.episode == b2.episode]
    return {
        "opens_on_conflict": hook.role in ("escalation", "turn") or hook.beat_type in CONFLICT_TYPES,
        "context_free_open": len(hook.unresolved_refs) == 0,
        "small_cast": len(speakers) <= r["max_speakers"],
        "ends_open": cliff.role != "payoff" and not payoff_after,
        "builds": last + 1e-9 >= mid,
        "no_dead_air": max(gaps, default=0.0) <= r["max_gap_s"],
        "features_lead": _features_lead(beats, leads or []),
    }


def judge_rubric(rub: dict[str, bool], cv: ColdViewer) -> dict[str, bool]:
    out = dict(rub)
    out["opens_on_conflict"] = bool(cv.opening_conflict)
    out["ends_open"] = bool(rub.get("ends_open", True)) and not cv.resolved_inside
    out["stakes_clear"] = bool(cv.stakes.strip())
    out["rootable"] = bool(cv.root_for.strip())
    return out


def multiplier(rub: dict[str, bool], cfg: dict) -> float:
    pen = cfg["rubric"]["penalties"]
    m = 1.0
    for k, ok in rub.items():
        if not ok:
            m *= float(pen.get(k, 1.0))
    return m


def summary(rub: dict[str, bool]) -> str:
    fails = [LABELS.get(k, k) for k, ok in rub.items() if not ok]
    return "passes every check" if not fails else "fails: " + ", ".join(fails)
