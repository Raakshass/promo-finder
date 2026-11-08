"""Per-promo explanations (spec 5.3): hook and cliffhanger labels, promise, score arithmetic, first-time viewer.
Owner: Lane P. Every number is read from the promo or the run's config; nothing is typed in.

Lines returned here are markdown: script- and model-written text inside them is escaped with fmt.md_esc.
"""
from __future__ import annotations

import re

from app.ui import fmt
from app.ui.components import AI
from app.ui.glossary import CHECKS, HOOK_TYPE_SHOWN, TP_WORDS, TYPE_WORDS, penalty

BRIEF_FAIL = ("Fails a brief requirement: the clip answers its own question before the cut. Trim it to end earlier, "
              "or skip it.")
HOOK_WINDOW_S = 3.0      # the brief's hook window; the run's own value is cfg.promo.hook_window_s
_NOTE = re.compile(r"episode\s+(\d+)\D+?(\d+)\s*\.\.\s*(\d+)")

try:  # rank.combine's factor for two moments of one episode stitched together
    from promo_finder.learn import TWO_SEGMENT_FACTOR as _TWO_SEGMENT
except Exception:  # noqa: BLE001
    _TWO_SEGMENT = 0.9


def _n(x: object) -> str:
    """A small number as written: 2.0 -> "2", 2.5 -> "2.5", 0.833 -> "0.83"."""
    try:
        v = round(float(x), 2)
    except (TypeError, ValueError):
        return ""
    return str(int(v)) if v == int(v) else f"{v:g}"


def _f2(x: float) -> str:
    """Two decimals, half-up: 0.833 -> "0.83"; 1 -> "1.00"."""
    return fmt.mult(x)[1:]


def _f3(x: float) -> str:
    """A multiplier at full stored precision, at least two decimals: 0.81 -> "0.81"; 0.765 -> "0.765"."""
    s = fmt.score(x)
    return s[:-1] if s.endswith("0") else s


def _num(x: object) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def hook_label(p: dict) -> str:
    """"Hook · first 3 s · starts 0.1 s in · AI 3/3 · a reveal"."""
    hook = p.get("hook") or {}
    flags = p.get("flags") or {}
    parts = ["Hook", "first 3 s"]
    at = _num(hook.get("at_s"))
    if at is not None:
        parts.append(f"starts {fmt.secs(at)} in")
    if _num(hook.get("score")) is not None:
        parts.append(f"AI {_n(hook.get('score'))}/3")
    if flags.get("hook_type") in HOOK_TYPE_SHOWN:
        parts.append(TYPE_WORDS[flags["hook_type"]])
    if flags.get("physical_hook"):
        parts.append("physical action in the first 3 s")
    return " · ".join(parts)


def cliff_label(p: dict) -> tuple[str, str | None]:
    """(label, warning or None). A clip that answers its own question gets "Closing line (resolves on screen)" and
    the brief-requirement warning."""
    cl = p.get("cliffhanger") or {}
    resolves = (p.get("rubric") or {}).get("ends_open") is False
    parts = ["Closing line (resolves on screen)"] if resolves else ["Cliffhanger", "at the cut"]
    at = _num(cl.get("at_s"))
    if at is not None:
        parts.append(f"line ends {fmt.secs(at)} in")
    if _num(cl.get("score")) is not None:
        parts.append(f"AI {_n(cl.get('score'))}/3")
    if p.get("writer_cue") or (p.get("flags") or {}).get("writer_cue"):
        parts.append("the writers' own episode-end cue")
    return " · ".join(parts), (BRIEF_FAIL if resolves else None)


def promise_phrase(p: dict) -> str:
    """The expander phrase for "Promise and payoff · {phrase}"."""
    pr = p.get("promise") or {}
    status = pr.get("status")
    n = pr.get("answered_in_episode")
    return {"same episode": "answered later in the same episode",
            "answered": f"answered in episode {n}" if n is not None else "answered",
            "behind the paywall": "answered behind the paywall",
            "rejected": "not verified",
            "unchecked": "not checked",
            "never answered": "possible fake hook"}.get(status, "not checked")


def rejected_sentence(p: dict) -> str:
    """"Not verified: the AI named episode 3, which is before this clip, so code rejected the answer. …"."""
    pr = p.get("promise") or {}
    m = _NOTE.search(str(pr.get("note") or ""))
    if m:
        k, lo, hi = (int(g) for g in m.groups())
        where = "before this clip" if k < lo else "after the last episode" if k > hi else "outside the episodes it could be"
        head = f"Not verified: the AI named episode {k}, which is {where}, so code rejected the answer."
    else:
        head = "Not verified: the AI named an episode the answer can't be in, so code rejected the answer."
    return head + " Check by hand where the show answers this."


def promise_sentences(p: dict, run_log: dict) -> list[str]:
    """The promise expander's lines: the question, when a new viewer reaches it, and where the show answers it."""
    pr = p.get("promise") or {}
    status = pr.get("status")
    out: list[str] = []
    if pr.get("question"):
        out.append(f"Leaves the viewer asking: {fmt.md_esc(pr['question'])}")
    reached = _num(pr.get("reached_at_min"))
    if reached is not None:
        out.append(f"A new viewer reaches this moment {fmt.minutes(reached * 60)} into the show.")
    conf = pr.get("confidence")
    conf_txt = f" ({conf} confidence)" if conf else ""
    n = pr.get("answered_in_episode")
    if status == "same episode":
        s = f"Answered later in the same episode{conf_txt}."
    elif status == "answered":
        wait = _num(pr.get("minutes_after_clip"))
        s = (f"Answered in episode {n}, which starts {fmt.minutes(wait * 60)} of watching after this clip ends{conf_txt}."
             if wait is not None else f"Answered in episode {n}{conf_txt}.")
    elif status == "behind the paywall":
        pay = ((run_log or {}).get("args") or {}).get("paywall")
        s = (f"Answered in episode {n}, behind the paywall (paid from episode {pay}): a conversion promo." if pay
             else f"Answered in episode {n}, behind the paywall: a conversion promo.")
    elif status == "rejected":
        s = rejected_sentence(p)
    elif status == "unchecked":
        s = "Not checked in this run."
    elif status == "never answered":
        s = "No episode summary answers it: possible fake hook."
    else:
        s = "Not checked: this run used no AI model."
    if conf == "low" and status in ("same episode", "answered", "behind the paywall"):
        s += " The AI marked this answer as a guess."
    out.append(s)
    return out


def _join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def score_lines(p: dict, cfg: dict, arc: dict) -> list[str] | None:
    """The score as plain arithmetic: line 1 the product, line 2 the weighted base, then one bullet per part.
    None when the stored parts don't multiply to the score within 0.001 (runs made by an earlier version)."""
    sub = p.get("subscores") or {}
    sc = (cfg or {}).get("scoring") or {}
    score = _num(p.get("score"))
    base = _num(sub.get("base"))
    weights = [_num(sc.get(k)) for k in ("w_hook", "w_cliff", "w_standalone", "w_density")]
    terms = [_num(sub.get(k)) for k in ("hook", "cliff", "standalone", "density")]
    if score is None or base is None or None in weights or None in terms:
        return None
    rm = _num(sub.get("rubric_mult"))
    am = _num(sub.get("arc_mult"))
    rm = 1.0 if rm is None else rm
    am = 1.0 if am is None else am
    flags = p.get("flags") or {}
    two = flags.get("kind") == "two_segment"
    if abs(base * rm * am * (_TWO_SEGMENT if two else 1.0) - score) > 0.001:
        return None

    head = f"Score {fmt.score(score)} = base {fmt.score(base)}"
    if round(rm, 3) != 1.0:
        head += f" × checks {_f3(rm)}"
    if round(am, 3) != 1.0:
        head += f" × turning point {_f3(am)}"
    if two:
        head += f" × {_f2(_TWO_SEGMENT)} two moments of one episode stitched"
    names = ("hook", "cliffhanger", "makes sense alone", "pace")
    base_line = f"Base {fmt.score(base)} = " + " + ".join(f"{_f2(w)} × {nm} {_f2(t)}" for w, nm, t in zip(weights, names, terms))
    lines = [head, base_line]

    hook = p.get("hook") or {}
    win = _n(((cfg or {}).get("promo") or {}).get("hook_window_s") or HOOK_WINDOW_S)
    h = f"- Hook {_f2(terms[0])}: AI line score {_n(hook.get('score'))}/3"
    if flags.get("physical_hook"):
        bonus = _num(sc.get("physical_hook_bonus")) or 0.0
        h += f", + {_n(bonus)} for a physical action in the first {win} s"
        if (_num(hook.get("score")) or 0.0) + bonus > 3:
            h += ", capped at 3"
    lines.append(h + ", ÷ 3.")

    cl = p.get("cliffhanger") or {}
    c = f"- Cliffhanger {_f2(terms[1])}: AI line score {_n(cl.get('score'))}/3"
    if p.get("writer_cue") or flags.get("writer_cue"):
        c += f", + {_n(_num(sc.get('writer_cue_bonus')) or 0.0)} next to the writers' cue"
    if flags.get("romance_cap"):
        c += ", capped at 2 for a love line"
    lines.append(c + ", ÷ 3.")

    cv = p.get("cold_viewer") or {}
    proxy = _num(sub.get("standalone_proxy"))
    if proxy is not None and cv.get("comprehensibility") is not None:
        lines.append(f"- Makes sense alone {_f2(terms[2])}: the first-time viewer (AI) follows it "
                     f"{_n(cv['comprehensibility'])}/5 (text-only estimate before that test: {_f2(proxy)}).")
    else:
        lines.append(f"- Makes sense alone {_f2(terms[2])}: a text-only estimate, the share of spoken lines that name "
                     "no one a stranger can't place (no first-time-viewer test for this clip).")

    share = _num(sub.get("dialogue_share"))
    prior = _num(sub.get("duration_prior"))
    pc = (cfg or {}).get("promo") or {}
    pace = f"- Pace {_f2(terms[3])}:"
    if share is not None:
        pace += f" speech fills {fmt.pct(share)} of the clip (full marks from 50%"
        floor = _num(sc.get("density_floor"))
        if floor is not None and share < floor:
            pace += f", halved below {fmt.pct(floor)}"
        pace += ")"
    if prior is not None:
        best = (f" (best at {_n(pc['preferred_min_s'])}–{_n(pc['preferred_max_s'])} s)"
                if pc.get("preferred_min_s") is not None and pc.get("preferred_max_s") is not None else "")
        pace += f"{',' if share is not None else ''} × length factor {_f2(prior)}{best}"
    lines.append(pace + ".")

    if round(rm, 3) != 1.0:
        fails = [k for k, ok in (p.get("rubric") or {}).items() if ok is False and k in CHECKS]
        named = [f"“{CHECKS[k].fail_label}” ({fmt.mult(penalty(cfg, k))})" for k in fails]
        lines.append(f"- Checks ×{_f3(rm)}: fails {_join_and(named)}." if named else f"- Checks ×{_f3(rm)}.")
    if round(am, 3) != 1.0:
        ep = flags.get("episode")
        tp = flags.get("turning_point")
        event = next((t.get("event") for t in (arc or {}).get("turning_points") or []
                      if isinstance(t, dict) and t.get("episode") == ep), None)
        what = TP_WORDS.get(tp, "")
        detail = f"{what}: {fmt.md_esc(event)}" if what and event else (what or fmt.md_esc(event or ""))
        lines.append(f"- Turning point ×{_f3(am)}: episode {ep} is a story turning point"
                     + (f" ({detail}). {AI} The AI's reading of the writers' notes." if event else
                        f" ({detail})." if detail else "."))
    if two:
        lines.append(f"- Two moments ×{_f2(_TWO_SEGMENT)}: two moments of one episode stitched together, a join not "
                     "yet proven to play.")
    return lines


def score_factors(p: dict) -> list[float]:
    """The factors the first score line prints, as rounded there (for tests and the arithmetic check)."""
    sub = p.get("subscores") or {}
    out = [float(fmt.score(sub["base"]))]
    for k in ("rubric_mult", "arc_mult"):
        v = _num(sub.get(k))
        if v is not None and round(v, 3) != 1.0:
            out.append(float(_f3(v)))
    if (p.get("flags") or {}).get("kind") == "two_segment":
        out.append(float(_f2(_TWO_SEGMENT)))
    return out


def viewer_rows(p: dict) -> list[tuple[str, str]]:
    """[(label, text)], empty when the clip had no first-time-viewer test. Text is the AI's, unescaped."""
    cv = p.get("cold_viewer") or {}
    if not cv or cv.get("comprehensibility") is None:
        return []
    rows: list[tuple[str, str]] = []
    if cv.get("what_happened"):
        rows.append(("What it saw", str(cv["what_happened"])))
    if cv.get("want_next"):
        rows.append(("Left asking", str(cv["want_next"])))
    refs = [str(r) for r in cv.get("unresolved_refs") or [] if r]
    if refs:
        rows.append(("Couldn't place", ", ".join(refs)))
    if cv.get("stakes"):
        rows.append(("Stakes as it read them", str(cv["stakes"])))
    if cv.get("root_for"):
        rows.append(("Sides with", str(cv["root_for"])))
    return rows
