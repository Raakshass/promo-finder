"""Quality checks used across views (spec 5.3 and 5.5). Owner: Lane R.

before_posting, weak_spots, stability_badge and fail_count are used by the Promos view and the drawer. problems,
measured_checks and flat_episodes feed the Quality view and the show map. Every number is read from the run's files;
nothing here is typed in except the 12-drama benchmark, the real reruns and the weight checks, which live only in
docs/research.md and the scripts named as their source (no run folder holds them).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.ui import fmt
from app.ui.glossary import CHECKS


@dataclass(frozen=True)
class Flag:
    short: str
    text: str


@dataclass(frozen=True)
class Problem:
    title: str
    body: str
    source: str


# ---------------------------------------------------------------- small private helpers

def _dec(x: float, places: int = 1) -> str:
    """Half-up decimal: _dec(3.8) -> \"3.8\", _dec(3.65, 2) -> \"3.65\", _dec(22.3, 0) -> \"22\"."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{q}"


def _window(data: dict) -> str:
    w = _d(_d(data.get("cfg")).get("eval")).get("cliff_window_s")
    return f"{_dec(w, 0)} s" if isinstance(w, (int, float)) else "closing seconds"


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _rubric(p: dict) -> dict:
    return {k: v for k, v in _d(p.get("rubric")).items() if isinstance(v, bool)}


def _fails(p: dict) -> list[str]:
    """Failed check keys, in CHECKS order (unknown keys last)."""
    rub = _rubric(p)
    known = [k for k in CHECKS if rub.get(k) is False]
    return known + [k for k, ok in rub.items() if ok is False and k not in CHECKS]


def _fail_label(key: str) -> str:
    c = CHECKS.get(key)
    return c.fail_label if c else key.replace("_", " ")


def _episodes(data: dict) -> list[int]:
    eps = [int(e) for e in _l(data.get("episodes_list")) if isinstance(e, (int, str)) and str(e).isdigit()]
    if eps:
        return sorted(eps)
    return sorted({int(s.get("episode")) for p in _l(data.get("promos")) for s in _l(p.get("segments"))
                   if isinstance(s.get("episode"), int)})


def _n_episodes(data: dict) -> int:
    m = _d(data.get("metrics"))
    if isinstance(m.get("episodes"), int) and m["episodes"] > 0:
        return m["episodes"]
    return len(_episodes(data))


def _shortlist(data: dict) -> int | None:
    m = _d(data.get("metrics"))
    if isinstance(m.get("finalists"), int):
        return m["finalists"]
    top = _d(_d(data.get("cfg")).get("judge")).get("top_n")
    return top if isinstance(top, int) else None


def _reps(data: dict) -> int | None:
    r = _d(data.get("stability")).get("reps")
    return r if isinstance(r, int) else None


def _rank_of(p: dict) -> str:
    r = p.get("rank")
    return f"#{r}" if isinstance(r, int) else ""


_NOTE_RE = re.compile(r"episode\s+(\d+)\D+?(\d+)\s*\.\.\s*(\d+)")


def rejected_sentence(p: dict) -> str:
    """Spec 5.3 'rejected' promise sentence, parsed from promise.note ("model named episode 3, outside 5..50")."""
    note = str(_d(p.get("promise")).get("note") or "")
    m = _NOTE_RE.search(note)
    if not m:
        return ("Not verified: the AI named an episode that can't answer it, so code rejected the answer. "
                "Check by hand where the show answers this.")
    k, lo, hi = (int(g) for g in m.groups())
    where = "before this clip" if k < lo else ("after the last episode" if k > hi else "outside the allowed episodes")
    return (f"Not verified: the AI named episode {k}, which is {where}, so code rejected the answer. "
            "Check by hand where the show answers this.")


def _reveal_episode(p: dict, data: dict) -> int | None:
    text = _d(p.get("flags")).get("major_reveal")
    seg_eps = [s.get("episode") for s in _l(p.get("segments")) if isinstance(s.get("episode"), int)]
    for r in _l(_d(data.get("arc")).get("major_reveals")):
        if isinstance(r, dict) and r.get("reveal") == text and isinstance(r.get("episode"), int):
            return r["episode"]
    for r in _l(_d(data.get("arc")).get("major_reveals")):
        if isinstance(r, dict) and r.get("episode") in seg_eps:
            return r["episode"]
    ep = _d(p.get("flags")).get("episode")
    return ep if isinstance(ep, int) else (seg_eps[0] if seg_eps else None)


def _spoiler_range(data: dict) -> str:
    eps = _episodes(data)
    if not eps:
        return "the last fifth"
    lo, hi = eps[0], eps[-1]
    frac = _d(_d(data.get("cfg")).get("scoring")).get("spoiler_last_fraction")
    try:
        cutoff = hi - float(frac) * (hi - lo)
    except (TypeError, ValueError):
        return "the last fifth"
    late = [e for e in eps if e > cutoff]
    return fmt.ep_range(late[0], late[-1]) if late else fmt.ep_range(hi, hi)


# ---------------------------------------------------------------- used by Promos and the drawer

def before_posting(p: dict, data: dict) -> list[Flag]:
    """Things to check on the video or in the story (spec 5.3), in the spec's order."""
    out: list[Flag] = []
    flags = _d(p.get("flags"))
    segs = [s for s in _l(p.get("segments")) if isinstance(s, dict)]
    if len(segs) > 1:
        a, b = segs[0], segs[1]
        out.append(Flag("Join", f"Joins the end of episode {a.get('episode')} ({a.get('start')}–{a.get('end')}) to the "
                                f"start of episode {b.get('episode')} ({b.get('start')}–{b.get('end')}): check the join "
                                "plays as one moment."))
    if flags.get("scene_change"):
        out.append(Flag("Scene change", "Crosses a scene change: check the cut reads as one moment."))
    if flags.get("major_reveal"):
        n = _reveal_episode(p, data)
        ep = f"Episode {n}" if n is not None else "This episode"
        out.append(Flag(f"Twist ep {n}" if n is not None else "Twist",
                        f"{ep} holds a major reveal (AI reading of the writers' notes: “{flags['major_reveal']}”): "
                        "check the clip doesn't give it away."))
    if flags.get("spoiler_risk"):
        out.append(Flag("Spoiler risk", f"From the last fifth of the show ({_spoiler_range(data)}): it may give away "
                                        "the ending."))
    promise = _d(p.get("promise"))
    status = promise.get("status")
    if status == "rejected":
        out.append(Flag("Answer not verified", rejected_sentence(p)))
    elif status == "never answered":
        out.append(Flag("Possible fake hook", "The show never answers the question this promo leaves: fake-hook risk."))
    elif status == "unchecked":
        out.append(Flag("Answer not checked", "Where the show answers this was not checked in this run."))
    if promise.get("confidence") == "low":
        n = promise.get("answered_in_episode")
        where = f" (episode {n})" if isinstance(n, int) else ""
        out.append(Flag("Answer is a guess", f"The AI marked its answer{where} as a low-confidence guess."))
    if _d(p.get("rubric")).get("ends_open") is False:
        out.append(Flag("Not a cliffhanger", "Ends after the payoff: the clip answers its own question before the cut. "
                                             "Trim the end or skip it."))
    return out


def weak_spots(p: dict, data: dict) -> list[str]:
    """Known weaknesses already priced into the score (spec 5.3), in the spec's order."""
    out: list[str] = []
    fails = _fails(p)
    if fails:
        total = len(_rubric(p))
        mult = _d(p.get("subscores")).get("rubric_mult")
        tail = f" (score {fmt.mult(mult)})" if isinstance(mult, (int, float)) else ""
        out.append(f"Fails {fmt.k_of_n(len(fails), total)} editor's checks: "
                   f"{', '.join(_lower1(_fail_label(k)) for k in fails)}{tail}")
    cv = _d(p.get("cold_viewer"))
    c = cv.get("comprehensibility")
    if isinstance(c, (int, float)) and c <= 3:
        names = [str(x) for x in _l(cv.get("unresolved_refs")) if x]
        extra = f" and couldn't place {', '.join(names)}" if names else ""
        out.append(f"The first-time viewer (AI) followed it only {fmt.num(c)}/5{extra}")
    keep = _d(_d(data.get("stability")).get("keep_rate_by_promo")).get(p.get("promo_id"))
    if isinstance(keep, (int, float)) and keep < 0.5:
        reps, short = _reps(data), _shortlist(data)
        on = f"on the {short}-clip shortlist" if short else "on the shortlist"
        of = f" of {reps} simulated re-runs" if reps else " of the simulated re-runs"
        out.append(f"Fragile pick: {on} in only {fmt.pct(keep)}{of}")
    return out


def stability_badge(p: dict, data: dict) -> tuple[str, str, str] | None:
    """(label, colour, help), or None when stability.json is missing or has no value for this promo."""
    keep = _d(_d(data.get("stability")).get("keep_rate_by_promo")).get(p.get("promo_id"))
    if not isinstance(keep, (int, float)):
        return None
    reps, short = _reps(data), _shortlist(data)
    help_ = (f"Share of {reps or 'the'} simulated re-runs (each AI line score redrawn from the model's measured "
             f"run-to-run noise) in which this clip stays on the {f'{short}-clip ' if short else ''}shortlist the "
             "first-time viewer sees. The 80% and 50% cut-offs are this app's display choice.")
    p_ = fmt.pct(keep)
    if keep >= 0.8:
        return f"Steady pick · {p_}", "gray", help_
    if keep >= 0.5:
        return f"Usually picked · {p_}", "gray", help_
    return f"Fragile pick · {p_}", "orange", help_


def fail_count(data: dict) -> int:
    """Promos with any false rubric check."""
    return sum(1 for p in _l(data.get("promos")) if _fails(p))


# ---------------------------------------------------------------- flat answers (map notes, Quality)

_STILL = re.compile(r"still flat answer \(max hook ([\d.]+), max cliff ([\d.]+)\)")


def flat_episodes(data: dict) -> list[dict]:
    """One row per episode the AI first answered flat: episode, still_flat, all_zero (every beat scored 0)."""
    issues = _d(_d(data.get("run_log")).get("id_issues"))
    beats = _d(data.get("beats"))
    rows = []
    for k, v in issues.items():
        note = str(_d(v).get("note") or "")
        if "flat answer" not in note:
            continue
        try:
            ep = int(k)
        except (TypeError, ValueError):
            continue
        still = "still flat" in note
        all_zero = False
        if still:
            bl = _l(_d(beats.get(ep) or beats.get(str(ep))).get("beats"))
            if bl:
                all_zero = all(float(b.get("hook") or 0) == 0 and float(b.get("cliff") or 0) == 0 for b in bl)
            else:
                m = _STILL.search(note)
                all_zero = bool(m) and float(m.group(1)) == 0 and float(m.group(2)) == 0
        rows.append({"episode": ep, "still_flat": still, "all_zero": all_zero})
    return sorted(rows, key=lambda r: r["episode"])


# ---------------------------------------------------------------- Quality: problems and measured checks

def _is_keyword_run(data: dict) -> bool:
    m = _d(_d(data.get("metrics")).get("model"))
    model = str(m.get("model") or _d(_d(data.get("run_log")).get("args")).get("model") or "").lower()
    return model in ("heuristic", "none") or str(m.get("provider") or "").lower() == "none"


def _is_upload(data: dict) -> bool:
    rid = str(_d(data.get("metrics")).get("run_id") or "")
    return "-upload" in rid


def _words_and(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def judge_name(model_id) -> str:
    """'Llama 3.1 8B run locally' for a local model; the short name otherwise."""
    name = fmt.model_short(str(model_id or ""))
    return name[:-len(" (local)")] + " run locally" if name.endswith(" (local)") else name


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:]


def _top10(promos: list[dict]) -> list[dict]:
    return [p for p in promos if isinstance(p.get("rank"), int) and p["rank"] <= 10]


# Fresh evidence that no run folder holds (verified 8 Nov). Source: docs/research.md, "The same benchmark on 12 dramas";
# scripts/weight_sensitivity.py; scripts/weights_vs_editors.py.
_BENCH12_SOURCE = 'docs/research.md, section "The same benchmark on 12 dramas"'
_BENCH12_BODY = ("On 12 dramas (11 have ads in the episodes tested), the raw best clips inside an episode appeared in "
                 "professional editors' ads 1.07× as often as an average second of that episode, against 1.03× for "
                 "random spots (+0.04, 95% CI 0.00 to 0.09; 8 of 11 dramas up). Across the show the top 20 score "
                 "1.39× against 1.08× for random clips (+0.31, CI +0.01 to +0.63), but in the top 10 simply taking "
                 "every episode's opening scores 1.74× against the slate's 1.44×.")
_RERUNS_TEXT = ("Two real uncached runs on 4 public dramas shared 68% of their top-10 candidates and gave 88% of lines "
                "the same hook score.")
_WEIGHTS_BODY = ("A 0.05 change to any one weight keeps 8.75 of the top 10 on average, and fitting to the professional "
                 "editors' picks cannot reject the weights (p = 0.12), so they hold up but are not proven best. "
                 "Your review decisions are meant to refit them.")
_WEIGHTS_SOURCE = "scripts/weight_sensitivity.py; scripts/weights_vs_editors.py (results in docs/research.md)"


def _problem_benchmark(data: dict) -> Problem | None:
    b = _d(data.get("benchmark"))
    if not _d(b.get("summary")) or _is_keyword_run(data) or _is_upload(data):
        return None
    if isinstance(b.get("dramas"), int) and b["dramas"] >= 12:
        s = _d(b["summary"])
        body = (f"On {b['dramas']} dramas, its picks inside an episode appeared in professional editors' ads "
                f"{fmt.lift(s.get('ours_within_lift', 0))} as often as an average second of that episode; random "
                f"clips scored {fmt.lift(s.get('random_within_lift', 0))}.")
        return Problem("Inside an episode, its picks beat chance only slightly.", body,
                       "eval/external_benchmark_slate.json → summary")
    return Problem("Inside an episode, its picks beat chance only slightly; episode openings still win the top 10.",
                   _BENCH12_BODY, _BENCH12_SOURCE)


def _problem_stability(data: dict) -> Problem | None:
    st_ = _d(data.get("stability"))
    if not isinstance(st_.get("expected_overlap_top10"), (int, float)) or _is_keyword_run(data) or _is_upload(data):
        return None
    t10, tn = st_["expected_overlap_top10"], st_.get("expected_overlap_top_n")
    reps = st_.get("reps")
    k = int(round(t10 * 10))
    body = (f"Re-ranking {reps} times with the model's measured run-to-run noise, two runs would share "
            f"{fmt.pct(t10)} of the top 10" + (f" and {fmt.pct(tn)} of the top 20." if isinstance(tn, (int, float))
                                                 else "."))
    src = "eval/stability.json"
    v2 = _d(_d(data.get("votes")).get("2"))
    if isinstance(v2.get("top10"), (int, float)):
        body += (f" Reading every episode twice and averaging raises that to {fmt.pct(v2['top10'])}"
                 + (f" and {fmt.pct(v2['top20'])}." if isinstance(v2.get("top20"), (int, float)) else "."))
        src += "; eval/votes_simulation.json"
    body += " " + _RERUNS_TEXT
    src += '; docs/research.md, section "The same benchmark on 12 dramas" (real reruns)'
    return Problem(f"Two runs agree on about {k} of their top 10.", body, src)


def _problem_rubric(data: dict) -> Problem | None:
    promos = _l(data.get("promos"))
    if not promos:
        return None
    if not any(_rubric(p) for p in promos):
        return Problem("This version had no editor's checks.", "Its promos were not checked against the editor's rules.",
                       "promos.json → rubric")
    n = len(promos)
    f = fail_count(data)
    top = _top10(promos)
    f10 = sum(1 for p in top if _fails(p))
    counts: dict[str, int] = {}
    for p in promos:
        for k in _fails(p):
            counts[k] = counts.get(k, 0) + 1
    common = sorted(counts.items(), key=lambda kv: (-kv[1], list(CHECKS).index(kv[0]) if kv[0] in CHECKS else 99))[:3]
    most = ", ".join(f"{_lower1(_fail_label(k))} ({c})" for k, c in common)
    top_txt = f" ({f10} of the top {len(top)})" if top else ""
    title = f"{f} of the {n} promos fail at least one editor's check{top_txt}."
    body = "They are ranked lower, not removed; each promo names its failed check."
    if most:
        body += f" Most often failed: {most}."
    return Problem(title, body, "promos.json → rubric")


def _problem_flat(data: dict) -> Problem | None:
    rows = flat_episodes(data)
    rep = _d(_d(data.get("metrics")).get("answer_repairs"))
    if not rows and not rep:
        return None
    if not rows:
        return None
    E = _n_episodes(data)
    fixed = sum(1 for r in rows if not r["still_flat"])
    zero = [r["episode"] for r in rows if r["still_flat"] and r["all_zero"]]
    other = [r["episode"] for r in rows if r["still_flat"] and not r["all_zero"]]
    body = (f"In {fmt.k_of_n(len(rows), E)} episodes, the first answer rated no line above 1 as a hook, or none above 1 "
            f"as a cliffhanger. Asking again one episode at a time fixed {fixed}.")
    tail = []
    if zero:
        tail.append(f"{'Episode' if len(zero) == 1 else 'Episodes'} {_words_and([str(e) for e in zero])} stayed at 0 "
                    "for every line")
    if other:
        tail.append(f"{'episode' if tail else 'Episode'} {_words_and([str(e) for e in other])} had no line strong "
                    "enough to open a promo")
    if tail:
        body += " " + ", and ".join(tail) + ", so no promo comes from them."
    return Problem("The free model sometimes answers flat.", body,
                   "run_log.json → id_issues; eval/metrics.json → answer_repairs")


def _problem_entailment(data: dict) -> Problem | None:
    ent = _d(_d(data.get("metrics")).get("entailment"))
    rows = _d(ent.get("rows"))
    args = _d(_d(data.get("run_log")).get("args"))
    if not rows:
        if args.get("no_entailment"):
            return Problem("Whether its best clip holds the writers' cliffhanger was not checked.",
                           "Not checked in this run.", "run_log.json → args")
        return None
    checked = {int(k): v for k, v in rows.items() if isinstance(_d(v).get("contains"), bool)}
    unchecked = sorted(int(k) for k, v in rows.items() if _d(v).get("contains") is None)
    misses = sum(1 for v in checked.values() if v["contains"] is False)
    title = (f"Its best clip misses the writers' own cliffhanger in {misses} of the {len(checked)} episodes it could "
             "check.")
    body = "Checked against the next episode's continuity note."
    if unchecked:
        body += (f" {len(unchecked)} more checks failed when the free tier's daily limit ran out (episodes "
                 f"{fmt.ranges(unchecked)})")
        top_eps = {s.get("episode") for p in _top10(_l(data.get("promos"))) for s in _l(p.get("segments"))}
        if top_eps and all(e in unchecked for e in top_eps if isinstance(e, int)):
            body += ", so none of the episodes behind the top 10 was checked."
        else:
            body += "."
    return Problem(title, body, "eval/metrics.json → entailment")


def _problem_cliff(data: dict) -> Problem | None:
    m = _d(data.get("metrics"))
    a, h = _d(m.get("cliff_localisation")), _d(m.get("cliff_localisation_heuristic"))
    if _is_keyword_run(data) or not _l(a.get("rows")) or not _l(h.get("rows")):
        return None
    ra, rh = _l(a["rows"]), _l(h["rows"])
    ka = sum(1 for r in ra if r.get("in_last_window"))
    kh = sum(1 for r in rh if r.get("in_last_window"))
    zero = sum(1 for r in ra if r.get("in_last_window") and not r.get("top_cliff_score"))
    w = _window(data)
    flat = [r["episode"] for r in flat_episodes(data) if r["still_flat"] and r["all_zero"]]
    if kh <= ka:
        title = "The AI finds the writers' episode-end cliffhanger at least as often as keyword rules."
    else:
        title = "Keyword rules find the writers' episode-end cliffhanger more often than the AI."
    body = (f"Is the line rated the strongest cliffhanger inside the episode's last {w}? Keyword rules: "
            f"{fmt.k_of_n(kh, len(rh))} episodes (they give every line in the last 5 s a bonus). AI: "
            f"{fmt.k_of_n(ka, len(ra))}")
    if zero:
        where = f", and {zero} of those are episodes {_words_and([str(e) for e in flat])}," if flat and len(flat) == zero \
            else f", and {zero} of those are episodes"
        body += f"{where} where every line scored 0 and ties go to the last line."
    else:
        body += "."
    return Problem(title, body, "eval/metrics.json → cliff_localisation, cliff_localisation_heuristic")


def _problem_ab(data: dict) -> Problem | None:
    ab = _d(data.get("ab"))
    if not isinstance(ab.get("a_wins"), int) or _is_keyword_run(data) or _is_upload(data):
        return None
    a, b, t = ab["a_wins"], ab["b_wins"], ab.get("ties", 0)
    if a > b:
        title = f"A blind judge preferred the first version, {a} to {b} with {t} ties."
    elif b > a:
        title = f"A blind judge preferred this version, {b} to {a} with {t} ties."
    else:
        title = f"A blind judge split the two versions, {a} each with {t} ties."
    k = ab.get("k") or len(_l(ab.get("rows")))
    pc = ab.get("position_consistency")
    judge = judge_name(ab.get("judge"))
    body = (f"Slot by slot over the top {k}, each pair was shown twice with the order swapped; a side wins only if "
            "picked both times.")
    if isinstance(pc, (int, float)):
        share = "only half" if abs(pc - 0.5) < 1e-9 else fmt.pct(pc)
        body += (f" The judge, {judge}, gave the same answer in both orders for {share} of the pairs, so this is weak "
                 "evidence either way.")
    late = sum(1 for r in _l(ab.get("rows")) if _l(r.get("a_eps")) and min(r["a_eps"]) >= 15)
    if late:
        body += (f" The first version took {late} of its top {k} from episode 15 on; this version posts the opening "
                 "first, as professional editors do.")
    return Problem(title, body, "eval/ab_vs_previous.json")


def _problem_weights(data: dict) -> Problem | None:
    if _is_keyword_run(data):
        return None
    return Problem("The weights are set by hand, not learned from data.", _WEIGHTS_BODY, _WEIGHTS_SOURCE)


_PROBLEM_NAMES = [
    ("benchmark", _problem_benchmark, "the external benchmark against professional ads"),
    ("stability", _problem_stability, "how much two runs agree"),
    ("rubric", _problem_rubric, "the editor's checks"),
    ("flat", _problem_flat, "flat answers"),
    ("entailment", _problem_entailment, "whether the best clip holds the writers' cliffhanger"),
    ("cliff", _problem_cliff, "the AI against keyword rules on episode-end cliffhangers"),
    ("ab", _problem_ab, "the blind A/B against the first version"),
    ("weights", _problem_weights, "how much the weights matter"),
]


def problems(data: dict) -> tuple[list[Problem], list[str]]:
    """(shown, names of items not measured), biggest first."""
    shown: list[Problem] = []
    missing: list[str] = []
    for _key, fn, name in _PROBLEM_NAMES:
        try:
            pr = fn(data)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pr = None
        if pr is None:
            missing.append(name)
        else:
            shown.append(pr)
    return shown, missing


NO_AI = "Not checked: this run used no AI model."


def _cv_row(data: dict) -> dict | None:
    promos = _l(data.get("promos"))
    if _is_keyword_run(data):
        return {"What was checked": "A first-time viewer (AI) follows the clip (1–5)", "Result": NO_AI,
                "Compared with": "", "How to read it": "", "Source": "promos.cold_viewer"}
    vals = [p["cold_viewer"]["comprehensibility"] for p in promos
            if isinstance(_d(p.get("cold_viewer")).get("comprehensibility"), (int, float))]
    if not vals:
        return None
    hi = sum(1 for v in vals if v >= 4)
    mean = sum(vals) / len(vals)
    res = f"{_dec(mean)} for the {len(vals)} promos ({hi} rated 4 or 5)"
    mcv = _d(_d(data.get("metrics")).get("cold_viewer"))
    comp = ""
    if isinstance(mcv.get("mean_comprehensibility"), (int, float)):
        comp = f"{_dec(mcv['mean_comprehensibility'], 2)} over all {mcv.get('n')} shortlisted clips"
    return {"What was checked": "A first-time viewer (AI) follows the clip (1–5)", "Result": res,
            "Compared with": comp,
            "How to read it": "The same model that scores the lines, reading only the clip's lines.",
            "Source": "promos.cold_viewer; metrics.cold_viewer"}


def _promise_row(data: dict) -> dict | None:
    if _is_keyword_run(data):
        return {"What was checked": "The show answers the question the promo leaves", "Result": NO_AI,
                "Compared with": "", "How to read it": "", "Source": "metrics.promise_ledger"}
    led = _d(_d(data.get("metrics")).get("promise_ledger"))
    if not isinstance(led.get("promos"), int):
        return None
    res = (f"{fmt.k_of_n(led.get('answered', 0), led['promos'])} ({led.get('answered_same_episode', 0)} later in the "
           f"same episode, {led.get('answered_later_episode', 0)} in a later one)")
    comp = (f"{led.get('never_answered', 0)} never answered; {led.get('unchecked_or_rejected', 0)} not verified")
    return {"What was checked": "The show answers the question the promo leaves", "Result": res, "Compared with": comp,
            "How to read it": "Found by one AI request over the writers' episode summaries; code rejects impossible "
                              "episodes.",
            "Source": "metrics.promise_ledger"}


def measured_checks(data: dict) -> list[dict]:
    """Rows: What was checked, Result, Compared with, How to read it, Source. A row whose file is missing is omitted."""
    rows: list[dict] = []
    m = _d(data.get("metrics"))
    promos = _l(data.get("promos"))
    N = len(promos)

    def row(what, result, comp, how, src):
        rows.append({"What was checked": what, "Result": result, "Compared with": comp, "How to read it": how,
                     "Source": src})

    inv = _d(m.get("invariants"))
    if isinstance(inv.get("checked"), int) and promos:
        ok = inv["checked"] - len(_l(inv.get("violations")))
        durs = [float(p["duration_s"]) for p in promos if isinstance(p.get("duration_s"), (int, float))]
        hooks = [float(_d(p.get("hook")).get("at_s")) for p in promos
                 if isinstance(_d(p.get("hook")).get("at_s"), (int, float))]
        how = "Guaranteed by the code that builds clips; not a quality result."
        if durs:
            how += f" Lengths {_dec(min(durs), 0)}–{_dec(max(durs), 0)} s"
            how += f"; hook at a median {fmt.secs(statistics.median(hooks))}." if hooks else "."
        row("Follows the brief's rules: 20–60 s, hook within 3 s, segments in order", fmt.k_of_n(ok, inv["checked"]),
            "", how, "metrics.invariants; promos.duration_s, hook.at_s")
    if any(_rubric(p) for p in promos):
        allpass = sum(1 for p in promos if _rubric(p) and not _fails(p))
        top = _top10(promos)
        tp = sum(1 for p in top if _rubric(p) and not _fails(p))
        n_checks = max(len(_rubric(p)) for p in promos)
        row(f"Passes all {n_checks} editor's checks", f"{fmt.k_of_n(allpass, N)} ({tp} of the top {len(top)})", "",
            f"The other {N - allpass} are ranked lower, not removed.", "promos.rubric; metrics.editor_rubric")
    blocks = _d(_d(m.get("editor_rubric")).get("top10_blocks"))
    if blocks:
        row("Top 10 by part of the show",
            " · ".join(f"{int(blocks.get(b, 0))} {b}" for b in ("opening", "middle", "late")), "",
            "The opening is posted first.", "metrics.editor_rubric.top10_blocks")
    for r in (_cv_row(data), _promise_row(data)):
        if r:
            rows.append(r)
    a, h = _d(m.get("cliff_localisation")), _d(m.get("cliff_localisation_heuristic"))
    if _l(a.get("rows")):
        ka = sum(1 for r in a["rows"] if r.get("in_last_window"))
        zero = sum(1 for r in a["rows"] if r.get("in_last_window") and not r.get("top_cliff_score"))
        comp = ""
        if _l(h.get("rows")) and not _is_keyword_run(data):
            comp = f"Keyword rules: {fmt.k_of_n(sum(1 for r in h['rows'] if r.get('in_last_window')), len(h['rows']))}"
        how = "Ties go to the later line"
        how += f", so {zero} of the {ka} are episodes where every line scored 0." if zero else "."
        row(f"Strongest cliffhanger line in the episode's last {_window(data)}", f"{fmt.k_of_n(ka, len(a['rows']))} episodes", comp,
            how, "metrics.cliff_localisation, cliff_localisation_heuristic")
    ent = _d(m.get("entailment"))
    if _d(ent.get("rows")):
        rws = _d(ent["rows"]).values()
        checked = [r for r in rws if isinstance(_d(r).get("contains"), bool)]
        caught = sum(1 for r in checked if r["contains"])
        unchecked = sum(1 for r in rws if _d(r).get("contains") is None)
        row("Each episode's best clip contains the writers' cliffhanger", f"{fmt.k_of_n(caught, len(checked))} checked",
            f"{unchecked} not checked (daily limit)" if unchecked else "",
            "Checked against the next episode's continuity note.", "metrics.entailment")
    b = _d(data.get("benchmark"))
    if _d(b.get("summary")) and not _is_keyword_run(data):
        s = _d(b["summary"])
        old = ""
        if isinstance(s.get("ours_within_lift"), (int, float)) and isinstance(s.get("random_within_lift"), (int, float)):
            old = (f" This run's own check on {b.get('dramas')} dramas found {fmt.lift(s['ours_within_lift'])} against "
                   f"{fmt.lift(s['random_within_lift'])} within an episode; with 12 dramas that result does not hold up.")
        row("Matches professional ads (12 dramas, 11 with ads in the tested episodes)",
            "Top 20: 1.39× an average second; inside an episode: 1.07×",
            "Random clips 1.08× and 1.03×; every episode's opening 1.74× in the top 10 (the slate 1.44×)",
            "Inside an episode +0.04 (95% CI 0.00 to 0.09; 8 of 11 dramas up)." + old,
            f"{_BENCH12_SOURCE}; eval/external_benchmark_slate.json")
    st_ = _d(data.get("stability"))
    if isinstance(st_.get("expected_overlap_top10"), (int, float)) and not _is_keyword_run(data):
        t10 = st_["expected_overlap_top10"]
        tn = st_.get("expected_overlap_top_n")
        res = f"About {int(round(t10 * 10))} of 10 ({fmt.pct(t10)})"
        if isinstance(tn, (int, float)):
            res += f"; top 20: {fmt.pct(tn)}"
        v2 = _d(_d(data.get("votes")).get("2"))
        comp = f"{fmt.pct(v2['top10'])} when each episode is read twice" if isinstance(v2.get("top10"), (int, float)) else ""
        row("Same top 10 on a re-run", res, comp,
            f"{st_.get('reps')} re-runs simulated from the model's measured noise; no new model calls. Real reruns on 4 "
            "public dramas shared 68% of top-10 candidates.",
            "stability.json; votes_simulation.json; docs/research.md (real reruns)")
    ab = _d(data.get("ab"))
    if isinstance(ab.get("a_wins"), int) and not _is_keyword_run(data):
        pc = ab.get("position_consistency")
        k = ab.get("k") or len(_l(ab.get("rows")))
        comp = (f"The judge agreed with itself in {int(round(pc * k))} of {k} pairs"
                if isinstance(pc, (int, float)) and k else "")
        row("Blind A/B against the first version",
            f"First version {ab['a_wins']}, this version {ab['b_wins']}, {ab.get('ties', 0)} ties", comp,
            f"Judge: {judge_name(ab.get('judge'))}. Weak evidence either way.",
            "ab_vs_previous.json")
    hv = _d(m.get("heuristic_vs_llm_top10"))
    if isinstance(hv.get("matched"), int) and not _is_keyword_run(data):
        row("Top 10 shared with the keyword-rules baseline", fmt.k_of_n(hv["matched"], hv.get("k", 10)), "",
            "The AI changes the picks; the A/B and the benchmark test whether that helps.",
            "metrics.heuristic_vs_llm_top10")
    return rows
