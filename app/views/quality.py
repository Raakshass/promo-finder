"""Quality view (spec 5.5). Owner: Lane R.

Biggest problems first, the measured checks with their denominators and sources, cases to check by hand, known limits
and the evidence runs behind the numbers. Every number comes from the run's files, except the fresh 12-drama, rerun and
weight evidence, which checks.py quotes from docs/research.md and the scripts it names.
"""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from app.ui import checks, fmt, state
from app.ui import data as D
from app.ui.components import AI, empty_state, view_header
from app.ui.data import RunInfo
from app.ui.glossary import CHECKS, penalty, term

try:
    from promo_finder.slate import quotas as _quotas
except Exception:  # noqa: BLE001
    _quotas = None

TITLE = "Where it picks badly"
PURPOSE = "How the tool checks its own picks, and where it falls short, biggest problems first."
LIMITS = [
    "It reads text only: it can't see acting, pacing or shot quality.",
    "Every timestamp assumes the final cut matches the script's timing.",
    "The first-time viewer is an AI model reading a transcript, five clips per request, not a person watching.",
    "Joins are made only between consecutive episodes.",
    "The weights are hand-set from research, not learned from data. They live in config.yaml; the decisions you "
    "download are meant to refit them.",
]
_RANK = re.compile(r"#(\d+)")


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _ranked(promos: list[dict]) -> list[dict]:
    return sorted((p for p in promos if isinstance(p.get("rank"), int)), key=lambda p: p["rank"])


def _open_from(key: str, pids: dict[int, str]) -> None:
    """ButtonColumn callback: parse the rank after '#' in the clicked label (robust to sorting) and open the drawer."""
    click = st.session_state.get(key)
    label = (click.get("label") if isinstance(click, dict) else getattr(click, "label", "")) or ""
    m = _RANK.search(str(label))
    if m and int(m.group(1)) in pids:
        state.open_promo(pids[int(m.group(1))])


def _promo_table(case: str, rows: list[dict], promos: list[dict], cols: dict | None = None) -> None:
    """A compact promo table whose first column is an 'Open #k' button (spec 5.5)."""
    if not rows:
        return
    pids = {p["rank"]: p["promo_id"] for p in promos if isinstance(p.get("rank"), int)}
    df = pd.DataFrame(rows)
    key = f"q-open-{case}"
    config = {"#": st.column_config.ButtonColumn("#", pinned=True, width=96, type="tertiary", key=key,
                                                 on_click=_open_from, args=(key, pids))}
    config.update(cols or {})
    st.dataframe(df, hide_index=True, column_order=list(df.columns), column_config=config,
                 row_height=35, height=38 + 35 * len(df), key=f"q-table-{case}")


def _table(rows: list[dict], cols: dict | None = None) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, column_order=list(df.columns), column_config=cols or {},
                 height=min(38 + 35 * len(df), 38 + 35 * 12))


def _fails(p: dict) -> list[str]:
    rub = _d(p.get("rubric"))
    return [k for k in CHECKS if rub.get(k) is False]


def _has_rubric(promos: list[dict]) -> bool:
    return any(_d(p.get("rubric")) for p in promos)


# ---------------------------------------------------------------- sections

def _problems(data: dict) -> None:
    st.subheader("Biggest problems first", anchor=False)
    shown, missing = checks.problems(data)
    with st.container(border=True):
        if not shown:
            st.caption("No problem could be measured for this run.")
        for i, pr in enumerate(shown, 1):
            st.markdown(f"{i}\\. **{pr.title}** {pr.body}", help=f"Source: {pr.source}")
        if missing:
            st.caption(f"Not measured for this run: {', '.join(missing)}. They come from the extra commands described "
                       "in Method.")


def _measured(data: dict) -> None:
    st.subheader("Measured checks", anchor=False)
    rows = checks.measured_checks(data)
    if not rows:
        st.caption("No measured checks in this run.")
        return
    st.table(pd.DataFrame(rows), hide_index=True, border="horizontal")


def _case_fails(data: dict, promos: list[dict]) -> None:
    failing = [p for p in _ranked(promos) if _fails(p)]
    if not _has_rubric(promos):
        return
    cfg = _d(data.get("cfg"))
    with st.expander(f"Fail an editor's check · {len(failing)}"):
        st.markdown("Each failed check lowers the score by a fixed factor; nothing is removed. The editor decides.")
        _promo_table("fails", [{"#": f"Open #{p['rank']}", "Segments": fmt.segs(p),
                                "Fails": [CHECKS[k].fail_label for k in _fails(p)],
                                "Score factor": fmt.mult(_d(p.get("subscores")).get("rubric_mult") or 1.0)}
                               for p in failing], promos,
                     {"Segments": st.column_config.ListColumn("Segments", width="medium"),
                      "Fails": st.column_config.ListColumn("Fails", width="large")})
        n = len(promos)
        leads = ", ".join(str(x).title() for x in _l(_d(data.get("arc")).get("leads"))) or "the leads"
        _table([{"Check": c.fail_label.replace("{leads}", leads),
                 "Fails": fmt.k_of_n(sum(1 for p in promos if _d(p.get("rubric")).get(k) is False), n),
                 "Penalty": fmt.mult(penalty(cfg, k)), "Decided by": c.decided_by} for k, c in CHECKS.items()])
        a = _d(_d(data.get("check_exp")).get("checks"))
        b = _d(_d(data.get("check_exp_ends_open")).get("checks"))

        def pf(src: dict, key: str) -> tuple[str, str] | None:
            c = _d(src.get(key))
            p_, f_ = _d(c.get("pass")).get("follows"), _d(c.get("fail")).get("follows")
            if isinstance(p_, (int, float)) and isinstance(f_, (int, float)):
                return f"{p_:.1f}", f"{f_:.1f}"
            return None
        cf, eo, sc = pf(a, "context_free_open"), pf(b, "ends_open") or pf(a, "ends_open"), pf(a, "small_cast")
        if cf and eo and sc:
            judge = checks.judge_name(_d(data.get("check_exp")).get("model"))
            st.caption(f"Do the checks matter? In a small controlled test with another model ({judge}, 5 failing clips "
                       f"against 5 passing, matched on base score), 'follows it' for passing vs failing clips was "
                       f"{cf[0]} vs {cf[1]} for the opening line needing no backstory and {eo[0]} vs {eo[1]} for "
                       f"stopping before the payoff, but {sc[0]} vs {sc[1]} for 3 or fewer speakers, which is not "
                       "confirmed. 'Stakes clear' and 'someone to root for' pass whenever the AI writes any answer.")


def _case_hard(promos: list[dict]) -> None:
    hard = [p for p in _ranked(promos)
            if isinstance(_d(p.get("cold_viewer")).get("comprehensibility"), (int, float))
            and p["cold_viewer"]["comprehensibility"] <= 3]
    if not hard:
        return
    with st.expander(f"Hard for a first-time viewer (AI) to follow · {len(hard)}"):
        st.markdown(f"{AI} Rated 3 of 5 or lower. Usually the clip names people a stranger hasn't met.")
        _promo_table("hard", [{"#": f"Open #{p['rank']}",
                               "Follows": f"{fmt.num(p['cold_viewer']['comprehensibility'])} of 5",
                               "Would tap": (f"{fmt.num(p['cold_viewer']['would_tap'])} of 5"
                                             if isinstance(p["cold_viewer"].get("would_tap"), (int, float)) else ""),
                               "Couldn't place": [str(x) for x in _l(p["cold_viewer"].get("unresolved_refs"))]}
                              for p in hard], promos,
                     {"Couldn't place": st.column_config.ListColumn("Couldn't place", width="large",
                                                                    help=term("Couldn't place"))})


def _case_edits(promos: list[dict]) -> None:
    joins = [p for p in _ranked(promos) if len(_l(p.get("segments"))) > 1]
    scenes = [p for p in _ranked(promos) if _d(p.get("flags")).get("scene_change")]
    if not joins and not scenes:
        return
    with st.expander(f"Edits to check on video · {fmt.plural(len(joins), 'join')}, "
                     f"{fmt.plural(len(scenes), 'scene change')}"):
        st.markdown("A join puts the end of one episode before the start of the next; check it plays. A scene change "
                    "inside a clip is fine if the cut reads as one moment.")
        rows = [{"#": f"Open #{p['rank']}", "Segments": fmt.segs(p),
                 "Kind": "Join" if p in joins and p not in scenes else
                 ("Scene change" if p not in joins else "Join and scene change")}
                for p in _ranked(promos) if p in joins or p in scenes]
        _promo_table("edits", rows, promos, {"Segments": st.column_config.ListColumn("Segments", width="medium")})


def _status_sentence(p: dict, data: dict) -> str:
    pr = _d(p.get("promise"))
    s = pr.get("status")
    if s == "rejected":
        return checks.rejected_sentence(p)
    if s == "never answered":
        return "No episode summary answers it: possible fake hook."
    if s == "unchecked":
        return "Not checked in this run."
    n = pr.get("answered_in_episode")
    out = "Answered later in the same episode." if s == "same episode" else (
        f"Answered in episode {n}." if isinstance(n, int) else "Answered.")
    if pr.get("confidence") == "low":
        out += " The AI marked this answer as a guess."
    return out


def _case_unconfirmed(data: dict, promos: list[dict]) -> None:
    rows = [p for p in _ranked(promos) if _d(p.get("promise")).get("status") in ("rejected", "never answered",
                                                                                   "unchecked")
            or _d(p.get("promise")).get("confidence") == "low"]
    if not rows:
        return
    with st.expander(f"Answers not confirmed · {len(rows)}"):
        st.markdown(f"{AI} The AI could not tie these questions to an answer it is sure of.")
        _promo_table("unconfirmed", [{"#": f"Open #{p['rank']}", "Question": str(p["promise"].get("question") or ""),
                                      "Status": _status_sentence(p, data)} for p in rows], promos,
                     {"Question": st.column_config.TextColumn("Question", width="large"),
                      "Status": st.column_config.TextColumn("Status", width="large")})


def _case_flat(data: dict) -> None:
    rows = checks.flat_episodes(data)
    if not rows:
        return
    still = sum(1 for r in rows if r["still_flat"])
    with st.expander(f"Episodes the AI answered flat · {len(rows)} ({still} still flat)"):
        st.markdown("A flat answer rates no line above 1 for hook or for cliffhanger, usually a lapse in a batched "
                    "answer. Each was asked again alone.")
        _table([{"Episode": f"Episode {r['episode']}",
                 "Result": "still flat: no clip from this episode" if r["still_flat"] else "fixed on asking again"}
                for r in rows])


def _case_entailment(data: dict) -> None:
    rows = _d(_d(_d(data.get("metrics")).get("entailment")).get("rows"))
    if not rows:
        return
    checked = {int(k): v for k, v in rows.items() if isinstance(_d(v).get("contains"), bool)}
    misses = {k: v for k, v in sorted(checked.items()) if v["contains"] is False}
    unchecked = sorted(int(k) for k, v in rows.items() if _d(v).get("contains") is None)
    with st.expander(f"Best clip misses the writers' cliffhanger · {fmt.k_of_n(len(misses), len(checked))} checked"):
        st.markdown("For each episode, the highest-scoring clip (on the slate or not) was checked against the next "
                    "episode's continuity note.")
        if misses:
            st.caption(f"{AI} What it misses is written by the AI.")
            _table([{"Episode": f"Episode {k}", "What it misses": str(v.get("reason") or "")} for k, v in misses.items()],
                   {"What it misses": st.column_config.TextColumn("What it misses (AI)", width="large")})
        if unchecked:
            st.caption(f"Not checked (daily limit): episodes {fmt.ranges(unchecked)}")


def _case_no_shortlist(data: dict) -> None:
    eps = [e for e in _l(_d(data.get("promoability")).get("episodes"))
           if isinstance(e.get("episode"), int) and e.get("finalists") == 0]
    if not eps:
        return
    nums = [e["episode"] for e in eps]
    with st.expander(f"No clip on the shortlist · {fmt.plural(len(nums), 'episode')}"):
        cfg = _d(data.get("cfg"))
        short = _d(data.get("metrics")).get("finalists")
        late_rng = ""
        if _quotas and isinstance(short, int) and _d(cfg.get("slate")).get("shares"):
            try:
                q = _quotas(short, cfg)
                from app.ui import explain_rank
                parts_ = explain_rank.parts(data.get("promos") or [], cfg, data.get("episodes_list") or [])
                late = next((p for p in parts_ if p.key == "late"), None)
                if late:
                    k = sum(1 for n in nums if late.first_ep <= n <= late.last_ep)
                    late_rng = (f"By design, the late part gets {q['late']} of the {short} shortlist places, so {k} of "
                                f"these {len(nums)} are in {fmt.ep_range(late.first_ep, late.last_ep)}. ")
            except (KeyError, TypeError, ValueError):
                late_rng = ""
        st.markdown(f"{late_rng}Their best clips are under Promos › Best clip in each episode.")
        st.caption(f"Episodes {fmt.ranges(nums)}")


def _case_promise(data: dict, promos: list[dict]) -> None:
    led = _d(_d(data.get("metrics")).get("promise_ledger"))
    if not isinstance(led.get("promos"), int) or not any(_d(p.get("promise")) for p in promos):
        return
    same, later = led.get("answered_same_episode", 0), led.get("answered_later_episode", 0)
    never, unver = led.get("never_answered", 0), led.get("unchecked_or_rejected", 0)
    med = led.get("median_minutes_clip_to_answer")
    with st.expander(f"Does the show answer each promo's question? · {fmt.k_of_n(led.get('answered', 0), led['promos'])}"):
        txt = (f"Every promo leaves a question. If the show never answers it, the promo is a fake hook. {same} are "
               f"answered later in the same episode and {later} in a later one")
        if isinstance(med, (int, float)) and later:
            txt += f", a median {fmt.minutes(med * 60)} of watching after the clip (over those {later})"
        txt += ". "
        if not never and not unver:
            txt += "Every question is answered."
        elif not never:
            txt += f"Every question is answered except {unver} that could not be verified."
        else:
            txt += f"{never} are never answered; {unver} could not be verified."
        st.markdown(f"{AI} {txt}")
        paywall = _d(_d(data.get("run_log")).get("args")).get("paywall")
        rows = []
        for p in _ranked(promos):
            pr = _d(p.get("promise"))
            n = pr.get("answered_in_episode")
            wait = pr.get("minutes_after_clip")
            row = {"#": f"Open #{p['rank']}", "Question it leaves": str(pr.get("question") or ""),
                   "Answered in": (f"Episode {n}" if isinstance(n, int) else "") if pr.get("status") != "rejected" else "",
                   "Wait after the clip": (fmt.minutes(wait * 60) if isinstance(wait, (int, float))
                                           and pr.get("status") == "answered" else ""),
                   "Confidence": str(pr.get("confidence") or ""),
                   "Status": {"same episode": "Same episode", "answered": "Later episode", "rejected": "Not verified",
                              "never answered": "Never answered", "unchecked": "Not checked",
                              "behind the paywall": "Behind the paywall"}.get(pr.get("status"), "Not checked")}
            if paywall:
                row["Behind the paywall"] = "Yes" if isinstance(n, int) and n >= int(paywall) else ""
            rows.append(row)
        _promo_table("promise", rows, promos,
                     {"Question it leaves": st.column_config.TextColumn("Question it leaves (AI)", width="large"),
                      "Wait after the clip": st.column_config.TextColumn("Wait after the clip",
                                                                         help=term("Wait after the clip")),
                      "Confidence": st.column_config.TextColumn("Confidence", help=term("Confidence"))})


def _case_difficulty(data: dict) -> None:
    pm = _d(_d(data.get("metrics")).get("promoability"))
    tiers = _d(pm.get("tiers"))
    eps = _l(_d(data.get("promoability")).get("episodes"))
    if not tiers or not eps:
        return
    label = " · ".join(f"{int(tiers.get(t, 0))} {t}" for t in ("easy", "medium", "hard"))
    with st.expander(f"Which episodes are hard to promo? · {label}"):
        verdict = "; ".join(str(v) for v in _l(pm.get("verdict")))
        flat_hard = [r["episode"] for r in checks.flat_episodes(data) if r["still_flat"]]
        hard = [e["episode"] for e in eps if e.get("tier") == "hard"]
        flat_in_hard = [e for e in flat_hard if e in hard]
        txt = f"{verdict[:1].upper()}{verdict[1:]}." if verdict else ""
        if flat_in_hard:
            txt += (f" {len(flat_in_hard)} of the {len(hard)} hard episodes ({', '.join(str(e) for e in flat_in_hard)}) "
                    "count as hard only because the AI answered them flat.")
        st.markdown(fmt.md_esc(txt))
        st.caption(term("Difficulty"))
        _table([{"Episode": f"Episode {e['episode']}",
                 "Difficulty": [e.get("tier") if e.get("episode") not in flat_hard else "not scored"],
                 "Conflict moments per min": e.get("conflict_beats_per_min"),
                 "Resolves vs builds": e.get("payoff_share"), "Needs backstory": e.get("context_load"),
                 "Speakers": e.get("speakers"), "Shortlisted clips": e.get("finalists"),
                 "Watch-outs": [str(w) for w in _l(e.get("why"))]} for e in eps],
               {"Difficulty": st.column_config.MultiselectColumn(
                   "Difficulty", options=["easy", "medium", "hard", "not scored"],
                   color=["green", "orange", "red", "gray"], disabled=True, help=term("Difficulty")),
                "Conflict moments per min": st.column_config.NumberColumn(format="%.1f"),
                "Resolves vs builds": st.column_config.NumberColumn(format="percent"),
                "Needs backstory": st.column_config.NumberColumn(format="percent"),
                "Watch-outs": st.column_config.ListColumn("Watch-outs", width="large")})


def _case_parse(data: dict) -> None:
    pr = _d(data.get("parse"))
    if not isinstance(pr.get("episodes"), int):
        return
    per = _l(pr.get("per_episode"))
    ok = sum(1 for e in per if e.get("parse_rate") == 1.0) if per else pr["episodes"]
    with st.expander(f"Did it read the scripts correctly? · {fmt.k_of_n(ok, pr['episodes'])} episodes parsed"):
        kinds = _d(pr.get("block_kinds"))
        txt = f"Read {fmt.k_of_n(ok, pr['episodes'])} episodes"
        if isinstance(pr.get("total_blocks"), int):
            txt += f": {fmt.num(pr['total_blocks'])} timed blocks"
            if kinds:
                txt += (f" ({fmt.num(kinds.get('dialogue', 0))} lines of dialogue, {fmt.num(kinds.get('action', 0))} "
                        f"actions, {fmt.num(kinds.get('sfx', 0))} sound cues, {fmt.num(kinds.get('camera', 0))} camera "
                        "cuts)")
            txt += ", all parsed" if ok == pr["episodes"] else ""
        txt += "."
        if isinstance(pr.get("duration_min_s"), (int, float)):
            txt += f" Episodes run {fmt.secs(pr['duration_min_s'])[:-2]}–{fmt.secs(pr['duration_max_s'])}."
        w = pr.get("episodes_with_warnings")
        if isinstance(w, int) and w:
            txt += f" {fmt.plural(w, 'episode')} had small format problems, all handled."
        st.markdown(txt)
        aliases = _d(_d(data.get("aliases")).get("aliases"))
        if aliases:
            st.markdown(f"Speaker names: the AI proposed merges; code applied {len(aliases)} where the short name sits "
                        "inside the label. The rest need a person.")
            _table([{"Written as": str(k), "Treated as": str(v)} for k, v in aliases.items()])


def _evidence(run: RunInfo, runs: list[RunInfo], data: dict) -> None:
    if run.role != "official":
        return
    by = {r.key: r for r in runs}
    rows = []
    ab = _d(data.get("ab"))
    if "first-version" in by and isinstance(ab.get("a_wins"), int):
        rows.append({"Run": by["first-version"].label, "What it tests": "Was the redesign better?",
                     "The number it backs": f"Blind A/B: first version {ab['a_wins']}, this version {ab['b_wins']}, "
                                            f"{ab.get('ties', 0)} ties."})
    m = _d(data.get("metrics"))
    hv, hl = _d(m.get("heuristic_vs_llm_top10")), _d(m.get("cliff_localisation_heuristic"))
    if "keyword-rules" in by and isinstance(hv.get("matched"), int):
        txt = f"Its top {hv.get('k', 10)} shares {hv['matched']} with the AI's"
        if _l(hl.get("rows")):
            txt += (f"; it finds the writers' cliffhanger in "
                    f"{fmt.k_of_n(sum(1 for r in hl['rows'] if r.get('in_last_window')), len(hl['rows']))} episodes")
        rows.append({"Run": by["keyword-rules"].label, "What it tests": "The floor the AI must beat.",
                     "The number it backs": txt + "."})
    if "cues-hidden" in by:
        try:
            r = by["cues-hidden"]
            other = D.load_run(str(r.dir), D.stamp(r.dir))
            ra = _d(_d(other.get("metrics")).get("cliff_localisation")).get("recall_all")
            mine = _d(m.get("cliff_localisation")).get("recall_all")
            if isinstance(ra, (int, float)) and isinstance(mine, (int, float)):
                rows.append({"Run": r.label, "What it tests": "Does the AI only read the word 'cliffhanger'?",
                             "The number it backs": f"{fmt.pct(ra)} of episodes with cues hidden against "
                                                    f"{fmt.pct(mine)} with them."})
        except D.RunReadError:
            pass
    st_ = _d(data.get("stability"))
    cons = [by[k] for k in ("consistency-a", "consistency-b") if k in by]
    if cons and isinstance(st_.get("expected_overlap_top10"), (int, float)):
        k = int(round(st_["expected_overlap_top10"] * 10))
        n = st_.get("noise_beats")
        over = f" over {fmt.num(n)} lines" if isinstance(n, int) else ""
        rows.append({"Run": " and ".join(r.label for r in cons),
                     "What it tests": "How much the model's line scores change between runs.",
                     "The number it backs": f"Their differences{over} drive 'two runs share about {k} of the top 10'."})
    if not rows:
        return
    st.subheader("Runs behind these numbers", anchor=False)
    st.caption("Evidence runs are read-only. Switch to one in the Results menu at the top.")
    _table(rows, {"The number it backs": st.column_config.TextColumn(width="large"),
                  "What it tests": st.column_config.TextColumn(width="medium")})


def render(run: RunInfo | None, data: dict, runs: list[RunInfo]) -> None:
    """Draw the view for the selected run. Never mutate `data`. Never write to st.sidebar."""
    view_header(TITLE, PURPOSE)
    if run is None:
        empty_state("No results to show yet", "Run the tool on a show in Run it.")
        return
    promos = data.get("promos") or []
    _problems(data)
    st.space("medium")
    _measured(data)
    st.space("medium")
    st.subheader("Cases to check by hand", anchor=False)
    _case_fails(data, promos)
    _case_hard(promos)
    _case_edits(promos)
    _case_unconfirmed(data, promos)
    _case_flat(data)
    _case_entailment(data)
    _case_no_shortlist(data)
    _case_promise(data, promos)
    _case_difficulty(data)
    _case_parse(data)
    st.space("medium")
    st.subheader("Known limits", anchor=False)
    st.markdown("\n".join(f"- {t}" for t in LIMITS))
    st.space("medium")
    _evidence(run, runs, data)
    if data.get("failures_md"):
        st.download_button("The generated report, unedited (failures.md)", data["failures_md"], file_name="failures.md",
                           mime="text/markdown", on_click="ignore", icon=":material/download:", key="q-failures-md")
