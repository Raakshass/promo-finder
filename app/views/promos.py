"""Promos view (spec 5.1). Owner: Lane P.

Top to bottom: title, the three brief cards, the AI line, the controls row (list switch, filter, Export), the
posting-order line, the review line, the table and its caption. "Best clip in each episode" swaps the last four for the
per-episode list. Every number comes from the selected run's files.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd
import streamlit as st

from app.ui import checks, exports, explain_rank, fmt, state
from app.ui.components import AI, empty_state, view_header
from app.ui.data import RunInfo
from app.ui.glossary import CHECKS, DECISIONS, PART_NAMES, term

TITLE = "The promos"
PURPOSE = ("Every promo the tool found: each segment's episode, in and out time, the hook, the cliffhanger and why it "
           "works. Open one to review it.")
AI_LINE = (f"{AI} The first-time viewer and the judges in this app are AI models reading the script, not people. The "
           "tool reads text only, so check every cut against the video.")
TABLE_CAPTION = ("Times are episode time from the script (minutes:seconds.hundredths). Check every cut against the "
                 "video before posting.")
NO_CLIP_RULES = ("No clip in this show fits the rules: 20–60 s, a strong line in the first 3 s, a strong line at the "
                 "cut, no line cut in half. Very short or very quiet episodes are the usual reason.")
FILTERS = ["all", "check", "open", "approve", "trim", "reject"]
FILTER_LABELS = {"all": "All", "check": "To check", "open": "Not decided", "approve": "Approved", "trim": "Trim",
                 "reject": "Rejected"}
LISTS = ["slate", "episodes"]
DIFFICULTY = ["easy", "medium", "hard", "not scored"]
JUDGE_ONLY = "First-time viewer (AI)"   # decided_by of the checks that exist only for shortlisted clips
_OPEN = re.compile(r"#(\d+)")


# ---------------------------------------------------------------------------------------------------------- helpers

def _n(x: object) -> str:
    """A small number as written: 20.0 -> "20", 2.5 -> "2.5"."""
    try:
        v = round(float(x), 2)
    except (TypeError, ValueError):
        return ""
    return str(int(v)) if v == int(v) else f"{v:g}"


def _promos(d: dict) -> list[dict]:
    return sorted((p for p in d.get("promos") or [] if isinstance(p, dict)), key=lambda p: p.get("rank") or 0)


def _rules(cfg: dict) -> tuple[str, str, str]:
    """(min s, max s, hook window s) from the run's config, with the brief's values when absent."""
    pc = (cfg or {}).get("promo") or {}
    return _n(pc.get("min_duration_s", 20)), _n(pc.get("max_duration_s", 60)), _n(pc.get("hook_window_s", 3))


def _has_rubric(promos: list[dict]) -> bool:
    return any(p.get("rubric") for p in promos)


def _has_viewer(promos: list[dict]) -> bool:
    return any((p.get("cold_viewer") or {}).get("comprehensibility") is not None for p in promos)


def _has_block(promos: list[dict]) -> bool:
    return bool(promos) and all((p.get("flags") or {}).get("block") in PART_NAMES for p in promos)


def _joins(p: dict) -> bool:
    return len({s.get("episode") for s in p.get("segments") or []}) > 1


def _rank_span(a: int, b: int) -> str:
    return f"#{a}" if a == b else f"#{a}–#{b}"


def _bare_range(a: int, b: int) -> str:
    return f"episode {a}" if a == b else f"{a}–{b}"


def _parts(d: dict, promos: list[dict]) -> list:
    return list(explain_rank.parts(promos, d.get("cfg") or {}, d.get("episodes_list") or []) or [])


def _episodes_total(run: RunInfo, d: dict) -> int:
    m = (d.get("metrics") or {}).get("episodes")
    if isinstance(m, int) and m > 0:
        return m
    return run.episodes or len(d.get("episodes_list") or [])


def _passes_rules(p: dict, cfg: dict) -> bool:
    lo, hi, win = (float(x) for x in _rules(cfg))
    try:
        return lo <= float(p.get("duration_s")) <= hi and float((p.get("hook") or {}).get("at_s")) <= win
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------------------ cards

def _card(kicker: str, headline: str, body: str, button: str, view: str, key: str) -> None:
    st.caption(kicker)
    st.markdown(f"#### {headline}")
    st.caption(body)
    st.button(button, type="secondary", width="stretch", on_click=state.goto, args=(view,), key=key)


def _card_find(run: RunInfo, d: dict, promos: list[dict]) -> tuple[str, str]:
    n = len(promos)
    e = _episodes_total(run, d)
    used = len({s.get("episode") for p in promos for s in p.get("segments") or []})
    head = f"{fmt.plural(n, 'promo')} from {fmt.k_of_n(used, e)} episodes" if n else "0 promos"
    if not n:
        return head, NO_CLIP_RULES
    s = sum(len(p.get("segments") or []) for p in promos)
    j = sum(1 for p in promos if _joins(p))
    lo, hi, win = _rules(d.get("cfg") or {})
    inv = (d.get("metrics") or {}).get("invariants") or {}
    ok = sum(1 for p in promos if _passes_rules(p, d.get("cfg") or {}))
    if inv.get("pass_rate") == 1.0 or (not inv and ok == n):
        who = f"All {n} promos run" if n > 1 else "It runs"
    else:
        who = f"{fmt.k_of_n(ok, n)} promos run"
    joins = f"{j} joins two episodes" if j == 1 else f"{j} join two episodes"
    body = f"{fmt.plural(s, 'segment')} with in and out times; {joins}. {who} {lo}–{hi} s with the hook in the first {win} s."
    return head, body


def _part_clause(parts_: list) -> str:
    bits = []
    for i, pt in enumerate(parts_):
        what = fmt.plural(pt.n, "promo") if i == 0 else str(pt.n)
        bits.append(f"{what} from {fmt.ep_range(pt.first_ep, pt.last_ep)}")
    return ", then ".join(bits)


def _card_rank(parts_: list) -> tuple[str, str]:
    tail = "# is the posting order, not a quality rank."
    if len(parts_) >= 2:
        first = parts_[0]
        return (f"Post {fmt.ep_range(first.first_ep, first.last_ep)} first",
                f"A viewer who taps a promo starts at episode 1: {_part_clause(parts_)}. {tail}")
    if len(parts_) == 1:
        return "Post from the top", f"A viewer who taps a promo starts at episode 1: {_part_clause(parts_)}. {tail}"
    return "Post from the top", tail


def _card_quality(d: dict, promos: list[dict]) -> tuple[str, str]:
    n = len(promos)
    if not n:
        head = "No promos to check"
    elif not _has_rubric(promos):
        head = "Not measured: this run predates the editor's checks"
    else:
        head = f"{fmt.k_of_n(checks.fail_count(d), n)} fail an editor's check"
    body = "Ranked lower, not removed."   # benchmark claims live in the Quality view, not on this card
    ov = (d.get("stability") or {}).get("expected_overlap_top10")
    if isinstance(ov, (int, float)):
        body += f" Two runs share about {round(ov * 10)} of their top 10."
    return head, body


def _cards(run: RunInfo, d: dict, promos: list[dict], parts_: list) -> None:
    c1, c2, c3 = st.columns(3, border=True)
    with c1:
        _card("1 · Find the promos", *_card_find(run, d, promos), "See them on the show map", "Ranking", "card-find")
    with c2:
        _card("2 · Rank them", *_card_rank(parts_), "See why each sits where it does", "Ranking", "card-rank")
    with c3:
        _card("3 · Check quality", *_card_quality(d, promos), "See where it picks badly", "Quality", "card-quality")


# ------------------------------------------------------------------------------------------------------ slate table

def _posting_line(parts_: list) -> str:
    if len(parts_) < 2:
        return "Post from the top."
    bits = [f"{_rank_span(pt.first_rank, pt.last_rank)} come from {fmt.ep_range(pt.first_ep, pt.last_ep)}"
            if i == 0 else f"{_rank_span(pt.first_rank, pt.last_rank)} from {_bare_range(pt.first_ep, pt.last_ep)}"
            for i, pt in enumerate(parts_)]
    return "Post from the top: " + ", ".join(bits) + ". Why: see **Ranking**."


def _review_line(run: RunInfo, promos: list[dict]) -> str:
    if run.role == "evidence":
        return "Evidence runs are read-only: decisions are off. Switch to the official results to review."
    c = state.review_counts(run.run_id, [p.get("promo_id") for p in promos])
    return (f"Your review: {fmt.k_of_n(c['decided'], len(promos))} decided · {c['approve']} approved · {c['trim']} to "
            f"trim · {c['reject']} rejected. Decisions stay in this browser tab until you reload; download your review "
            "to keep it.")


def _open_from_table(rank_to_pid: dict[int, str]) -> None:
    click = st.session_state.get("promos_open")
    label = (click or {}).get("label") if isinstance(click, dict) else getattr(click, "label", None)
    m = _OPEN.search(label or "")
    if m and int(m.group(1)) in rank_to_pid:
        state.open_promo(rank_to_pid[int(m.group(1))])


def _passed(p: dict) -> tuple[int, int]:
    rub = {k: v for k, v in (p.get("rubric") or {}).items() if k in CHECKS}
    return sum(1 for v in rub.values() if v), len(rub)


def _row(p: dict, call: str | None, flags_: list) -> dict:
    hook = p.get("hook") or {}
    cl = p.get("cliffhanger") or {}
    cliff = fmt.said(cl.get("speaker"), cl.get("text") or "")
    if (p.get("rubric") or {}).get("ends_open") is False:
        cliff = "Resolves: " + cliff
    k, n = _passed(p)
    return {
        "#": f"Open #{p.get('rank')}",
        "Part": PART_NAMES.get((p.get("flags") or {}).get("block"), ""),
        "Segments (episode time)": fmt.segs(p),
        "Length": float(p.get("duration_s") or 0.0),
        "Hook (first 3 s)": fmt.said(hook.get("speaker"), hook.get("text") or ""),
        "Cliffhanger (at the cut)": cliff,
        "Why it works": str(p.get("why_it_works") or ""),
        "Checks": fmt.k_of_n(k, n) if n else "",
        "To check": [f.short for f in flags_],
        "Your call": DECISIONS[call][1] if call in DECISIONS else "",
    }


def _slate_table(run: RunInfo, d: dict, promos: list[dict], flt: str) -> None:
    review = state.review_all(run.run_id)
    rows, rank_to_pid = [], {}
    for p in promos:
        call = (review.get(p.get("promo_id")) or {}).get("call")
        flags_ = checks.before_posting(p, d)
        keep = (flt == "all" or (flt == "check" and flags_) or (flt == "open" and call is None)
                or (flt in DECISIONS and call == flt))
        if not keep:
            continue
        rows.append(_row(p, call, flags_))
        if isinstance(p.get("rank"), int):
            rank_to_pid[p["rank"]] = p.get("promo_id")
    if not rows:
        st.caption("No promo matches this filter.")
        return
    df = pd.DataFrame(rows)
    has_rubric = _has_rubric(promos)
    why_label = "Why it works (AI)" if _has_viewer(promos) else "Why it works (template, no AI)"
    order = ["#"] + (["Part"] if _has_block(promos) else []) + ["Segments (episode time)", "Length", "Hook (first 3 s)",
                                                               "Cliffhanger (at the cut)", "Why it works"]
    order += (["Checks", "To check"] if has_rubric else []) + ["Your call"]
    cc = st.column_config
    config = {
        "#": cc.ButtonColumn("#", pinned=True, width=96, type="tertiary", key="promos_open", on_click=_open_from_table,
                             args=(rank_to_pid,),
                             help="Posting order: post #1 first. Click to open everything about the promo."),
        "Part": cc.TextColumn("Part", width="small", help=term("Part of the show")),
        "Segments (episode time)": cc.ListColumn(
            "Segments (episode time)", width="medium",
            help="Each segment's episode, in and out time, in playing order. Minutes:seconds.hundredths from the script."),
        "Length": cc.NumberColumn("Length", format="%.1f s", width="small",
                                  help="Total length. The brief asks for 20–60 s."),
        "Hook (first 3 s)": cc.TextColumn("Hook (first 3 s)", width="large", help=term("Hook")),
        "Cliffhanger (at the cut)": cc.TextColumn("Cliffhanger (at the cut)", width="large", help=term("Cliffhanger")),
        "Why it works": cc.TextColumn(why_label, width="large",
                                      help="One line from the first-time viewer (AI). AI-written; it can be wrong."
                                      if _has_viewer(promos) else "Written from a template: this run had no AI."),
        "Checks": cc.TextColumn("Checks", width="small", help=term("Editor's checks")),
        "To check": cc.ListColumn("To check", width="medium", help=term("Before posting")),
        "Your call": cc.TextColumn("Your call", width="small",
                                   help="Your decision, set when you open the promo. Kept in this browser tab only."),
    }
    st.dataframe(df, hide_index=True, row_height=35, height=38 + 35 * len(df), key="promos_table",
                 column_order=order, column_config={k: v for k, v in config.items() if k in order})
    st.caption(TABLE_CAPTION)
    if not has_rubric:
        st.caption("This run predates the editor's checks.")


# ------------------------------------------------------------------------------------------------- best clip list

def _check_keys_by_label() -> dict[str, str]:
    try:
        from promo_finder.rubric import LABELS
    except Exception:  # noqa: BLE001
        return {}
    return {str(v).lower(): k for k, v in LABELS.items()}


def _script_checks(summary: str, n_script: int) -> str:
    """"passes every check" -> "All 7 pass"; "fails: a, b" -> "Fails 2 of 7: <labels>"."""
    s = (summary or "").strip()
    if not s:
        return ""
    if s.lower().startswith("passes"):
        return f"All {n_script} pass"
    names = [x.strip() for x in s.split(":", 1)[-1].split(",") if x.strip()]
    keys = _check_keys_by_label()
    shown = [CHECKS[keys[x.lower()]].fail_label if x.lower() in keys and keys[x.lower()] in CHECKS else x for x in names]
    return f"Fails {fmt.k_of_n(len(names), n_script)}: " + "; ".join(shown)


def _block_of(ep: int, d: dict) -> str:
    eps = d.get("episodes_list") or []
    if not eps:
        return ""
    try:
        from promo_finder.slate import block_of
        return PART_NAMES.get(block_of(ep, min(eps), max(eps), d.get("cfg") or {}), "")
    except Exception:  # noqa: BLE001
        return ""


def _episode_rows(d: dict) -> tuple[list[dict], int, int]:
    """(rows, episodes with a clip, of those passing every script check)."""
    text = d.get("per_episode_csv") or ""
    clips = {}
    for r in csv.DictReader(io.StringIO(text, newline="")):
        try:
            clips[int(r.get("episode") or "")] = r
        except ValueError:
            continue
    tiers = {e.get("episode"): e for e in (d.get("promoability") or {}).get("episodes") or [] if isinstance(e, dict)}
    flat = {f.get("episode"): f for f in checks.flat_episodes(d) or [] if isinstance(f, dict)}
    eps = sorted(set(d.get("episodes_list") or []) | set(clips) | {e for e in tiers if isinstance(e, int)})
    n_script = sum(1 for c in CHECKS.values() if c.decided_by != JUDGE_ONLY)
    rows, with_clip, pass_all = [], 0, 0
    for ep in eps:
        r = clips.get(ep)
        tier = tiers.get(ep) or {}
        why = [str(w) for w in tier.get("why") or [] if w]
        fl = flat.get(ep) or {}
        if r and (r.get("in") or "").strip():
            with_clip += 1
            summary = r.get("editor_checks") or ""
            pass_all += summary.strip().lower().startswith("passes")
            try:
                length = fmt.secs(float(r.get("duration_s") or ""))
            except ValueError:
                length = ""
            rows.append({"Episode": ep, "Part": PART_NAMES.get(r.get("show_block") or "", "") or _block_of(ep, d),
                         "In–out": f"{r.get('in')}–{r.get('out')}", "Length": length,
                         "Hook": r.get("hook_line") or "", "Cliffhanger": r.get("cliffhanger_line") or "",
                         "Script checks": _script_checks(summary, n_script),
                         "Difficulty": [tier["tier"]] if tier.get("tier") in DIFFICULTY else [],
                         "Watch-outs": why})
            continue
        if fl.get("still_flat") and fl.get("all_zero"):
            reason, diff = "No clip: the AI scored every line 0, even when asked again.", ["not scored"]
        elif fl.get("still_flat"):
            reason, diff = "No clip: no line was strong enough to open a promo, even when asked again.", ["not scored"]
        else:
            lo, hi, win = _rules(d.get("cfg") or {})
            reason = (f"No clip fits the rules: {lo}–{hi} s, a strong line in the first {win} s, a strong line at the "
                      "cut, no line cut in half.")
            diff = [tier["tier"]] if tier.get("tier") in DIFFICULTY else []
        rows.append({"Episode": ep, "Part": _block_of(ep, d), "In–out": "", "Length": "", "Hook": reason,
                     "Cliffhanger": "", "Script checks": "", "Difficulty": diff, "Watch-outs": why})
    return rows, with_clip, pass_all


def _best_clip_list(run: RunInfo, d: dict, promos: list[dict]) -> None:
    rows, k, p = _episode_rows(d)
    e = _episodes_total(run, d)
    n_script = sum(1 for c in CHECKS.values() if c.decided_by != JUDGE_ONLY)
    shortlist = ((d.get("cfg") or {}).get("judge") or {}).get("top_n")
    st.caption(f"The best single clip from each episode, for 'new episode out' posts. Not part of the {len(promos)} and "
               f"not tested on the first-time viewer (AI). {fmt.k_of_n(k, e)} episodes have one; {p} of those pass "
               "every script check.")
    if not rows:
        st.caption("This run has no per-episode list.")
        return
    cc = st.column_config
    judge = len(CHECKS) - n_script
    checks_help = (f"The {n_script} checks code runs on every clip. The {judge} that need the first-time viewer (AI) "
                   + (f"exist only for the {shortlist} shortlisted clips." if shortlist
                      else "exist only for the shortlisted clips."))
    order = ["Episode", "Part", "In–out", "Length", "Hook", "Cliffhanger", "Script checks", "Difficulty", "Watch-outs"]
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, row_height=35, height=38 + 35 * min(len(rows), 20), key="episodes_table",
        column_order=order,
        column_config={
            "Episode": cc.NumberColumn("Episode", format="%d", width="small"),
            "Part": cc.TextColumn("Part", width="small", help=term("Part of the show")),
            "In–out": cc.TextColumn("In–out", width="medium", help=term("Episode time")),
            "Length": cc.TextColumn("Length", width="small"),
            "Hook": cc.TextColumn("Hook", width="large", help=term("Hook")),
            "Cliffhanger": cc.TextColumn("Cliffhanger", width="large", help=term("Cliffhanger")),
            "Script checks": cc.TextColumn("Script checks", width="medium", help=checks_help),
            "Difficulty": cc.MultiselectColumn("Difficulty", options=DIFFICULTY,
                                               color=["green", "orange", "red", "gray"], disabled=True,
                                               help=term("Difficulty")),
            "Watch-outs": cc.ListColumn("Watch-outs", width="large"),
        })


# ----------------------------------------------------------------------------------------------------------- export

def _file_bytes(run: RunInfo, rel: str, fallback: str) -> bytes | None:
    p = run.dir / rel
    try:
        return p.read_bytes() if p.is_file() else (fallback.encode("utf-8") if fallback else None)
    except OSError:
        return fallback.encode("utf-8") if fallback else None


def _slug(run: RunInfo) -> str:
    return run.dir.parent.name or "promos"


def _export(run: RunInfo, d: dict, promos: list[dict]) -> None:
    review = state.review_all(run.run_id)
    decided = any((e or {}).get("call") for e in review.values())
    with st.popover("Export", icon=":material/download:"):
        only = st.toggle("Only promos you approved or marked to trim", key="exp_only_kept", disabled=not decided)
        st.download_button("Cut list (CSV, one row per segment)", on_click="ignore", key="exp-cut",
                           data=exports.cut_list_csv(d, review, only_kept=bool(only and decided)),
                           file_name=f"{_slug(run)}-cut-list.csv", mime="text/csv", icon=":material/download:",
                           help="One row per segment, in playing order. In and out are episode time from the script.")
        sh = _file_bytes(run, "cut_promos.sh", d.get("cut_script") or "")
        if sh:
            st.download_button("ffmpeg script (cut_promos.sh)", data=sh, file_name="cut_promos.sh", mime="text/x-shellscript",
                               on_click="ignore", key="exp-sh", icon=":material/download:",
                               help="Cuts each promo from your episode videos (named ep-001.mp4 …) with ffmpeg. Not yet "
                                    "tried on real footage.")
        md = _file_bytes(run, "promos.md", d.get("promos_md") or "")
        if md:
            st.download_button("Readable slate (promos.md)", data=md, file_name="promos.md", mime="text/markdown",
                               on_click="ignore", key="exp-md", icon=":material/download:")
        pe = _file_bytes(run, "promos_by_episode.csv", d.get("per_episode_csv") or "")
        if pe:
            st.download_button("Best clip in each episode (CSV)", data=pe, file_name="promos_by_episode.csv",
                               mime="text/csv", on_click="ignore", key="exp-pe", icon=":material/download:")
        if run.role != "evidence":
            st.download_button("Your review (CSV)", data=exports.review_csv(run, d, review), disabled=not decided,
                               file_name=f"{_slug(run)}-review-{run.key}.csv", mime="text/csv", on_click="ignore",
                               key="exp-review", icon=":material/download:")
        with st.expander("Copy all timecodes"):
            st.code(exports.timecodes_text(promos), language=None)


# ----------------------------------------------------------------------------------------------------------- render

def render(run: RunInfo | None, data: dict, runs: list[RunInfo]) -> None:
    """Draw the view for the selected run. Never mutate `data`. Never write to st.sidebar."""
    view_header(TITLE, PURPOSE)
    if run is None:
        empty_state("No results to show yet", "Run the tool on a show in Run it.",
                    action=("Go to Run it", state.goto, ("Run it",)))
        return
    d = data or {}
    promos = _promos(d)
    parts_ = _parts(d, promos) if promos else []
    _cards(run, d, promos, parts_)
    st.caption(AI_LINE)
    if not promos:
        empty_state("No clip fits the rules", NO_CLIP_RULES, action=("See the input check", state.goto, ("Quality",)))
        return
    e = _episodes_total(run, d)
    if 0 < e < 10:
        st.caption(f"With {e} episodes the opening, middle and late split is coarse, and clips from the last fifth are "
                   "flagged as possible spoilers.")

    with st.container(horizontal=True, vertical_alignment="center"):
        which = st.segmented_control("List", LISTS, key="promos_list", default="slate", label_visibility="collapsed",
                                     format_func=lambda k: f"The {len(promos)} promos" if k == "slate"
                                     else "Best clip in each episode")
        which = which or "slate"
        flt = "all"
        if which == "slate":
            flt = st.pills("Show", FILTERS, key="promos_filter", default="all", label_visibility="collapsed",
                           format_func=FILTER_LABELS.get) or "all"
        st.space("stretch")
        _export(run, d, promos)

    if which == "episodes":
        _best_clip_list(run, d, promos)
        return
    st.markdown(_posting_line(parts_))
    st.caption(_review_line(run, promos))
    _slate_table(run, d, promos, flt)
