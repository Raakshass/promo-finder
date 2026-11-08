"""The promo drawer (spec 5.3). Owner: Lane P.

maybe_open() is called once, last in streamlit_app.py. The drawer is a dialog, so it is a fragment: a click inside it
reruns only the drawer. Previous and Next set ss[OPEN]; closing calls state.close_promo and reruns the page.
Cut from P0: keyboard shortcuts and the read-along section.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui import checks, components, data, explain_promo, explain_rank, fmt, state
from app.ui.data import RunInfo
from app.ui.glossary import CHECKS, DECISIONS, PART_NAMES, REJECT_REASONS, penalty

VIEWER_INTRO = ("The same AI model, shown only this clip's spoken lines and actions, answers as a stranger scrolling "
                "past. It is a stand-in, not a person.")
PROMISE_CAPTION = ("Found by one AI request over the writers' one-line episode summaries; code only checks that the "
                   "episode comes after the clip.")
NOT_REPLAYABLE = "This run was ordered by an earlier version of the tool, so its order can't be replayed here."
NOT_MULTIPLYING = "These parts don't multiply to this run's score: it was made by an earlier version of the tool."


def _find_run(run_key: str) -> RunInfo | None:
    return next((r for r in data.list_runs() if r.key == run_key), None)


def _leads(arc: dict) -> str:
    leads = [str(x) for x in (arc or {}).get("leads") or [] if x]
    return " and ".join(leads) if leads else "the two leads"


def _set_call(run_id: str, pid: str, call: str | None) -> None:
    state.review_set(run_id, pid, call)


def _save_note(run_id: str, pid: str, key: str) -> None:
    entry = state.review_get(run_id, pid) or {}
    if entry.get("call"):
        state.review_set(run_id, pid, entry["call"], note=st.session_state.get(key) or "")


def _save_reason(run_id: str, pid: str, key: str) -> None:
    entry = state.review_get(run_id, pid) or {}
    if entry.get("call") == "reject":
        state.review_set(run_id, pid, "reject", reason=st.session_state.get(key) or "")


def _seg_secs(s: dict) -> str:
    try:
        return fmt.secs(float(s["end_s"]) - float(s["start_s"]))
    except (KeyError, TypeError, ValueError):
        return ""


def _cut_lines(p: dict) -> str:
    return "\n".join(f"Ep {s.get('episode')}  {s.get('start')}–{s.get('end')}  {_seg_secs(s)}".rstrip()
                     for s in p.get("segments") or [])


def _heading(p: dict, d: dict, parts_: list) -> None:
    rank = p.get("rank")
    segs = p.get("segments") or []
    st.subheader(f"#{rank} · {fmt.plural(len(segs), 'segment')} · {fmt.secs(p.get('duration_s') or 0)}", anchor=False)
    flags = p.get("flags") or {}
    bits: list[str] = []
    block = flags.get("block")
    if block in PART_NAMES:
        part = next((x for x in parts_ if getattr(x, "key", None) == block), None)
        bits.append(f"{PART_NAMES[block]} part ({fmt.ep_range(part.first_ep, part.last_ep)})" if part
                    else f"{PART_NAMES[block]} part")
    n = len(d.get("promos") or [])
    if rank is not None:
        bits.append(f"posted {fmt.ordinal(rank)} of {n}")
    if flags.get("minutes_into_show") is not None:
        bits.append(f"{fmt.minutes(float(flags['minutes_into_show']) * 60)} into the show")
    if p.get("score") is not None:
        bits.append(f"score {fmt.score(p['score'])}")
    if bits:
        st.caption(" · ".join(bits))
    badge = checks.stability_badge(p, d)
    if badge:
        label, colour, help_ = badge
        st.badge(label, color=colour, help=help_, icon=":material/trending_down:" if colour == "orange" else None)


def _decisions(run: RunInfo, run_key: str, p: dict, promos: list[dict]) -> None:
    pid = p["promo_id"]
    order = [q["promo_id"] for q in promos]
    i = order.index(pid)
    entry = state.review_get(run.run_id, pid) or {}
    call = entry.get("call")
    read_only = run.role == "evidence"
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button("Previous", icon=":material/arrow_upward:", key=f"dr-prev-{run_key}", disabled=i == 0,
                  on_click=state.open_promo, args=(order[max(i - 1, 0)],))
        st.button("Next", icon=":material/arrow_downward:", key=f"dr-next-{run_key}", disabled=i == len(order) - 1,
                  on_click=state.open_promo, args=(order[min(i + 1, len(order) - 1)],))
        st.space("stretch")
        if not read_only:
            for key, (button, _past, _colour, icon, _short) in DECISIONS.items():
                st.button(button, icon=icon, key=f"dr-{key}-{run_key}-{pid}",
                          type="primary" if call == key else "secondary",
                          on_click=_set_call, args=(run.run_id, pid, key))
            st.button("Clear", key=f"dr-clear-{run_key}-{pid}", type="tertiary", disabled=call is None,
                      on_click=_set_call, args=(run.run_id, pid, None))
    if read_only:
        st.caption("Read-only: this is an evidence run.")
        return
    past = DECISIONS[call][1] if call in DECISIONS else "not decided"
    st.caption(f"Your call: {past} · kept in this browser tab until you reload.")
    note_key = f"note-{run_key}-{pid}"
    if call == "reject":
        why_key = f"why-{run_key}-{pid}"
        st.pills("Why reject it?", REJECT_REASONS, key=why_key,
                 default=entry.get("reason") if entry.get("reason") in REJECT_REASONS else None,
                 on_change=_save_reason, args=(run.run_id, pid, why_key))
    if call in ("trim", "reject"):
        st.text_input("What to change (for the video editor)", value=entry.get("note") or "",
                      placeholder="For example: start 2 s later, on the slap", key=note_key,
                      on_change=_save_note, args=(run.run_id, pid, note_key))


def _lines_block(p: dict) -> None:
    hook = p.get("hook") or {}
    cl = p.get("cliffhanger") or {}
    st.markdown(f"**{fmt.md_esc(explain_promo.hook_label(p))}**  \n"
                f"{fmt.md_esc(fmt.said(hook.get('speaker'), hook.get('text') or ''))}")
    label, warn = explain_promo.cliff_label(p)
    st.markdown(f"**{fmt.md_esc(label)}**  \n{fmt.md_esc(fmt.said(cl.get('speaker'), cl.get('text') or ''))}")
    if warn:
        st.badge(warn, color="orange", icon=":material/flag:")
    why = p.get("why_it_works")
    if why:
        if (p.get("cold_viewer") or {}).get("comprehensibility") is not None:
            components.ai_text(why, "Why it works")
        else:
            st.markdown(f"**Why it works (template, no AI)** {fmt.md_esc(why)}")


def _before_posting(p: dict, d: dict) -> None:
    flags_ = checks.before_posting(p, d)
    if flags_:
        with st.container(border=True):
            st.markdown(f":orange[:material/flag:] **Before posting · {len(flags_)}**")
            for f in flags_:
                st.markdown(f"- {fmt.md_esc(f.text)}")
    else:
        st.badge("Nothing to check before posting", color="green", icon=":material/check_circle:")


def _why_here(p: dict, d: dict, parts_: list) -> None:
    promos = d.get("promos") or []
    cfg = d.get("cfg") or {}
    rank = p.get("rank")
    sentence = ""
    if explain_rank.replayable(promos, cfg):
        sentence = explain_rank.why_sentence(p["promo_id"], explain_rank.replay(promos, cfg), promos, parts_)
        if not sentence:
            return
    st.markdown(f"**Why #{rank} here.** " + (sentence or NOT_REPLAYABLE))


def _weak_spots(p: dict, d: dict) -> None:
    ws = checks.weak_spots(p, d)
    if ws:
        st.caption(":material/trending_down: Already priced into the score: " + "; ".join(fmt.md_esc(w) for w in ws))


def _editor_checks(p: dict, d: dict) -> None:
    rub = p.get("rubric") or {}
    known = [k for k in CHECKS if k in rub]
    if not known:
        st.caption("This run predates the editor's checks.")
        return
    cfg = d.get("cfg") or {}
    passed = [k for k in known if rub[k]]
    failed = [k for k in known if not rub[k]]
    rm = (p.get("subscores") or {}).get("rubric_mult")
    label = f"Editor's checks · {fmt.k_of_n(len(passed), len(known))} pass"
    if rm is not None:
        label += f" · score {fmt.mult(rm)}"
    leads = _leads(d.get("arc") or {})
    with st.expander(label):
        for k in failed:
            c = CHECKS[k]
            st.badge(f"{c.fail_label} · {fmt.mult(penalty(cfg, k))}", color="orange", icon=":material/flag:",
                     help=c.rule.format(leads=leads) + " Decided by: " + c.decided_by)
        if passed:
            st.caption("Passes: " + ", ".join(CHECKS[k].pass_label for k in passed))
        st.caption("A failed check lowers the score by a fixed factor; nothing is removed. The editor decides.")


def _viewer(p: dict) -> None:
    cv = p.get("cold_viewer") or {}
    rows = explain_promo.viewer_rows(p)
    if not rows and cv.get("comprehensibility") is None:
        st.caption("First-time viewer (AI): not run in this run (no first-time-viewer test).")
        return
    label = f"First-time viewer (AI) · follows it {explain_promo._n(cv.get('comprehensibility'))}/5"
    if cv.get("would_tap") is not None:
        label += f" · would tap {explain_promo._n(cv.get('would_tap'))}/5"
    with st.expander(label):
        components.ai_text(VIEWER_INTRO)
        st.markdown("  \n".join(f"**{lab}:** {fmt.md_esc(text)}" for lab, text in rows))


def _promise(run_key: str, p: dict, d: dict) -> None:
    pr = p.get("promise") or {}
    with st.expander(f"Promise and payoff · {explain_promo.promise_phrase(p)}"):
        lines = explain_promo.promise_sentences(p, d.get("run_log") or {})
        if pr.get("question") and lines:  # the question is the AI's
            lines = [f"{components.AI} {lines[0]}"] + lines[1:]
        st.markdown("  \n".join(lines))
        if pr.get("answer"):
            if st.toggle("Show the answer (spoiler)", key=f"spoil-{run_key}-{p['promo_id']}"):
                components.ai_text(pr["answer"], "Answer")
        if pr.get("status"):
            st.caption(PROMISE_CAPTION)


def _arithmetic(p: dict, d: dict) -> None:
    with st.expander(f"Score arithmetic · {fmt.score(p.get('score') or 0)}"):
        lines = explain_promo.score_lines(p, d.get("cfg") or {}, d.get("arc") or {})
        if lines is None:
            st.caption(NOT_MULTIPLYING)
            return
        st.markdown("  \n".join(lines[:2]))
        st.markdown("\n".join(lines[2:]))


def _where(p: dict, t: float) -> str:
    """Clip time -> "Ep 10 01:25.00" (segment start + offset)."""
    acc = 0.0
    segs = p.get("segments") or []
    for i, s in enumerate(segs):
        try:
            a, b = float(s["start_s"]), float(s["end_s"])
        except (KeyError, TypeError, ValueError):
            return ""
        if t < acc + (b - a) or i == len(segs) - 1:
            x = a + max(0.0, t - acc)
            m, sec = divmod(x, 60)
            return f"Ep {s.get('episode')} {int(m):02d}:{sec:05.2f}"
        acc += b - a
    return ""


def _transcript(p: dict) -> None:
    lines = [ln for ln in p.get("lines") or [] if isinstance(ln, dict)]
    if not lines:
        return
    with st.expander(f"Transcript · {fmt.plural(len(lines), 'line')}"):
        rows = []
        for ln in lines:
            t = float(ln.get("t") or 0.0)
            kind = ln.get("kind")
            who = ln.get("speaker") or ("Sound" if kind == "sfx" else "On screen")
            rows.append({"Clip time": fmt.secs(t), "Episode · time": _where(p, t), "Who": str(who),
                         "Line": str(ln.get("text") or "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_order=["Clip time", "Episode · time", "Who", "Line"])
        st.caption("Sound cues were not shown to the first-time viewer (AI).")


def _pitch(p: dict) -> None:
    if not p.get("post_first_reason"):
        return
    with st.expander("AI pitch (top 10 only)"):
        components.ai_text(p["post_first_reason"])
        st.caption("Written by the AI after code had set the order. It is not the reason for the position, and it can "
                   "contain mistakes.")


@st.dialog("Review promo", position="right", width="large", on_dismiss=state.close_promo)
def _drawer(run_key: str) -> None:
    run = _find_run(run_key)
    pid = st.session_state.get(state.OPEN)
    if run is None or pid is None:
        st.caption("This promo is no longer available.")
        return
    d = data.load_run(str(run.dir), data.stamp(run.dir))
    promos = sorted(d.get("promos") or [], key=lambda q: q.get("rank") or 0)
    p = next((q for q in promos if q.get("promo_id") == pid), None)
    if p is None:
        st.caption("This promo is no longer available.")
        return
    parts_ = explain_rank.parts(promos, d.get("cfg") or {}, d.get("episodes_list") or [])

    _heading(p, d, parts_)
    _decisions(run, run_key, p, promos)
    st.code(_cut_lines(p), language=None, wrap_lines=True)
    st.caption("Episode time from the script. Copy these into your editor; check each cut against the video.")
    tl = components.timeline_html(p)
    if tl:
        st.html(tl)
        st.caption(components.TIMELINE_LEGEND)
    _lines_block(p)
    _before_posting(p, d)
    _why_here(p, d, parts_)
    _weak_spots(p, d)
    _editor_checks(p, d)
    _viewer(p)
    _promise(run_key, p, d)
    _arithmetic(p, d)
    _transcript(p)
    _pitch(p)
    st.caption(f"Promo id {pid} · run {run.run_id}")


def maybe_open(run: RunInfo, data: dict) -> None:
    """Open the drawer when ss[OPEN] is a promo of this run; otherwise drop ss[OPEN]."""
    pid = st.session_state.get(state.OPEN)
    if pid is None:
        return
    if any(p.get("promo_id") == pid for p in (data or {}).get("promos") or []):
        _drawer(run.key)
    else:
        st.session_state.pop(state.OPEN, None)
