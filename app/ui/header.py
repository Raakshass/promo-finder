"""The frame around every view (spec 2.3 and 6): access gate, header rows 1-3 with The story, Results and About
popovers, the non-official banner and the footer. Owner: E0."""
from __future__ import annotations

import hmac

import streamlit as st

from app.ui import data as data_mod
from app.ui import fmt, state, theme
from app.ui.components import AI
from app.ui.data import RunInfo

DEFAULT_REPO = "https://github.com/Raakshass/promo-finder"
TAGLINE = ("Finds 20–60 s promos in a micro-drama's timestamped scripts. The AI scores the lines; code sets every cut "
           "and the order; an editor makes the final call.")
PICKER_NOTE = ("Official results are the run the solution doc reports. Evidence runs back one measured claim each. "
               "Your runs are ones you started in Run it.")
FOOTER = "Contains excerpts from a confidential script, shown only to review this assignment. Please don't share this link."


def _wordmark_html(title: str = "") -> str:
    """Clapper icon, the wordmark and the escaped show title (the page's only h1)."""
    t = (f"<span style='color:{theme.INK_3}'>·</span>"
         f"<h1 style='font-size:22px;font-weight:700;line-height:1.2;margin:0;padding:0;color:{theme.INK}'>"
         f"{fmt.esc(title)}</h1>") if title else ""
    return (f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-family:{theme.FONT_STACK}'>"
            f"{theme.icon('clapper', 22, theme.BRAND)}"
            f"<span style='font-size:17px;font-weight:600;color:{theme.INK}'>promo-finder</span>{t}</div>")


def gate() -> bool:
    """True when the app may render. With APP_PASSWORD set, draws the wordmark row and the code field until the right
    code is entered (logic unchanged from the previous app)."""
    code = data_mod.secret("APP_PASSWORD")
    if not code or st.session_state.get(state.AUTHED):
        return True
    st.html(_wordmark_html())
    entered = st.text_input("Access code", type="password", help="It is in the submission form.")
    st.caption("The results quote a confidential script, so this demo asks for the code given in the submission form.")
    if entered and hmac.compare_digest(entered.encode(), code.encode()):
        st.session_state[state.AUTHED] = True
        st.rerun()
    elif entered:
        st.error("That code doesn't match. Check the submission form.")
    return False


def _names(names: list) -> str:
    xs = [str(n).strip().title() for n in names or [] if str(n).strip()]
    return xs[0] if len(xs) == 1 else (", ".join(xs[:-1]) + " and " + xs[-1] if xs else "")


def _current(runs: list[RunInfo]) -> RunInfo:
    """The run the radio will select: Session State, else a valid ?run= key, else the first (official) run.
    Runs before the radio exists, so assigning its key here is allowed.

    ?run= is synced by hand, not with bind="query-params": a bound radio writes its formatted label to the URL
    ("?run=Evidence · No AI: keyword rules") and rejects the run key, so the URL could not carry ?run=keyword-rules."""
    by_key = {r.key: r for r in runs}
    ss = st.session_state
    if state.RUN not in ss and st.query_params.get(state.RUN) in by_key:
        ss[state.RUN] = st.query_params[state.RUN]
    if state.RUN in ss and ss[state.RUN] not in by_key:
        ss[state.RUN] = runs[0].key  # e.g. an upload lost in a restart
    return by_key.get(ss.get(state.RUN), runs[0])


def _sync_url(runs: list[RunInfo], key: str) -> None:
    """?run=<key>, removed when the first (official) run is selected or the key is unknown."""
    if key == runs[0].key or key not in {r.key for r in runs}:
        if state.RUN in st.query_params:
            del st.query_params[state.RUN]
    elif st.query_params.get(state.RUN) != key:
        st.query_params[state.RUN] = key


def run_line(run: RunInfo) -> str:
    """Header row 2: "{label} · {E} episodes · scored by {model}{ (free tier)} · run {d Mon yyyy, HH:MM}"."""
    if run.model == "none":
        scored = "no AI (keyword rules)"
    else:
        scored = f"scored by {fmt.model_short(run.model)}" + (" (free tier)" if fmt.is_free(run.model) else "")
    when = fmt.run_time(run.run_id)
    lead = {"upload": f"Your run · {run.show_title}", "local": f"Local run · {run.show_title}"}.get(run.role, run.label)
    parts = [lead, fmt.plural(run.episodes, "episode"), scored] + ([f"run {when}"] if when else [])
    return " · ".join(parts)


def _story(d: dict) -> None:
    arc = d.get("arc") or {}
    if not arc.get("premise"):
        return
    with st.popover("The story", icon=":material/menu_book:"):
        st.markdown("**The story so far**")
        st.markdown(fmt.md_esc(arc["premise"]))
        bits = []
        if _names(arc.get("leads")):
            bits.append(f"Leads: {_names(arc.get('leads'))}.")
        if _names(arc.get("antagonists")):
            bits.append(f"Antagonists: {_names(arc.get('antagonists'))}.")
        if bits:
            st.markdown(fmt.md_esc(" ".join(bits)))
        st.caption(f"{AI} Written by the AI from the writers' episode notes. It can contain mistakes.")


def _results(runs: list[RunInfo], current: RunInfo) -> str:
    labels = {r.key: r.label for r in runs}
    with st.popover(f"Results: {current.short}", icon=":material/swap_horiz:"):
        key = st.radio("Results shown", [r.key for r in runs], format_func=lambda k: labels.get(k, k),
                       captions=[r.caption for r in runs], key=state.RUN, label_visibility="collapsed")
        st.caption(PICKER_NOTE)
    _sync_url(runs, key)
    return key


def _about(run: RunInfo | None, d: dict) -> None:
    repo = (data_mod.secret("REPO_URL") or DEFAULT_REPO).rstrip("/")
    with st.popover("About", icon=":material/info:"):
        st.markdown("promo-finder reads a micro-drama's timestamped scripts and finds 20–60 s promos: where to cut, the "
                    "hook, the cliffhanger and why each works. Built for Kuku's AI Generalist assignment, Problem 1.")
        st.link_button("Code on GitHub", repo, icon=":material/code:", width="stretch")
        st.link_button("Method and sources", f"{repo}/blob/main/docs/research.md", icon=":material/menu_book:", width="stretch")
        st.link_button("Assumptions", f"{repo}/blob/main/docs/assumptions.md", icon=":material/rule:", width="stretch")
        doc = data_mod.secret("DOC_URL")
        if doc:
            st.link_button("Solution doc", doc, icon=":material/description:", width="stretch")
        raw = data_mod.secret("RAW_OUTPUTS_URL")
        if raw:
            st.link_button("Raw outputs", raw, icon=":material/folder_zip:", width="stretch")
        if run is not None:
            promos = d.get("promos") or []
            pv = ((promos[0].get("provenance") or {}).get("prompt_version") if promos else None) \
                or ((d.get("metrics") or {}).get("model") or {}).get("prompt_version")
            st.caption(" · ".join(([f"Prompt version {pv}"] if pv else []) + [f"run id {run.run_id}"]))


def render_header(runs: list[RunInfo]) -> RunInfo | None:
    """Draws rows 1-3 and the popovers; returns the selected run (None when there are no runs)."""
    current = _current(runs) if runs else None
    d: dict = {}
    if current is not None:
        try:
            d = data_mod.load_run(str(current.dir), data_mod.stamp(current.dir))
        except data_mod.RunReadError:
            d = {}  # streamlit_app shows the error; the picker stays usable
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(_wordmark_html(current.show_title if current else ""), width="content")
        st.space("stretch")
        selected = current
        if current is not None:
            _story(d)
            key = _results(runs, current)
            selected = next((r for r in runs if r.key == key), current)
        _about(selected, d if selected == current else {})
    if selected is not None:
        st.caption(run_line(selected))
    st.caption(TAGLINE)
    return selected


def render_banner(run: RunInfo) -> None:
    """Non-official runs only."""
    if run.role == "evidence":
        st.warning(f"**You are viewing an evidence run, not the official results:** {run.label}. {run.caption}",
                   icon=":material/science:")
    elif run.role == "upload":
        st.warning(f"**You are viewing your own run:** {run.label}. It stays on this demo server until the app restarts, "
                   "so download what you need.", icon=":material/upload_file:")
    elif run.role == "local":
        st.warning(f"**You are viewing a local run, not the official results:** {run.label}. {run.caption}",
                   icon=":material/science:")
        return
    else:
        return
    st.button("Back to the official results", icon=":material/arrow_back:", on_click=state.switch_run,
              args=("official",), key="banner-back")


def render_footer() -> None:
    st.caption(FOOTER)
