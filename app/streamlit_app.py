"""promo-finder: the Streamlit app (Community Cloud main module).

Order on the page (spec 2.3): access gate, header, banner (non-official runs), tabs bound to ?view=, the open view,
footer, then the promo drawer when one is open. Only the open tab's view runs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # guard: the script reruns on every interaction
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui import components, data, dialog, header, state, theme  # noqa: E402
from app.views import method, promos, quality, ranking, run_it  # noqa: E402

VIEWS = ["Promos", "Ranking", "Quality", "Method", "Run it"]
RENDER = {"Promos": promos.render, "Ranking": ranking.render, "Quality": quality.render, "Method": method.render,
          "Run it": run_it.render}
FETCH_BACKOFF_S = 60


def _retry_fetch() -> None:
    st.session_state.pop(state.FETCH_FAILED_AT, None)
    st.session_state.pop(state.FETCH_ERROR, None)


def _fetch() -> None:
    """Pull the official results from OUTPUTS_URL once per container; remember a failure for 60 s."""
    ss = st.session_state
    if not data.secret("OUTPUTS_URL"):
        return
    failed_at = ss.get(state.FETCH_FAILED_AT)
    if failed_at is None or time.time() - failed_at >= FETCH_BACKOFF_S:
        with st.spinner("Loading the official results (about 10 seconds on the first visit)…"):
            err = data.fetch_outputs()
        if err:
            ss[state.FETCH_FAILED_AT] = time.time()
            ss[state.FETCH_ERROR] = err
        else:
            _retry_fetch()
    if ss.get(state.FETCH_FAILED_AT) is not None:
        st.error("The download link didn't respond. Try again in a minute, or run the tool on a show of your own in Run it.",
                 title="The official results couldn't be loaded", icon=":material/cloud_off:")
        st.button("Try again", on_click=_retry_fetch, key="fetch-retry")
        with st.expander("Technical details"):
            st.code(str(ss.get(state.FETCH_ERROR) or ""), language=None, wrap_lines=True)


st.set_page_config(page_title="promo-finder", page_icon=":material/movie:", layout="wide",
                   initial_sidebar_state="collapsed")
st.html(theme.CSS)
state.init()
state.apply_pending_nav()

if not header.gate():
    st.stop()

_fetch()

runs = data.list_runs(state.space())
run = header.render_header(runs)

d: dict = {}
loaded = False
if run is not None:
    try:
        d = data.load_run(str(run.dir), data.stamp(run.dir))
        loaded = True
    except data.RunReadError as e:
        st.error(f"This run's files could not be read: {e.file}.", icon=":material/error:")
        with st.expander("Technical details"):
            st.code(str(e), language=None, wrap_lines=True)
    header.render_banner(run)

tabs = st.tabs(VIEWS, key=state.VIEW, bind="query-params", default="Promos")
for label, tab in zip(VIEWS, tabs):
    if not tab.open:
        continue
    with tab:
        if label == "Run it":
            run_it.render(run, d if loaded else {}, runs)
        elif run is None:
            components.empty_state("No results to show yet", "Run the tool on a show in Run it.",
                                   action=("Go to Run it", state.goto, ("Run it",)))
        elif loaded:
            RENDER[label](run, d, runs)

header.render_footer()
if run is not None and loaded:
    dialog.maybe_open(run, d)
