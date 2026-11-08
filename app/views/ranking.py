"""Ranking view (spec 5.4). Owner: Lane R.

The rule box, the show map and one reason row per promo. Every number comes from the run's files and config.yaml.
"""
from __future__ import annotations

import streamlit as st

from app.ui import explain_rank, fmt, showmap
from app.ui.components import empty_state, open_button, view_header
from app.ui.data import RunInfo

TITLE = "Which to post first, and why"
PURPOSE = "Each promo's number (#1, #2, …) is its posting order. Here is the rule, the show map it follows, and the reason for every position."
NOT_REPLAYABLE = ("This run was made by an earlier version that ordered promos differently, so the order can't be "
                  "explained here.")
LIST_CAPTION = ("Score: 0 to 1, built from the hook, the cliffhanger, how well a stranger follows and pace, × a penalty "
                "per failed check (open a promo for its arithmetic). Counted as: the score after the variety rule, at "
                "the moment the promo was placed. The order is chosen among the {short} shortlisted clips; only the {n} "
                "that made the slate are named here.")


def _ep(p: dict):
    e = (p.get("flags") or {}).get("episode")
    if isinstance(e, int):
        return e
    segs = p.get("segments") or []
    return segs[0].get("episode") if segs else None


def _reasons(promos: list[dict], data: dict, parts_: list) -> None:
    placements = explain_rank.replay(promos, data.get("cfg") or {})
    by_rank = sorted((p for p in promos if isinstance(p.get("rank"), int)), key=lambda p: p["rank"])
    for part in parts_:
        st.markdown(f"##### {part.name} · {fmt.ep_range(part.first_ep, part.last_ep)} · {fmt.plural(part.n, 'place')}")
        for p in by_rank:
            if (p.get("flags") or {}).get("block") != part.key:
                continue
            pid = p["promo_id"]
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**#{p['rank']}** · Ep {_ep(p)} · {fmt.score(p.get('score') or 0)}", width=170)
                st.markdown(explain_rank.why_sentence(pid, placements, promos, parts_), width="stretch")
                open_button(pid, key=f"rk-{p['rank']}")  # rank, not the promo id: widget ids are scanned
    metrics = data.get("metrics") or {}
    short = metrics.get("finalists") or ((data.get("cfg") or {}).get("judge") or {}).get("top_n")
    cap = LIST_CAPTION.format(short=short, n=len(promos)) if isinstance(short, int) else \
        LIST_CAPTION.replace(" {short}", "").format(n=len(promos))
    st.caption(cap)


def render(run: RunInfo | None, data: dict, runs: list[RunInfo]) -> None:
    """Draw the view for the selected run. Never mutate `data`. Never write to st.sidebar."""
    view_header(TITLE, PURPOSE)
    if run is None:
        empty_state("No results to show yet", "Run the tool on a show in Run it.")
        return
    promos = data.get("promos") or []
    cfg = data.get("cfg") or {}
    parts_ = explain_rank.parts(promos, cfg, data.get("episodes_list") or [])
    ok = explain_rank.replayable(promos, cfg)
    with st.container(border=True):
        if ok:
            st.markdown("\n\n".join(explain_rank.rule_lines(data, parts_)))
        else:
            st.markdown(NOT_REPLAYABLE)
    st.space("medium")
    st.subheader("Show map: where the promos come from", anchor=False)
    showmap.render_map(run, data, parts_)
    if ok and promos:
        st.space("medium")
        st.subheader("Why each promo sits where it does", anchor=False)
        _reasons(promos, data, parts_)
