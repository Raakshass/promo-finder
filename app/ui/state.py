"""Session keys, navigation callbacks and the review store (spec 2.4 and 4.4).

Navigation rule: a widget's key can't be assigned after the widget exists in the same run. Callbacks (on_click) may set
VIEW and RUN directly. Code running inside the script after the tabs exist calls goto_later(), which parks the target in
_goto_* and reruns; apply_pending_nav() moves it into place at the top of the next run, before the widgets exist.
"""
from __future__ import annotations

import re
import secrets as _secrets
from datetime import datetime

import streamlit as st

VIEW, RUN, OPEN, REVIEW, SPACE = "view", "run", "open_pid", "review", "space"
GOTO_VIEW, GOTO_RUN = "_goto_view", "_goto_run"
AUTHED = "authed"
FETCH_FAILED_AT, FETCH_ERROR = "_fetch_failed_at", "_fetch_error"

CALLS = ("approve", "trim", "reject")
_SPACE_RE = re.compile(r"[0-9a-f]{8}")


def init() -> None:
    """setdefault for the non-widget keys. Widget keys (view, run, promos_list, promos_filter, map_sel) are left to
    their widgets; open_pid, _goto_* and _fetch_* are absent until set."""
    ss = st.session_state
    ss.setdefault(REVIEW, {})
    ss.setdefault(AUTHED, False)


def apply_pending_nav() -> None:
    """Call first, before st.tabs and the run radio exist."""
    ss = st.session_state
    view = ss.pop(GOTO_VIEW, None)
    run = ss.pop(GOTO_RUN, None)
    if run is not None:
        ss[RUN] = run
        ss.pop(OPEN, None)
    if view is not None:
        ss[VIEW] = view


def goto(view: str) -> None:
    """on_click callback: open a tab."""
    st.session_state[VIEW] = view


def goto_later(view: str, run_key: str | None = None) -> None:
    """In-script navigation (after the tabs exist): park the target, then rerun. Does not return."""
    st.session_state[GOTO_VIEW] = view
    if run_key is not None:
        st.session_state[GOTO_RUN] = run_key
    st.rerun()


def switch_run(key: str) -> None:
    """on_click callback: show another run and close the drawer."""
    st.session_state[RUN] = key
    st.session_state.pop(OPEN, None)


def open_promo(pid: str) -> None:
    """on_click callback: open the drawer on a promo."""
    st.session_state[OPEN] = pid


def close_promo() -> None:
    """The drawer's on_dismiss."""
    st.session_state.pop(OPEN, None)


def space(create: bool = False) -> str | None:
    """The ?space= value (8 hex characters), or None. Kept for the frozen API; upload scoping is cut from P0 and
    list_runs ignores it."""
    v = st.query_params.get(SPACE)
    if v and _SPACE_RE.fullmatch(v):
        st.session_state[SPACE] = v
        return v
    if create:
        v = _secrets.token_hex(4)
        st.query_params[SPACE] = v
        st.session_state[SPACE] = v
        return v
    return None


def _runs_store() -> dict:
    store = st.session_state.get(REVIEW)
    if not isinstance(store, dict):
        store = {}
        st.session_state[REVIEW] = store
    return store


def review_get(run_id: str, pid: str) -> dict | None:
    """{"call", "note", "reason", "at"} for one promo, or None when not decided."""
    entry = _runs_store().get(run_id, {}).get(pid)
    return dict(entry) if entry else None


def review_set(run_id: str, pid: str, call: str | None, note: str | None = None, reason: str | None = None) -> None:
    """Record a decision. call None removes the entry. note/reason None keep the stored value ("" when new); the
    reason is cleared when the call is not "reject". Stamps "at" with the local time in ISO 8601."""
    store = _runs_store()
    decisions = store.setdefault(run_id, {})
    if call is None:
        decisions.pop(pid, None)
        return
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS} or None, got {call!r}")
    old = decisions.get(pid) or {}
    decisions[pid] = {
        "call": call,
        "note": note if note is not None else old.get("note", ""),
        "reason": (reason if reason is not None else old.get("reason", "")) if call == "reject" else "",
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def review_all(run_id: str) -> dict[str, dict]:
    """Every decision for one run, keyed by promo id (a copy)."""
    return {pid: dict(e) for pid, e in _runs_store().get(run_id, {}).items()}


def review_counts(run_id: str, pids: list[str]) -> dict[str, int]:
    """Counts over the given promos. Keys: decided, approve, trim, reject, open."""
    decisions = _runs_store().get(run_id, {})
    counts = {"decided": 0, "approve": 0, "trim": 0, "reject": 0, "open": 0}
    for pid in pids:
        call = (decisions.get(pid) or {}).get("call")
        if call in CALLS:
            counts[call] += 1
            counts["decided"] += 1
        else:
            counts["open"] += 1
    return counts
