"""Shared renderers (spec 4.5). Owner: Lane P.

E0 implemented the one-line helpers every lane needs (AI badge, view_header, ai_text, help_src, empty_state, not_measured,
call_badge, open_button). Lane P builds timeline_html. read_along is cut from P0: keep the stub, don't call it.
"""
from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from app.ui import fmt, state, theme
from app.ui.glossary import DECISIONS

AI = ":violet-badge[:material/smart_toy: AI]"


def view_header(title: str, purpose: str) -> None:
    st.header(title, anchor=False)
    st.caption(purpose)


def ai_text(text: str, label: str = "") -> None:
    """AI-written text with the violet badge. Markdown specials in `text` are escaped."""
    st.markdown(f"{AI} **{label}** {fmt.md_esc(text)}" if label else f"{AI} {fmt.md_esc(text)}")


def help_src(text: str, source: str) -> str:
    """For help= arguments: the explanation, a blank line, then its source."""
    return f"{text}\n\nSource: {source}"


def empty_state(title: str, body: str, icon: str = ":material/info:",
                action: tuple[str, Callable, tuple] | None = None) -> None:
    """A bordered container: icon + bold title + body + an optional button (label, on_click, args)."""
    with st.container(border=True):
        st.markdown(f"{icon} **{title}**")
        st.caption(body)
        if action:
            label, cb, args = action
            st.button(label, on_click=cb, args=args, key=f"empty-{title}-{label}")


def not_measured(what: str, why: str = "") -> None:
    st.caption(f"Not measured in this run: {what}." + (f" {why}" if why else ""))


def call_badge(call: str | None) -> str:
    """":green-badge[:material/check_circle: Approved]", or "" when not decided."""
    if call not in DECISIONS:
        return ""
    _button, past, colour, icon, _key = DECISIONS[call]
    return f":{colour}-badge[{icon} {past}]"


TIMELINE_LEGEND = ("Pink band: the first 3 s, where the hook must land. Red line: the hook line starts. Blue line: the "
                   "cliffhanger line ends. Dashed line: the join between episodes.")
_TL_WIDTH_PX = 640      # assumed drawer width, only to estimate label overlap
_TL_CHAR_PX = 7         # estimated width of one 12 px character
_TL_HOOK_WINDOW_S = 3.0  # the brief's hook window


def _tl_float(x: object) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _tl_label(text: str, pct: float, top: int, colour: str, placed: list[tuple[float, float]], title: str) -> str:
    """One text label centred on pct (clamped to the bar), or "" when it would overlap a label already placed."""
    w = len(text) * _TL_CHAR_PX
    x = pct / 100 * _TL_WIDTH_PX
    if x - w / 2 < 0:
        lo, shift = x, "0"
    elif x + w / 2 > _TL_WIDTH_PX:
        lo, shift = x - w, "-100%"
    else:
        lo, shift = x - w / 2, "-50%"
    hi = lo + w
    if any(lo < b + 6 and hi > a - 6 for a, b in placed):
        return ""
    placed.append((lo, hi))
    return (f'<span title="{fmt.esc(title)}" style="position:absolute;top:{top}px;left:{pct:.3f}%;'
            f'transform:translateX({shift});white-space:nowrap;color:{colour}">{fmt.esc(text)}</span>')


def timeline_html(p: dict) -> str:
    """Spec 5.3 "Timeline": plain HTML, full width, 64 px tall. The bar shows each segment, the 3 s hook band, the hook
    and cliffhanger markers and the joins between episodes. Labels that would overlap are dropped."""
    segs_ = [s for s in p.get("segments") or [] if isinstance(s, dict)]
    lens = []
    for s in segs_:
        a, b = _tl_float(s.get("start_s")), _tl_float(s.get("end_s"))
        lens.append(max(0.0, b - a) if a is not None and b is not None else 0.0)
    dur = _tl_float(p.get("duration_s")) or sum(lens)
    if not dur or dur <= 0:
        return ""

    def pct(t: float) -> float:
        return max(0.0, min(100.0, t / dur * 100))

    bar_top, bar_h = 22, 14
    label_below = bar_top + bar_h + 6
    aria = fmt.esc(f"Timeline of promo #{p.get('rank', '')}")
    out = [f'<div class="pf-num" role="img" aria-label="{aria}" style="position:relative;height:64px;width:100%;'
           f'font-family:{fmt.esc(theme.FONT_STACK)};font-size:12px;line-height:14px;color:{theme.INK}">']
    t0 = 0.0
    for i, (s, ln) in enumerate(zip(segs_, lens)):  # segments, alternating greys
        if ln > 0:
            colour = "#e5e7eb" if i % 2 == 0 else "#d1d5db"
            out.append(f'<div title="{fmt.esc(fmt.seg(s))}" style="position:absolute;top:{bar_top}px;'
                       f'height:{bar_h}px;left:{pct(t0):.3f}%;width:{pct(t0 + ln) - pct(t0):.3f}%;'
                       f'background:{colour}"></div>')
        t0 += ln
    out.append(f'<div title="The first {fmt.secs(_TL_HOOK_WINDOW_S)}, where the hook must land" '
               f'style="position:absolute;top:{bar_top}px;height:{bar_h}px;left:0;'
               f'width:{pct(_TL_HOOK_WINDOW_S):.3f}%;background:{theme.HOOK_1};opacity:.45"></div>')
    above: list[tuple[float, float]] = []
    below: list[tuple[float, float]] = []
    labels: list[str] = []

    def marker(t: float, colour: str, title: str, dashed: bool = False) -> str:
        style = f"border-left:1px dashed {theme.INK}" if dashed else f"background:{colour};width:2px"
        left = f"calc({pct(t):.3f}% - {0 if dashed else 1}px)"
        return (f'<div title="{fmt.esc(title)}" style="position:absolute;top:{bar_top - 4}px;'
                f'height:{bar_h + 8}px;left:{left};{style}"></div>')

    hook_at = _tl_float((p.get("hook") or {}).get("at_s"))
    cliff_at = _tl_float((p.get("cliffhanger") or {}).get("at_s"))
    if hook_at is not None:
        title = f"Hook line starts {fmt.secs(hook_at)} into the clip"
        out.append(marker(hook_at, theme.HOOK_3, title))
        labels.append(_tl_label(f"Hook {fmt.secs(hook_at)}", pct(hook_at), 2, theme.HOOK_TEXT, above, title))
    if cliff_at is not None:
        title = f"Cliffhanger line ends {fmt.secs(cliff_at)} into the clip"
        out.append(marker(min(cliff_at, dur), theme.CLIFF_3, title))
        labels.append(_tl_label(f"Cliffhanger {fmt.secs(cliff_at)}", pct(cliff_at), 2, theme.CLIFF_TEXT, above, title))
    labels.append(_tl_label("0 s", 0.0, label_below, theme.INK_2, below, "Start of the clip"))
    labels.append(_tl_label(f"{dur:.0f} s", 100.0, label_below, theme.INK_2, below, f"End of the clip: {fmt.secs(dur)}"))
    t0 = 0.0
    for i, (s, ln) in enumerate(zip(segs_, lens)):  # joins between episodes
        if i > 0:
            title = f"Join: episode {s.get('episode')} starts here, {fmt.secs(t0)} into the clip"
            out.append(marker(t0, theme.INK, title, dashed=True))
            labels.append(_tl_label(f"Ep {s.get('episode')} starts", pct(t0), label_below, theme.INK_2, below, title))
        t0 += ln
    out += [lb for lb in labels if lb]
    out.append("</div>")
    return "".join(out)


def read_along(p: dict, height: int = 600) -> None:
    """Cut from P0. Do not call."""
    return None


def open_button(pid: str, key: str, label: str = "Open") -> None:
    st.button(label, key=key, type="tertiary", icon=":material/open_in_new:", on_click=state.open_promo, args=(pid,))
