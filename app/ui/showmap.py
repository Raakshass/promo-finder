"""The show map: arrays, figure, legend and takeaways (spec 5.4). Owner: Lane R.

The selected-promo panel is cut from P0: hover plus the reasons list's Open buttons are enough, so the chart is drawn
without on_select. Every number in the takeaways is computed from the run's files.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import plotly.graph_objects as go
import streamlit as st

from app.ui import checks, fmt, theme
from app.ui import data as D
from app.ui.components import empty_state
from app.ui.data import RunInfo
from app.ui.glossary import ROLE_WORDS

_HOOK = (theme.ZERO, theme.HOOK_1, theme.HOOK_2, theme.HOOK_3)
_CLIFF = (theme.ZERO, theme.CLIFF_1, theme.CLIFF_2, theme.CLIFF_3)
_FONT = 12
_MUTED = theme.INK_2


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _dec1(x: float) -> str:
    return f"{Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-1), rounding=ROUND_HALF_UP)}"


def _sc(x: float) -> str:
    """A 0-3 score: "2" for whole numbers, the exact value when --votes produced fractions ("2.5")."""
    x = float(x or 0)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _scale(cols: tuple[str, str, str, str]) -> list:
    """Stepped scale: each colour holds a quarter of the range (spec 5.4)."""
    n = len(cols)
    return [[b, c] for i, c in enumerate(cols) for b in (i / n, (i + 1) / n)]


def _who(b: dict) -> str:
    if b.get("speaker"):
        return str(b["speaker"])
    return "Sound" if str(b.get("kind") or "") in ("sfx", "sound") else "On screen"


def _beats(data: dict) -> dict[int, dict]:
    out = {}
    for k, v in _d(data.get("beats")).items():
        try:
            out[int(k)] = _d(v)
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def _promo_spans(promos: list[dict]) -> dict[int, list[tuple[float, float, int]]]:
    spans: dict[int, list[tuple[float, float, int]]] = {}
    for p in promos:
        for s in _l(p.get("segments")):
            if isinstance(s.get("episode"), int) and isinstance(s.get("start_s"), (int, float)):
                spans.setdefault(s["episode"], []).append((float(s["start_s"]), float(s["end_s"]), int(p["rank"])))
    return spans


def build_arrays(data: dict) -> dict:
    """The uncached work behind map_arrays, from a load_run dict."""
    beats = _beats(data)
    if not beats:
        return {}
    eps = list(beats)
    durations = [float(beats[e].get("duration_s") or 0) or
                 max((float(b.get("t1") or 0) for b in _l(beats[e].get("beats"))), default=0.0) for e in eps]
    max_s = int(math.ceil(max(durations, default=0)))
    width = max(max_s, 1)
    spans = _promo_spans(_l(data.get("promos")))
    z_hook = [[None] * width for _ in range(2 * len(eps))]
    z_cliff = [[None] * width for _ in range(2 * len(eps))]
    hv_hook = [[""] * width for _ in range(2 * len(eps))]
    hv_cliff = [[""] * width for _ in range(2 * len(eps))]
    for i, (ep, dur) in enumerate(zip(eps, durations)):
        cells = int(math.ceil(dur))
        best_h: list[dict | None] = [None] * cells
        best_c: list[dict | None] = [None] * cells
        for b in _l(beats[ep].get("beats")):
            t0, t1 = float(b.get("t0") or 0), float(b.get("t1") or 0)
            for t in range(max(int(math.floor(t0)), 0), min(int(math.ceil(max(t1, t0 + 1e-6))), cells)):
                if best_h[t] is None or float(b.get("hook") or 0) > float(best_h[t].get("hook") or 0):
                    best_h[t] = b
                if best_c[t] is None or float(b.get("cliff") or 0) > float(best_c[t].get("cliff") or 0):
                    best_c[t] = b
        for t in range(cells):
            ranks = sorted({r for a, z, r in spans.get(ep, []) if a <= t + 0.5 < z})
            promo = ""
            if ranks:
                promo = (" · in promo #" if len(ranks) == 1 else " · in promos ") + \
                    (f"{ranks[0]}" if len(ranks) == 1 else ", ".join(f"#{r}" for r in ranks))
            for row, best, z, hv, key in ((2 * i, best_h, z_hook, hv_hook, "hook"),
                                          (2 * i + 1, best_c, z_cliff, hv_cliff, "cliff")):
                b = best[t]
                if b is None:
                    z[row][t] = 0
                    hv[row][t] = f"Ep {ep} · {fmt.mmss(t)} · no line"
                    continue
                z[row][t] = round(float(b.get(key) or 0))
                role = ROLE_WORDS.get(str(b.get("role") or ""))
                hv[row][t] = (f"Ep {ep} · {fmt.clock(float(b.get('t0') or 0))}–{fmt.clock(float(b.get('t1') or 0))} · "
                              f"{_who(b)} · hook {_sc(b.get('hook'))}/3 · cliffhanger {_sc(b.get('cliff'))}/3"
                              f"{f' · {role}' if role else ''}{promo}")
    return {"z_hook": z_hook, "z_cliff": z_cliff, "hover_hook": hv_hook, "hover_cliff": hv_cliff, "eps": eps,
            "durations": durations, "max_s": max_s}


@st.cache_data(show_spinner=False, max_entries=4)
def map_arrays(run_dir: str, stamp: float) -> dict:
    """z_hook, z_cliff, hover_hook, hover_cliff, eps, durations, max_s. Rows 2i (hook) and 2i+1 (cliffhanger) per
    episode; one column per second; None after the episode ends."""
    return build_arrays(D.load_run(run_dir, stamp))


def _flat_notes(data: dict, eps: list[int]) -> dict[int, str]:
    notes: dict[int, str] = {}
    for r in checks.flat_episodes(data):
        if r["still_flat"]:
            notes[r["episode"]] = ("AI scored every line 0, twice" if r["all_zero"]
                                   else "no line strong enough to open a promo, after asking twice")
    for e in _l(_d(data.get("promoability")).get("episodes")):
        ep = e.get("episode")
        if isinstance(ep, int) and ep in eps and ep not in notes and e.get("windows") == 0:
            notes[ep] = "no 20–60 s clip fits the rules"
    return notes


def _checks_phrase(p: dict) -> str:
    rub = {k: v for k, v in _d(p.get("rubric")).items() if isinstance(v, bool)}
    if not rub:
        return ""
    fails = sum(1 for v in rub.values() if v is False)
    return f" · passes all {len(rub)} checks" if not fails else f" · fails {fmt.k_of_n(fails, len(rub))}"


def figure(data: dict, parts_: list, arrays: dict | None = None) -> go.Figure:
    """The map figure (spec 5.4). `arrays` is map_arrays' result; built from `data` when not given."""
    a = arrays if arrays else build_arrays(data)
    fig = go.Figure()
    if not a:
        return fig
    eps, max_s = a["eps"], a["max_s"]
    width = len(a["z_hook"][0]) if a["z_hook"] else 1
    xs = [t + 0.5 for t in range(width)]
    ys = list(range(2 * len(eps)))
    for z, hv, cols in ((a["z_hook"], a["hover_hook"], _HOOK), (a["z_cliff"], a["hover_cliff"], _CLIFF)):
        fig.add_trace(go.Heatmap(z=z, x=xs, y=ys, text=hv, hovertemplate="%{text}<extra></extra>", zmin=-0.5,
                                 zmax=3.5, colorscale=_scale(cols), showscale=False, xgap=0, ygap=1, hoverongaps=False))
    row_of = {e: i for i, e in enumerate(eps)}
    promos = sorted((p for p in _l(data.get("promos")) if isinstance(p.get("rank"), int)), key=lambda p: p["rank"])
    bx, by, btxt, bfill, bfont, bhover = [], [], [], [], [], []
    for p in promos:
        top = p["rank"] <= 10
        segs = [s for s in _l(p.get("segments")) if s.get("episode") in row_of]
        hover = (f"#{p['rank']} · {' + '.join(fmt.segs(p))} · {fmt.secs(p.get('duration_s') or 0)}"
                 f"{_checks_phrase(p)}")
        for s in segs:
            i = row_of[s["episode"]]
            fig.add_shape(type="rect", x0=float(s["start_s"]), x1=float(s["end_s"]), y0=2 * i - 0.5, y1=2 * i + 1.5,
                          line=dict(color=theme.INK, width=2 if top else 1), fillcolor="rgba(0,0,0,0)", layer="above")
            bx.append(float(s["start_s"]))
            by.append(2 * i + 0.5)
            btxt.append(fmt.num(p["rank"]))
            bfill.append(theme.INK if top else "#ffffff")
            bfont.append("#ffffff" if top else theme.INK)
            bhover.append(hover)
    if bx:
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers+text", text=btxt, hovertext=bhover,
                                 hovertemplate="%{hovertext}<extra></extra>", textposition="middle center",
                                 textfont=dict(size=_FONT, color=bfont, family=theme.FONT_STACK, weight="bold"),
                                 marker=dict(symbol="square", size=18, color=bfill,
                                             line=dict(color=theme.INK, width=1)),
                                 showlegend=False, cliponaxis=False))
    arc = _d(data.get("arc"))
    turns = {t.get("episode") for t in _l(arc.get("turning_points")) if isinstance(t, dict)}
    reveals = {r.get("episode") for r in _l(arc.get("major_reveals")) if isinstance(r, dict)}
    ticks = [f"Ep {e}" + (" · turn" if e in turns else "") + (" · reveal" if e in reveals else "") for e in eps]
    if len(eps) >= 3:
        for k, part in enumerate(parts_):
            rows = [row_of[e] for e in eps if part.first_ep <= e <= part.last_ep]
            if not rows:
                continue
            if k < len(parts_) - 1:
                y = 2 * max(rows) + 1.5
                fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=y, y1=y,
                              line=dict(color=theme.INK, width=1, dash="dot"))
            fig.add_annotation(xref="paper", x=1.01, y=2 * min(rows) - 0.5, xanchor="left", yanchor="top",
                               showarrow=False, align="left", font=dict(size=_FONT, color=theme.INK),
                               text=f"{part.name} · {fmt.ep_range(part.first_ep, part.last_ep)}<br>"
                                    f"{fmt.plural(part.n, 'promo')}")
    for ep, note in _flat_notes(data, eps).items():
        fig.add_annotation(x=4, y=2 * row_of[ep] + 0.5, xanchor="left", showarrow=False, text=note,
                           font=dict(size=_FONT, color=_MUTED))
    step = 30
    tick_x = list(range(0, max_s + 1, step))
    fig.update_layout(
        height=18 * len(eps) + 120, margin=dict(l=90, r=200, t=60, b=20), dragmode=False,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", showlegend=False,
        font=dict(family=theme.FONT_STACK, size=_FONT, color=theme.INK),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=theme.LINE, font=dict(size=13, color=theme.INK,
                                                                               family=theme.FONT_STACK)),
        xaxis=dict(side="top", range=[-0.035 * max(max_s, 1), max(max_s, 1)],  # a little room so a badge at 0:00 clears the episode labels
                   tickvals=tick_x, ticktext=[fmt.mmss(t) for t in tick_x],
                   title=dict(text="Time into the episode (min:sec)", font=dict(size=_FONT)),
                   tickfont=dict(size=_FONT), fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickvals=[2 * i + 0.5 for i in range(len(eps))], ticktext=ticks,
                   tickfont=dict(size=_FONT), fixedrange=True, showgrid=False, zeroline=False),
    )
    return fig


def legend_html() -> str:
    """One row of inline swatches; the words are in ink, never in colour."""
    def sw(colors, border=False):
        return "".join(f"<span style='display:inline-block;width:12px;height:12px;background:{c};"
                       f"{'border:1px solid ' + theme.LINE + ';' if border else ''}margin-right:1px;"
                       "vertical-align:-1px'></span>" for c in colors)
    badge_filled = (f"<span style='display:inline-block;min-width:16px;height:16px;line-height:16px;text-align:center;"
                    f"background:{theme.INK};color:#fff;font-size:12px;font-weight:700;margin-right:2px'>1</span>")
    badge_open = (f"<span style='display:inline-block;min-width:14px;height:14px;line-height:14px;text-align:center;"
                  f"background:#fff;color:{theme.INK};border:1px solid {theme.INK};font-size:12px;font-weight:700;"
                  "margin-right:4px'>12</span>")
    items = [
        (sw(_HOOK[1:]), "Top lane: how well a promo could open on this line (AI hook score 1–3; darker is stronger)."),
        (sw(_CLIFF[1:]), "Bottom lane: how well a promo could end on it (AI cliffhanger score 1–3)."),
        (sw([theme.ZERO]) + sw(["#ffffff"], border=True),
         "Grey: inside the episode, nothing scored 1 or more. White: the episode has ended."),
        (badge_filled + badge_open, "Numbered box: a promo, in posting order; a filled badge is a top-10 promo. The "
                                    "same number on two rows is one promo running from the end of an episode into the "
                                    "next."),
        ("", "turn / reveal next to an episode: a story turning point or a major reveal (AI reading of the writers' "
             "notes)."),
    ]
    cells = "".join(f"<span style='display:inline-flex;align-items:baseline;gap:6px;max-width:100%'>"
                    f"<span style='white-space:nowrap'>{s}</span><span>{fmt.esc(t)}</span></span>" for s, t in items)
    return (f"<div style='display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;line-height:1.45;color:{theme.INK};"
            f"font-family:{theme.FONT_STACK}'>{cells}</div>")


def _cum_minutes(beats: dict[int, dict], upto: int) -> float:
    return sum(float(b.get("duration_s") or 0) for e, b in beats.items() if e <= upto)


def _words_and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def takeaways(data: dict, parts_: list) -> list[str]:
    """Up to four computed bullets (spec 5.4)."""
    out: list[str] = []
    beats = _beats(data)
    promos = _l(data.get("promos"))
    if parts_ and beats and promos:
        p0 = parts_[0]
        out.append(f"{p0.n} of the {len(promos)} promos come from {fmt.ep_range(p0.first_ep, p0.last_ep)}, the first "
                   f"{fmt.minutes(_cum_minutes(beats, p0.last_ep))} a new viewer watches.")
    hmin = _d(_d(data.get("cfg")).get("candidates")).get("hook_min_score")
    if len(parts_) >= 2 and beats and isinstance(hmin, (int, float)):
        rates = {}
        for part in parts_:
            eps = [e for e in beats if part.first_ep <= e <= part.last_ep]
            secs = sum(float(beats[e].get("duration_s") or 0) for e in eps)
            strong = sum(1 for e in eps for b in _l(beats[e].get("beats")) if float(b.get("hook") or 0) >= hmin)
            rates[part.key] = strong / (secs / 60) if secs else 0.0
        last, first = parts_[-1], parts_[0]
        if rates[last.key] == max(rates.values()) and rates[last.key] > rates[first.key]:
            before = _cum_minutes(beats, last.first_ep - 1)
            out.append(f"{fmt.ep_range(last.first_ep, last.last_ep).capitalize()} have the most lines strong enough to "
                       f"open a promo ({_dec1(rates[last.key])} a minute, against {_dec1(rates[first.key])} in "
                       f"{fmt.ep_range(first.first_ep, first.last_ep)}), but get {last.n} places: a new viewer reaches "
                       f"them only after {fmt.minutes(before)} of watching, and late clips can give the story away.")
    rows = _l(_d(_d(data.get("metrics")).get("cliff_localisation")).get("rows"))
    win = _d(_d(data.get("cfg")).get("eval")).get("cliff_window_s")
    if rows and isinstance(win, (int, float)):
        k = sum(1 for r in rows if r.get("in_last_window") and (r.get("top_cliff_score") or 0) > 0)
        out.append(f"In {k} of {len(rows)} rows the strongest cliffhanger sits in the last {fmt.num(win)} seconds: the "
                   "writers end episodes on one.")
    flat = checks.flat_episodes(data)
    zero = [str(r["episode"]) for r in flat if r["still_flat"] and r["all_zero"]]
    other = [str(r["episode"]) for r in flat if r["still_flat"] and not r["all_zero"]]
    if zero or other:
        bits = []
        if zero:
            bits.append(f"{'Row' if len(zero) == 1 else 'Rows'} {_words_and(zero)} "
                        f"{'stays' if len(zero) == 1 else 'stay'} grey: the AI scored every line 0, even when asked "
                        "again.")
        if other:
            bits.append(f"{'Row' if len(other) == 1 else 'Rows'} {_words_and(other)} "
                        f"{'has' if len(other) == 1 else 'have'} no line strong enough to open a promo.")
        out.append(" ".join(bits))
    return out


def render_map(run: RunInfo, data: dict, parts_: list) -> None:
    """Takeaways, legend, chart. (No selected panel in P0.)"""
    if not _d(data.get("beats")):
        empty_state("No line scores in this run", "The map needs beats/ in the run folder.")
        return
    if (run.model or "").lower() in ("none", "heuristic", ""):
        st.caption("Scores here come from keyword rules, not the AI.")
    tk = takeaways(data, parts_)
    if tk:
        st.markdown("\n".join(f"- {t}" for t in tk))
    st.html(legend_html())
    arrays = map_arrays(str(run.dir), D.stamp(run.dir))
    fig = figure(data, parts_, arrays=arrays)
    st.plotly_chart(fig, theme=None, key="showmap", config={"displayModeBar": False},
                    alt="Show map: hook and cliffhanger scores for every line of every episode, with the promos "
                        "boxed and numbered")
    st.caption("Hover a numbered badge for the promo's segments and checks; open it from the list below.")
