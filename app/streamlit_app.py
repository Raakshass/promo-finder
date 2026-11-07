"""promo-finder — Streamlit UI. Demo mode reads a committed run under outputs/; the Run tab parses an
uploaded pack in seconds and, with a pasted key, runs the full pipeline. The input scripts are never bundled."""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

st.set_page_config(page_title="promo-finder", page_icon=":material/movie:", layout="wide")

# Inline icons from Lucide (lucide.dev, ISC licence): crisp at any size, theme-coloured, no emoji anywhere.
_LUCIDE = {
    "clapper": '<path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z"/><path d="m6.2 5.3 3.1 3.9"/><path d="m12.4 3.4 3.1 4"/><path d="M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/><path d="M14.8 14.8 20 20"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "pen": '<path d="M21.17 6.81a1 1 0 0 0-3.98-3.98L3.84 16.17a2 2 0 0 0-.5.83l-1.32 4.35a.5.5 0 0 0 .62.62l4.35-1.32a2 2 0 0 0 .83-.5z"/>',
    "pin": '<path d="M20 10c0 4.99-5.54 10.19-7.4 11.8a1 1 0 0 1-1.2 0C9.54 20.19 4 14.99 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/>',
    "eye_off": '<path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.53 13.53 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" x2="22" y1="2" y2="22"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
}


def icon(name: str, size: int = 14, color: str = "currentColor") -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{color}' "
            f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:-2px;margin-right:4px'>{_LUCIDE[name]}</svg>")


def chip(name: str, text: str, bg: str = "#f3f4f6", fg: str = "#374151") -> str:
    return (f"<span style='background:{bg};color:{fg};border-radius:999px;padding:3px 10px;margin:0 6px 6px 0;display:inline-block;"
            f"font-size:0.8rem;font-weight:500'>{icon(name, 13, fg)}{text}</span>")

# ----------------------------------------------------------------------------- data

def _fetch_outputs_if_configured() -> None:
    """Deployed demo: outputs/ is not in the public repo (it quotes the show), so the app can pull an
    outputs.zip from a private link set as the OUTPUTS_URL secret or env var, once per container."""
    import io
    import os
    import re
    import urllib.request
    import zipfile

    url = os.environ.get("OUTPUTS_URL") or (st.secrets.get("OUTPUTS_URL") if hasattr(st, "secrets") and _has_secret("OUTPUTS_URL") else None)
    if not url or any((OUT / "_fetched").glob("*/*/promos.json")):
        return
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:  # Google Drive share link -> direct download
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    try:
        with st.spinner("fetching outputs…"):
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "promo-finder"}), timeout=120).read()
            (OUT / "_fetched").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(OUT / "_fetched")
    except Exception as e:  # noqa: BLE001
        st.warning(f"could not fetch outputs from OUTPUTS_URL: {e}")


def _has_secret(name: str) -> bool:
    try:
        return name in st.secrets
    except Exception:  # noqa: BLE001 - no secrets file locally
        return False


def list_runs() -> list[Path]:
    _fetch_outputs_if_configured()
    runs = [p for p in OUT.glob("*/*") if (p / "promos.json").exists()]
    runs += [p for p in OUT.glob("_fetched/**/promos.json")]
    runs = [p.parent if p.name == "promos.json" else p for p in runs]
    # the run marked OFFICIAL (written by scripts/make_submission.py) opens first; then judged runs; then newest
    def key(p: Path):
        judged = (p / "llm_raw" / "cold_viewer").exists()
        return ((p / "OFFICIAL").exists(), judged, p.name)
    return sorted(set(runs), key=key, reverse=True)


@st.cache_data(show_spinner=False)
def load_run(run_dir: str) -> dict:
    d = Path(run_dir)
    data = {
        "promos": json.loads((d / "promos.json").read_text()),
        "metrics": json.loads((d / "eval" / "metrics.json").read_text()) if (d / "eval" / "metrics.json").exists() else {},
        "failures": (d / "eval" / "failures.md").read_text() if (d / "eval" / "failures.md").exists() else "",
        "parse": json.loads((d / "parse_report.json").read_text()) if (d / "parse_report.json").exists() else {},
        "aliases": yaml.safe_load((d / "aliases.yaml").read_text()) if (d / "aliases.yaml").exists() else {},
        "svg": (d / "show_map.svg").read_text() if (d / "show_map.svg").exists() else "",
        "run_log": json.loads((d / "run_log.json").read_text()) if (d / "run_log.json").exists() else {},
        "beats": {},
        "promoability": json.loads((d / "eval" / "promoability.json").read_text()) if (d / "eval" / "promoability.json").exists() else {},
        "promoability_md": (d / "eval" / "promoability.md").read_text() if (d / "eval" / "promoability.md").exists() else "",
        "ab": json.loads((d / "eval" / "ab_vs_previous.json").read_text()) if (d / "eval" / "ab_vs_previous.json").exists() else {},
        "arc": json.loads((d / "show_arc.json").read_text()) if (d / "show_arc.json").exists() else {},
        "per_episode": (d / "promos_by_episode.csv").read_text() if (d / "promos_by_episode.csv").exists() else "",
        "cut_script": (d / "cut_promos.sh").read_text() if (d / "cut_promos.sh").exists() else "",
        "stability": json.loads((d / "eval" / "stability.json").read_text()) if (d / "eval" / "stability.json").exists() else {},
        "cfg": yaml.safe_load((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {},
    }
    for f in sorted((d / "beats").glob("ep-*.json")) if (d / "beats").exists() else []:
        j = json.loads(f.read_text())
        data["beats"][j["episode"]] = j
    return data


# ----------------------------------------------------------------------------- widgets

def phone_preview(p: dict, height: int = 640) -> None:
    lines = [l for l in p["lines"] if l["kind"] in ("dialogue", "action")]
    dur = p["duration_s"]
    cliff_at = p["cliffhanger"]["at_s"]
    html = f"""
<style>
 .wrap{{display:flex;justify-content:center;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
 .phone{{width:300px;height:560px;border-radius:36px;background:#0b0b0e;border:6px solid #1c1c22;position:relative;overflow:hidden;box-shadow:0 20px 40px rgba(0,0,0,.35)}}
 .screen{{position:absolute;inset:0;background:radial-gradient(120% 90% at 50% 30%, #2a2130 0%, #0b0b0e 70%)}}
 .notch{{position:absolute;top:10px;left:50%;transform:translateX(-50%);width:110px;height:22px;background:#1c1c22;border-radius:12px}}
 .badge{{position:absolute;top:48px;left:14px;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.5px;background:#ff3b5c;color:#fff;opacity:0;transition:opacity .2s}}
 .cliff{{position:absolute;top:48px;right:14px;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:700;background:#3b82f6;color:#fff;opacity:0;transition:opacity .2s}}
 .spk{{position:absolute;left:18px;right:18px;bottom:150px;color:#ffd166;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px}}
 .cap{{position:absolute;left:18px;right:18px;bottom:78px;color:#fff;font-size:20px;line-height:1.25;font-weight:600;text-shadow:0 2px 8px rgba(0,0,0,.8);min-height:60px}}
 .cap.action{{color:#cfd8dc;font-style:italic;font-weight:400;font-size:15px}}
 .timer{{position:absolute;right:18px;bottom:52px;color:#aaa;font-size:11px;font-variant-numeric:tabular-nums}}
 .bar{{position:absolute;left:18px;right:18px;bottom:40px;height:4px;background:#333;border-radius:2px}}
 .fill{{height:100%;width:0;background:#fff;border-radius:2px}}
 .hookzone{{position:absolute;left:0;top:0;height:100%;width:{min(100, 300/ max(dur,1))}%;background:#ff3b5c;opacity:.7}}
 .btn{{position:absolute;left:50%;bottom:8px;transform:translateX(-50%);background:#fff;color:#000;border:0;border-radius:999px;padding:5px 14px;font-size:12px;font-weight:700;cursor:pointer}}
 .end{{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;background:rgba(0,0,0,.72);color:#fff;text-align:center;padding:24px}}
 .end b{{font-size:22px}}
</style>
<div class="wrap"><div class="phone"><div class="screen">
 <div class="notch"></div>
 <div class="badge" id="badge">HOOK · first 3s</div>
 <div class="cliff" id="cliffb">CLIFFHANGER</div>
 <div class="spk" id="spk"></div>
 <div class="cap" id="cap"></div>
 <div class="timer" id="timer">0.0 / {dur:.0f}s</div>
 <div class="bar"><div class="hookzone"></div><div class="fill" id="fill"></div></div>
 <button class="btn" id="btn">▶ Play</button>
 <div class="end" id="end"><b>…cut.</b><div style="margin-top:8px;font-size:13px">Would you tap to watch the show?</div></div>
</div></div></div>
<script>
const lines = {json.dumps(lines)};
const dur = {dur};
const cliffAt = {cliff_at};
let start = null, raf = null;
const cap = document.getElementById('cap'), spk = document.getElementById('spk'), fill = document.getElementById('fill');
const timer = document.getElementById('timer'), badge = document.getElementById('badge'), cliffb = document.getElementById('cliffb'), endEl = document.getElementById('end');
function show(t) {{
  let cur = null;
  for (const l of lines) {{ if (t >= l.t && t < l.t + Math.max(l.dur, 1.2)) cur = l; }}
  if (!cur) {{ for (const l of lines) {{ if (l.t <= t) cur = l; }} }}
  if (!cur && lines.length) cur = lines[0];  // before the first line starts, preview the hook
  if (cur) {{
    cap.textContent = cur.kind === 'dialogue' ? '“' + cur.text + '”' : cur.text;
    cap.className = 'cap' + (cur.kind === 'action' ? ' action' : '');
    spk.textContent = cur.kind === 'dialogue' ? (cur.speaker || '') : 'on screen';
  }}
  badge.style.opacity = t <= 3 ? 1 : 0;
  cliffb.style.opacity = t >= Math.max(0, cliffAt - 0.5) ? 1 : 0;
  fill.style.width = Math.min(100, 100 * t / dur) + '%';
  timer.textContent = t.toFixed(1) + ' / ' + dur.toFixed(0) + 's';
}}
function tick(now) {{
  if (start === null) start = now;
  const t = (now - start) / 1000;
  if (t >= dur) {{ show(dur); endEl.style.display = 'flex'; return; }}
  show(t); raf = requestAnimationFrame(tick);
}}
document.getElementById('btn').onclick = () => {{ if (raf) cancelAnimationFrame(raf); start = null; endEl.style.display = 'none'; raf = requestAnimationFrame(tick); }};
show(0);
</script>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def evidence_strip(p: dict) -> str:
    dur = p["duration_s"]
    W, H = 640, 74
    sx = (W - 20) / max(dur, 1)
    hook_x = 10 + p["hook"]["at_s"] * sx
    cliff_x = 10 + p["cliffhanger"]["at_s"] * sx
    seg_rects = ""
    off = 0.0
    for i, s in enumerate(p["segments"]):
        w = (s["end_s"] - s["start_s"]) * sx
        seg_rects += f'<rect x="{10 + off * sx:.1f}" y="28" width="{w:.1f}" height="14" fill="{"#e5e7eb" if i % 2 == 0 else "#d1d5db"}"/>'
        seg_rects += f'<text x="{10 + off * sx + 3:.1f}" y="39" font-size="9" fill="#374151">ep {s["episode"]} {s["start"]}→{s["end"]}</text>'
        off += s["end_s"] - s["start_s"]
    cue = f'<text x="{W / 2:.0f}" y="62" font-size="10" text-anchor="middle" fill="#2563eb">ends on the writers\' own episode-end cue</text>' if p.get("writer_cue") else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="Inter,Helvetica,Arial,sans-serif">
<rect x="10" y="28" width="{dur * sx:.1f}" height="14" fill="#e5e7eb"/>{seg_rects}
<rect x="10" y="28" width="{3 * sx:.1f}" height="14" fill="rgba(255,59,92,.35)"/>
<line x1="{hook_x:.1f}" y1="22" x2="{hook_x:.1f}" y2="48" stroke="#ff3b5c" stroke-width="2"/>
<text x="{hook_x + 3:.1f}" y="20" font-size="10" fill="#ff3b5c" font-weight="bold">hook +{p["hook"]["at_s"]:.1f}s</text>
<line x1="{cliff_x:.1f}" y1="22" x2="{cliff_x:.1f}" y2="48" stroke="#3b82f6" stroke-width="2"/>
<text x="{cliff_x - 3:.1f}" y="20" font-size="10" fill="#3b82f6" text-anchor="end" font-weight="bold">cliff +{p["cliffhanger"]["at_s"]:.1f}s</text>
<text x="10" y="62" font-size="10" fill="#6b7280">0s</text><text x="{10 + dur * sx:.1f}" y="62" font-size="10" fill="#6b7280" text-anchor="end">{dur:.0f}s</text>
{cue}</svg>"""


RUBRIC_LABELS = {
    "opens_on_conflict": "opens on conflict", "context_free_open": "no backstory needed", "small_cast": "≤3 speakers",
    "ends_open": "ends before the payoff", "builds": "builds to the end", "no_dead_air": "no dead air",
    "stakes_clear": "stakes clear", "rootable": "someone to root for", "features_lead": "features a lead",
}


def rubric_pills(rub: dict) -> str:
    """The editor's checklist as pills: green = pass, red = fail (each failure costs a fixed share of the score)."""
    pills = []
    for k, ok in rub.items():
        bg, fg, mark = ("#dcfce7", "#166534", "check") if ok else ("#fee2e2", "#991b1b", "x")
        pills.append(f"<span style='background:{bg};color:{fg};border-radius:999px;padding:2px 9px;margin:0 4px 4px 0;"
                     f"display:inline-block;font-size:0.78em'>{icon(mark, 12, fg)}{RUBRIC_LABELS.get(k, k)}</span>")
    return "<div style='margin:6px 0 4px'><b style='font-size:0.85em'>Editor's checks</b><br>" + "".join(pills) + "</div>"


def promo_card(p: dict, show_phone: bool = True) -> None:
    segs = " + ".join(f"Ep {s['episode']} **{s['start']} → {s['end']}**" for s in p["segments"])
    flags = p.get("flags", {})
    chips = []
    block = flags.get("block")
    if block:
        mins = flags.get("minutes_into_show")
        chips.append(chip("layers", {"opening": "Opening of the show", "middle": "Middle of the show", "late": "Late in the show"}.get(block, block)
                          + (f" · {mins} min in" if mins not in (None, "") else ""), "#ede9fe", "#5b21b6"))
    conf = (data.get("stability") or {}).get("keep_rate_by_promo", {}).get(p["promo_id"]) if data else None
    if conf is not None:
        chips.append(chip("shield", f"on the shortlist in {conf:.0%} of simulated re-runs", "#ecfeff", "#155e75"))
    if flags.get("turning_point"):
        chips.append(chip("pin", f"turning point: {str(flags['turning_point']).replace('_', ' ')}", "#fef9c3", "#854d0e"))
    if flags.get("spoiler_risk"):
        chips.append(chip("alert", "spoiler risk", "#fee2e2", "#991b1b"))
    if flags.get("major_reveal"):
        chips.append(chip("eye_off", "episode holds a major reveal", "#fee2e2", "#991b1b"))
    if flags.get("segments") == 2:
        chips.append(chip("scissors", "2 segments"))
    if flags.get("physical_hook"):
        chips.append(chip("zap", "physical hook"))
    if p.get("writer_cue"):
        chips.append(chip("pen", "writers' cue"))
    st.markdown(f"### #{p['rank']} · {segs} · {p['duration_s']:.0f}s", unsafe_allow_html=True)
    st.markdown("".join(chips), unsafe_allow_html=True)
    if show_phone:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            phone_preview(p)
    else:
        c2 = st.container()
    with c2:
        st.markdown(evidence_strip(p), unsafe_allow_html=True)
        if p.get("rubric"):
            st.markdown(rubric_pills(p["rubric"]), unsafe_allow_html=True)
        st.markdown(f"**Hook** ({flags.get('hook_type', '')}): {p['hook']['speaker'] or 'ACTION'} — “{p['hook']['text']}”")
        st.markdown(f"**Cliffhanger**: {p['cliffhanger']['speaker'] or 'ACTION'} — “{p['cliffhanger']['text']}”")
        st.markdown(f"**Why it works**: {p['why_it_works']}")
        if p.get("post_first_reason"):
            st.markdown(f"**Post-first reason**: {p['post_first_reason']}")
        cv = p.get("cold_viewer")
        if cv:
            st.info(f"**Cold viewer** (never saw the show) — follows it **{cv['comprehensibility']}/5**, would tap **{cv['would_tap']}/5**\n\n"
                    f"“{cv['what_happened']}”\n\nLeft wondering: *{cv['want_next']}*" + (f"\n\nDidn't understand: {', '.join(cv['unresolved_refs'])}" if cv.get("unresolved_refs") else "")
                    + (f"\n\nStakes as a stranger reads them: **{cv['stakes']}**" if cv.get("stakes") else "")
                    + (f" · roots for: **{cv['root_for']}**" if cv.get("root_for") else ""), icon=":material/visibility:")
        pr = p.get("promise") or {}
        if pr.get("question"):
            ans = (("answers it <b>later in the same episode</b>" if pr.get("status") == "same episode" else
                    f"answers it in <b>episode {pr['answered_in_episode']}</b>, {pr.get('minutes_after_clip', '?')} min after the clip")
                   if pr.get("answered_in_episode") else f"<b>{pr.get('status', 'not checked')}</b>")
            st.markdown(f"<div style='border-left:3px solid #7c3aed;padding:6px 10px;margin:6px 0;background:#faf5ff;border-radius:6px;font-size:0.92rem'>"
                        f"{icon('help', 14, '#6d28d9')}<b>Plants the question:</b> {pr['question']}<br>"
                        f"{icon('clock', 14, '#6d28d9')}a new viewer reaches this moment <b>{pr.get('reached_at_min', '?')} min</b> into the show · "
                        f"the show {ans}</div>", unsafe_allow_html=True)
        ss = p["subscores"]
        w = ((data or {}).get("cfg") or {}).get("scoring", {}) if data else {}
        wh, wc, wsa, wd = (w.get("w_hook", 0.35), w.get("w_cliff", 0.30), w.get("w_standalone", 0.25), w.get("w_density", 0.10))
        if "base" in ss:
            fails = [RUBRIC_LABELS.get(k, k) for k, ok in (p.get("rubric") or {}).items() if not ok]
            st.markdown(
                f"<div style='font-size:0.86rem;color:#374151;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:6px 10px;margin:6px 0'>"
                f"<b>Why this rank.</b> base {ss['base']:.3f} = {wh}×hook {ss['hook']:.2f} + {wc}×cliff {ss['cliff']:.2f} + "
                f"{wsa}×stands alone {ss['standalone']:.2f} + {wd}×pace {ss['density']:.2f}; "
                f"× checks {ss.get('rubric_mult', 1):.2f}" + (f" ({', '.join(fails)})" if fails else "") +
                f" × turning point {ss.get('arc_mult', 1):.2f}" + (" × 0.9 two-part join" if p["flags"].get("kind") == "two_segment" else "") +
                f" = <b>{p['score']:.3f}</b>. Placed in the <b>{p['flags'].get('block', '?')}</b> part of the slate (opening, middle and late "
                f"each get a fixed share of places; best-first inside each, at most 2 per episode in the top 10).</div>",
                unsafe_allow_html=True)
        cols = st.columns(4)
        for col, k in zip(cols, ("hook", "cliff", "standalone", "density")):
            col.metric(k, f"{ss.get(k, 0):.2f}")
            col.progress(min(1.0, float(ss.get(k, 0))))
        d1, d2, d3 = st.columns(3)
        key = f"dec-{p['promo_id']}"
        if d1.button("Approve", key=key + "a", icon=":material/check_circle:"):
            st.session_state.decisions[p["promo_id"]] = "approved"
        if d2.button("Reject", key=key + "r", icon=":material/cancel:"):
            st.session_state.decisions[p["promo_id"]] = "rejected"
        if d3.button("Needs trim", key=key + "t", icon=":material/content_cut:"):
            st.session_state.decisions[p["promo_id"]] = "trim"
        if p["promo_id"] in st.session_state.decisions:
            st.caption(f"your decision: **{st.session_state.decisions[p['promo_id']]}**")
    with st.expander("full clip transcript (what the cold viewer saw)"):
        for l in p["lines"]:
            if l["kind"] == "dialogue":
                st.markdown(f"`{l['t']:5.1f}s` **{l['speaker']}**: “{l['text']}”")
            elif l["kind"] == "action":
                st.markdown(f"`{l['t']:5.1f}s` *{l['text']}*")


def show_map(data: dict, promos: list[dict]) -> None:
    beats = data["beats"]
    if not beats:
        st.info("no beats in this run")
        return
    eps = sorted(beats)
    max_d = max(b["duration_s"] for b in beats.values())
    cols = int(max_d) + 1
    z, text = [], []
    for e in eps:
        row = [None] * cols
        trow = [""] * cols
        for b in beats[e]["beats"]:
            v = b["hook"] if b["hook"] >= b["cliff"] else -b["cliff"]
            for t in range(int(b["t0"]), min(cols, int(b["t1"]) + 1)):
                if row[t] is None or abs(v) > abs(row[t]):
                    row[t] = v
                    trow[t] = f"ep{e} {b['beat_id']} hook {b['hook']} cliff {b['cliff']}<br>{(b['speaker'] or b['kind'])}"
        for t in range(int(beats[e]["duration_s"]) + 1, cols):
            row[t] = None
        z.append([0 if v is None and t <= beats[e]["duration_s"] else v for t, v in enumerate(row)])
        text.append(trow)
    fig = go.Figure(go.Heatmap(z=z, x=list(range(cols)), y=[f"ep {e}" for e in eps], text=text, hoverinfo="text",
                               colorscale=[[0, "#1f77b4"], [0.5, "#f3f4f6"], [1, "#d62728"]], zmin=-3, zmax=3, showscale=False))
    ymap = {e: i for i, e in enumerate(eps)}
    for p in promos:
        for s in p["segments"]:
            y = ymap.get(s["episode"])
            if y is None:
                continue
            fig.add_shape(type="rect", x0=s["start_s"], x1=s["end_s"], y0=y - 0.5, y1=y + 0.5, line=dict(color="black", width=2 if p["rank"] <= 10 else 1))
            if s is p["segments"][0]:
                fig.add_annotation(x=s["start_s"], y=y, text=str(p["rank"]), showarrow=False, font=dict(size=9, color="black"), xanchor="left")
    fig.update_layout(height=22 * len(eps) + 80, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="seconds into the episode",
                      yaxis=dict(autorange="reversed"), title="Every beat of every episode — red = hook potential, blue = cliffhanger potential, boxes = selected promos")
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- app

if "decisions" not in st.session_state:
    st.session_state.decisions = {}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero { padding: 22px 26px; border-radius: 18px; background: linear-gradient(120deg, #1b1033 0%, #3a1450 55%, #ff3b5c 140%); color: #fff; margin-bottom: 14px; }
.hero h1 { color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0 0 4px; }
.hero p { color: #f1e9ff; margin: 0; font-size: 1.02rem; }
.hero .premise { margin-top: 10px; font-size: .95rem; color: #ffe1e7; }
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 6px 0 16px; }
.kpi { border: 1px solid #ececf2; border-radius: 14px; padding: 12px 14px; background: #fff; }
.kpi .v { font-size: 1.45rem; font-weight: 800; color: #111118; line-height: 1.1; }
.kpi .l { font-size: .78rem; color: #6b7280; margin-top: 3px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>""", unsafe_allow_html=True)


def _secret(name: str) -> str | None:
    import os
    return os.environ.get(name) or (st.secrets.get(name) if _has_secret(name) else None)


# optional access code: the promo transcripts quote the show, so a deployed app can be limited to the people given the code
_code = _secret("APP_PASSWORD")
if _code and not st.session_state.get("authed"):
    st.markdown("<div class='hero'><h1>" + icon("clapper", 30, "#ffffff") + "promo-finder</h1><p>Finds the promo clips in a micro-drama's scripts.</p></div>", unsafe_allow_html=True)
    entered = st.text_input("Access code (in the submission form)", type="password")
    if entered and entered == _code:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("That code is not right.")
    st.stop()

runs = list_runs()
with st.sidebar:
    st.header("Run")
    if runs:
        _picked = st.session_state.get("picked_run")
        _idx = next((k for k, r in enumerate(runs) if str(r) == _picked), 0)
        run_dir = st.selectbox("results to browse", runs, index=_idx, format_func=lambda p: ("official · " if (p / "OFFICIAL").exists() else "") + p.name)
    else:
        run_dir = None
        st.warning("no runs under outputs/ yet")
    st.markdown("---")
    st.markdown("**Swap the model**: `PROMO_MODEL=provider/model`\n\n`nvidia/…` (free, no daily cap) · `openrouter/…` · `gemini/…` · `groq/…` · `mistral/…` · `openai/…` · `anthropic/…` · `custom/…` (any OpenAI-compatible endpoint via `PROMO_BASE_URL`) · `none`")


data = load_run(str(run_dir)) if run_dir else None

_premise = (data or {}).get("arc", {}).get("premise", "") if data else ""
st.markdown("<div class='hero'><h1>" + icon("clapper", 30, "#ffffff") + "promo-finder</h1><p>Finds the 20–60 s promo clips in a micro-drama's scripts: exact cut points, the hook, the cliffhanger, why it works, and its own quality check. The AI judges drama; code owns every timestamp.</p>"
            + (f"<div class='premise'><b>This show:</b> {_premise}</div>" if _premise else "") + "</div>", unsafe_allow_html=True)
if data:
    _m = data["metrics"]
    _er = _m.get("editor_rubric", {})
    _cv = _m.get("cold_viewer", {})
    _llm = _m.get("llm", {})
    _ab = data.get("ab") or {}
    _tiles = [
        (f"{len(data['promos'])}", f"ranked promos from {_m.get('episodes', '?')} episodes"),
        (f"{_er.get('top10_all_checks_pass', '–')}/10", "top 10 passing every editor's check"),
        (f"{_cv.get('mean_comprehensibility') or '–'}/5", "a first-time viewer follows the clip"),
        (f"{_ab['b_wins']}–{_ab['a_wins']}" if _ab else f"{_m.get('invariants', {}).get('pass_rate', 0):.0%}",
         f"blind A/B wins, new vs first version ({_ab.get('ties', 0)} ties)" if _ab else "clips that follow every rule"),
        ((f"{_m['promise_ledger']['answered']}/{_m['promise_ledger']['promos']}" if _m.get("promise_ledger") else "–"),
         "promos whose question the show answers"),
        (f"${_llm.get('est_cost_usd_at_paid_rates', 0) or 0:.2f}", f"cost · {_llm.get('calls_total', '?')} AI calls"),
    ]
    st.markdown("<div class='kpis'>" + "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in _tiles) + "</div>", unsafe_allow_html=True)

tabs = st.tabs([":material/movie: Promos", ":material/help: Promise & payoff", ":material/map: Show map", ":material/insights: Promo-ability",
                ":material/report: Where it picks badly", ":material/fact_check: Data QA", ":material/play_circle: Run it"])

with tabs[0]:
    if not data:
        st.info("Run the tool first (see the Run tab).")
    else:
        m = data["metrics"]
        if not data["promos"]:
            st.info("No clip in this show fits the rules (20-60 s, a hook in the first 3 s, no line cut in half). "
                    "Very short episodes are the usual reason; see the Promo-ability and Data QA tabs.", icon=":material/info:")
        else:
            with st.expander("How this ranking is computed", icon=":material/calculate:"):
                st.markdown(
                    "1. **Candidates:** every strong opening line paired with a later strong closing line becomes a 20–60 s clip that starts "
                    "on a camera cut, with the hook inside 3 s.\n"
                    "2. **Sub-scores, 0–1:** *hook* and *cliff* are the AI's 0–3 scores of the opening and closing lines; *stands alone* is the "
                    "first-time viewer's rating; *pace* is the share of the clip filled with dialogue, times a length factor.\n"
                    "3. **Score:** `0.35·hook + 0.30·cliff + 0.25·stands alone + 0.10·pace`, × a fixed penalty for each failed editor's check, "
                    "× 1.05 for a turning-point episode.\n"
                    "4. **De-duplication:** overlapping clips and clips that share a key line are dropped.\n"
                    "5. **The slate:** 55% opening, 33% middle, 12% late (the shape of professional ads), best-first in each part, at most 2 "
                    "per episode in the top 10. The opening part is posted first.\n\n"
                    "Every card below shows its own numbers. The weights are hand-set from the research and live in `config.yaml`; "
                    "they are not learned from data.")
            st.markdown("#### Post these first")
            top3 = data["promos"][:3]
            cols = st.columns(len(top3))
            for col, p in zip(cols, top3):
                with col:
                    segs = " + ".join(f"ep {s['episode']} {s['start']}→{s['end']}" for s in p["segments"])
                    st.markdown(f"**#{p['rank']}** · {segs} · {p['duration_s']:.0f}s")
                    phone_preview(p, height=600)
                    st.caption(p.get("post_first_reason") or p.get("why_it_works", ""))
            st.markdown("---")
            n = len(data["promos"]) if len(data["promos"]) <= 3 else st.slider("show top", 3, len(data["promos"]), min(8, len(data["promos"])))
            phones = st.toggle("phone preview on every card", value=False)
            for p in data["promos"][:n]:
                with st.container(border=True):
                    promo_card(p, phones)
            df = pd.DataFrame([{"rank": p["rank"], "episode": "+".join(str(s["episode"]) for s in p["segments"]),
                                "in": p["segments"][0]["start"], "out": p["segments"][-1]["end"], "s": p["duration_s"],
                                "hook": p["hook"]["text"], "cliffhanger": p["cliffhanger"]["text"], "score": p["score"],
                                "decision": st.session_state.decisions.get(p["promo_id"], "")} for p in data["promos"]])
            st.download_button("promos.csv (in/out points for the edit)", icon=":material/download:", data=df.to_csv(index=False), file_name="promos.csv", mime="text/csv")
            st.download_button("decisions.json", icon=":material/download:", data=json.dumps(st.session_state.decisions, indent=1), file_name="decisions.json", mime="application/json")
            if data.get("cut_script"):
                st.download_button("cut_promos.sh (renders every clip with ffmpeg)", icon=":material/download:", data=data["cut_script"], file_name="cut_promos.sh", mime="text/x-shellscript")
            if data.get("per_episode"):
                with st.expander("One pick per episode (for 'new episode out' posts)"):
                    import io as _io
                    st.dataframe(pd.read_csv(_io.StringIO(data["per_episode"])), width="stretch", height=380, hide_index=True)
                    st.download_button("promos_by_episode.csv", icon=":material/download:", data=data["per_episode"], file_name="promos_by_episode.csv", mime="text/csv")

with tabs[1]:
    if data:
        st.markdown("### Every promo plants a question. Does the show answer it, and when?")
        st.caption("The question is the first-time viewer's own words after watching the clip. 'Reached' is how many minutes of watching "
                   "from episode 1 it takes a new viewer to arrive at the moment; the answer episode is found by the model in the writers' "
                   "episode summaries and checked by code (an answer before the clip is rejected). A question the show never answers is a fake hook.")
        rows = []
        for p in data["promos"]:
            pr = p.get("promise") or {}
            rows.append({"rank": p["rank"], "episode": p["segments"][0]["episode"], "block": p["flags"].get("block", ""),
                         "question it plants": pr.get("question", ""), "reached (min into show)": pr.get("reached_at_min"),
                         "answered in episode": pr.get("answered_in_episode"), "minutes after the clip": pr.get("minutes_after_clip"),
                         "status": pr.get("status", ""), "how": pr.get("answer", "")})
        if rows and any(r["question it plants"] for r in rows):
            led = data["metrics"].get("promise_ledger") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("answered by the show", f"{led.get('answered', 0)}/{led.get('promos', 0)}")
            c2.metric("never answered (fake-hook risk)", led.get("never_answered", 0))
            c3.metric("median minutes from clip to answer", led.get("median_minutes_clip_to_answer") or "–")
            c4.metric("answers behind the paywall", led.get("behind_paywall", 0) if (data.get("run_log", {}).get("args", {}) or {}).get("paywall") else "–",
                      help="needs --paywall N; an answer behind the paywall is a conversion promo, a free one builds trust")
            st.dataframe(pd.DataFrame(rows), width="stretch", height=520, hide_index=True)
        else:
            st.info("This run has no ledger yet (it needs the model and the cold-viewer test).")

with tabs[2]:
    if data:
        show_map(data, data["promos"])
        if data["svg"]:
            with st.expander("static version (show_map.svg, as used in the doc)"):
                st.markdown(data["svg"], unsafe_allow_html=True)

with tabs[3]:
    if data:
        pa = data.get("promoability") or {}
        if not pa:
            st.info("This run predates the promo-ability report; re-run the tool.")
        else:
            sh = pa["show"]
            st.markdown(f"### Is this show easy to promo?\n**{'; '.join(sh['verdict'])}.**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("easy episodes", sh["tiers"]["easy"], help="a clip with a strong hook and cliff passes every code check")
            c2.metric("medium", sh["tiers"]["medium"], help="opens on conflict and ends open, but fails a softer check")
            c3.metric("hard", sh["tiers"]["hard"], help="no strong clip both opens on conflict and ends before the payoff")
            c4.metric("top-2 speakers' share of lines", f"{sh['top2_dialogue_share']:.0%}", help="two-handers are easier for a stranger to follow than ensembles")
            rows = pd.DataFrame([{"ep": r["episode"], "tier": r["tier"], "conflict beats/min": r["conflict_beats_per_min"],
                                  "payoff share": r["payoff_share"], "backstory load": r["context_load"], "speakers": r["speakers"],
                                  "finalists": r["finalists"], "why": "; ".join(r["why"])} for r in pa["episodes"]])
            st.dataframe(rows, width="stretch", height=420, hide_index=True)
            with st.expander("What makes a scene easy or hard to promo (the research behind these signals)"):
                st.markdown("""
| Easy to promo | Why | Hard to promo | Why |
|---|---|---|---|
| Public humiliation, cut **before** the payback | a sympathetic underdog + an unresolved threat | Exposition, recaps, greetings | no gap, no emotion |
| Slap, confrontation, threat, ultimatum | conflict visible in the golden 3 seconds | Quiet romance | low arousal, payoff needs history |
| Identity / status reveal, betrayal discovered | a reversal of power in one line | Lines that name people or past events a stranger lacks | curiosity needs a gap the viewer can see |
| Two people, one clear "X wants Y, but Z" | a stranger can track 2–3 people, not 6 | Ensemble scenes, 4+ speakers | over working-memory load |
| Scene that ends on a question / interrupted action | the outcome is still uncertain | Scenes that resolve on screen (apology, rescue, answer) | the clip spends its own payoff |

Show level: revenge / hidden-identity / contract-marriage two-handers are the easiest; slow-burn romance, whodunits (every reveal is a spoiler), ensembles and voice-over-heavy shows are the hardest. Sources: HIVE/DramaAD (EMNLP 2025), Papalampidi et al. (turning points), Loewenstein (curiosity gap), Zillmann (suspense), Netflix artwork research, Google ABCD, micro-drama ad-intelligence reports — see `docs/research.md`.
""")

with tabs[4]:
    if data:
        m = data["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("invariants", f"{m.get('invariants', {}).get('pass_rate', 0):.0%}", help="20–60 s, hook within 3 s of the cut, segments ordered")
        cl = m.get("cliff_localisation", {})
        c2.metric("known-cliff recall", f"{cl.get('recall_all', 0):.0%}", help="writers end every episode on a cliff: is the model's top cliff beat in the last 10 s? (heuristic twin: %s)" % f"{m.get('cliff_localisation_heuristic', {}).get('recall_all', 0):.0%}")
        ent = m.get("entailment", {})
        c3.metric("cliff capture", f"{ent.get('capture_rate'):.0%}" if ent.get("capture_rate") is not None else "–", help="does the episode's best promo contain the moment the writers' own continuity note describes?")
        hv = m.get("heuristic_vs_llm_top10") or {}
        c4.metric("LLM vs heuristic top-10 overlap", f"{hv.get('overlap', 0):.0%}" if hv else "–", help="how different the model's picks are from the lexicon baseline")
        er = m.get("editor_rubric", {})
        if er:
            st.markdown("**Editor's checks — pass rate in the top 10**")
            st.markdown(rubric_pills({k: v >= 0.8 for k, v in er.get("top10_pass_rates", {}).items()}), unsafe_allow_html=True)
            st.caption(" · ".join(f"{RUBRIC_LABELS.get(k, k)} {v:.0%}" for k, v in er.get("top10_pass_rates", {}).items())
                       + f" · lanes in the top 10: {er.get('top10_lanes')}")
        ab = data.get("ab") or {}
        if ab:
            st.markdown(f"**Blind A/B vs the previous version** (judge picks the clip it would tap, both orders): "
                        f"new wins **{ab['b_wins']}**, old wins **{ab['a_wins']}**, ties {ab['ties']}, same clip {ab['same_clip']}")
        st.markdown(data["failures"])
        with st.expander("raw metrics.json"):
            st.json(m)

with tabs[5]:
    if data:
        pr = data["parse"]
        st.markdown(f"**{pr.get('episodes')} episodes · {pr.get('total_blocks')} timed blocks · {pr.get('block_kinds')} · durations {pr.get('duration_min_s')}–{pr.get('duration_max_s')} s**")
        st.dataframe(pd.DataFrame(pr.get("per_episode", []))[["episode", "duration_s", "scenes", "dialogue_lines", "parse_rate", "warnings"]], width="stretch", height=360)
        drift = pr.get("drift", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Speaker labels as written** (name drift shows up here)")
            st.dataframe(pd.DataFrame(list(drift.get("speakers", {}).items()), columns=["label", "lines"]), height=300)
            if data["aliases"].get("aliases"):
                st.markdown("**Aliases the model proposed** (editable `aliases.yaml`)")
                st.json(data["aliases"])
        with c2:
            st.markdown("**Spelling clusters** (the same proper noun spelled several ways)")
            st.json(drift.get("spelling_clusters", {}))
            st.markdown("**LLM calls this run**")
            st.json(data["run_log"].get("llm", {}))

with tabs[6]:
    import os
    import subprocess
    import time as _time

    st.markdown("### Run it on a new show")
    st.markdown("Upload a `.zip` of timestamped screenplays (or one file holding every episode). Parsing takes a second; the AI run streams its progress below and the results open in every tab when it finishes.")
    up = st.file_uploader("Screenplay pack", type=["zip", "txt", "md", "fountain"])
    nvidia_key = _secret("NVIDIA_API_KEY")
    server_keys = [k for k in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_API_KEY2", "OPENROUTER_API_KEY3") if _secret(k)]
    # Preferred: NVIDIA's free API runs the same open-weights model as the official run, with no daily cap (~40 requests/min).
    # Fallback: OpenRouter's free tier (about 50 calls a day per key), so free runs there cover the opening episodes only.
    demo_model = _secret("PROMO_DEMO_MODEL") or ("nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b" if nvidia_key else "openrouter/nvidia/nemotron-3.5-super-vl:free")
    demo_cap = int(_secret("DEMO_MAX_EPISODES") or (0 if nvidia_key else 15))
    modes = ["Free hosted open model (no key needed)", "My own key or endpoint", "No model (instant keyword twin)"]
    mode = st.radio("Model", modes, index=0 if server_keys else 2)
    env = dict(os.environ)
    cmd_model, extra = "none", []
    if mode == modes[0]:
        if not server_keys:
            st.warning("This deployment has no free model key configured (NVIDIA_API_KEY or OPENROUTER_API_KEY in the app's secrets). "
                       "Use your own key, or the keyword twin.")
        if nvidia_key:
            st.caption(f"`{demo_model}`: NVIDIA Nemotron 3.5 Super, the open-weights model behind the official results, served free by "
                       "NVIDIA's API catalog. No daily cap, about 40 requests a minute shared by everyone using this app, so a 50-episode "
                       "show takes a few minutes.")
        else:
            st.caption(f"`{demo_model}`: NVIDIA Nemotron 3.5 Super, an open-weights model served free by OpenRouter. That free tier allows "
                       f"about 50 calls a day per key, so a free run covers the first {demo_cap} episodes (the opening of a show, where professional "
                       f"editors take most ad footage from). When the day's quota is used up the run stops and says so.")
        st.warning("Free tiers log what you send them (NVIDIA's free trial API says so explicitly, and some free OpenRouter "
                   "providers do too). Don't upload confidential scripts here: use your own key or endpoint for those.",
                   icon=":material/privacy_tip:")
        limit = st.number_input("episodes to process (0 = all)", 0, 500, demo_cap)
        for k in server_keys:
            env[k] = _secret(k) or ""
        cmd_model = demo_model if server_keys else "none"
        if nvidia_key and demo_model.startswith("nvidia/"):
            extra = ["--reasoning-off", "--rpm", "35", "--concurrency", "4", "--no-entailment", "--judge-top-n", "20"]
        else:
            extra = ["--batch", "5", "--reasoning-off", "--rpm", "12", "--concurrency", "2", "--no-entailment", "--judge-top-n", "20"]
    elif mode == modes[1]:
        c1, c2 = st.columns([2, 1])
        cmd_model = c1.text_input("provider/model", value="openrouter/nvidia/nemotron-3.5-super-vl:free",
                                  help="nvidia/… (free) · openrouter/… · gemini/… · groq/… · mistral/… · openai/… · anthropic/… · custom/<model> for any OpenAI-compatible endpoint")
        key = c2.text_input("API key", type="password", help="used for this run only, never stored")
        base = st.text_input("Base URL (only for custom/…)", value="", placeholder="https://your-gateway.example.com/v1") if cmd_model.startswith("custom/") else ""
        c3, c4, c5 = st.columns(3)
        limit = c3.number_input("episodes to process (0 = all)", 0, 500, 0)
        fast = c4.checkbox("fast mode (uncapped API: 8 calls in parallel)", value=not cmd_model.endswith(":free"))
        votes = c5.selectbox("readings per episode", [1, 2, 3], index=0, help="2 = steadier ranking for twice the calls")
        prov = cmd_model.split("/")[0].lower()
        var = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "openrouter": "OPENROUTER_API_KEY", "aimlapi": "AIMLAPI_API_KEY", "custom": "PROMO_API_KEY",
               "nvidia": "NVIDIA_API_KEY", "mistral": "MISTRAL_API_KEY"}.get(prov)
        for k in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY2", "OPENROUTER_API_KEY3", "NVIDIA_API_KEY"):
            env.pop(k, None)  # never mix a visitor's run with this deployment's keys
        if var and key:
            env[var] = key
        if base:
            env["PROMO_BASE_URL"] = base
        extra = (["--fast"] if fast else ["--batch", "5", "--reasoning-off", "--rpm", "12"]) + ["--votes", str(votes)]
        if not key and prov not in ("ollama", "none"):
            st.info("Paste a key for that provider to run with it.")
    else:
        limit = st.number_input("episodes to process (0 = all)", 0, 500, 0)
        cmd_model = "none"

    if up:
        from promo_finder.parse import parse_report, parse_show

        tmp = Path(tempfile.mkdtemp(prefix="upload-"))
        if up.name.lower().endswith(".zip"):
            with zipfile.ZipFile(up) as z:
                z.extractall(tmp)
        else:
            (tmp / up.name).write_bytes(up.getvalue())
        try:
            skipped: list = []
            eps = parse_show(tmp, skipped)
            rep = parse_report(eps)
            st.success(f"Parsed {rep['episodes']} episodes, {rep['total_blocks']} timed blocks, {rep['duration_min_s']}-{rep['duration_max_s']} s each"
                       + (f"; skipped {len(skipped)} non-screenplay file(s)" if skipped else ""), icon=":material/task_alt:")
            with st.expander("parse report"):
                st.dataframe(pd.DataFrame(rep["per_episode"])[["episode", "duration_s", "scenes", "dialogue_lines", "parse_rate", "warnings"]], height=280)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read this pack: {e}", icon=":material/error:")
            eps = []
        lock = Path(tempfile.gettempdir()) / "promo_finder_run.lock"
        busy = lock.exists() and _time.time() - lock.stat().st_mtime < 900
        if eps and st.button("Run promo-finder", type="primary", icon=":material/play_arrow:", disabled=busy):
            lock.write_text(str(os.getpid()))
            cmd = [sys.executable, "-m", "promo_finder.cli", "run", str(tmp), "--model", cmd_model, "--judge-model", cmd_model,
                   "--out", str(OUT), "--tag", "upload"] + extra + (["--limit", str(int(limit))] if limit else [])
            log_lines: list[str] = []
            try:
                with st.status("Running…", expanded=True) as box:
                    env["COLUMNS"] = "220"  # keep the CLI's messages on one line each
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(ROOT))
                    for line in proc.stdout:  # type: ignore[union-attr]
                        line = line.rstrip()
                        if line:
                            log_lines.append(line)
                            box.write(line[:200])
                    proc.wait()
                    box.update(label="Done" if proc.returncode == 0 else "Stopped with an error", state="complete" if proc.returncode == 0 else "error")
            finally:
                lock.unlink(missing_ok=True)
            st.cache_data.clear()
            new = sorted((p for p in OUT.glob("*/*-upload") if (p / "promos.json").exists()), key=lambda p: p.stat().st_mtime)
            joined = " ".join("\n".join(log_lines).split())  # the CLI wraps long lines; match on flattened text
            if proc.returncode == 0 and new:
                st.session_state["picked_run"] = str(new[-1])
                st.rerun()
            elif "daily quota" in joined:
                st.error("The free hosted model's daily quota is used up for today (it resets at 05:30 IST). Nothing was run with it, "
                         "and no keyword-twin results were passed off as model results. Use your own key above, or the keyword twin.",
                         icon=":material/hourglass_empty:")
            elif "HTTP 401" in joined or "HTTP 403" in joined or "no API key" in joined:
                st.error("The provider rejected the key (or no key was given). Check the provider/model name and the key.", icon=":material/key_off:")
            else:
                st.error("The run stopped with an error; the last lines of its log are below.", icon=":material/error:")
                st.code("\n".join(log_lines[-40:]))
        elif busy:
            st.info("Another run is in progress on this deployment; try again in a minute.")
    st.markdown("**Run it locally instead** (any model, no limits): `uv sync && uv run promo-finder run <folder> --model <provider/model>`. See the README.")
