"""Every formatter the app uses (spec 4.1). No Python object reaches the screen through str(): it goes through here.

Rounding is half-up on the value as written (1.395 -> "1.40"), which is what a reader checking the JSON expects.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

NOT_MEASURED = "not measured in this run"

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_SMALL_WORDS = {"a", "an", "the", "of", "and", "to", "in", "on"}
_RUN_TIME = re.compile(r"(?:^|__)(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
_MD_SPECIALS = re.compile(r"([\\`*_\[\]<>#~$|])")


def _q(x: float, places: int) -> Decimal:
    """Half-up rounding of the value as Python prints it."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def esc(x: object) -> str:
    """HTML-escape anything bound for st.html or unsafe_allow_html. "" for None."""
    return "" if x is None else html.escape(str(x))


def md_esc(x: object) -> str:
    """Escape markdown specials (\\ ` * _ [ ] < > # ~ $ |) so script text never turns into formatting, links or LaTeX
    in st.markdown / st.caption. "" for None. Do not use it inside st.html (use esc)."""
    return "" if x is None else _MD_SPECIALS.sub(r"\\\1", str(x))


def score(x: float) -> str:
    """0.7583 -> "0.758"."""
    return f"{_q(x, 3)}"


def mult(x: float) -> str:
    """0.85 -> "×0.85"."""
    return f"×{_q(x, 2)}"


def pct(x: float) -> str:
    """0.4 -> "40%"; 0.913 -> "91%"."""
    return f"{_q(float(x) * 100, 0)}%"


def lift(x: float) -> str:
    """1.395 -> "1.40×"."""
    return f"{_q(x, 2)}×"


def k_of_n(k: int, n: int) -> str:
    """"16 of 24"."""
    return f"{int(k)} of {int(n)}"


def num(n: int) -> str:
    """Whole numbers with thousands separators: 158057 -> "158,057"."""
    return f"{int(n):,}"


def plural(n: int, word: str, many: str | None = None) -> str:
    """plural(1, "segment") -> "1 segment"; plural(2, "segment") -> "2 segments"."""
    return f"{int(n)} {word if int(n) == 1 else (many or word + 's')}"


def secs(x: float) -> str:
    """24.0 -> "24.0 s"."""
    return f"{_q(x, 1)} s"


def clock(x: float) -> str:
    """82.5 -> "1:22.5" (m:ss.s, for map hovers and transcripts)."""
    tenths = int(_q(float(x) * 10, 0))
    m, t = divmod(max(tenths, 0), 600)
    return f"{m}:{t // 10:02d}.{t % 10}"


def mmss(x: float) -> str:
    """90 -> "1:30" (axis ticks)."""
    total = int(_q(x, 0))
    m, s = divmod(max(total, 0), 60)
    return f"{m}:{s:02d}"


def minutes(x_s: float) -> str:
    """1176 -> "19.6 min"."""
    return f"{_q(float(x_s) / 60, 1)} min"


def ranges(nums: Iterable[int]) -> str:
    """[1,2,3,4,5,6,7,9,10,11,24] -> "1–7, 9–11, 24"."""
    xs = sorted({int(n) for n in nums})
    out: list[str] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}–{xs[j]}")
        i = j + 1
    return ", ".join(out)


def ep_range(a: int, b: int) -> str:
    """"episodes 1–10"; "episode 4" when a == b."""
    return f"episode {a}" if int(a) == int(b) else f"episodes {a}–{b}"


def seg(s: dict) -> str:
    """{"episode":10,"start":"01:22.50","end":"01:34.00"} -> "Ep 10 01:22.50–01:34.00"."""
    return f"Ep {s.get('episode', '')} {s.get('start', '')}–{s.get('end', '')}"


def segs(p: dict) -> list[str]:
    """One string per segment, in playing order."""
    return [seg(s) for s in p.get("segments") or []]


def ordinal(n: int) -> str:
    """3 -> "3rd"; 11 -> "11th"; 22 -> "22nd"."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def said(speaker: str | None, text: str) -> str:
    """'CAPTAIN: “Hold the gate!”'; speaker None -> "On screen: <text>" (no quotes)."""
    text = "" if text is None else str(text)
    return f"{speaker}: “{text}”" if speaker else f"On screen: {text}"


def show_title(slug: str) -> str:
    """"i-married-a-broken-soldier" -> "I Married a Broken Soldier"."""
    words = [w for w in re.split(r"[-_\s]+", slug or "") if w]
    return " ".join(w.capitalize() if i == 0 or w.lower() not in _SMALL_WORDS else w.lower() for i, w in enumerate(words))


def _parse_run_time(run_id: str) -> tuple[int, int, int, int, int] | None:
    m = _RUN_TIME.search(run_id or "")
    if not m:
        return None
    y, mo, d, hh, mm, _ = (int(g) for g in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31 and hh < 24 and mm < 60):
        return None
    return y, mo, d, hh, mm


def run_time(run_id: str) -> str:
    """"20261107-182015-…" -> "7 Nov 2026, 18:20"; "" if unparsable."""
    t = _parse_run_time(run_id)
    return f"{t[2]} {_MONTHS[t[1] - 1]} {t[0]}, {t[3]:02d}:{t[4]:02d}" if t else ""


def run_time_short(run_id: str) -> str:
    """"7 Nov 18:20"; "" if unparsable."""
    t = _parse_run_time(run_id)
    return f"{t[2]} {_MONTHS[t[1] - 1]} {t[3]:02d}:{t[4]:02d}" if t else ""


def run_date(run_id: str) -> str:
    """"7 Nov"; "" if unparsable. Used by the "First version (6 Nov)" label."""
    t = _parse_run_time(run_id)
    return f"{t[2]} {_MONTHS[t[1] - 1]}" if t else ""


def model_short(model_id: str) -> str:
    """Short display name for a model id (spec 6.3)."""
    m = (model_id or "").strip()
    low = m.lower()
    if low in ("", "none") or "heuristic" in low:
        return "No AI (keyword rules)"
    if "nemotron-3.5-super" in low:
        return "Nemotron 3.5 Super"
    if "llama-3.1-8b" in low:
        return "Llama 3.1 8B (local)"
    if "qwen3-4b" in low:
        return "Qwen3 4B (local)"
    last = m.rstrip("/").split("/")[-1]
    return last[: -len(":free")] if last.endswith(":free") else last


def is_free(model_id: str, provider: str = "") -> bool:
    """True when the id ends with ":free", or the provider (or the id itself) is nvidia."""
    m = (model_id or "").strip()
    full = f"{provider}/{m}" if provider else m
    return m.endswith(":free") or full.lower().startswith("nvidia/")
