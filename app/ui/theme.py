"""Colour tokens (spec 8.2), the one CSS string the app injects (spec 8.4) and the inline Lucide icons used inside custom HTML.

Two colour families never share an element: data colours encode scores (map, timeline), status colours mark decisions and
checks. Status colours always come with an icon and a word.
"""
from __future__ import annotations

# Ink and surfaces
INK = "#111118"        # body text, promo outlines, top-10 badges
INK_2 = "#4b5563"      # secondary text; any text on the panel colour
INK_3 = "#6b7280"      # captions on white only
PRIMARY = "#18181b"    # interactive accent only: selected tab, primary buttons, the chosen decision
PANEL = "#f6f6f8"
LINE = "#e5e7eb"

# Data colours (charts only)
HOOK_1, HOOK_2, HOOK_3 = "#f0908f", "#d93b3b", "#8f1414"
CLIFF_1, CLIFF_2, CLIFF_3 = "#86b6ef", "#2a78d6", "#104281"
ZERO = "#e2e4e9"       # map: inside the episode, nothing scored 1 or more
HOOK_TEXT = "#b91c1c"  # "Hook" label in the timeline
CLIFF_TEXT = "#1d4ed8"  # "Cliffhanger" label in the timeline

# Status colours as (text, background). The matching Streamlit names are "green", "orange", "gray", "violet", "red".
GREEN = ("#15803d", "#f0fdf4")    # passes, Approved, nothing to check
ORANGE = ("#92400e", "#fef3c7")   # needs a look: failed check, before posting, Trim, fragile pick, non-official banner
GRAY = ("#475569", "#f1f5f9")     # Rejected, not measured, metadata
VIOLET = ("#6d28d9", "#f5f3ff")   # AI-written text and AI judgements
RED = ("#b91c1c", "#fef2f2")      # system errors only (st.error)

BRAND = "#ff3b5c"      # the wordmark's clapper icon only (a graphic, never text)

FONT_STACK = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

CSS = "<style>.block-container{max-width:1280px;padding-top:1.25rem}.pf-num{font-variant-numeric:tabular-nums}</style>"

# Lucide icons (lucide.dev, ISC licence), only inside custom HTML. Streamlit-drawn icons use ":material/<name>:".
LUCIDE = {
    "clapper": '<path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z"/><path d="m6.2 5.3 3.1 3.9"/>'
               '<path d="m12.4 3.4 3.1 4"/><path d="M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
}


def icon(name: str, size: int = 14, color: str = "currentColor") -> str:
    """Inline SVG for custom HTML. `name` is a key of LUCIDE."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true' "
            f"style='vertical-align:-2px;margin-right:6px'>{LUCIDE[name]}</svg>")
