"""E0: no emoji in app/**, no banned words in app/**, and no raw Python values or ids in the rendered page (spec 11,
checks 39 and 40).

Banned words come from the env var PROMO_BANNED_WORDS (comma-separated, case-insensitive); that check skips when unset.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from apphelpers import rendered_text

ROOT = Path(__file__).resolve().parents[2]
SCANNED = sorted(p for p in (ROOT / "app").rglob("*") if p.is_file() and p.suffix in (".py", ".toml", ".css", ".html", ".svg",
                                                                                         ".md", ".json", ".txt"))
SCANNED += [ROOT / ".streamlit" / "config.toml"]

EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U000025A0-\U000025FF\U00002300-\U000023FF"
    "\U0000FE0F\U0000200D\U00002139\U000024C2\U00003030\U0000303D\U00003297\U00003299]")

# Raw values that must never reach the screen (check 40). Promo ids are allowed only in the drawer footer and exports,
# and the drawer is closed in these runs.
FORBIDDEN = {
    "None": re.compile(r"\bNone\b"), "nan": re.compile(r"\bnan\b"), "0/0": re.compile(r"(?<![\d.])0/0(?![\d.])"),
    "–/5": re.compile("–/5"), "dict literal": re.compile(r"\{'"), "list literal": re.compile(r"\[\d+, \d+"),
    "promo id": re.compile(r"\bP\d{2}\b"), "beat id": re.compile(r"\be\d{3}-b\d{2}\b"),
}
# Spec copy that legitimately contains a forbidden token.
ALLOWED_PHRASES = ("None is never answered",)


def test_scanned_files_exist():
    assert any(p.name == "streamlit_app.py" for p in SCANNED)


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_emoji(path):
    text = path.read_text(encoding="utf-8")
    hits = [(i + 1, f"U+{ord(m.group()):04X}") for i, line in enumerate(text.splitlines()) for m in EMOJI.finditer(line)]
    assert not hits, f"emoji code points in {path.name}: {hits[:10]}"


def test_no_banned_words():
    raw = os.environ.get("PROMO_BANNED_WORDS", "")
    words = [w.strip().lower() for w in raw.split(",") if w.strip()]
    if not words:
        pytest.skip("PROMO_BANNED_WORDS is not set")
    hits = [(str(p.relative_to(ROOT)), w) for p in SCANNED for w in words if w in p.read_text(encoding="utf-8").lower()]
    assert not hits, f"banned words found in {len(hits)} place(s): {sorted({f for f, _ in hits})}"


def test_rendered_text_is_clean(app_test):
    at = app_test.run()
    problems: dict[str, list[str]] = {}
    for view in ("Promos", "Ranking", "Quality", "Method", "Run it"):
        at.session_state["view"] = view
        at.run()
        assert not at.exception, f"{view}: {at.exception}"
        text = rendered_text(at)
        for phrase in ALLOWED_PHRASES:
            text = text.replace(phrase, "")
        for name, rx in FORBIDDEN.items():
            for m in rx.finditer(text):
                problems.setdefault(view, []).append(f"{name}: …{text[max(0, m.start() - 40):m.end() + 40]}…")
    assert not problems, problems
