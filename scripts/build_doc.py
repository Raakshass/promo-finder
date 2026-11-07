"""docs/solution.md -> docs/solution.pdf via headless Chrome (no LaTeX, no pandoc). Checks the 2-page limit."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "solution.md"
HTML = ROOT / "docs" / "solution.html"
PDF = ROOT / "docs" / "solution.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 9.5mm 12mm 9.5mm 12mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 8.9pt; line-height: 1.29; color: #1f2328; }
h1 { font-size: 21pt; font-weight: 700; letter-spacing: -0.3px; margin: 0; color: #111827; }
.sub { font-size: 11pt; color: #374151; margin: 1px 0 2px; font-weight: 500; }
.byline { font-size: 8.3pt; color: #6b7280; margin: 0 0 4px; }
h2 { font-size: 11pt; font-weight: 700; color: #3b5bdb; margin: 7px 0 2px; padding-bottom: 2px; border-bottom: 1.5px solid #e5e7eb; }
p { margin: 3px 0; }
ul { margin: 3px 0 3px 15px; padding: 0; } li { margin: 1.5px 0; }
table { border-collapse: collapse; width: 100%; margin: 3px 0 2px; font-size: 8.1pt; }
th, td { border: 1px solid #e5e7eb; padding: 2px 5px; vertical-align: top; text-align: left; }
th { background: #f3f4f6; font-weight: 600; color: #111827; }
tr:nth-child(even) td { background: #fafafa; }
tr { break-inside: avoid; page-break-inside: avoid; } h2 { break-after: avoid; page-break-after: avoid; }
code { font-family: Menlo, monospace; font-size: 8pt; background: #f3f4f6; padding: 0 3px; border-radius: 3px; }
strong { color: #111827; }
.diagram { margin: 4px 0 2px; } .diagram svg { width: 100%; height: auto; display: block; }
a { color: #3b5bdb; text-decoration: none; }
"""


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    unfilled = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", md)))
    if unfilled:
        print(f"warning: unfilled placeholders: {unfilled}")
    body = markdown.markdown(md, extensions=["tables"])
    svg = (ROOT / "docs" / "diagram.svg").read_text(encoding="utf-8") if (ROOT / "docs" / "diagram.svg").exists() else ""
    body = body.replace("<p>DIAGRAM_HERE</p>", f"<div class='diagram'>{svg}</div>")
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True)
    try:
        from pypdf import PdfReader  # optional
        n = len(PdfReader(str(PDF)).pages)
    except Exception:  # noqa: BLE001
        n = PDF.read_bytes().count(b"/Type /Page") - PDF.read_bytes().count(b"/Type /Pages")
    print(f"wrote {PDF} ({n} pages){' — OVER THE 2-PAGE LIMIT' if n > 2 else ''}")
    sys.exit(1 if n > 2 else 0)


if __name__ == "__main__":
    main()
