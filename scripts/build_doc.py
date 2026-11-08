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
@page { size: A4; margin: 9mm 12mm 8.5mm 12mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 8.6pt; line-height: 1.3; color: #27272a; }
h1 { font-size: 20pt; font-weight: 700; letter-spacing: -0.4px; margin: 0; color: #18181b; }
.sub { font-size: 10.2pt; color: #3f3f46; margin: 2px 0 3px; font-weight: 500; line-height: 1.3; }
.byline { font-size: 7.9pt; color: #71717a; margin: 0 0 5px; padding-bottom: 5px; border-bottom: 2px solid #18181b; }
h2 { font-size: 10.4pt; font-weight: 700; color: #18181b; margin: 7px 0 3px; padding-left: 7px; border-left: 3px solid #4f46e5; line-height: 1.2; }
p { margin: 3px 0; }
ul, ol { margin: 3px 0 3px 16px; padding: 0; } li { margin: 1.5px 0; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 3px; font-size: 7.9pt; }
th, td { border-bottom: 1px solid #e4e4e7; padding: 2.2px 5px; vertical-align: top; text-align: left; }
th { font-weight: 600; color: #18181b; border-bottom: 1.5px solid #a1a1aa; }
tr { break-inside: avoid; page-break-inside: avoid; } h2 { break-after: avoid; page-break-after: avoid; }
code { font-family: Menlo, monospace; font-size: 7.7pt; background: #f4f4f5; padding: 0 3px; border-radius: 3px; color: #3f3f46; }
strong { color: #18181b; }
.formula { font-family: Menlo, monospace; font-size: 8pt; background: #eef2ff; border-left: 3px solid #4f46e5; padding: 5px 8px; margin: 4px 0 4px; border-radius: 0 4px 4px 0; color: #1e1b4b; }
.diagram { margin: 4px 0 2px; } .diagram svg { width: 100%; height: auto; display: block; }
a { color: #4f46e5; text-decoration: none; }
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
