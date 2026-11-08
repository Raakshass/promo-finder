"""Helpers shared by the app tests (import as `from apphelpers import ...`; conftest.py can't be imported by name).

- find_official / find_evidence locate runs by marker or folder prefix, never by full folder name, so a new official
  run needs no test edits.
- rendered_text(at) collects every string an AppTest run put on the page (labels, bodies, captions, options, help,
  table and dataframe cells of the displayed columns), as UTF-8 text.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submission" / "outputs"
APP = ROOT / "app" / "streamlit_app.py"


def find_official(out: Path = BUNDLE) -> Path | None:
    marked = sorted(p.parent for p in out.glob("*/*/OFFICIAL") if (p.parent / "promos.json").exists())
    return marked[-1] if marked else None


def find_evidence(prefix: str, out: Path = BUNDLE) -> Path | None:
    """prefix: previous_version, heuristic_only, cue_blind_ablation, stability_base or stability_rerun."""
    found = sorted(p.parent for p in out.glob(f"*/{prefix}__*/promos.json"))
    return found[-1] if found else None


def _frame_text(node) -> list[str]:
    try:
        df = node.value
    except Exception:  # noqa: BLE001
        return []
    order = list(getattr(node.proto, "column_order", []) or [])
    cols = [c for c in order if c in df.columns] or list(df.columns)
    out = [str(c) for c in cols]
    for c in cols:
        out += [repr(v) if v is None else str(v) for v in df[c].tolist()]
    return out


def rendered_text(at) -> str:
    """Every string on the page after at.run(), one per line."""
    from google.protobuf import text_format

    parts: list[str] = []
    for node in at._tree:
        kind = type(node).__name__
        proto = getattr(node, "proto", None)
        if kind in ("Dataframe", "Table"):
            parts += _frame_text(node)
            continue
        if proto is None or proto.DESCRIPTOR.name == "PlotlyChart":
            continue  # chart specs carry customdata (promo ids) that is never shown as text; test hovers directly
        try:
            parts.append(text_format.MessageToString(proto, as_utf8=True))
        except Exception:  # noqa: BLE001
            continue
    return "\n".join(parts)
