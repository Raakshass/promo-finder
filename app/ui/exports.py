"""Downloads (spec 5.2). Owner: Lane P. run_zip is P1 (cut).

- cut_list_csv: the run's own promos.csv (one row per segment), written back unchanged, plus three columns for the
  editor's decisions. Never one row per promo: a two-part promo has two cut rows.
- review_csv: only the promos the editor decided.
- timecodes_text: plain ASCII lines that paste anywhere.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from app.ui import fmt
from app.ui.data import RunInfo

CUT_COLUMNS = ["rank", "promo_id", "segment", "episode", "in", "out", "duration_s", "hook_type", "hook_line",
               "cliffhanger_line", "why_it_works", "score", "cold_viewer_comprehensibility", "spoiler_risk", "show_block",
               "minutes_into_show", "editor_checks"]
CALL_COLUMNS = ["your_call", "trim_note", "reject_reason"]
REVIEW_COLUMNS = ["run_id", "show", "promo_id", "rank", "segments", "your_call", "trim_note", "reject_reason",
                  "decided_at"]
KEPT = ("approve", "trim")


def _cell(x: object) -> str:
    """A CSV cell: "" for None, Python's own text otherwise (as the pipeline's csv.writer wrote it)."""
    return "" if x is None else str(x)


def _editor_checks(rubric: dict) -> str:
    """The pipeline's rubric summary, for runs that have no promos.csv."""
    if not rubric:
        return ""
    try:
        from promo_finder.rubric import LABELS
    except Exception:  # noqa: BLE001
        LABELS = {}
    fails = [LABELS.get(k, k) for k, ok in rubric.items() if not ok]
    return "passes every check" if not fails else "fails: " + ", ".join(fails)


def _rows_from_json(promos: list[dict]) -> list[list[str]]:
    """The 17 promos.csv columns, one row per segment, rebuilt from promos.json. Unknown fields are left blank."""
    rows = [list(CUT_COLUMNS)]
    for p in promos:
        flags = p.get("flags") or {}
        cv = p.get("cold_viewer") or {}
        for j, s in enumerate(p.get("segments") or [], 1):
            try:
                dur = round(float(s["end_s"]) - float(s["start_s"]), 2)
            except (KeyError, TypeError, ValueError):
                dur = ""
            rows.append([_cell(x) for x in (
                p.get("rank"), p.get("promo_id"), j, s.get("episode"), s.get("start"), s.get("end"), dur,
                flags.get("hook_type"), (p.get("hook") or {}).get("text"), (p.get("cliffhanger") or {}).get("text"),
                p.get("why_it_works"), p.get("score"), cv.get("comprehensibility"), flags.get("spoiler_risk"),
                flags.get("block"), flags.get("minutes_into_show"), _editor_checks(p.get("rubric") or {}))])
    return rows


def cut_list_csv(data: dict, review: dict[str, dict], only_kept: bool = False) -> str:
    """The run's per-segment cut list plus your_call, trim_note and reject_reason, joined on promo_id.
    The run's own columns are written back with csv.writer (default dialect), so they stay byte-identical."""
    text = (data or {}).get("promos_csv") or ""
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r] if text.strip() else []
    if not rows:
        rows = _rows_from_json((data or {}).get("promos") or [])
    header, body = rows[0], rows[1:]
    pid_col = header.index("promo_id") if "promo_id" in header else 1
    review = review or {}
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(header + CALL_COLUMNS)
    for r in body:
        pid = r[pid_col] if pid_col < len(r) else ""
        entry = review.get(pid) or {}
        call = entry.get("call") or ""
        if only_kept and call not in KEPT:
            continue
        reason = (entry.get("reason") or "") if call == "reject" else ""
        wr.writerow(r + [call, entry.get("note") or "", reason])
    return buf.getvalue()


def review_csv(run: RunInfo, data: dict, review: dict[str, dict]) -> str:
    """One row per decided promo, in posting order."""
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(REVIEW_COLUMNS)
    review = review or {}
    for p in sorted((data or {}).get("promos") or [], key=lambda p: p.get("rank") or 0):
        entry = review.get(p.get("promo_id")) or {}
        call = entry.get("call")
        if not call:
            continue
        reason = (entry.get("reason") or "") if call == "reject" else ""
        wr.writerow([run.run_id, run.show_title, _cell(p.get("promo_id")), _cell(p.get("rank")), " + ".join(fmt.segs(p)),
                     call, entry.get("note") or "", reason, entry.get("at") or ""])
    return buf.getvalue()


def _seg_len(s: dict) -> str:
    try:
        return fmt.secs(float(s["end_s"]) - float(s["start_s"]))
    except (KeyError, TypeError, ValueError):
        return ""


def timecodes_text(promos: list[dict]) -> str:
    """One line per segment: "#5  Ep 10  01:22.50–01:34.00  11.5 s"; continuation segments are indented under their
    promo. Spaces only, so it pastes anywhere."""
    lines: list[str] = []
    for p in sorted(promos or [], key=lambda p: p.get("rank") or 0):
        tag = f"#{p.get('rank')}"
        for j, s in enumerate(p.get("segments") or []):
            lead = tag if j == 0 else " " * len(tag)
            lines.append(f"{lead}  Ep {s.get('episode')}  {s.get('start')}–{s.get('end')}  {_seg_len(s)}".rstrip())
    return "\n".join(lines)


def run_zip(run_dir: Path) -> bytes:
    """P1, cut. Do not build."""
    return b""
