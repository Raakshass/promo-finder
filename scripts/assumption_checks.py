"""Every assumption, tested where it can be tested. Writes <run>/eval/assumption_checks.{json,md}.

  uv run python scripts/assumption_checks.py <official_run> [--extra-runs RUN ...] [--cue-blind RUN] [--dramaad-stats FILE]

Each row: the assumption, the evidence (computed here from the pack, the runs or the external benchmark), the result,
and what changes if it is wrong (the config knob). Rubric checks are validated against the first-time-viewer verdicts
of every judged clip in the given runs: if a check matters, clips that pass it should be followed / tapped more.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promo_finder.parse import parse_show  # noqa: E402


def judged_clips(runs: list[Path]) -> list[dict]:
    seen, out = set(), []
    for r in runs:
        for p in json.loads((r / "promos.json").read_text()):
            cv = p.get("cold_viewer")
            if not cv or not p.get("rubric"):
                continue
            key = tuple((s["episode"], round(s["start_s"]), round(s["end_s"])) for s in p["segments"])
            if key in seen:
                continue
            seen.add(key)
            speakers = {l.get("speaker") for l in p.get("lines", []) if l.get("kind") == "dialogue" and l.get("speaker")}
            out.append({"rubric": p["rubric"], "comp": cv["comprehensibility"], "tap": cv["would_tap"], "speakers": len(speakers)})
    return out


def split(clips: list[dict], key: str) -> str:
    ok = [c for c in clips if c["rubric"].get(key) is True]
    bad = [c for c in clips if c["rubric"].get(key) is False]
    if len(ok) < 3 or len(bad) < 3:
        return f"too few failing clips to test (pass {len(ok)}, fail {len(bad)})"
    return (f"follows {st.mean(c['comp'] for c in ok):.2f} vs {st.mean(c['comp'] for c in bad):.2f}, would tap "
            f"{st.mean(c['tap'] for c in ok):.2f} vs {st.mean(c['tap'] for c in bad):.2f} (pass n={len(ok)} / fail n={len(bad)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--extra-runs", type=Path, nargs="*", default=[])
    ap.add_argument("--cue-blind", type=Path)
    ap.add_argument("--dramaad-stats", type=Path, help="JSON from scripts/benchmark_dramaad.py stats")
    a = ap.parse_args()
    run = a.run
    m = json.loads((run / "eval" / "metrics.json").read_text())
    promos = json.loads((run / "promos.json").read_text())
    args = json.loads((run / "run_log.json").read_text())["args"]
    eps = parse_show(Path(args["show_dir"]))
    rows = []

    def row(assumption, evidence, result, knob):
        rows.append({"assumption": assumption, "evidence": evidence, "result": result, "if_wrong": knob})

    # 1. timing
    blocks = sum(e.blocks_seen for e in eps)
    parsed = sum(e.blocks_parsed for e in eps)
    durs = [e.duration for e in eps]
    row("Script timestamps are the edit's timestamps.",
        "Internal consistency of the pack (no video to compare against).",
        f"{parsed}/{blocks} timed blocks parse; episodes {min(durs):.0f}-{max(durs):.0f} s (median {st.median(durs):.0f} s), the 1-3 min length of a micro-drama episode. "
        "Cannot be proven from text: every clip is rendered by cut_promos.sh for a human check.",
        "Re-time with a speech aligner (WhisperX, free) before cutting.")
    # 2. camera cuts
    cam_starts = {(e.number, round(ev.t0, 2)) for e in eps for ev in e.events if ev.kind == "camera"}
    line_starts = {(e.number, round(ev.t0, 2)) for e in eps for ev in e.events}
    seg_starts = [(s["episode"], round(s["start_s"], 2)) for p in promos for s in p["segments"]]
    on_cam = sum(1 for x in seg_starts if x in cam_starts)
    on_block = sum(1 for x in seg_starts if x in line_starts)
    shots = sorted(ev.t1 - ev.t0 for e in eps for ev in e.events if ev.kind == "camera")
    row("CAMERA blocks are shot boundaries, so a clip that starts on one starts on a cut.",
        f"{len(shots)} CAMERA blocks, median shot {st.median(shots):.1f} s.",
        f"{on_cam}/{len(seg_starts)} promo segments start exactly on a CAMERA block, {on_block}/{len(seg_starts)} on a timed block of any kind.",
        "promo.cut_snap_s (how far back a start may snap to a cut).")
    # 3. hook within 3 s
    hook_at = [p["hook"]["at_s"] for p in promos]
    row("The hook must land in the first 3 seconds (brief; Meta: up to 47% of an ad's value is in the first 3 s).",
        "Measured on the final slate.", f"invariants {m['invariants']['pass_rate']:.0%}; hook lands at median +{st.median(hook_at):.1f} s, max +{max(hook_at):.1f} s.",
        "promo.hook_window_s.")
    # 4. continuity
    notes = [e.continuity.lower() for e in eps if e.continuity]
    direct = sum(1 for n in notes if re.search(r"picks up|directly|continues|follows|immediately|moments after|same scene", n))
    row("Episodes continue straight into each other, so a tail-of-N + head-of-N+1 join is a real continuous moment.",
        "The writers' own continuity notes.", f"{direct}/{len(notes)} notes say the episode picks up directly or immediately; joins are only built between consecutive numbers.",
        "candidates.boundary_span_s = 0 disables joins.")
    # 5. stage directions
    if a.cue_blind:
        mb = json.loads((a.cue_blind / "eval" / "metrics.json").read_text())
        row("Writers' stage directions ('cliffhanger', 'cut to black') are not what the model keys on.",
            "Cue-blind ablation: the same run with those words hidden from the model.",
            f"known-cliff recall {mb['cliff_localisation']['recall_all']:.0%} blind vs {m['cliff_localisation']['recall_all']:.0%} with cues.",
            "scoring.writer_cue_bonus.")
    # 6. episode positions
    if a.dramaad_stats and a.dramaad_stats.exists():
        ds = json.loads(a.dramaad_stats.read_text())
        sh = ds["share_footage_seconds"]
        row("Promos should come mostly from the opening of the show.",
            f"DramaAD: {ds['ads']} ads cut by professional editors from {ds['dramas']} short dramas.",
            f"ad footage: {sh['opening']:.0%} from the first 20% of episodes, {sh['middle']:.0%} from 20-50%, {sh['late']:.0%} from the second half; "
            f"{ds['ads_opening_in_first_20pct']:.0%} of ads open in the first 20%. The slate copies this shape: {m.get('editor_rubric', {}).get('slate_blocks')}.",
            "slate.shares / slate.opening_end, or --paywall N.")
    # 7-10. rubric checks vs the first-time viewer
    clips = judged_clips([run] + list(a.extra_runs))
    for key, text in [("ends_open", "A clip that answers its own question is not a promo (ends before the payoff)."),
                      ("small_cast", "A stranger can follow at most 3 speakers."),
                      ("context_free_open", "The opening line must not lean on backstory."),
                      ("opens_on_conflict", "Open on conflict, not setup.")]:
        row(text, f"First-time-viewer verdicts on {len(clips)} distinct judged clips from {1 + len(a.extra_runs)} runs: pass vs fail.", split(clips, key),
            f"rubric.penalties.{key}")
    sp = [c for c in clips if c["speakers"] >= 4]
    if sp:
        row("(detail) 4+ speakers confuse a stranger.", "Same verdicts, by speaker count.",
            f"follows {st.mean(c['comp'] for c in clips if c['speakers'] <= 3):.2f} with <=3 speakers vs {st.mean(c['comp'] for c in sp):.2f} with 4+ (n={len(sp)}).",
            "rubric.max_speakers.")
    # 11. model reliability
    rep = m.get("answer_repairs") or {}
    row("The free model's answers can be trusted as they come.",
        "Every answer is validated; flat (all-zero) episodes are detected.",
        f"FALSE as stated: {rep.get('flat_answers', 0)}/{m['episodes']} episodes came back flat inside a batch; {rep.get('flat_fixed_on_retry', 0)} were fixed by re-asking alone, "
        f"the rest confirmed. {rep.get('episodes_on_heuristic', 0)} episodes fell back to the keyword twin.",
        "--batch 1 avoids most flat answers; --votes 2 averages two readings.")
    # 12. durations
    d = [p["duration_s"] for p in promos]
    row("A promo is 20-60 s (brief).", "Measured on the final slate.", f"{min(d):.0f}-{max(d):.0f} s, median {st.median(d):.0f} s.", "promo.min/max_duration_s.")
    out = {"run": str(run), "rows": rows}
    (run / "eval" / "assumption_checks.json").write_text(json.dumps(out, indent=1))
    md = ["# Assumptions, tested", "", "| Assumption | Evidence | Result | If it is wrong |", "|---|---|---|---|"]
    md += [f"| {r['assumption']} | {r['evidence']} | {r['result']} | {r['if_wrong']} |" for r in rows]
    (run / "eval" / "assumption_checks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
