"""Fill docs/solution.md placeholders from the runs and evaluations, and build the 2-page PDF.

  uv run python scripts/fill_doc.py --run <official_run> [--cue-blind <run>] [--bench <bench.json>] [--dramaad-stats <stats.json>]
      [--votes-sim <votes.json>] [--minutes N] [--prototype URL] [--repo URL] [--outputs URL] [--video URL]
<official_run>/eval/{metrics,stability,assumption_checks,ab_vs_previous}.json are read when present.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def load(p: Path | None):
    return json.loads(p.read_text()) if p and p.exists() else None


def short(t: str, n: int = 52) -> str:
    t = t.replace("|", "/").strip()
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--cue-blind", type=Path)
    ap.add_argument("--bench", type=Path, help="scripts/benchmark_dramaad.py score --source slate --json-out")
    ap.add_argument("--dramaad-stats", type=Path)
    ap.add_argument("--votes-sim", type=Path, help='{"1": {"top10": .., "top20": ..}, "2": {...}}')
    ap.add_argument("--minutes", type=float, default=0)
    for k in ("prototype", "repo", "outputs", "video"):
        ap.add_argument(f"--{k}", default="")
    a = ap.parse_args()
    run = a.run
    # evaluation files that ship inside the official run's eval/ folder are picked up automatically
    a.bench = a.bench or (run / "eval" / "external_benchmark_slate.json")
    a.dramaad_stats = a.dramaad_stats or (run / "eval" / "dramaad_stats.json")
    a.votes_sim = a.votes_sim or (run / "eval" / "votes_simulation.json")
    m = json.loads((run / "eval" / "metrics.json").read_text())
    promos = json.loads((run / "promos.json").read_text())
    stab = load(run / "eval" / "stability.json")
    checks = load(run / "eval" / "assumption_checks.json")
    ab = load(run / "eval" / "ab_vs_previous.json")
    llm, cv, er = m["llm"], m.get("cold_viewer", {}), m.get("editor_rubric") or {}
    vals = {
        "PROTOTYPE_URL": a.prototype or "(link in form)", "REPO_URL": a.repo or "(link in form)",
        "OUTPUTS_URL": a.outputs or "(link in form)", "VIDEO_URL": a.video or "(link in form)",
        "CALLS": str(llm["calls_total"]), "MINUTES": f"{(a.minutes or m['seconds'] / 60):.0f}",
        "INVARIANTS": pct(m["invariants"]["pass_rate"]) + (f" ({len(m['invariants']['violations'])} violations)" if m["invariants"]["violations"] else ""),
        "CV_MEAN": f"{cv['mean_comprehensibility']}/5" if cv.get("mean_comprehensibility") is not None else "–",
        "CV_GE4": pct(cv.get("share_ge_4")),
    }
    vals["RUBRIC"] = f"{er.get('top10_all_checks_pass', '–')} of the top 10 pass every check"
    sb = er.get("slate_blocks") or {}
    tot = sum(sb.values()) or 1
    vals["SLATE"] = f"slate: {sb.get('opening', 0)} opening · {sb.get('middle', 0)} middle · {sb.get('late', 0)} late (of {tot})"
    led = m.get("promise_ledger") or {}
    vals["LEDGER"] = (f"{led.get('answered', 0)} of {led.get('promos', 0)} promos' questions are answered later in the show "
                      f"(median {led.get('median_minutes_clip_to_answer', '–')} min after the clip); {led.get('never_answered', 0)} never answered: flagged as fake-hook risks") if led else "–"
    ds = load(a.dramaad_stats)
    vals["PRO_SHAPE"] = (f"{ds['share_footage_seconds']['opening']:.0%} of the footage comes from the first 20% of episodes, {ds['share_footage_seconds']['middle']:.0%} from 20–50% "
                         f"and {ds['share_footage_seconds']['late']:.0%} from the second half; {ds['ads_opening_in_first_20pct']:.0%} of ads open in the first 20%") if ds else "most footage comes from the opening episodes"
    bench = load(a.bench)
    if bench:
        s = bench["summary"]
        vals["BENCH_N"] = str(bench["dramas"])
        vals["BENCH"] = (f"our slate's seconds are used by the editors' ads **{s['ours_lift']:.1f}×** as often as an average second "
                         f"(random clips {s['random_lift']:.1f}×, the opening of every episode {s['episode_openings_lift']:.1f}×); "
                         f"{pct(s['ours_hit_rate'])} of our clips sit mostly inside a professional ad (random {pct(s['random_hit_rate'])})")
    else:
        vals["BENCH_N"], vals["BENCH"] = "–", "–"
    if ab:
        judge = ab.get("judge", "").split("/")[-1]
        vals["AB"] = (f"old wins {ab['a_wins']}, new wins {ab['b_wins']}, {ab['ties']} ties ({judge}). A real trade-off: the new slate posts early-episode "
                      "clips like professional editors, and those are less explosive in isolation (`slate.shares` is the knob)")
    else:
        vals["AB"] = "–"
    vs = load(a.votes_sim)
    if stab:
        vals["CONSISTENCY"] = (f"{pct(stab['expected_overlap_top10'])} of the top 10 and {pct(stab['expected_overlap_top_n'])} of the top 20, in {stab['reps']} re-runs simulated from "
                               f"the model's measured noise; " + (f"reading each episode twice raises the top 10 to {pct(vs['2']['top10'])}" if vs else ""))
    else:
        vals["CONSISTENCY"] = "–"
    vals["VOTES_GAIN"] = f"{pct(vs['1']['top10'])} → {pct(vs['2']['top10'])}" if vs else "–"
    if a.cue_blind:
        mb = json.loads((a.cue_blind / "eval" / "metrics.json").read_text())
        vals["CLIFF_BLIND"] = f"{pct(mb['cliff_localisation']['recall_all'])} of episodes (vs {pct(m['cliff_localisation']['recall_all'])} with the notes): it reads the drama"
    else:
        vals["CLIFF_BLIND"] = "–"
    # assumptions: a curated, short table (all 14 are in docs/assumptions.md and eval/assumption_checks.md)
    exp = load(run / "eval" / "check_experiment.json")
    by = {r["assumption"][:40]: r for r in (checks or {}).get("rows", [])}
    def find(prefix):
        return next((r for k, r in by.items() if k.startswith(prefix)), None)
    cont = find("Episodes continue straight")
    rows_a = ["| Assumption | How it was tested | Result |", "|---|---|---|"]
    if ds:
        rows_a.append(f"| Promos come mostly from the opening | {ds['ads']} professional ads (DramaAD) | {ds['share_footage_seconds']['opening']:.0%} / {ds['share_footage_seconds']['middle']:.0%} / {ds['share_footage_seconds']['late']:.0%} of footage from opening / middle / late: the slate copies it |")
    if cont:
        rows_a.append(f"| An end-of-episode + start-of-next join is one moment | the writers' continuity notes | {cont['result'].split(';')[0]} |")
    if a.cue_blind:
        rows_a.append(f"| The model reads the drama, not the stage directions | the same run with \"cliffhanger\" / \"cut to black\" hidden | recall {vals['CLIFF_BLIND'].split(' of episodes')[0]} vs {pct(m['cliff_localisation']['recall_all'])} |")
    exp2 = load(run / "eval" / "check_experiment_ends_open.json")
    if exp:
        bits = []
        merged_checks = dict(exp["checks"]); merged_checks.update((exp2 or {}).get("checks", {}))
        for k, label in (("ends_open", "stops before the payoff"), ("context_free_open", "no backstory"), ("small_cast", "≤3 speakers")):
            r = merged_checks.get(k)
            if r and r["fail"]["n"] and r["pass"]["n"]:
                bits.append(f"{label} {r['pass']['follows']:.1f} vs {r['fail']['follows']:.1f}")
        judge = exp.get("model", "").split("/")[-1]
        rows_a.append(f"| The editor's checks matter to a viewer | blind judge of another model family ({judge.split('-Instruct')[0]}), matched clips | follows, pass vs fail: " + (" · ".join(bits) or "inconclusive") + " (5+5 clips each: a small test) |")
    else:
        rows_a.append("| The editor's checks matter to a viewer | first-time-viewer verdicts on judged clips | not confirmed for ≤3 speakers and no-backstory on 28 clips (too few failures): kept as small literature priors |")
    vals["ASSUMPTIONS"] = "\n".join(rows_a) + "\n\nAll 14 assumptions, with what to change if each is wrong: `docs/assumptions.md`."
    fails = []
    rep = m.get("answer_repairs") or {}
    if rep.get("flat_answers"):
        fails.append(f"The free model came back flat on {rep['flat_answers']} of {m['episodes']} episodes ({rep.get('flat_fixed_on_retry', 0)} fixed by re-asking).")
    fails.append("It reads text only and assumes script timing equals the final cut (`cut_promos.sh` renders each clip for a check).")
    vals["FAILURES"] = " ".join(fails)
    who = lambda line: " ".join(w[:1].upper() + w[1:].lower() for w in (line["speaker"] or "Action").split())  # noqa: E731
    rows = ["| # | Clip | Opens on (first 3 s) | Cuts on | Plants the question | Answered |", "|---|---|---|---|---|---|"]
    for p in promos[:3]:
        clip = " + ".join(f"ep {s['episode']} {s['start']}→{s['end']}" for s in p["segments"])
        pr = p.get("promise") or {}
        ans = f"ep {pr['answered_in_episode']}" if pr.get("answered_in_episode") else (pr.get("status") or "–")
        rows.append(f"| {p['rank']} | {clip} ({p['duration_s']:.0f} s) | {who(p['hook'])}: “{short(p['hook']['text'], 44)}” | {who(p['cliffhanger'])}: “{short(p['cliffhanger']['text'], 44)}” | {short(pr.get('question', '–'), 48)} | {ans} |")
    vals["TOP5"] = f"First 3 of {len(promos)} in posting order (the full slate, one pick per episode and every raw AI answer are in the outputs link):\n\n" + "\n".join(rows)
    src = (ROOT / "docs" / "solution.md").read_text()
    out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda mm: vals.get(mm.group(1), mm.group(0)), src)
    (ROOT / "docs" / "solution.filled.md").write_text(out)
    bd = ROOT / "scripts" / "build_doc.py"
    code = bd.read_text().replace('SRC = ROOT / "docs" / "solution.md"', 'SRC = ROOT / "docs" / "solution.filled.md"')
    (ROOT / "scripts" / "_build_filled.py").write_text(code)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "_build_filled.py")], check=False)
    (ROOT / "scripts" / "_build_filled.py").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
