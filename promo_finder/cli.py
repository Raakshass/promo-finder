"""promo-finder command line."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import evaluate as ev
from .beat_scoring import PROMPT_VERSION, build_aliases, score_show
from . import ledger as led
from . import promoability
from .judge import ab_test, cold_viewer, entailment, rank_reasons
from .llm import LLMError, RunLog, make_llm, parse_model_string
from .parse import parse_report, parse_show
from .rank import apply_judge, diversity_order, rank_windows
from . import slate as sl
from .report import per_episode_csv, to_promos, write_run
from .show_arc import build_show_arc

app = typer.Typer(add_completion=False, help="Find and rank promo clips in timestamped microdrama screenplays.")
console = Console(emoji=False)
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None) -> dict:
    p = path or ROOT / "config.yaml"
    return yaml.safe_load(p.read_text())


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


@app.command()
def parse(show_dir: Path, json_out: bool = typer.Option(False, "--json", help="print the full parse report as JSON")):
    """Sanity-check an input pack in seconds: counts, durations, warnings, name drift."""
    skipped: list[dict] = []
    try:
        eps = parse_show(show_dir, skipped)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]cannot read this show: {e}[/red]")
        raise typer.Exit(1)
    for sk in skipped:
        console.print(f"[yellow]skipped {sk['file']}: {sk['reason']}[/yellow]")
    rep = parse_report(eps)
    if json_out:
        print(json.dumps(rep, indent=1))
        return
    t = Table(title=f"{show_dir.name}: {rep['episodes']} episodes, {rep['total_blocks']} blocks, {rep['block_kinds']}")
    t.add_column("ep", justify="right"); t.add_column("dur s", justify="right"); t.add_column("scenes", justify="right")
    t.add_column("lines", justify="right"); t.add_column("parse", justify="right"); t.add_column("warnings")
    for r in rep["per_episode"]:
        t.add_row(str(r["episode"]), f"{r['duration_s']:.1f}", str(r["scenes"]), str(r["dialogue_lines"]), f"{r['parse_rate']:.0%}", "; ".join(r["warnings"])[:90])
    console.print(t)
    console.print(f"speaker labels: {len(rep['drift']['speakers'])}; spelling clusters: {rep['drift']['spelling_clusters']}")


@app.command()
def models(provider: str = typer.Option("gemini", help="gemini | groq | openrouter | openai | ollama | anthropic")):
    """List the models your key can reach (so a judge can pick one live)."""
    load_dotenv(ROOT / ".env")
    try:
        llm = make_llm(f"{provider}/_", load_config(None)["llm"], None, RunLog())
        for m in llm.list_models():
            print(m)
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def check(model: str = typer.Option(None, "--model", "-m", help="provider/model (default: $PROMO_MODEL)")):
    """One tiny call: is this model usable with the keys in .env? Prints the provider's own message if not."""
    from pydantic import BaseModel

    class Ping(BaseModel):
        ok: bool

    load_dotenv(ROOT / ".env")
    model_s = model or os.environ.get("PROMO_MODEL") or "none"
    try:
        llm = make_llm(model_s, {**load_config(None)["llm"], "max_retries": 1}, None, RunLog())
        if llm is None:
            console.print("model is 'none': heuristic-only mode, nothing to check")
            return
        out = llm.complete_json("Answer with JSON only.", 'Return exactly {"ok": true}', Ping, "check", "ping")
        rec = llm.log.calls[-1]
        console.print(f"[green]OK[/green] {llm.name} answered (ok={out.ok}; {rec.input_tokens}/{rec.output_tokens} tokens)")
    except LLMError as e:
        console.print(f"[red]NOT USABLE[/red] {e}")
        raise typer.Exit(1)


@app.command()
def run(
    show_dir: Path,
    model: str = typer.Option(None, "--model", "-m", help="provider/model, or 'none' (default: $PROMO_MODEL)"),
    judge_model: str = typer.Option(None, "--judge-model", help="provider/model for the cold-viewer test (default: $PROMO_JUDGE_MODEL or --model)"),
    out: Path = typer.Option(None, "--out", help="output root (default: outputs/)"),
    config: Path = typer.Option(None, "--config"),
    limit: int = typer.Option(0, "--limit", help="only the first N episodes (quick test)"),
    no_judge: bool = typer.Option(False, "--no-judge"),
    no_cache: bool = typer.Option(False, "--no-cache", help="ignore the prompt cache (stability runs)"),
    cue_blind: bool = typer.Option(False, "--cue-blind", help="ablation: hide the writers' 'cliffhanger/cut to black' cues from the model"),
    tag: str = typer.Option("", "--tag", help="suffix for the run id"),
    no_entailment: bool = typer.Option(False, "--no-entailment", help="skip the continuity-note entailment check (saves ~1 call per episode)"),
    rpm: float = typer.Option(0, "--rpm", help="override llm.rpm from config.yaml (requests per minute throttle)"),
    judge_top_n: int = typer.Option(0, "--judge-top-n", help="override judge.top_n (how many finalists get the cold-viewer test)"),
    concurrency: int = typer.Option(0, "--concurrency", help="override llm.concurrency (parallel model calls; use 1 on tight credit limits)"),
    batch: int = typer.Option(0, "--batch", help="episodes per scoring call and clips per judge call (pairs per entailment call = 2x); fewer calls for request-capped free tiers"),
    max_tokens: int = typer.Option(0, "--max-tokens", help="override the per-call output cap (default 4096; batched calls need more)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="print every retry and transport error as it happens"),
    reasoning_off: bool = typer.Option(False, "--reasoning-off", help="ask OpenAI-compatible providers to disable model 'thinking' (cheaper, faster)"),
    paywall: int = typer.Option(0, "--paywall", help="first paid episode: the opening block of the slate becomes the free episodes before it"),
    votes: int = typer.Option(1, "--votes", help="independent readings per episode, averaged (2 = steadier ranking for 2x the calls)"),
    fast: bool = typer.Option(False, "--fast", help="interview preset for an uncapped API: one episode per call, 8 calls in parallel, judge only the final slate, skip the continuity evaluation"),
):
    """Run the full pipeline on a show folder and write outputs/<show>/<run_id>/."""
    load_dotenv(ROOT / ".env")
    if verbose:
        os.environ["PROMO_VERBOSE"] = "1"
    cfg = load_config(config)
    if rpm:
        cfg["llm"]["rpm"] = rpm
    if no_entailment:
        cfg["eval"]["entailment"] = False
    if judge_top_n:
        cfg["judge"]["top_n"] = judge_top_n
    if concurrency:
        cfg["llm"]["concurrency"] = concurrency
    if batch:
        cfg["llm"]["episodes_per_call"] = batch
        cfg["judge"]["clips_per_call"] = batch
        cfg["eval"]["pairs_per_call"] = 2 * batch
        cfg["llm"].setdefault("max_tokens", 4096 * batch)
    if max_tokens:
        cfg["llm"]["max_tokens"] = max_tokens
    if reasoning_off:
        cfg["llm"]["reasoning_off"] = True
    if paywall:
        cfg["slate"]["paywall"] = paywall
    if fast:
        cfg["llm"]["episodes_per_call"] = cfg["llm"].get("episodes_per_call", 1) if batch else 1
        cfg["llm"]["concurrency"] = concurrency or 8
        cfg["llm"]["rpm"] = rpm or 240
        cfg["judge"]["top_n"] = judge_top_n or cfg["output"]["final_top_n"]
        cfg["judge"]["clips_per_call"] = cfg["judge"].get("clips_per_call", 5) if batch else 5
        cfg["eval"]["entailment"] = False
        if not max_tokens:  # long episodes overflow 4,096 output tokens (3 of 240 benchmark episodes did)
            cfg["llm"]["max_tokens"] = max(int(cfg["llm"].get("max_tokens") or 0), 8192)
    t_start = time.time()
    skipped: list[dict] = []
    try:
        eps = parse_show(show_dir, skipped)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]cannot read this show: {e}[/red]")
        raise typer.Exit(1)
    for sk in skipped:
        console.print(f"[yellow]skipped {sk['file']}: {sk['reason']}[/yellow]")
    if limit:
        eps = eps[:limit]
    parse_rep = parse_report(eps)
    parse_rep["skipped_files"] = skipped
    show = slug(show_dir.name)
    model_s = model or os.environ.get("PROMO_MODEL") or "none"
    parsed = parse_model_string(model_s)
    model_slug = slug(f"{parsed[0]}-{parsed[1]}") if parsed else "heuristic"
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{model_slug}" + ("-cueblind" if cue_blind else "") + (f"-{tag}" if tag else "")
    run_dir = (out or ROOT / "outputs") / show / run_id
    raw_dir = run_dir / "llm_raw"
    cache_dir = None if no_cache else ROOT / ".cache"
    log = RunLog()
    try:
        llm = make_llm(model_s, cfg["llm"], cache_dir, log)
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    # the judge follows --model unless --judge-model is given: a model chosen on the command line (a test, a mock, a
    # free model) must never be paired silently with a different, possibly paid, judge from .env
    judge_s = judge_model or (model_s if model else (os.environ.get("PROMO_JUDGE_MODEL") or model_s))
    if llm is None and not judge_model:
        judge_s = "none"  # heuristic-only mode makes zero model calls unless a judge is named explicitly
    if parse_model_string(judge_s) == parsed:
        judge_llm = llm
    else:
        try:
            judge_llm = make_llm(judge_s, cfg["llm"], cache_dir, log)
        except LLMError as e:
            console.print(f"[yellow]judge model unavailable ({e}); using the main model as judge[/yellow]")
            judge_s, judge_llm = model_s, llm
    if judge_llm is None:
        judge_llm = llm
    console.print(f"[bold]{show}[/bold]: {len(eps)} episodes · model: {llm.name if llm else 'none (heuristic twin)'} · judge: {judge_llm.name if judge_llm else 'none'} · run {run_id}")
    if parse_rep["episodes_with_warnings"]:
        console.print(f"parser warnings in {parse_rep['episodes_with_warnings']} episodes (see parse_report.json)")

    try:
        aliases, notes = build_aliases(eps, llm, raw_dir)
    except LLMError as e:
        console.print(f"[red]model unusable, aborting before the full run: {e}[/red]\nTry `promo-finder check --model <provider/model>` or `--model none`.")
        raise typer.Exit(2)
    if aliases:
        console.print(f"aliases merged: {len(aliases)} labels → {len(set(aliases.values()))} canonical names")
    try:
        arc, arc_note = build_show_arc(eps, llm, raw_dir, aliases)
    except LLMError as e:
        console.print(f"[red]model unusable, aborting before the full run: {e}[/red]")
        raise typer.Exit(2)
    console.print(f"show arc ({arc.source}): leads {arc.leads} · {len(arc.turning_points)} turning points · {len(arc.major_reveals)} major reveals"
                  + (f" · {arc_note}" if arc_note != "ok" else ""))

    with Progress(TextColumn("scoring beats"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("score", total=len(eps))
        try:
            ctx = score_show(eps, llm, cfg, raw_dir, strip_cues=cue_blind, progress=lambda _n: prog.advance(task), aliases=aliases, alias_notes=notes, votes=votes)
        except LLMError as e:
            console.print(f"[red]model unusable, aborting: {e}[/red]")
            raise typer.Exit(2)
    if ctx.llm_failures:
        console.print(f"[yellow]{len(ctx.llm_failures)} episodes fell back to heuristics: {sorted(ctx.llm_failures)}[/yellow]")
    source = "llm" if ctx.has_llm else "heuristic"
    ctx.arc = arc

    finalists, all_windows = rank_windows(ctx, cfg, source)
    heur_finalists, _ = rank_windows(ctx, cfg, "heuristic")
    console.print(f"candidate windows: {len(all_windows)} → finalists: {len(finalists)}")

    judged, judge_errors = {}, {}
    if judge_llm is not None and cfg["judge"]["enabled"] and not no_judge:
        with console.status(f"cold-viewer test on {len(finalists)} finalists ({judge_llm.name})"):
            judged, judge_errors = cold_viewer(finalists, ctx, judge_llm, raw_dir, per_call=int(cfg['judge'].get('clips_per_call', 1)))
        apply_judge(finalists, judged, cfg)
    final_n = cfg["output"]["final_top_n"]
    # the posted slate copies the professional shape (slate.py): quotas per block of the show, best-first inside each
    final = sl.build(finalists, cfg, final_n, diversity_order)

    provenance = {"provider": llm.provider if llm else "none", "model": llm.model if llm else "heuristic",
                  "judge": judge_llm.name if judge_llm else "none", "prompt_version": PROMPT_VERSION, "run_id": run_id}
    promos = to_promos(final, ctx, judged, {}, provenance, source)
    if llm is not None and not no_judge:
        cards = [{"promo_id": p.promo_id, "rank": p.rank, "episode": p.segments[0].episode, "duration_s": p.duration_s,
                  "hook_type": p.flags.get("hook_type"), "hook": p.hook.text, "cliff": p.cliffhanger.text, "subscores": p.subscores,
                  "cold_viewer": (p.cold_viewer.what_happened + " / " + p.cold_viewer.want_next) if p.cold_viewer else "n/a"} for p in promos[:10]]
        reasons = rank_reasons(cards, llm, raw_dir)
        for p in promos:
            p.post_first_reason = reasons.get(p.promo_id, "")

    # promise & payoff: the question each promo plants, when a new viewer reaches it, where the show answers it
    ledger_rows = led.build(promos, ctx, None if no_judge else (judge_llm if judge_llm is not None else llm), raw_dir, cfg["slate"].get("paywall"))
    for p in promos:
        p.promise = ledger_rows.get(p.promo_id, {})

    ent = {}
    if llm is not None and cfg["eval"]["entailment"]:
        best_by_ep = {}
        for w in sorted(all_windows, key=lambda x: -x.score):
            n = w.segments[0].episode
            if n not in best_by_ep and w.kind == "single":
                best_by_ep[n] = w
        with console.status(f"continuity-note entailment on {len(best_by_ep)} episodes"):
            ent = entailment(ctx, best_by_ep, llm, raw_dir, per_call=int(cfg['eval'].get('pairs_per_call', 1)))

    metrics = {
        "run_id": run_id, "model": provenance, "episodes": len(eps), "windows": len(all_windows), "finalists": len(finalists),
        "invariants": ev.invariants(promos, cfg),
        "cliff_localisation": ev.cliff_localisation(ctx, cfg, source),
        "cliff_localisation_heuristic": ev.cliff_localisation(ctx, cfg, "heuristic"),
        "heuristic_vs_llm_top10": ev.ranking_overlap(finalists, heur_finalists, 10) if source == "llm" else None,
        "logline_overlap": ev.logline_overlap(ctx, finalists),
        "cold_viewer": {
            "n": len(judged),
            "mean_comprehensibility": round(sum(j.comprehensibility for j in judged.values()) / len(judged), 2) if judged else None,
            "share_ge_4": round(sum(j.comprehensibility >= 4 for j in judged.values()) / len(judged), 3) if judged else None,
            "mean_would_tap": round(sum(j.would_tap for j in judged.values()) / len(judged), 2) if judged else None,
            "errors": len(judge_errors),
        },
        "entailment": entail_summary(ent),
        "editor_rubric": rubric_metrics(promos),
        "promise_ledger": led.summary(ledger_rows) if any("status" in r for r in ledger_rows.values()) else None,
        "show_arc": {"source": arc.source, "premise": arc.premise, "leads": arc.leads,
                     "turning_points": [t.model_dump() for t in arc.turning_points], "major_reveals": len(arc.major_reveals)},
        "answer_repairs": {
            "flat_answers": sum(1 for v in ctx.id_issues.values() if "flat answer" in v.get("note", "")),
            "flat_fixed_on_retry": sum(1 for v in ctx.id_issues.values() if "flat answer" in v.get("note", "") and v["note"].endswith("fixed")),
            "dropped_from_batch": sum(1 for v in ctx.id_issues.values() if "missing from batch" in v.get("note", "") or "empty in batch" in v.get("note", "")),
            "episodes_on_heuristic": len(ctx.llm_failures),
        },
        "llm": log.summary(),
        "seconds": round(time.time() - t_start, 1),
    }
    promo_rep = promoability.report(ctx, all_windows, finalists, source)
    metrics["promoability"] = promo_rep["show"]
    failures_md = ev.failures_markdown(promos, finalists, ctx, cfg, {"entailment": ent})
    write_run(run_dir, show=show_dir.name, promos=promos, ctx=ctx, metrics=metrics, failures_md=failures_md, parse_rep=parse_rep,
              cfg=cfg, log_summary=log.summary(), args={"show_dir": str(show_dir), "model": model_s, "judge": judge_s, "limit": limit, "cue_blind": cue_blind, "no_cache": no_cache, "no_judge": no_judge, "no_entailment": no_entailment, "rpm": cfg["llm"]["rpm"], "batch": batch, "max_tokens": cfg["llm"].get("max_tokens"), "votes": votes, "fast": fast, "paywall": cfg["slate"].get("paywall")},
              judged_errors=judge_errors, source=source)

    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "promoability.json").write_text(json.dumps(promo_rep, indent=1), encoding="utf-8")
    (run_dir / "eval" / "promoability.md").write_text(promoability.markdown(promo_rep), encoding="utf-8")
    (run_dir / "promos_by_episode.csv").write_text(per_episode_csv(all_windows, ctx, source), encoding="utf-8")
    from .rank import nms as _nms
    with open(run_dir / "candidates.jsonl", "w", encoding="utf-8") as fh:  # every de-duplicated candidate, best first (benchmarks, learn)
        for w in sorted(_nms(all_windows, cfg), key=lambda x: -x.score)[:300]:
            fh.write(json.dumps({"cand_id": w.cand_id, "score": w.score, "block": w.flags.get("block"), "kind": w.kind,
                                 "segments": [{"episode": g.episode, "start_s": round(g.start_s, 2), "end_s": round(g.end_s, 2)} for g in w.segments],
                                 # the four terms and the multiplier's parts, so `promo-finder learn` can re-score any candidate
                                 "subscores": {k: w.subscores[k] for k in ("hook", "cliff", "standalone", "density", "base", "rubric_mult", "arc_mult") if k in w.subscores},
                                 "judged": "standalone_proxy" in w.subscores, "hook_beat": w.hook_beat, "cliff_beat": w.cliff_beat,
                                 "hook_type": w.flags.get("hook_type"), "spoiler_risk": bool(w.flags.get("spoiler_risk"))}) + "\n")
    (run_dir / "eval" / "promise_ledger.json").write_text(json.dumps(ledger_rows, indent=1, ensure_ascii=False), encoding="utf-8")

    t = Table(title=f"Top promos → {run_dir}")
    for col in ("#", "episode", "in → out", "s", "hook", "cliffhanger", "cold", "score"):
        t.add_column(col)
    for p in promos[:12]:
        segs = " + ".join(f"{s.start}→{s.end}" for s in p.segments)
        t.add_row(str(p.rank), "+".join(str(s.episode) for s in p.segments), segs, f"{p.duration_s:.0f}", p.hook.text[:38], p.cliffhanger.text[:38],
                  f"{p.cold_viewer.comprehensibility}/5" if p.cold_viewer else "-", f"{p.score:.3f}")
    console.print(t)
    m = metrics
    ps = m["llm"].get("per_stage") or {}
    if ps:
        console.print("time per stage (live calls): " + " · ".join(f"{k} {v['live']}×{v['median_live_seconds']}s" for k, v in ps.items() if v["live"]))
    console.print(f"invariants {m['invariants']['pass_rate']:.0%} · cliff localisation {m['cliff_localisation']['recall_all']:.0%} (heuristic {m['cliff_localisation_heuristic']['recall_all']:.0%})"
                  + (f" · cold viewer {m['cold_viewer']['mean_comprehensibility']}/5" if m['cold_viewer']['mean_comprehensibility'] else "")
                  + (f" · cliff capture {m['entailment']['capture_rate']:.0%}" if m['entailment']['capture_rate'] is not None else "")
                  + f" · LLM calls {m['llm']['calls_live']} live / {m['llm']['calls_cached']} cached · est ${m['llm']['est_cost_usd_at_paid_rates']} at paid rates · {m['seconds']}s")


def entail_summary(ent: dict) -> dict:
    """Capture rate over the checks that actually ran: a failed call (quota, network) is not a miss."""
    checked = [r for r in ent.values() if r.get("contains") is not None]
    return {"n": len(checked), "attempted": len(ent), "errors": len(ent) - len(checked),
            "capture_rate": round(sum(1 for r in checked if r["contains"]) / len(checked), 3) if checked else None, "rows": ent}


def rubric_metrics(promos) -> dict:
    """Editor's-rubric pass rates on the final list and the top 10, and where in the show the slate comes from."""
    def rates(ps):
        keys = sorted({k for p in ps for k in p.rubric})
        return {k: round(sum(1 for p in ps if p.rubric.get(k)) / max(sum(1 for p in ps if k in p.rubric), 1), 3) for k in keys}
    top = promos[:10]
    return {
        "top10_pass_rates": rates(top),
        "final_pass_rates": rates(promos),
        "top10_all_checks_pass": sum(1 for p in top if p.rubric and all(p.rubric.values())),
        "top10_blocks": {b: sum(1 for p in top if p.flags.get("block") == b) for b in ("opening", "middle", "late")},
        "slate_blocks": {b: sum(1 for p in promos if p.flags.get("block") == b) for b in ("opening", "middle", "late")},
    }


@app.command()
def ab(
    run_a: Path = typer.Argument(..., help="run folder A (e.g. the previous official run)"),
    run_b: Path = typer.Argument(..., help="run folder B (e.g. the new run)"),
    model: str = typer.Option(None, "--model", help="provider/model for the blind judge (default: $PROMO_JUDGE_MODEL)"),
    k: int = typer.Option(10, "--k", help="compare the top-k slots"),
    reasoning_off: bool = typer.Option(False, "--reasoning-off"),
    config: Path = typer.Option(ROOT / "config.yaml", "--config"),
    tag: str = typer.Option("", "--tag", help="suffix for the result file (e.g. the judge's family), so several judges can be kept"),
    max_tokens: int = typer.Option(3000, "--max-tokens", help="output cap; some free models reject large caps with HTTP 400"),
):
    """Blind A/B: a first-time-viewer judge picks the clip it would tap, slot by slot, in both orders (position-bias controlled)."""
    load_dotenv(ROOT / ".env")
    cfg = load_config(config)
    if reasoning_off:
        cfg["llm"]["reasoning_off"] = True
    cfg["llm"]["max_tokens"] = max_tokens
    log = RunLog()
    llm = make_llm(model or os.environ.get("PROMO_JUDGE_MODEL") or os.environ.get("PROMO_MODEL") or "none", cfg["llm"], ROOT / ".cache", log)
    if llm is None:
        console.print("[red]the A/B test needs a model[/red]")
        raise typer.Exit(1)
    pa = json.loads((run_a / "promos.json").read_text(encoding="utf-8"))
    pb = json.loads((run_b / "promos.json").read_text(encoding="utf-8"))
    res = ab_test(pa, pb, llm, run_b / "llm_raw", k)
    res.update({"run_a": str(run_a), "run_b": str(run_b), "judge": llm.name, "llm": log.summary()})
    (run_b / "eval").mkdir(parents=True, exist_ok=True)
    fname = f"ab_vs_previous{'-' + tag if tag else ''}.json"
    (run_b / "eval" / fname).write_text(json.dumps(res, indent=1), encoding="utf-8")
    console.print(f"B (new) wins {res['b_wins']} · A (old) wins {res['a_wins']} · ties {res['ties']} · same clip {res['same_clip']} "
                  f"· judge position-consistent on {res['position_consistency']} of judged pairs → {run_b / 'eval' / fname}")


@app.command()
def stability(
    run: Path = typer.Argument(..., help="the run to report confidence for"),
    noise_a: Path = typer.Option(None, "--noise-a", help="first of two independent runs of the same model (measures the noise)"),
    noise_b: Path = typer.Option(None, "--noise-b", help="second independent run"),
    reps: int = typer.Option(100, "--reps"),
    votes: int = typer.Option(1, "--votes", help="simulate this many averaged readings per episode"),
    config: Path = typer.Option(ROOT / "config.yaml", "--config"),
):
    """Confidence via simulated re-runs (no model calls): redraw every score from the model's measured run-to-run noise,
    re-rank, and report how often each promo on the slate survives, plus the expected overlap of two runs."""
    from . import stability as stb
    cfg = load_config(config)
    ctx = stb.replay(run)
    noise_file = ROOT / "promo_finder" / "data" / "noise_model.json"
    if noise_a and noise_b:
        noise = stb.noise_model(noise_a, noise_b)
        noise_file.parent.mkdir(parents=True, exist_ok=True)
        noise_file.write_text(json.dumps(noise, indent=1))
    elif noise_file.exists():
        noise = json.loads(noise_file.read_text())  # measured on the official pack; a reasonable prior for a new show
    else:
        console.print("[red]no noise model: pass --noise-a/--noise-b (two independent runs of the same model)[/red]")
        raise typer.Exit(1)
    n = cfg["output"]["final_top_n"]
    res = stb.simulate(ctx, cfg, noise, reps=reps, n=n, votes=votes)
    # per promo: does it stay on the judge's shortlist (judge.top_n clips) when the scores are redrawn?
    short = stb.simulate(ctx, cfg, noise, reps=reps, n=cfg["judge"]["top_n"], votes=votes, seed=11)
    from .rank import iou as _iou
    from .models import Segment as _Seg, Window as _Win
    promos = json.loads((run / "promos.json").read_text())
    by_promo = {}
    for p in promos:
        pw = _Win(cand_id=p["promo_id"], segments=[_Seg(episode=s["episode"], start_s=s["start_s"], end_s=s["end_s"]) for s in p["segments"]],
                  hook_beat=p["hook"]["beat_id"], cliff_beat=p["cliffhanger"]["beat_id"], beat_ids=[], duration_s=p["duration_s"])
        hits = sum(1 for sim in short["sims"] if any(w.hook_beat == pw.hook_beat or _iou(w, pw) >= 0.5 for w in sim))
        by_promo[p["promo_id"]] = round(hits / max(len(short["sims"]), 1), 3)
    out = {"reps": reps, "votes": votes, "noise_beats": noise.get("beats"), "expected_overlap_top_n": res["expected_overlap_top_n"],
           "expected_overlap_top10": res["expected_overlap_top10"], "keep_rate_by_promo": by_promo,
           "keep_rate_definition": f"share of simulated re-runs in which the promo stays on the {cfg['judge']['top_n']}-clip shortlist the judge sees"}
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / "stability.json").write_text(json.dumps(out, indent=1))
    ks = sorted(by_promo.values())
    console.print(f"{reps} simulated re-runs (votes={votes}): two runs share {res['expected_overlap_top_n']:.0%} of the top {n} and "
                  f"{res['expected_overlap_top10']:.0%} of the top 10; each promo stays on the {cfg['judge']['top_n']}-clip shortlist in a median {ks[len(ks) // 2]:.0%} of re-runs → {run / 'eval' / 'stability.json'}")


@app.command()
def compare(run_a: Path, run_b: Path):
    """Compare two runs (stability, cue-blind ablation, model swap): top-k overlap and metric deltas."""
    print(json.dumps(ev.compare_runs(run_a, run_b), indent=1))


@app.command()
def learn(
    run: Path = typer.Argument(..., help="a finished run folder (promos.json, candidates.jsonl, config.yaml)"),
    outcomes: Path = typer.Option(..., "--outcomes", help="CSV with one row per posted (or reviewed) promo: promo_id as in promos.csv "
                                  "(or rank, or a cand_id) and a metric column; or the app's decisions.json"),
    metric: str = typer.Option(None, "--metric", help="the outcome column (default: detected, e.g. hook_rate, hold_rate, ctr, approved)"),
    kind: str = typer.Option("auto", "--kind", help="auto | continuous (hook rate, hold rate, CTR) | binary (approved/kept 0/1)"),
    prior_strength: float = typer.Option(30.0, "--prior-strength", help="how many posts the current weights are worth (30: about 30 posts move them halfway; less for a noisy metric)"),
    out: Path = typer.Option(None, "--out", help="write a config with the learned weights to this file (never over a config.yaml)"),
    plan: int = typer.Option(0, "--plan", help="also pick the next K promos to post (Thompson sampling over the weights)"),
    draws: int = typer.Option(1000, "--draws", help="posterior draws behind --plan"),
    seed: int = typer.Option(7, "--seed", help="random seed of the draws"),
    config: Path = typer.Option(ROOT / "config.yaml", "--config", help="the config whose other settings --out copies"),
    show_dir: Path = typer.Option(None, "--show-dir", help="the show folder, if it moved (only older runs need it, to rebuild candidates' terms)"),
    json_out: bool = typer.Option(False, "--json", help="print the result as JSON"),
):
    """Refit the four ranking weights from how posted promos performed (hook rate, hold rate, CTR, or approved/kept), with the
    current weights as a prior worth --prior-strength posts. --plan K picks the next K promos by Thompson sampling. No model calls."""
    from . import learn as lr
    try:
        if out:
            lr.check_out_path(out, config)
        pool = lr.load_pool(run, show_dir)
        obs = lr.read_outcomes(outcomes, metric, kind)
        joined = lr.join(obs, pool)
        res = lr.fit(joined.Z, joined.y, pool.weights, obs.kind, prior_strength, pool.Z)
        pl = lr.plan(res, pool, plan, joined.clips, draws, seed) if plan > 0 else None
        written = None
        if out:
            written = lr.write_config(out, config, res, {"run": f"{run.parent.name}/{run.name}", "outcomes": outcomes.name,
                                                         "metric": obs.metric, "unknown": len(joined.unknown)})
    except lr.LearnError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    notes = pool.notes + obs.notes + ([f"{obs.skipped} rows had no {obs.metric} value and were skipped"] if obs.skipped else [])
    if json_out:
        print(json.dumps({"run": run.name, "metric": obs.metric, "id_column": obs.id_column, "rows": len(obs.ids), "unknown_ids": joined.unknown,
                          "fit": res.as_dict(), "plan": pl, "written": str(out) if written else None, "notes": notes}, indent=1))
        return
    for n_ in notes:
        console.print(f"[dim]{n_}[/dim]")
    if joined.unknown:
        console.print(f"[yellow]{len(joined.unknown)} id{'s are' if len(joined.unknown) != 1 else ' is'} not in this run and skipped: {', '.join(joined.unknown[:8])}"
                      + (" ..." if len(joined.unknown) > 8 else "") + "[/yellow]")
    console.print(f"[bold]Ranking weights learned from {obs.metric}[/bold] ({obs.kind}, {res.n} posts) · prior: the current weights, "
                  f"worth {res.prior_strength:g} posts")
    t = Table()
    for col in ("term", "old", "new", "± sd", "change"):
        t.add_column(col, justify="left" if col == "term" else "right")
    for term, o, n_, s in zip(lr.TERMS, res.old, res.new, res.sd):
        t.add_row(term, f"{o:.3f}", f"{n_:.3f}", f"{s:.3f}", f"{n_ - o:+.3f}")
    console.print(t)
    st = res.stats
    fit_line = (f"old score vs {obs.metric}: r = {st.get('r_old_score')} · learned score: r = {st.get('r_new_score_in_sample')} (in-sample)"
                if res.kind == "continuous" else
                f"old score AUC {st.get('auc_old_score')} · learned score AUC {st.get('auc_new_score_in_sample')} (in-sample) · base rate {st.get('base_rate')}")
    console.print(f"the posts carry {res.data_share:.0%} of the information (the prior the rest); {res.moved:.1%} of the weight moved between terms · {fit_line}")
    for w_ in res.warnings:
        console.print(f"[yellow]warning: {w_}[/yellow]")
    console.print(f"[green]wrote {out}[/green] (use it with: promo-finder run <show> --config {out})" if written
                  else "not written: pass --out <file.yaml> to save a config with these weights")
    if pl is not None:
        console.print(f"[bold]Next {plan} promos to post[/bold] · Thompson sampling: the slate under one posterior draw (seed {seed}); "
                      f"'in slate' and 'top pick' are shares of {pl.get('valid_draws', 0)} draws")
        pt = Table()
        for col in ("#", "clip", "promo", "episode", "block", "score", "in slate", "top pick", "pick", "judged"):
            pt.add_column(col, justify="right" if col in ("#", "score", "in slate", "top pick") else "left")
        for r in pl["rows"]:
            pt.add_row(str(r["slot"]), r["cand_id"], r["promo_id"] or "-", "+".join(map(str, r["episodes"])), r["block"] or "-",
                       f"{r['score_new']:.3f}", f"{r['in_slate_share']:.0%}", f"{r['top_share']:.0%}", r["pick"], "yes" if r["judged"] else "no")
        console.print(pt)
        console.print(f"[dim]{pl['eligible']} of {pl['pool']} clips eligible ({pl['excluded_as_posted']} already posted or near-duplicates of a post). "
                      f"Rules applied: {pl['rules']}. Out of scope: {pl['out_of_scope']}.[/dim]")
        if pl.get("note"):
            console.print(f"[yellow]{pl['note']}[/yellow]")


if __name__ == "__main__":
    app()
