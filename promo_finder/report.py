"""Writers for the raw-outputs folder and the show map."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from .beat_scoring import ScoredShow
from .judge import clip_lines
from . import rubric as rb
from .models import ColdViewer, Promo, PromoLine, PromoSegment, Window, fmt_ts


def to_promos(ranked: list[Window], ctx: ScoredShow, judged: dict[str, ColdViewer], reasons: dict[str, str],
              provenance: dict[str, str], source: str = "llm") -> list[Promo]:
    promos: list[Promo] = []
    for i, w in enumerate(ranked, 1):
        hb, cb = ctx.beat(w.hook_beat), ctx.beat(w.cliff_beat)
        hs, cs = ctx.score(w.hook_beat, source), ctx.score(w.cliff_beat, source)
        pid = f"P{i:02d}"
        cv = judged.get(w.cand_id)
        why = cv.why_it_works if cv else f"Opens on a {hs.beat_type} beat (\"{hb.text[:60]}\") and cuts on a {cs.beat_type} beat (\"{cb.text[:60]}\")."
        promos.append(Promo(
            promo_id=pid, rank=i, score=w.score,
            segments=[PromoSegment(episode=s.episode, start=s.start, end=s.end, start_s=round(s.start_s, 2), end_s=round(s.end_s, 2)) for s in w.segments],
            duration_s=w.duration_s,
            hook=PromoLine(beat_id=hb.global_id, at_s=float(w.flags.get("hook_at_s", 0.0)), speaker=hb.speaker, text=hb.text, score=hs.hook),
            cliffhanger=PromoLine(beat_id=cb.global_id, at_s=round(w.duration_s - (w.segments[-1].end_s - cb.t1), 2), speaker=cb.speaker, text=cb.text, score=cs.cliff),
            writer_cue=bool(w.flags.get("writer_cue")),
            subscores=w.subscores,
            rubric=w.rubric,
            lines=clip_lines(w, ctx),
            cold_viewer=cv,
            why_it_works=why,
            post_first_reason=reasons.get(pid, ""),
            flags={k: v for k, v in w.flags.items() if k not in ("hook_at_s",)} | {"cand_id": w.cand_id, "kind": w.kind},
            provenance=provenance,
        ))
    return promos


def promos_csv(promos: list[Promo]) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["rank", "promo_id", "segment", "episode", "in", "out", "duration_s", "hook_type", "hook_line", "cliffhanger_line", "why_it_works", "score", "cold_viewer_comprehensibility", "spoiler_risk", "show_block", "minutes_into_show", "editor_checks"])
    for p in promos:
        for j, s in enumerate(p.segments, 1):
            wr.writerow([p.rank, p.promo_id, j, s.episode, s.start, s.end, round(s.end_s - s.start_s, 2), p.flags.get("hook_type"),
                         p.hook.text, p.cliffhanger.text, p.why_it_works, p.score,
                         p.cold_viewer.comprehensibility if p.cold_viewer else "", p.flags.get("spoiler_risk"), p.flags.get("block", ""), p.flags.get("minutes_into_show", ""),
                         rb.summary(p.rubric) if p.rubric else ""])
    return buf.getvalue()


def promos_markdown(promos: list[Promo], show: str) -> str:
    out = [f"# Promos — {show}", "", "Ranked: post from the top. Timestamps are episode-relative (MM:SS.ff), cut-aligned.", ""]
    for p in promos:
        segs = " + ".join(f"Ep {s.episode} {s.start} → {s.end}" for s in p.segments)
        out.append(f"## {p.rank}. {p.promo_id} — {segs} ({p.duration_s:.0f}s)")
        out.append(f"- **Hook** (+{p.hook.at_s}s, {p.flags.get('hook_type')}): {p.hook.speaker or 'ACTION'} — \"{p.hook.text}\"")
        out.append(f"- **Cliffhanger**: {p.cliffhanger.speaker or 'ACTION'} — \"{p.cliffhanger.text}\"" + (" *(writers' own episode-end cue)*" if p.writer_cue else ""))
        out.append(f"- **Why it works**: {p.why_it_works}")
        if p.post_first_reason:
            out.append(f"- **Post-first reason**: {p.post_first_reason}")
        pr = p.promise or {}
        if pr.get("question"):
            ans = (f"answered in episode {pr['answered_in_episode']} ({pr.get('minutes_after_clip', '?')} min after the clip)" if pr.get("answered_in_episode")
                   else pr.get("status", "unchecked"))
            out.append(f"- **Promise & payoff**: plants \"{pr['question']}\" · a new viewer reaches it {pr.get('reached_at_min', '?')} min into the show · {ans}")
        if p.cold_viewer:
            cv = p.cold_viewer
            out.append(f"- **Cold viewer** ({cv.comprehensibility}/5 follows it, {cv.would_tap}/5 would tap): \"{cv.what_happened}\" — wants to know: {cv.want_next}")
            if cv.stakes or cv.root_for:
                out.append(f"- **Stakes** (as a stranger reads them): {cv.stakes or '—'} · **roots for**: {cv.root_for or 'nobody'}")
        if p.rubric:
            out.append(f"- **Editor's checks** ({sum(p.rubric.values())}/{len(p.rubric)}, {p.flags.get('block', '?')} of the show, {p.flags.get('minutes_into_show', '?')} min in): {rb.summary(p.rubric)}")
        flags = [k for k in ("spoiler_risk", "physical_hook") if p.flags.get(k)] + ([f"{p.flags['segments']} segments"] if p.flags.get("segments") == 2 else [])
        out.append(f"- score {p.score} · hook {p.subscores['hook']} · cliff {p.subscores['cliff']} · standalone {p.subscores['standalone']} · density {p.subscores['density']}" + (f" · flags: {', '.join(flags)}" if flags else ""))
        out.append("")
    return "\n".join(out)


def show_map_svg(ctx: ScoredShow, promos: list[Promo], source: str = "llm", width: int = 1100) -> str:
    eps = sorted(ctx.episodes, key=lambda e: e.number)
    max_d = max(e.duration for e in eps)
    row_h, left, top = 11, 46, 28
    h = top + row_h * len(eps) + 30
    sx = (width - left - 20) / max_d
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" font-family="Helvetica, Arial, sans-serif" font-size="9">',
           f'<rect width="{width}" height="{h}" fill="white"/>',
           f'<text x="{left}" y="14" font-size="12" font-weight="bold">Show map — every beat of every episode, coloured by promo potential (red = hook, blue = cliffhanger). Brackets = top promos.</text>']
    for i, ep in enumerate(eps):
        y = top + i * row_h
        out.append(f'<text x="2" y="{y + 9}" fill="#444">ep {ep.number:02d}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{ep.duration * sx:.1f}" height="{row_h - 2}" fill="#f1f1f1"/>')
        scores = ctx.scores(ep.number, source)
        for b in ctx.beats_by_ep[ep.number]:
            s = scores[b.beat_id]
            v = max(s.hook, s.cliff)
            if v == 0:
                continue
            alpha = 0.25 + 0.25 * v
            color = f"rgba(214,39,40,{alpha:.2f})" if s.hook >= s.cliff else f"rgba(31,119,180,{alpha:.2f})"
            out.append(f'<rect x="{left + b.t0 * sx:.1f}" y="{y}" width="{max(1.0, (b.t1 - b.t0) * sx):.1f}" height="{row_h - 2}" fill="{color}"/>')
    for p in promos[:20]:
        for s in p.segments:
            i = next(k for k, e in enumerate(eps) if e.number == s.episode)
            y = top + i * row_h
            out.append(f'<rect x="{left + s.start_s * sx:.1f}" y="{y - 1}" width="{(s.end_s - s.start_s) * sx:.1f}" height="{row_h}" fill="none" stroke="#111" stroke-width="{2 if p.rank <= 10 else 1}"/>')
            if s is p.segments[0]:
                out.append(f'<text x="{left + s.start_s * sx + 2:.1f}" y="{y + 8}" font-size="8" font-weight="bold" fill="#111">{p.rank}</text>')
    for t in range(0, int(max_d) + 1, 30):
        out.append(f'<text x="{left + t * sx:.1f}" y="{h - 12}" fill="#666">{fmt_ts(t)[:5]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_run(run_dir: Path, *, show: str, promos: list[Promo], ctx: ScoredShow, metrics: dict, failures_md: str,
              parse_rep: dict, cfg: dict, log_summary: dict, args: dict, judged_errors: dict, source: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "promos.json").write_text(json.dumps([p.model_dump() for p in promos], indent=1))
    (run_dir / "promos.csv").write_text(promos_csv(promos))
    (run_dir / "promos.md").write_text(promos_markdown(promos, show))
    (run_dir / "parse_report.json").write_text(json.dumps(parse_rep, indent=1))
    (run_dir / "aliases.yaml").write_text(yaml.safe_dump({"aliases": ctx.aliases, "notes": ctx.alias_notes}, sort_keys=True))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "run_log.json").write_text(json.dumps({"args": args, "llm": log_summary, "llm_failures": ctx.llm_failures, "id_issues": ctx.id_issues, "judge_errors": judged_errors}, indent=1))
    bd = run_dir / "beats"
    bd.mkdir(exist_ok=True)
    for ep in ctx.episodes:
        rows = []
        sc = ctx.scores(ep.number, source)
        hs = ctx.heur_by_ep[ep.number]
        for b in ctx.beats_by_ep[ep.number]:
            s = sc[b.beat_id]
            rows.append({"beat_id": b.global_id, "t0": b.t0, "t1": b.t1, "kind": b.kind, "speaker": b.speaker,
                         "hook": s.hook, "cliff": s.cliff, "beat_type": s.beat_type, "unresolved_refs": s.unresolved_refs,
                         "role": s.role, "intensity": s.intensity,
                         "source": s.source, "heuristic_hook": hs[b.beat_id].hook, "heuristic_cliff": hs[b.beat_id].cliff})
        (bd / f"ep-{ep.number:03d}.json").write_text(json.dumps({"episode": ep.number, "duration_s": ep.duration, "beats": rows}, indent=0))
    ev = run_dir / "eval"
    ev.mkdir(exist_ok=True)
    (ev / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (ev / "failures.md").write_text(failures_md)
    (run_dir / "show_map.svg").write_text(show_map_svg(ctx, promos, source))
    if ctx.arc is not None:
        (run_dir / "show_arc.json").write_text(ctx.arc.model_dump_json(indent=1))  # type: ignore[attr-defined]
    (run_dir / "cut_promos.sh").write_text(ffmpeg_script(promos))


def per_episode_csv(windows: list[Window], ctx: ScoredShow, source: str = "llm") -> str:
    """The best clip of every episode (for 'new episode out' posts), not just the show-level top list."""
    best: dict[int, Window] = {}
    for w in sorted(windows, key=lambda x: -x.score):
        n = w.segments[0].episode
        if n not in best and w.kind == "single":
            best[n] = w
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["episode", "in", "out", "duration_s", "hook_line", "cliffhanger_line", "score", "show_block", "editor_checks"])
    for ep in sorted(ctx.episodes, key=lambda e: e.number):
        w = best.get(ep.number)
        if w is None:
            wr.writerow([ep.number, "", "", "", "no 20-60 s clip fits the rules", "", "", "", ""])
            continue
        s = w.segments[0]
        wr.writerow([ep.number, s.start, s.end, round(w.duration_s, 2), ctx.beat(w.hook_beat).text, ctx.beat(w.cliff_beat).text,
                     w.score, w.flags.get("block", ""), rb.summary(w.rubric)])
    return buf.getvalue()


def ffmpeg_script(promos: list[Promo]) -> str:
    """From the ranked list to rendered clips with free tools: one ffmpeg cut per segment, joined when there are two.
    Point EPISODE_DIR at the episode video files (ep-001.mp4, ...)."""
    out = ["#!/usr/bin/env bash", "# Cut every promo from the episode videos. Usage: EPISODE_DIR=/path/to/episodes bash cut_promos.sh",
           "set -euo pipefail", 'EPISODE_DIR="${EPISODE_DIR:-episodes}"', 'OUT="${OUT:-promo_clips}"', 'mkdir -p "$OUT"', ""]
    for p in promos:
        parts = []
        for j, s in enumerate(p.segments, 1):
            part = f'"$OUT/{p.promo_id}-part{j}.mp4"'
            out.append(f'ffmpeg -y -loglevel error -ss {s.start_s:.2f} -to {s.end_s:.2f} -i "$EPISODE_DIR/ep-{s.episode:03d}.mp4" '
                       f'-c:v libx264 -preset veryfast -crf 20 -c:a aac {part}')
            parts.append(part)
        if len(parts) == 1:
            out.append(f'mv {parts[0]} "$OUT/{p.rank:02d}-{p.promo_id}.mp4"')
        else:
            lst = f'"$OUT/{p.promo_id}.txt"'
            out.append(f"printf \"file '%s'\\n\" " + " ".join(f"$(basename {x})" for x in parts) + f" > {lst}")
            out.append(f'(cd "$OUT" && ffmpeg -y -loglevel error -f concat -safe 0 -i {p.promo_id}.txt -c copy {p.rank:02d}-{p.promo_id}.mp4)')
            out.append(f'rm -f {" ".join(parts)} {lst}')
        out.append("")
    out.append('echo "done: $(ls "$OUT"/*.mp4 | wc -l) clips in $OUT"')
    return "\n".join(out) + "\n"
