"""Quality checks that need no labels beyond what the pack already carries."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .beat_scoring import ScoredShow
from .features import WRITER_CUE
from .models import Promo, Window
from .rank import iou


def invariants(promos: list[Promo], cfg: dict) -> dict:
    p = cfg["promo"]
    bad: list[str] = []
    for pr in promos:
        if not (p["min_duration_s"] - 0.01 <= pr.duration_s <= p["max_duration_s"] + 0.01):
            bad.append(f"{pr.promo_id}: duration {pr.duration_s}")
        if pr.hook.at_s > p["hook_window_s"] + 0.01:
            bad.append(f"{pr.promo_id}: hook at {pr.hook.at_s}s")
        for a, b in zip(pr.segments, pr.segments[1:]):
            if (b.episode, b.start_s) < (a.episode, a.end_s):
                bad.append(f"{pr.promo_id}: segments out of order")
    return {"checked": len(promos), "violations": bad, "pass_rate": round(1 - len(bad) / max(len(promos), 1), 3)}


def writer_cue_episodes(ctx: ScoredShow) -> list[int]:
    out = []
    for ep in ctx.episodes:
        tail = [e for e in ep.events if e.t1 >= ep.duration - 6.0 and e.kind != "dialogue"]
        if any(re.search(WRITER_CUE, e.text) for e in tail):
            out.append(ep.number)
    return out


def cliff_localisation(ctx: ScoredShow, cfg: dict, source: str = "llm") -> dict:
    """Writers end every episode on a cliff. Is the model's top cliff beat inside the last N seconds?"""
    win = cfg["eval"]["cliff_window_s"]
    cue_eps = set(writer_cue_episodes(ctx))
    rows = []
    for ep in ctx.episodes:
        scores = ctx.scores(ep.number, source)
        beats = ctx.beats_by_ep[ep.number]
        best = max(beats, key=lambda b: (scores[b.beat_id].cliff, b.t1))
        rows.append({
            "episode": ep.number, "top_cliff_beat": best.global_id, "t1": round(best.t1, 2),
            "in_last_window": best.t1 >= ep.duration - win, "writer_cue": ep.number in cue_eps,
            "top_cliff_score": scores[best.beat_id].cliff,
        })
    hit = sum(r["in_last_window"] for r in rows)
    cue_rows = [r for r in rows if r["writer_cue"]]
    return {
        "source": source,
        "recall_all": round(hit / len(rows), 3),
        "recall_writer_cue_episodes": round(sum(r["in_last_window"] for r in cue_rows) / max(len(cue_rows), 1), 3),
        "n_writer_cue_episodes": len(cue_rows),
        "rows": rows,
    }


def logline_overlap(ctx: ScoredShow, finalists: list[Window]) -> dict:
    """Weak proxy: content-word overlap between an episode's logline and its best windows' text."""
    stop = set("the a an and or of to in on at for with his her their she he they it is was be as by from that this into".split())

    def words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in stop and len(w) > 3}

    rows = []
    for ep in ctx.episodes:
        ws = [w for w in finalists if w.segments[0].episode == ep.number][:3]
        if not ws or not ep.logline:
            continue
        lw = words(ep.logline)
        best = 0.0
        for w in ws:
            tw = set()
            for gid in w.beat_ids:
                tw |= words(ctx.beat(gid).text)
            best = max(best, len(lw & tw) / max(len(lw), 1))
        rows.append({"episode": ep.number, "overlap": round(best, 3)})
    return {"mean_overlap": round(sum(r["overlap"] for r in rows) / max(len(rows), 1), 3), "rows": rows}


def ranking_overlap(a: list[Window], b: list[Window], k: int = 10, thr: float = 0.5) -> dict:
    """How many of A's top-k have a matching window (IoU ≥ thr) in B's top-k."""
    ta, tb = a[:k], b[:k]
    matched = sum(1 for w in ta if any(iou(w, v) >= thr for v in tb))
    return {"k": k, "matched": matched, "overlap": round(matched / max(len(ta), 1), 3)}


def failures_markdown(promos: list[Promo], finalists: list[Window], ctx: ScoredShow, cfg: dict, extra: dict) -> str:
    lines = ["# Where the tool picks badly", ""]
    lines.append("Generated from this run, unedited. Each item is a concrete case a reviewer can check against the scripts.")
    lines.append("")
    judged = [p for p in promos if p.cold_viewer]
    if judged:
        worst = sorted(judged, key=lambda p: (p.cold_viewer.comprehensibility, p.cold_viewer.would_tap))[:5]
        lines.append("## Finalists a first-time viewer rated lowest (cold-viewer test)")
        for p in worst:
            cv = p.cold_viewer
            seg = ", ".join(f"ep{s.episode} {s.start}-{s.end}" for s in p.segments)
            lines.append(f"- **{p.promo_id}** ({seg}) comprehensibility {cv.comprehensibility}/5, would tap {cv.would_tap}/5. Viewer: \"{cv.what_happened}\" Unresolved: {', '.join(cv.unresolved_refs) or 'none'}.")
        lines.append("")
    from .rubric import LABELS
    failing = [p for p in promos if p.rubric and not all(p.rubric.values())]
    if failing:
        lines.append("## Final promos that fail an editor's check (ranked anyway, at a penalty; the editor decides)")
        for p in failing:
            seg = ", ".join(f"ep{s.episode} {s.start}-{s.end}" for s in p.segments)
            fails = ", ".join(LABELS.get(k, k) for k, ok in p.rubric.items() if not ok)
            lines.append(f"- **{p.promo_id}** (#{p.rank}, {seg}): fails {fails}")
        lines.append("")
    late = [p for p in promos if p.flags.get("spoiler_risk")]
    if late:
        lines.append("## Spoiler risk (from the last 20% of episodes; the tool flags, a human decides)")
        lines += [f"- {p.promo_id}: ep{p.segments[0].episode} — hook \"{p.hook.text[:80]}\"" for p in late]
        lines.append("")
    slow = [p for p in promos if p.hook.at_s > 2.0]
    if slow:
        lines.append("## Hooks that land late (>2 s after the cut; the brief wants ≤3 s)")
        lines += [f"- {p.promo_id}: hook at +{p.hook.at_s}s" for p in slow]
        lines.append("")
    sc = [p for p in promos if p.flags.get("scene_change")]
    if sc:
        lines.append("## Clips that cross a scene boundary (fine for a promo, but the editor should check the cut)")
        lines += [f"- {p.promo_id}: " + ", ".join(f"ep{s.episode} {s.start}-{s.end}" for s in p.segments) for p in sc]
        lines.append("")
    two = [p for p in promos if p.flags.get("segments") == 2]
    if two:
        lines.append("## Multi-segment promos (joins are unproven; check the cut plays)")
        lines += [f"- {p.promo_id}: " + " + ".join(f"ep{s.episode} {s.start}-{s.end}" for s in p.segments) for p in two]
        lines.append("")
    if ctx.llm_failures:
        lines.append("## Episodes where the model call failed and the heuristic twin was used")
        lines += [f"- ep{n}: {e}" for n, e in sorted(ctx.llm_failures.items())]
        lines.append("")
    if ctx.id_issues:
        lines.append("## Episodes where the model's answer needed repair (unknown/missing beat ids, dropped from a batch, or a flat all-zero answer)")
        lines += [f"- ep{n}: " + "; ".join(x for x in [
            f"unknown ids {v['unknown_ids'][:5]}" if v.get("unknown_ids") else "",
            f"missing ids filled from heuristics {v['missing_ids_filled_from_heuristic'][:5]}" if v.get("missing_ids_filled_from_heuristic") else "",
            v.get("note", "")] if x) for n, v in sorted(ctx.id_issues.items())]
        lines.append("")
    ent = extra.get("entailment") or {}
    missed = [n for n, r in ent.items() if r.get("contains") is False]
    if missed:
        lines.append("## Episodes whose best promo does not contain the writers' own cliffhanger")
        lines += [f"- ep{n}: {ent[n]['reason']}" for n in missed[:15]]
        lines.append("")
    no_promo = [ep.number for ep in ctx.episodes if not any(w.segments[0].episode == ep.number for w in finalists)]
    if no_promo:
        lines.append(f"## Episodes with no finalist at all: {no_promo}")
        lines.append("")
    lines.append("## Known limits of the method")
    lines += [
        "- Text only: it cannot see performance, pacing or shot quality; a weak slap on screen still scores as a slap.",
        "- Timestamps assume the final cut matches the screenplay timing.",
        "- Self-containment is judged by a model reading a transcript, not by a real viewer watching a clip.",
        "- Non-adjacent joins across episodes are not attempted; only direct continuations across an episode boundary.",
    ]
    return "\n".join(lines)


def compare_runs(a: Path, b: Path) -> dict:
    pa = json.loads((a / "promos.json").read_text())
    pb = json.loads((b / "promos.json").read_text())

    def to_win(p: dict) -> Window:
        from .models import Segment
        return Window(cand_id=p["promo_id"], segments=[Segment(episode=s["episode"], start_s=s["start_s"], end_s=s["end_s"]) for s in p["segments"]],
                      hook_beat=p["hook"]["beat_id"], cliff_beat=p["cliffhanger"]["beat_id"], beat_ids=[], duration_s=p["duration_s"])

    wa, wb = [to_win(p) for p in pa], [to_win(p) for p in pb]
    out = {"a": str(a), "b": str(b), "top10": ranking_overlap(wa, wb, 10), "top20": ranking_overlap(wa, wb, 20)}
    # looser view: the same opening beat chosen in both runs (clips may differ by a few seconds)
    ha, hb = {w.hook_beat for w in wa[:10]}, {w.hook_beat for w in wb[:10]}
    out["top10_same_hook_beat"] = {"matched": len(ha & hb), "overlap": round(len(ha & hb) / max(len(ha), 1), 3)}
    ha20, hb20 = {w.hook_beat for w in wa[:20]}, {w.hook_beat for w in wb[:20]}
    out["top20_same_hook_beat"] = {"matched": len(ha20 & hb20), "overlap": round(len(ha20 & hb20) / max(len(ha20), 1), 3)}
    # beat-level agreement: the cleanest measure of model nondeterminism, independent of the judge
    ba, bb = a / "beats", b / "beats"
    if ba.exists() and bb.exists():
        same = total = hook_same = cliff_same = hook_near = cliff_near = 0
        flat_eps: list[int] = []

        def flat(rows: dict) -> bool:  # same rule as beat_scoring.degenerate: a lapse, not a judgement
            llm = [r for r in rows.values() if r.get("source") == "llm"]
            return bool(llm) and (max(r["hook"] for r in llm) <= 1 or max(r["cliff"] for r in llm) <= 1)

        for fa in sorted(ba.glob("ep-*.json")):
            fb = bb / fa.name
            if not fb.exists():
                continue
            ra = {r["beat_id"]: r for r in json.loads(fa.read_text())["beats"]}
            rb = {r["beat_id"]: r for r in json.loads(fb.read_text())["beats"]}
            if flat(ra) or flat(rb):
                flat_eps.append(int(fa.stem.split("-")[1]))
                continue
            for k, r in ra.items():
                if k in rb:
                    total += 1
                    hook_same += r["hook"] == rb[k]["hook"]
                    cliff_same += r["cliff"] == rb[k]["cliff"]
                    hook_near += abs(r["hook"] - rb[k]["hook"]) <= 1
                    cliff_near += abs(r["cliff"] - rb[k]["cliff"]) <= 1
                    same += r["hook"] == rb[k]["hook"] and r["cliff"] == rb[k]["cliff"]
        if total:
            out["beat_agreement"] = {"beats": total, "episodes_excluded_flat": flat_eps, "hook_exact": round(hook_same / total, 3), "cliff_exact": round(cliff_same / total, 3), "both_exact": round(same / total, 3),
                                     "hook_within_1": round(hook_near / total, 3), "cliff_within_1": round(cliff_near / total, 3)}
    ma, mb = a / "eval" / "metrics.json", b / "eval" / "metrics.json"
    if ma.exists() and mb.exists():
        ja, jb = json.loads(ma.read_text()), json.loads(mb.read_text())
        out["cliff_localisation"] = {"a": ja.get("cliff_localisation", {}).get("recall_all"), "b": jb.get("cliff_localisation", {}).get("recall_all")}
        out["cold_viewer_mean"] = {"a": ja.get("cold_viewer", {}).get("mean_comprehensibility"), "b": jb.get("cold_viewer", {}).get("mean_comprehensibility")}
    return out
