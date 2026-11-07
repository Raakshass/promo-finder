"""Promo-ability: which episodes (and which shows) are easy or hard to cut a standalone promo from, and why.

The research behind the signals (docs/research.md):
  - conflict density: micro-drama ads are cut from confrontation / humiliation / reveal / threat beats (HIVE/DramaAD,
    AppGrowing, SocialPeta); an episode with few of them has little to cut.
  - payoff share: compress-and-release writing spends tension on screen; an episode that resolves more than it
    escalates yields clips that end on an answer, not a question.
  - context load: lines that name people or past events a stranger lacks (Loewenstein: curiosity needs a gap the
    viewer can see; no context, no curiosity).
  - cast size: >3 people per frame underperforms (Netflix artwork research); working memory holds 3-5 chunks.
  - off-screen voice share: promos must work on mute with captions (Meta/TikTok); V.O./O.S. lines carry no face.
Tiers are absolute (does a clip pass the editor's checks with a strong hook and cliff?), not show-relative, so a
strong show can have zero hard episodes and a weak one many.
"""
from __future__ import annotations

from statistics import median

from .beat_scoring import ScoredShow
from .models import Window
from .rubric import CODE_CHECKS, CONFLICT_TYPES, LABELS


def _episode_stats(ctx: ScoredShow, n: int, source: str) -> dict:
    ep = ctx.episode(n)
    beats = ctx.beats_by_ep[n]
    sc = ctx.scores(n, source)
    dia = [b for b in beats if b.kind == "dialogue"]
    conflict = [b for b in beats if sc[b.beat_id].hook >= 2 and (sc[b.beat_id].role in ("escalation", "turn") or sc[b.beat_id].beat_type in CONFLICT_TYPES)]
    payoffs = [b for b in beats if sc[b.beat_id].role == "payoff"]
    pushes = [b for b in beats if sc[b.beat_id].role in ("escalation", "turn")]
    dia_ev = [e for e in ep.events if e.kind == "dialogue"]
    offscreen = [e for e in dia_ev if e.mode]
    minutes = max(ep.duration / 60.0, 1e-6)
    return {
        "episode": n,
        "duration_s": round(ep.duration, 1),
        "beats": len(beats),
        "conflict_beats_per_min": round(len(conflict) / minutes, 2),
        "payoff_share": round(len(payoffs) / max(len(payoffs) + len(pushes), 1), 3),
        "speakers": len({b.speaker for b in dia if b.speaker}),
        "context_load": round(sum(1 for b in dia if sc[b.beat_id].unresolved_refs) / max(len(dia), 1), 3),
        "offscreen_share": round(len(offscreen) / max(len(dia_ev), 1), 3),
        "peak_hook": max((sc[b.beat_id].hook for b in beats), default=0),
        "peak_cliff": max((sc[b.beat_id].cliff for b in beats), default=0),
    }


def _tier(windows: list[Window]) -> tuple[str, dict | None]:
    strong = [w for w in windows if w.subscores.get("hook", 0) >= 0.66 and w.subscores.get("cliff", 0) >= 0.66]
    clean = [w for w in strong if all(w.rubric.get(k, True) for k in CODE_CHECKS)]
    core = [w for w in strong if w.rubric.get("ends_open", True) and w.rubric.get("opens_on_conflict", True)]
    best = max(windows, key=lambda w: w.score, default=None)
    info = None
    if best is not None:
        info = {"cand_id": best.cand_id, "score": best.score, "rubric_pass": best.subscores.get("rubric_pass"),
                "failed_checks": [k for k, ok in best.rubric.items() if not ok]}
    if clean:
        return "easy", info
    if core:
        return "medium", info
    return "hard", info


def report(ctx: ScoredShow, windows: list[Window], finalists: list[Window], source: str = "llm") -> dict:
    by_ep: dict[int, list[Window]] = {}
    for w in windows:
        by_ep.setdefault(w.segments[0].episode, []).append(w)
    fin_count: dict[int, int] = {}
    for w in finalists:
        fin_count[w.segments[0].episode] = fin_count.get(w.segments[0].episode, 0) + 1
    rows = []
    for ep in sorted(ctx.episodes, key=lambda e: e.number):
        st = _episode_stats(ctx, ep.number, source)
        tier, best = _tier(by_ep.get(ep.number, []))
        st.update({"tier": tier, "best_window": best, "finalists": fin_count.get(ep.number, 0), "windows": len(by_ep.get(ep.number, []))})
        rows.append(st)
    med = {k: round(median(r[k] for r in rows), 2) for k in ("conflict_beats_per_min", "payoff_share", "speakers", "context_load", "offscreen_share")}
    for r in rows:
        why = []
        if r["windows"] == 0:
            why.append("no 20-60 s window fits the rules (episode too short or no strong opening line)")
        if r["conflict_beats_per_min"] < 0.5 * med["conflict_beats_per_min"]:
            why.append(f"little open conflict ({r['conflict_beats_per_min']}/min vs show median {med['conflict_beats_per_min']})")
        if r["payoff_share"] > max(0.4, 1.5 * med["payoff_share"]):
            why.append(f"resolves more than it escalates (payoff share {r['payoff_share']:.0%})")
        if r["context_load"] > max(0.4, 1.5 * med["context_load"]):
            why.append(f"lines lean on backstory ({r['context_load']:.0%} of lines name people/events a stranger lacks)")
        if r["speakers"] > 5:
            why.append(f"large cast ({r['speakers']} speakers)")
        if r["offscreen_share"] > 0.3:
            why.append(f"voice-over / off-screen heavy ({r['offscreen_share']:.0%} of lines)")
        if r["peak_cliff"] < 2:
            why.append("no beat leaves a real question open")
        if r["tier"] != "easy" and r["best_window"] and r["best_window"]["failed_checks"]:
            why.append("best clip fails: " + ", ".join(LABELS.get(k, k) for k in r["best_window"]["failed_checks"]))
        r["why"] = why
    counts = {t: sum(1 for r in rows if r["tier"] == t) for t in ("easy", "medium", "hard")}
    # show-level: is this a two-hander (easy) or an ensemble (hard)? how much of the dialogue do the top two carry?
    from collections import Counter

    raw = Counter(b.speaker for bs in ctx.beats_by_ep.values() for b in bs if b.kind == "dialogue" and b.speaker)
    titles = {"LADY", "LORD", "SIR", "DUKE", "DUCHESS", "MISS", "MRS", "MR", "MADAM", "CAPTAIN", "KING", "QUEEN", "PRINCE", "PRINCESS", "OLD", "YOUNG"}

    def canon(label: str) -> str:  # "LADY ELEANOR" -> "ELEANOR" when ELEANOR is itself a label (same conservative rule as aliases)
        words = label.split()
        rest = [w for w in words if w not in titles]
        cand = " ".join(rest)
        return cand if rest and len(rest) < len(words) and cand in raw else label

    talk: Counter = Counter()
    for lab, c in raw.items():
        talk[canon(lab)] += c
    total = sum(talk.values()) or 1
    top2 = sum(c for _, c in talk.most_common(2)) / total
    show = {
        "episodes": len(rows), "tiers": counts,
        "median_conflict_beats_per_min": med["conflict_beats_per_min"],
        "median_payoff_share": med["payoff_share"],
        "median_context_load": med["context_load"],
        "median_speakers_per_episode": med["speakers"],
        "offscreen_share": round(sum(r["offscreen_share"] for r in rows) / max(len(rows), 1), 3),
        "top2_dialogue_share": round(top2, 3),
        "top_speakers": [s for s, _ in talk.most_common(4)],
    }
    verdict = []
    if counts["easy"] >= 0.5 * len(rows):
        verdict.append("easy to promo: most episodes contain at least one clip that passes every editor's check")
    elif counts["hard"] >= 0.3 * len(rows):
        verdict.append("hard to promo: many episodes have no clip that both opens on conflict and ends before the payoff")
    else:
        verdict.append("promo-able with care: most episodes give a usable clip, but many need an editor's trim")
    verdict.append("two-hander-led (the top two speakers carry {:.0%} of the lines): easy for a stranger to follow".format(top2) if top2 >= 0.4
                   else "ensemble-led (top two speakers carry only {:.0%} of the lines): harder for a stranger to follow".format(top2))
    show["verdict"] = verdict
    return {"show": show, "episodes": rows}


def markdown(rep: dict) -> str:
    s = rep["show"]
    out = ["# Promo-ability: which episodes are easy or hard to promo, and why", "",
           f"**Show verdict:** {'; '.join(s['verdict'])}.", "",
           f"Tiers: easy {s['tiers']['easy']} · medium {s['tiers']['medium']} · hard {s['tiers']['hard']} (of {s['episodes']}). "
           f"Median conflict beats/min {s['median_conflict_beats_per_min']}, payoff share {s['median_payoff_share']:.0%}, "
           f"context load {s['median_context_load']:.0%}, speakers/episode {s['median_speakers_per_episode']}, off-screen lines {s['offscreen_share']:.0%}.", "",
           "- **easy**: at least one clip with a strong hook and cliff passes every code check of the editor's rubric",
           "- **medium**: a strong clip opens on conflict and ends before the payoff, but fails a softer check (cast, build, backstory, dead air)",
           "- **hard**: no strong clip both opens on conflict and ends open; an editor should expect to trim by hand or skip", "",
           "| ep | tier | conflict/min | payoff share | context load | speakers | finalists | why |", "|---|---|---|---|---|---|---|---|"]
    for r in rep["episodes"]:
        out.append(f"| {r['episode']} | {r['tier']} | {r['conflict_beats_per_min']} | {r['payoff_share']:.0%} | {r['context_load']:.0%} | {r['speakers']} | {r['finalists']} | {'; '.join(r['why']) or '—'} |")
    return "\n".join(out) + "\n"
