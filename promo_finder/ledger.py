"""Promise & payoff ledger: every promo plants a question; where does the show answer it?

For each promo on the slate: the question a first-time viewer is left with (the cold-viewer test), when a new viewer
REACHES the promo's moment (minutes of watching from episode 1), WHERE the show answers the question (one model call
over the writers' episode summaries), how long after the clip that is, and whether the answer sits before or behind
the paywall. Code does all the arithmetic; the model only names an episode, and an impossible episode is rejected.

Why it matters (docs/research.md): an ad's promise must be paid off or viewers "feel lied to" (MrBeast production
guide; fake-hook micro-drama ads drew backlash and takedowns). An answer in the free episodes builds trust; an answer
just behind the paywall is a conversion promo; a question the show never answers is a fake hook and is flagged.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .beat_scoring import ScoredShow, load_prompt
from .llm import LLM, LLMError
from .models import Promo


class LedgerItem(BaseModel):
    promo_id: str
    answered_in_episode: int | None = None
    answer: str = ""
    confidence: str = "low"


class LedgerOut(BaseModel):
    items: list[LedgerItem]


def question_of(p: Promo) -> str:
    if p.cold_viewer and p.cold_viewer.want_next and "none" not in p.cold_viewer.want_next.lower()[:10]:
        return p.cold_viewer.want_next.strip()
    return f"What happens after: \"{p.cliffhanger.text[:120]}\"?"


def build(promos: list[Promo], ctx: ScoredShow, llm: LLM | None, raw_dir: Path | None, paywall: int | None = None) -> dict[str, dict]:
    starts = ctx.starts
    eps = sorted(ctx.episodes, key=lambda e: e.number)
    last = eps[-1].number
    out: dict[str, dict] = {}
    for p in promos:
        s0 = p.segments[0]
        out[p.promo_id] = {
            "question": question_of(p),
            "reached_at_min": round((starts[s0.episode] + s0.start_s) / 60.0, 1),
            "clip_episode": s0.episode,
        }
    if llm is None or not promos:
        return out
    user = ("EPISODES:\n" + "\n".join(f"Ep {e.number}: {e.logline}" + (f" | continuity: {e.continuity}" if e.continuity else "") for e in eps)
            + "\n\nPROMOS:\n" + "\n".join(f"{p.promo_id} (from episode {p.segments[-1].episode}): question = {out[p.promo_id]['question']} | last line = \"{p.cliffhanger.text[:160]}\""
                                       for p in promos))
    try:
        res = llm.complete_json(load_prompt("ledger"), user, LedgerOut, "ledger", "slate",
                                raw_path=(raw_dir / "ledger" / "slate.json") if raw_dir else None)
    except LLMError as e:
        if e.fatal:
            raise
        for v in out.values():
            v["status"] = "unchecked"
        return out
    got = {it.promo_id.strip(): it for it in res.items}
    for p in promos:
        row, it = out[p.promo_id], got.get(p.promo_id)
        end_ep = p.segments[-1].episode
        if it is None:
            row["status"] = "unchecked"
            continue
        n = it.answered_in_episode
        if n is not None and not (end_ep <= n <= last):
            row.update({"status": "rejected", "note": f"model named episode {n}, outside {end_ep}..{last}"})
            continue
        row.update({"answer": it.answer.strip(), "confidence": it.confidence.strip().lower()})
        if n is None:
            row.update({"answered_in_episode": None, "status": "never answered"})
            continue
        clip_end_abs = starts[end_ep] + p.segments[-1].end_s
        answer_abs = starts[n] + (0.0 if n > end_ep else p.segments[-1].end_s)
        row.update({
            "answered_in_episode": n,
            "answered_at_min": round(answer_abs / 60.0, 1),
            "minutes_after_clip": round(max(0.0, answer_abs - clip_end_abs) / 60.0, 1),
            "status": ("same episode" if n == end_ep else ("behind the paywall" if paywall and n >= paywall else "answered")),
        })
        if paywall:
            row["answer_is_free"] = n < paywall
    return out


def summary(ledger: dict[str, dict]) -> dict:
    rows = list(ledger.values())
    answered = [r for r in rows if r.get("answered_in_episode")]
    later = [r for r in answered if r.get("status") != "same episode"]
    mins = sorted(r["minutes_after_clip"] for r in later if "minutes_after_clip" in r)
    return {
        "promos": len(rows),
        "answered": len(answered),
        "never_answered": sum(1 for r in rows if r.get("status") == "never answered"),
        "unchecked_or_rejected": sum(1 for r in rows if r.get("status") in ("unchecked", "rejected")),
        "answered_same_episode": len(answered) - len(later),
        "answered_later_episode": len(later),
        "median_minutes_clip_to_answer": mins[len(mins) // 2] if mins else None,  # over answers in a later episode
        "behind_paywall": sum(1 for r in rows if r.get("answer_is_free") is False),
    }
