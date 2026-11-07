"""LLM passes that run on finalists only: the cold-viewer test, the continuity-note entailment check,
and the one-line 'post first' reasons."""
from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel, Field

from .beat_scoring import ScoredShow, load_prompt
from .llm import LLM, LLMError
from .models import ColdViewer, Window


def clip_lines(w: Window, ctx: ScoredShow) -> list[dict]:
    """Lines inside the clip with time relative to the clip start (segments are concatenated)."""
    out: list[dict] = []
    offset = 0.0
    for seg in w.segments:
        for gid in w.beat_ids:
            b = ctx.beat(gid)
            if b.episode != seg.episode or b.t0 < seg.start_s - 0.01 or b.t1 > seg.end_s + 0.01:
                continue
            out.append({
                "t": round(offset + (b.t0 - seg.start_s), 2),
                "dur": round(b.t1 - b.t0, 2),
                "kind": b.kind,
                "speaker": b.speaker,
                "text": b.text,
                "emotions": b.emotions,
                "beat_id": gid,
            })
        offset += seg.duration_s
    out.sort(key=lambda x: x["t"])
    return out


def transcript(w: Window, ctx: ScoredShow, include_sfx: bool = False) -> str:
    lines = []
    for l in clip_lines(w, ctx):
        if l["kind"] == "dialogue":
            lines.append(f"{l['speaker'] or 'VOICE'}: \"{l['text']}\"")
        elif l["kind"] == "action":
            lines.append(f"[{l['text']}]")
        elif include_sfx:
            lines.append(f"(sound: {l['text']})")
    return "\n".join(lines)


class ColdViewerItem(ColdViewer):
    clip_id: str


class ColdViewerBatchOut(BaseModel):
    clips: list[ColdViewerItem]


CLIP_BATCH_NOTE = (
    "\n\nYou will see {n} separate clips, each headed 'CLIP k (id=...)'. Judge each one on its own, as if it were the "
    "only thing you saw. Return JSON only: {{\"clips\": [{{\"clip_id\": \"...\", \"comprehensibility\": 3, \"what_happened\": \"...\", "
    "\"want_next\": \"...\", \"unresolved_refs\": [], \"would_tap\": 3, \"why_it_works\": \"...\"}}, ...]}} — one entry per clip id."
)


def cold_viewer(windows: list[Window], ctx: ScoredShow, llm: LLM, raw_dir: Path | None, per_call: int = 1) -> tuple[dict[str, ColdViewer], dict[str, str]]:
    system = load_prompt("cold_viewer")
    order = list(windows)
    random.Random(7).shuffle(order)  # order independence: judge never sees the ranking
    results: dict[str, ColdViewer] = {}
    errors: dict[str, str] = {}
    if per_call > 1:
        for i in range(0, len(order), per_call):
            chunk = order[i:i + per_call]
            tag = f"batch-{i // per_call + 1:02d}"
            user = "\n\n".join(f"CLIP {k + 1} (id={w.cand_id}, {w.duration_s:.0f} seconds):\n{transcript(w, ctx)}" for k, w in enumerate(chunk))
            try:
                out = llm.complete_json(system + CLIP_BATCH_NOTE.format(n=len(chunk)), user, ColdViewerBatchOut, "cold_viewer", tag,
                                        raw_path=(raw_dir / "cold_viewer" / f"{tag}.json") if raw_dir else None)
                got = {c.clip_id.strip(): c for c in out.clips}
                for w in chunk:
                    c = got.get(w.cand_id)
                    if c is None:
                        errors[w.cand_id] = f"missing from batch {tag}"
                    else:
                        results[w.cand_id] = ColdViewer(**{k: getattr(c, k) for k in ColdViewer.model_fields})
            except LLMError as e:
                for w in chunk:
                    errors[w.cand_id] = str(e)
        return results, errors
    for w in order:
        user = f"CLIP ({w.duration_s:.0f} seconds):\n{transcript(w, ctx)}"
        try:
            results[w.cand_id] = llm.complete_json(system, user, ColdViewer, "cold_viewer", w.cand_id,
                                                   raw_path=(raw_dir / "cold_viewer" / f"{w.cand_id}.json") if raw_dir else None)
        except LLMError as e:
            errors[w.cand_id] = str(e)
    return results, errors


class EntailOut(BaseModel):
    contains: bool
    reason: str = ""


class EntailItem(BaseModel):
    episode: int
    contains: bool
    reason: str = ""


class EntailBatchOut(BaseModel):
    pairs: list[EntailItem]


PAIR_BATCH_NOTE = (
    "\n\nYou will see {n} independent pairs, each headed 'PAIR k — EPISODE N'. Decide each on its own. Return JSON only: "
    "{{\"pairs\": [{{\"episode\": N, \"contains\": true, \"reason\": \"...\"}}, ...]}} — one entry per pair."
)


def entailment(ctx: ScoredShow, best_by_ep: dict[int, Window], llm: LLM, raw_dir: Path | None, per_call: int = 1) -> dict[int, dict]:
    """For each episode N whose successor's continuity note describes N's ending: does N's best promo contain it?"""
    system = load_prompt("entailment")
    by_num = {e.number: e for e in ctx.episodes}
    out: dict[int, dict] = {}
    if per_call > 1:
        items = [(n, w, by_num[n + 1]) for n, w in sorted(best_by_ep.items()) if by_num.get(n + 1) and by_num[n + 1].continuity]
        for i in range(0, len(items), per_call):
            chunk = items[i:i + per_call]
            tag = f"batch-{i // per_call + 1:02d}"
            user = "\n\n".join(
                f"PAIR {k + 1} — EPISODE {n}\n(A) WRITERS' NOTE ON HOW EPISODE {n} ENDED:\n{nxt.continuity}\n(B) PROMO CLIP TRANSCRIPT FROM EPISODE {n}:\n{transcript(w, ctx)}"
                for k, (n, w, nxt) in enumerate(chunk))
            try:
                res = llm.complete_json(system + PAIR_BATCH_NOTE.format(n=len(chunk)), user, EntailBatchOut, "entailment", tag,
                                        raw_path=(raw_dir / "entailment" / f"{tag}.json") if raw_dir else None)
                got = {r.episode: r for r in res.pairs}
                for n, w, nxt in chunk:
                    r = got.get(n)
                    out[n] = {"contains": r.contains if r else None, "reason": r.reason if r else f"missing from batch {tag}", "cand_id": w.cand_id, "note": nxt.continuity}
            except LLMError as e:
                for n, w, nxt in chunk:
                    out[n] = {"contains": None, "reason": f"error: {e}", "cand_id": w.cand_id, "note": nxt.continuity}
        return out
    for n, w in sorted(best_by_ep.items()):
        nxt = by_num.get(n + 1)
        if not nxt or not nxt.continuity:
            continue
        user = f"(A) WRITERS' NOTE ON HOW EPISODE {n} ENDED:\n{nxt.continuity}\n\n(B) PROMO CLIP TRANSCRIPT FROM EPISODE {n}:\n{transcript(w, ctx)}"
        try:
            r = llm.complete_json(system, user, EntailOut, "entailment", f"ep-{n:03d}",
                                  raw_path=(raw_dir / "entailment" / f"ep-{n:03d}.json") if raw_dir else None)
            out[n] = {"contains": r.contains, "reason": r.reason, "cand_id": w.cand_id, "note": nxt.continuity}
        except LLMError as e:
            out[n] = {"contains": None, "reason": f"error: {e}", "cand_id": w.cand_id, "note": nxt.continuity}
    return out


class ReasonsOut(BaseModel):
    reasons: dict[str, str] = Field(default_factory=dict)


def rank_reasons(cards: list[dict], llm: LLM, raw_dir: Path | None) -> dict[str, str]:
    system = load_prompt("rank_reasons")
    user = "\n\n".join(
        f"{c['promo_id']} (rank {c['rank']}, ep {c['episode']}, {c['duration_s']:.0f}s)\n"
        f"  HOOK ({c['hook_type']}): {c['hook']}\n  CLIFF: {c['cliff']}\n  scores: {c['subscores']}\n"
        f"  first-time viewer: {c['cold_viewer']}"
        for c in cards
    )
    try:
        return llm.complete_json(system, user, ReasonsOut, "rank_reasons", "top",
                                 raw_path=(raw_dir / "rank_reasons" / "top.json") if raw_dir else None).reasons
    except LLMError:
        return {}


class ABItem(BaseModel):
    pair: int
    winner: str
    reason: str = ""


class ABOut(BaseModel):
    pairs: list[ABItem]


def promo_transcript(p: dict) -> str:
    """Transcript of a promo from promos.json (what a viewer hears and sees)."""
    lines = []
    for l in p.get("lines", []):
        if l.get("kind") == "dialogue":
            lines.append(f"{l.get('speaker') or 'VOICE'}: \"{l['text']}\"")
        elif l.get("kind") == "action":
            lines.append(f"[{l['text']}]")
    return "\n".join(lines)


def _key(p: dict) -> tuple:
    return tuple((s["episode"], round(s["start_s"], 1), round(s["end_s"], 1)) for s in p["segments"])


def ab_test(promos_a: list[dict], promos_b: list[dict], llm: LLM, raw_dir: Path | None, k: int = 10) -> dict:
    """Blind pairwise test, slot by slot (A's #i vs B's #i). Each pair is judged twice with the clip order swapped; a
    side wins only if it wins in both orders, otherwise the pair is a tie (the judge was swayed by position)."""
    system = load_prompt("ab")
    pairs = []
    for i, (a, b) in enumerate(zip(promos_a[:k], promos_b[:k]), 1):
        pairs.append({"slot": i, "a": a, "b": b, "identical": _key(a) == _key(b)})
    live = [p for p in pairs if not p["identical"]]
    verdicts: dict[str, dict[int, str]] = {}
    reasons: dict[str, dict[int, str]] = {}
    for order in ("ab", "ba"):
        user = []
        for j, p in enumerate(live, 1):
            first, second = (p["a"], p["b"]) if order == "ab" else (p["b"], p["a"])
            user.append(f"PAIR {j}\nCLIP 1 ({first['duration_s']:.0f} s):\n{promo_transcript(first)}\n\nCLIP 2 ({second['duration_s']:.0f} s):\n{promo_transcript(second)}")
        try:
            out = llm.complete_json(system, "\n\n".join(user), ABOut, "ab", f"order-{order}",
                                    raw_path=(raw_dir / "ab" / f"order-{order}.json") if raw_dir else None)
            got = {it.pair: it for it in out.pairs}
        except LLMError as e:
            got = {}
            reasons[order] = {0: f"error: {e}"}
        verdicts[order] = {}
        reasons.setdefault(order, {})
        for j, p in enumerate(live, 1):
            it = got.get(j)
            w = (it.winner.strip().strip('"') if it else "")
            if w not in ("1", "2"):
                verdicts[order][p["slot"]] = "?"
                continue
            first_is_a = order == "ab"
            verdicts[order][p["slot"]] = ("a" if (w == "1") == first_is_a else "b")
            reasons[order][p["slot"]] = it.reason
    rows = []
    for p in pairs:
        if p["identical"]:
            rows.append({"slot": p["slot"], "result": "same clip"})
            continue
        v1, v2 = verdicts["ab"].get(p["slot"], "?"), verdicts["ba"].get(p["slot"], "?")
        res = v1 if v1 == v2 and v1 in ("a", "b") else "tie"
        rows.append({"slot": p["slot"], "result": res, "order_ab": v1, "order_ba": v2,
                     "a": p["a"]["promo_id"], "b": p["b"]["promo_id"],
                     "a_eps": [s["episode"] for s in p["a"]["segments"]], "b_eps": [s["episode"] for s in p["b"]["segments"]],
                     "reason_ab": reasons["ab"].get(p["slot"], ""), "reason_ba": reasons["ba"].get(p["slot"], "")})
    a_w = sum(r["result"] == "a" for r in rows)
    b_w = sum(r["result"] == "b" for r in rows)
    judged = [r for r in rows if r["result"] not in ("same clip",)]
    consistent = sum(1 for r in judged if r.get("order_ab") == r.get("order_ba") and r.get("order_ab") in ("a", "b"))
    return {"k": k, "a_wins": a_w, "b_wins": b_w, "ties": sum(r["result"] == "tie" for r in rows),
            "same_clip": sum(r["result"] == "same clip" for r in rows),
            "position_consistency": round(consistent / len(judged), 3) if judged else None, "rows": rows}
