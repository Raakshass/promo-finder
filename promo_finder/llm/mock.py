"""Deterministic stand-in model for tests and demos (`--model mock/dev`). It answers every schema with
plausible, valid JSON derived from the prompt itself, so the entire LLM path (cache, raw dumps, judge,
entailment, reasons) can run with zero keys. It has no taste: never use it for real outputs."""
from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel

from .base import LLM


def _h(s: str, mod: int) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % mod


class MockLLM(LLM):
    def _raw_call(self, system, user, schema: type[BaseModel]):
        name = schema.__name__
        if name == "BatchScoresOut":
            eps = []
            for sec in re.split(r"^=== EPISODE (\d+) ===$", user, flags=re.M)[1:]:
                pass
            parts = re.split(r"^=== EPISODE (\d+) ===$", user, flags=re.M)
            for num, body in zip(parts[1::2], parts[2::2]):
                sub, _, _ = self._raw_call(system, body, type("BeatScoresOut", (BaseModel,), {}))
                eps.append({"episode": int(num), "beats": json.loads(sub)["beats"]})
            out = {"episodes": eps}
        elif name == "ColdViewerBatchOut":
            ids = re.findall(r"^CLIP \d+ \(id=([^,]+),", user, flags=re.M)
            out = {"clips": [{"clip_id": i, "comprehensibility": 3 + _h(i, 3), "what_happened": "Someone is accused and threatened.",
                              "want_next": "Will the threat be carried out?", "unresolved_refs": [], "would_tap": 2 + _h(i, 3),
                              "why_it_works": "Opens on a confrontation and cuts before the answer.",
                              "stakes": "She wants the truth, but he threatens her." if _h(i, 4) else "", "root_for": "the accused woman",
                              "resolved_inside": _h(i, 5) == 0, "opening_conflict": True} for i in ids]}
        elif name == "LedgerOut":
            items = []
            for pid, ep in re.findall(r"^(P\d+) \(from episode (\d+)\)", user, flags=re.M):
                k = _h(pid, 3)
                items.append({"promo_id": pid, "answered_in_episode": None if k == 0 else int(ep) + k - 1, "answer": "mock", "confidence": "low"})
            out = {"items": items}
        elif name == "ABOut":
            n = len(re.findall(r"^PAIR \d+$", user, flags=re.M))
            out = {"pairs": [{"pair": j, "winner": "1" if _h(user + str(j), 2) else "2", "reason": "mock"} for j in range(1, n + 1)]}
        elif name == "EntailBatchOut":
            nums = re.findall(r"^PAIR \d+ — EPISODE (\d+)$", user, flags=re.M)
            out = {"pairs": [{"episode": int(n), "contains": _h(n, 4) != 0, "reason": "mock"} for n in nums]}
        elif name == "BeatScoresOut":
            beats = []
            for m in re.finditer(r"^\[(b\d+) [^\]]*\] (.*)$", user, flags=re.M):
                bid, line = m.group(1), m.group(2)
                low = line.lower()
                hook = 3 if re.search(r"slap|strike|marry|hang|arrest|poison|knew|liar|dead", low) else (2 if "!" in line else _h(bid + line, 2))
                cliff = 3 if low.rstrip('"').endswith("?") or re.search(r"or (else|I)|tell me|choose|if you", low) else _h(line, 3)
                beats.append({"beat_id": bid, "hook": hook, "cliff": cliff, "beat_type": "threat" if "hang" in low else "line",
                              "unresolved_refs": [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", line)[:1]]})
            out = {"beats": beats}
        elif name == "AliasesOut":
            labels = re.findall(r"^\s*\d+\s+(.+)$", user.split("LOGLINES")[0], flags=re.M)
            aliases = {}
            for l in labels:
                base = re.sub(r"^(LADY|LORD|THE) ", "", l.strip())
                if base != l.strip() and base in labels:
                    aliases[l.strip()] = base
            out = {"aliases": aliases, "notes": "mock: stripped titles only"}
        elif name == "ColdViewer":
            k = _h(user, 3)
            out = {"comprehensibility": 3 + k, "what_happened": "Someone is accused and threatened; another person refuses to back down.",
                   "want_next": "Will the threat be carried out?", "unresolved_refs": [], "would_tap": 2 + k,
                   "why_it_works": "Opens on a confrontation and cuts before the answer."}
        elif name == "EntailOut":
            out = {"contains": _h(user, 4) != 0, "reason": "mock judgement"}
        elif name == "ReasonsOut":
            ids = re.findall(r"^(P\d\d) ", user, flags=re.M)
            out = {"reasons": {i: "Mock reason: strong open, unresolved close." for i in ids}}
        else:
            out = {}
        text = json.dumps(out)
        return text, len(user) // 4, len(text) // 4

    def list_models(self) -> list[str]:
        return ["dev"]
