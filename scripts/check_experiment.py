"""Controlled test of the editor's checks: do clips that FAIL a check really do worse with a first-time viewer?

The judged finalists cannot answer this (the ranking already filtered failing clips out: selection bias). So for each
check: take strong candidates that fail it but pass every other code check, pair each with the passing candidate of the
nearest base score (hook + cliff + standalone + density, before penalties), and send all clips, shuffled, to the blind
cold-viewer judge. Writes <run>/eval/check_experiment.json.

  uv run python scripts/check_experiment.py <official_run> --checks ends_open,small_cast --pairs 6 --model <provider/model>
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

from promo_finder.cli import load_config  # noqa: E402
from promo_finder.judge import cold_viewer  # noqa: E402
from promo_finder.llm import RunLog, make_llm  # noqa: E402
from promo_finder.rank import flag_spoilers, generate_windows, nms, score_window  # noqa: E402
from promo_finder.rubric import CODE_CHECKS  # noqa: E402
from promo_finder.stability import replay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--checks", default="ends_open,small_cast")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--relaxed", action="store_true", help="match on base score only (other checks may fail on either side)")
    ap.add_argument("--out-name", default="check_experiment.json")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config(ROOT / "config.yaml")
    cfg["llm"]["reasoning_off"] = True
    cfg["llm"]["max_tokens"] = 8192
    ctx = replay(a.run)
    windows = generate_windows(ctx, cfg, "llm")
    for w in windows:
        score_window(w, ctx, cfg, "llm")
    flag_spoilers(windows, ctx, cfg)
    pool = [w for w in nms(windows, cfg) if w.kind == "single"]
    rng = random.Random(a.seed)
    chosen: dict[str, dict[str, list]] = {}
    used: set[str] = set()
    for check in a.checks.split(","):
        others = [c for c in CODE_CHECKS if c != check]
        if a.relaxed:
            fail = sorted([w for w in pool if not w.rubric.get(check, True)], key=lambda w: -w.subscores["base"])
            passing = [w for w in pool if w.rubric.get(check, True)]
        else:
            fail = sorted([w for w in pool if not w.rubric.get(check, True) and all(w.rubric.get(c, True) for c in others)],
                          key=lambda w: -w.subscores["base"])
            passing = [w for w in pool if all(w.rubric.get(c, True) for c in CODE_CHECKS)]
        pairs = []
        for f in fail:
            if len(pairs) >= a.pairs or f.cand_id in used:
                continue
            cands = [p for p in passing if p.cand_id not in used and p.flags.get("block") == f.flags.get("block")] or [p for p in passing if p.cand_id not in used]
            if not cands:
                break
            p = min(cands, key=lambda x: abs(x.subscores["base"] - f.subscores["base"]))
            used.update({f.cand_id, p.cand_id})
            pairs.append((f, p))
        chosen[check] = {"fail": [f for f, _ in pairs], "pass": [p for _, p in pairs]}
    clips = [w for c in chosen.values() for side in c.values() for w in side]
    rng.shuffle(clips)
    log = RunLog()
    llm = make_llm(a.model, cfg["llm"], ROOT / ".cache", log)
    judged, errors = cold_viewer(clips, ctx, llm, a.run / "llm_raw" / "check_experiment", per_call=5)
    out = {"model": a.model, "calls": log.summary(), "checks": {}}
    for check, c in chosen.items():
        res = {}
        for side in ("fail", "pass"):
            js = [judged[w.cand_id] for w in c[side] if w.cand_id in judged]
            res[side] = {"n": len(js), "follows": round(st.mean(j.comprehensibility for j in js), 2) if js else None,
                         "would_tap": round(st.mean(j.would_tap for j in js), 2) if js else None,
                         "base_score": round(st.mean(w.subscores["base"] for w in c[side]), 3) if c[side] else None}
        out["checks"][check] = res
        print(f"{check}: fail n={res['fail']['n']} follows {res['fail']['follows']} tap {res['fail']['would_tap']} (base {res['fail']['base_score']}) | "
              f"pass n={res['pass']['n']} follows {res['pass']['follows']} tap {res['pass']['would_tap']} (base {res['pass']['base_score']})")
    out["errors"] = errors
    (a.run / "eval").mkdir(exist_ok=True)
    out["relaxed"] = a.relaxed
    (a.run / "eval" / a.out_name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
