"""External benchmark: does promo-finder pick the moments professional editors pick, on dramas it has never seen?

Data: DramaAD (released with arXiv 2507.02790, EMNLP 2025 Industry track; https://huggingface.co/datasets/wonderful9462/DramaAD).
28 Chinese short dramas (72-109 episodes). Per drama, multimodal_narration_understanding.json holds every episode's
timed dialogue (with character names), timed scene descriptions and a one-line summary; script.json lists, for each of
the ~20 ads cut by professional editors, the episode and start/end milliseconds of every segment the ad uses.

  uv run python scripts/benchmark_dramaad.py stats   <dramaad_dir>                      # where editors take footage from
  uv run python scripts/benchmark_dramaad.py convert <dramaad_dir> <out_dir> --dramas ID,ID --episodes 20
  uv run python scripts/benchmark_dramaad.py score   <dramaad_dir> <runs_dir> --episodes 20 [--k 10]

Download (text only, ~25 MB): for each drama id, fetch
  https://huggingface.co/datasets/wonderful9462/DramaAD/resolve/main/<id>/{multimodal_narration_understanding.json,script.json}

Scoring. Editors often reuse whole early episodes, so naive overlap rewards "pick early episodes", not "pick good
moments". usage(t) = how many of the drama's professional ads contain second t. For our top-k clips we report:
  lift            mean usage of our clips' seconds / mean usage of all seconds in the evaluated episodes (random = 1.0)
  within_lift     the same, but each clip compared with its own episode's mean (controls for episode position)
  hit@k           share of our clips with >= 50% of their seconds inside some professional ad
  episode_rho     Spearman correlation, across evaluated episodes, between our best clip score and editors' usage
Baselines on the same footing: random clips (expected lift 1.0, measured by sampling) and "episode openings"
(the first N seconds of each episode, a strong trivial strategy because editors often start segments at 0).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

TS = re.compile(r"^第([\d.]+)~([\d.]+)秒\s*(.*)$")


def fmt(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def load(drama: Path) -> tuple[dict, dict]:
    mnu = json.loads((drama / "multimodal_narration_understanding.json").read_text(encoding="utf-8"))
    ads = json.loads((drama / "script.json").read_text(encoding="utf-8"))
    return mnu, ads


def episode_lines(ep: dict) -> tuple[list[tuple[float, float, str, str]], list[tuple[float, float, str, str]]]:
    """(dialogue lines, scenes) as (t0, t1, who/where, text)."""
    dia, scenes = [], []
    for line in (ep.get("dialog") or "").splitlines():
        m = TS.match(line.strip())
        if not m:
            continue
        t0, t1, rest = float(m.group(1)), float(m.group(2)), m.group(3)
        who, _, text = rest.partition(":")
        if not text:
            who, _, text = rest.partition("：")
        if text.strip():
            dia.append((t0, t1, who.strip() or "VOICE", text.strip()))
    for line in (ep.get("scene_desc") or "").splitlines():
        m = TS.match(line.strip())
        if not m:
            continue
        t0, t1, rest = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        where, _, desc = rest.partition(" ")
        scenes.append((t0, t1, where.strip(), desc.strip()))
    return dia, scenes


def to_screenplay(n: int, ep: dict) -> str:
    """One DramaAD episode in promo-finder's input format (timed dialogue + scene descriptions as ACTION blocks)."""
    dia, scenes = episode_lines(ep)
    if not scenes and dia:
        scenes = [(0.0, max(t1 for _, t1, _, _ in dia), "", "")]
    out = [f"EPISODE NUMBER: Episode {n}", f"EPISODE LOGLINE: {(ep.get('summary') or '').strip()}", "CONTINUITY NOTES: ", "", "--- SCRIPT START ---", ""]
    for k, (s0, s1, where, desc) in enumerate(scenes, 1):
        out += [f"SCENE {k:02d}", f"Scene heading: {where or 'SCENE'}", f"Scene start: {fmt(s0)}", f"Scene end: {fmt(s1)}", "", "Actions and dialogue:", ""]
        if desc:
            first = re.split(r"(?<=[。！？])", desc)[0][:120]  # the scene's first sentence, as on-screen action
            out += ["**ACTION:**", f"Start: ({fmt(s0)}) | End: ({fmt(min(s1, s0 + 3.0))})", first, ""]
        for t0, t1, who, text in dia:
            if s0 - 0.05 <= t0 < s1 or (k == len(scenes) and t0 >= s1):
                label = f"{who} (V.O.)" if who in ("旁白", "VOICE") else who
                out += [f"**{label}:**", f"Start: ({fmt(t0)}) | End: ({fmt(max(t1, t0 + 0.3))})", f'"{text}"', ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out) + "\n"


def cmd_convert(a) -> None:
    root, out = Path(a.dramaad), Path(a.out)
    ids = a.dramas.split(",") if a.dramas else sorted(p.name for p in root.iterdir() if p.is_dir())
    for d in ids:
        mnu, _ = load(root / d)
        dst = out / d
        dst.mkdir(parents=True, exist_ok=True)
        eps = mnu["episode_info"]
        for k in sorted(eps, key=int)[: a.episodes]:
            (dst / f"episode-{int(k):03d}.txt").write_text(to_screenplay(int(k), eps[k]), encoding="utf-8")
        print(f"{d}: {mnu['drama_info'].get('title')} ({mnu['drama_info'].get('category')}) -> {min(a.episodes, len(eps))} episodes in {dst}")


def cmd_stats(a) -> None:
    root = Path(a.dramaad)
    pos, first, by_block, secs = [], [], {"opening": 0, "middle": 0, "late": 0}, {"opening": 0.0, "middle": 0.0, "late": 0.0}
    ads_n = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mnu, ads = load(d)
        n = len(mnu["episode_info"])
        for segs in ads.values():
            ads_n += 1
            segs = sorted(segs, key=lambda s: (s["epid"], s["start"]))
            first.append((segs[0]["epid"] - 1) / (n - 1))
            for s in segs:
                p = (s["epid"] - 1) / (n - 1)
                pos.append(p)
                b = "opening" if p < 0.2 else ("middle" if p < 0.5 else "late")
                by_block[b] += 1
                secs[b] += (s["end"] - s["start"]) / 1000
    tot, ts = len(pos), sum(secs.values())
    res = {"dramas": len([p for p in root.iterdir() if p.is_dir()]), "ads": ads_n, "segments": tot,
           "share_footage_seconds": {k: round(v / ts, 3) for k, v in secs.items()},
           "share_segments": {k: round(v / tot, 3) for k, v in by_block.items()},
           "share_first_10pct": round(sum(p <= 0.1 for p in pos) / tot, 3),
           "ads_opening_in_first_20pct": round(sum(p < 0.2 for p in first) / len(first), 3),
           "decile_histogram": [sum(1 for p in pos if min(9, int(p * 10)) == i) for i in range(10)]}
    print(json.dumps(res, indent=1))


def usage_map(ads: dict, episodes: set[int]) -> dict[int, dict[int, int]]:
    """usage[ep][second] = number of professional ads containing that second."""
    u: dict[int, dict[int, int]] = {}
    for segs in ads.values():
        seen: set[tuple[int, int]] = set()
        for s in segs:
            if s["epid"] not in episodes:
                continue
            for t in range(int(s["start"] // 1000), int(s["end"] // 1000) + 1):
                seen.add((s["epid"], t))
        for ep, t in seen:
            u.setdefault(ep, {})
            u[ep][t] = u[ep].get(t, 0) + 1
    return u


def clip_seconds(segs: list[dict]) -> list[tuple[int, int]]:
    return [(s["episode"], t) for s in segs for t in range(int(s["start_s"]), int(s["end_s"]) + 1)]


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def evaluate_run(run: Path, ads: dict, n_eps: int, dur: dict[int, float], k: int, seed: int = 0, source: str = "candidates") -> dict:
    """Score our top-k moments (all scored windows after overlap suppression, best first) against professional usage."""
    eps = set(range(1, n_eps + 1))
    u = usage_map(ads, eps)
    all_secs = [(e, t) for e in eps for t in range(int(dur.get(e, 0)) + 1)]
    mean_all = st.mean(u.get(e, {}).get(t, 0) for e, t in all_secs)
    ep_mean = {e: st.mean([u.get(e, {}).get(t, 0) for t in range(int(dur.get(e, 0)) + 1)] or [0]) for e in eps}
    cands = [json.loads(l) for l in (run / "candidates.jsonl").read_text(encoding="utf-8").splitlines()] if (run / "candidates.jsonl").exists() else []
    slate = [{"segments": p["segments"], "score": p["score"]} for p in json.loads((run / "promos.json").read_text(encoding="utf-8"))]
    if not cands:
        cands = slate
    top = slate[:k] if source == "slate" else sorted(cands, key=lambda c: -c["score"])[:k]
    # can within-episode choice be measured at all? only where editors used part of an episode, not all of it
    varying = sum(1 for e in eps if len({u.get(e, {}).get(t, 0) for t in range(int(dur.get(e, 0)) + 1)}) > 1)

    def stats_for(clips):
        lifts, wl, hits = [], [], 0
        for c in clips:
            secs = [(e, t) for e, t in clip_seconds(c["segments"]) if e in eps]
            if not secs:
                continue
            vals = [u.get(e, {}).get(t, 0) for e, t in secs]
            lifts.append(st.mean(vals) / mean_all if mean_all else 0)
            base = st.mean(ep_mean[e] for e, _ in secs)
            wl.append(st.mean(vals) / base if base else 1.0)
            hits += sum(1 for v in vals if v > 0) / len(vals) >= 0.5
        return {"lift": round(st.mean(lifts), 3) if lifts else None, "within_lift": round(st.mean(wl), 3) if wl else None,
                "hit_rate": round(hits / len(clips), 3) if clips else None}

    ours = stats_for(top)
    rng = random.Random(seed)
    lengths = [sum(s["end_s"] - s["start_s"] for s in c["segments"]) for c in top] or [35.0]
    rand_runs = []
    for _ in range(200):
        clips = []
        for L in lengths:
            e = rng.choice(sorted(eps))
            s0 = rng.uniform(0, max(0.0, dur.get(e, 0) - L))
            clips.append({"segments": [{"episode": e, "start_s": s0, "end_s": s0 + L}]})
        rand_runs.append(stats_for(clips))
    random_base = {m: round(st.mean(r[m] for r in rand_runs if r[m] is not None), 3) for m in ("lift", "within_lift", "hit_rate")}
    openings = [{"segments": [{"episode": e, "start_s": 0.0, "end_s": min(dur.get(e, 0), st.median(lengths))}]} for e in sorted(eps)[:k]]
    best_by_ep: dict[int, float] = {}
    for c in cands:
        e = c["segments"][0]["episode"]
        best_by_ep[e] = max(best_by_ep.get(e, 0.0), c["score"])
    ev = sorted(eps)
    rho = spearman([best_by_ep.get(e, 0.0) for e in ev], [ep_mean[e] for e in ev])
    return {"source": source, "ours": ours, "random": random_base, "episode_openings": stats_for(openings), "episode_rho": rho, "k": k,
            "episodes": n_eps, "episodes_with_partial_use": varying}


def cmd_score(a) -> None:
    root, runs = Path(a.dramaad), Path(a.runs)
    results = {}
    for run in sorted(p for p in runs.glob("*/*") if (p / "promos.json").exists()):
        drama = run.parent.name
        if not (root / drama).exists():
            continue
        mnu, ads = load(root / drama)
        dur = {}
        for kk, ep in mnu["episode_info"].items():
            dia, scenes = episode_lines(ep)
            dur[int(kk)] = max([t1 for _, t1, _, _ in dia] + [t1 for _, t1, _, _ in scenes] + [0.0])
        results[drama] = {"run": run.name, "title": mnu["drama_info"].get("title"), "category": mnu["drama_info"].get("category"),
                          **evaluate_run(run, ads, a.episodes, dur, a.k, source=a.source)}
    agg = {}
    for key in ("lift", "within_lift", "hit_rate"):
        for who in ("ours", "random", "episode_openings"):
            vals = [r[who][key] for r in results.values() if r[who][key] is not None]
            agg[f"{who}_{key}"] = round(st.mean(vals), 3) if vals else None
    rhos = [r["episode_rho"] for r in results.values() if r["episode_rho"] is not None]
    agg["episode_rho_mean"] = round(st.mean(rhos), 3) if rhos else None
    agg["episodes_with_partial_use"] = sum(r["episodes_with_partial_use"] for r in results.values())
    agg["episodes_evaluated"] = sum(r["episodes"] for r in results.values())
    out = {"dramas": len(results), "summary": agg, "per_drama": results}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("dramaad"); c.add_argument("out"); c.add_argument("--dramas", default=""); c.add_argument("--episodes", type=int, default=20)
    s = sub.add_parser("stats"); s.add_argument("dramaad")
    sc = sub.add_parser("score"); sc.add_argument("dramaad"); sc.add_argument("runs"); sc.add_argument("--episodes", type=int, default=20); sc.add_argument("--k", type=int, default=10)
    sc.add_argument("--json-out", default="")
    sc.add_argument("--source", choices=["candidates", "slate"], default="candidates", help="score the raw best moments, or the published slate")
    a = ap.parse_args()
    {"convert": cmd_convert, "stats": cmd_stats, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
