"""Assemble submission/: the raw outputs of the named runs (unedited), RUNS.txt, and outputs.zip for Drive.

  uv run python scripts/make_submission.py --official <run_dir> [--cue-blind <run_dir>] [--stability <run_dir>]
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, type=Path)
    ap.add_argument("--cue-blind", type=Path)
    ap.add_argument("--stability", type=Path, help="uncached re-run (same code version as --stability-base)")
    ap.add_argument("--stability-base", type=Path, help="the run --stability is compared with")
    ap.add_argument("--previous", type=Path, help="the previous version's official run (the A/B baseline)")
    ap.add_argument("--heuristic", type=Path, help="optional zero-key run, as evidence the fallback works")
    a = ap.parse_args()
    out = SUB / "outputs"
    if out.exists():
        shutil.rmtree(out)
    runs = {"official": a.official, "cue_blind_ablation": a.cue_blind, "stability_base": a.stability_base, "stability_rerun": a.stability,
            "previous_version": a.previous, "heuristic_only": a.heuristic}
    lines = ["Raw, unedited outputs of promo-finder on the full 50-episode input pack.", ""]
    for label, src in runs.items():
        if not src:
            continue
        dst = out / src.parent.name / f"{label}__{src.name}"
        shutil.copytree(src, dst)
        (dst / "OFFICIAL").unlink(missing_ok=True)
        if label == "official":
            (dst / "OFFICIAL").write_text("the run the solution doc reports\n")  # the app opens this run first
        lines.append(f"{label}: {src.parent.name}/{src.name}")
    (out / "README.txt").write_text("\n".join(lines + ["", "Start with official__*/promos.md (ranked promos), official__*/promos_by_episode.csv (one pick per episode),",
                                                      "official__*/eval/promoability.md (which episodes are easy/hard to promo) and official__*/eval/failures.md (where it picks badly).",
                                                      "official__*/eval/ab_vs_previous.json is the blind A/B against previous_version__*; cut_promos.sh renders every clip with ffmpeg.",
                                                      "llm_raw/ holds every raw model response; beats/ holds every beat's scores; eval/metrics.json holds the numbers in the doc."]))
    (SUB / "RUNS.txt").write_text("\n".join(f"{k}={v}" for k, v in runs.items() if v))
    zpath = SUB / "outputs.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print(f"wrote {out} and {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
