#!/usr/bin/env bash
# Rebuild the 2-page PDF with the submission links. Usage: bash scripts/final_doc.sh "<app url>" "<outputs link>" "<video link or ->"
set -euo pipefail
cd "$(dirname "$0")/.."
APP="${1:-}"; OUTS="${2:-}"; VID="${3:-}"
get() { grep "^$1=" submission/RUNS.txt | cut -d= -f2-; }
uv run python scripts/fill_doc.py --run "$(get official)" --cue-blind "$(get cue_blind_ablation)" --minutes 10 \
  --repo "https://github.com/Raakshass/promo-finder" --prototype "$APP" --outputs "$OUTS" --video "$VID"
cp docs/solution.pdf submission/solution.pdf
echo "submission/solution.pdf updated"
