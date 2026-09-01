#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PREDICTIONS [OUTPUT_DIR] [CONFIG]"
  exit 2
fi

PREDICTIONS=$1
OUTPUT_DIR=${2:-outputs/visualizations}
CONFIG=${3:-configs/psgformer/psgformer_r50_psg.py}

PYTHONPATH=. python tools/vis_results.py "$CONFIG" "$PREDICTIONS" \
  "$OUTPUT_DIR" --topk 20
