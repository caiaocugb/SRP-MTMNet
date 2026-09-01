#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [CONFIG]"
  exit 2
fi

CHECKPOINT=$1
CONFIG=${2:-configs/psgformer/psgformer_r50_psg.py}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/psgformer_r50_psg/evaluation}

mkdir -p "$OUTPUT_DIR"
PYTHONPATH=. python tools/test.py "$CONFIG" "$CHECKPOINT" \
  --out "$OUTPUT_DIR/results.pkl" \
  --work-dir "$OUTPUT_DIR" \
  --eval sgdet
