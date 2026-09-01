#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/psgformer/psgformer_r50_psg.py}
WORK_DIR=${2:-work_dirs/psgformer_r50_psg}

PYTHONPATH=. python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" \
  --seed 42 --deterministic
