#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../.venvs/mediatek-ver4/bin/python}"
CACHE_DIR="${CACHE_DIR:-$ROOT/../cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs_ver4_protocol_5runs}"
export PYTHONPATH="$ROOT/vendor:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GPU_ID="${GPU_ID:-0}"
REPEATS="${REPEATS:-5}"
BASE_SEED="${BASE_SEED:-25252}"

declare -A MAX_TRANSITIONS=(
  [all_beauty]=500000
  [baby_products]=1500000
  [sports_and_outdoors]=1000000
  [toys_and_games]=1500000
)

if (( $# )); then
  DATASETS=("$@")
else
  DATASETS=(all_beauty baby_products sports_and_outdoors toys_and_games)
fi

for ((repeat_index = 0; repeat_index < REPEATS; repeat_index++)); do
  seed=$((BASE_SEED + repeat_index))
  for dataset in "${DATASETS[@]}"; do
    "$PYTHON_BIN" "$ROOT/prepare_ver4_data.py" "$dataset" --cache-dir "$CACHE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$ROOT/run_ver4_protocol.py" "$dataset" \
      --device cuda:0 \
      --seed "$seed" \
      --epochs 18 \
      --output-dir "$OUTPUT_DIR" \
      --max-transitions "${MAX_TRANSITIONS[$dataset]}" \
      --save-model-weights
  done
done
