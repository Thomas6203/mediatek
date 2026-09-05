#!/usr/bin/env bash
# Run with: bash run.sh [CUDA_IDS]
# Examples: bash run.sh          # GPU 0
#           bash run.sh 0        # GPU 0
#           bash run.sh 0,1,2    # three-GPU DDP
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="25252"
# Edit this value to control training and full-catalog evaluation batch size.
BATCH_SIZE="128"
CUDA_IDS="${1:-0}"

if [[ $# -gt 1 || ! "$CUDA_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Usage: bash run.sh [CUDA_IDS], e.g. bash run.sh 0,1" >&2
  exit 2
fi

# These environment variables also cover any Hugging Face subcomponents.
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"

IFS=',' read -ra GPU_IDS <<< "$CUDA_IDS"
GPU_COUNT="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
cd "$PROJECT_ROOT"

TRAIN_ARGS=(
  --cache-dir "$CACHE_DIR" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --subset raw_review_All_Beauty \
  --epochs 20 \
  --device cuda
)

if [[ "$GPU_COUNT" -eq 1 ]]; then
  "$PYTHON_BIN" -m src.train "${TRAIN_ARGS[@]}"
else
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    -m src.train "${TRAIN_ARGS[@]}" --distributed
fi
