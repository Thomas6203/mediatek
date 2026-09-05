#!/usr/bin/env bash
# REINFORCE + LoRA Mamba launcher. Usage: bash run_rl.sh [CUDA_IDS]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="25252"
# Lower than DL because this path backpropagates through Mamba 2.8B.
BATCH_SIZE="4"
CANDIDATES_PER_STEP="64"
EPOCHS="3"
VALIDATE_EVERY_STEPS="500"
CUDA_IDS="${1:-0}"

if [[ $# -gt 1 || ! "$CUDA_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Usage: bash run_rl.sh [CUDA_IDS], e.g. bash run_rl.sh 0,1" >&2
  exit 2
fi

export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
IFS=',' read -ra GPU_IDS <<< "$CUDA_IDS"
GPU_COUNT="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
cd "$PROJECT_ROOT"

TRAIN_ARGS=(
  --cache-dir "$CACHE_DIR"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --candidates-per-step "$CANDIDATES_PER_STEP"
  --epochs "$EPOCHS"
  --validate-every-steps "$VALIDATE_EVERY_STEPS"
  --subset raw_review_All_Beauty
  --device cuda
)

if [[ "$GPU_COUNT" -eq 1 ]]; then
  "$PYTHON_BIN" -m src.train_rl "${TRAIN_ARGS[@]}"
else
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    -m src.train_rl "${TRAIN_ARGS[@]}" --distributed
fi
