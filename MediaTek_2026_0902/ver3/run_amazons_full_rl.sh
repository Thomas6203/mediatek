#!/usr/bin/env bash
# Run from anywhere: bash /workspace/P78123011/MediaTek_2026_project/run_amazons_full_rl.sh
# Every subset uses the same model architecture. Training-only hyperparameters
# scale with catalog size; every validation and test evaluates all eligible users.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/outputs_mamba_rl/amazons_full"
PYTHON_BIN="${PYTHON_BIN:-/home/P78123011/miniforge3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
GPU_IDS="${GPU_IDS:-0,1}"

COMMON_ENV=(
  "PYTHON_BIN=$PYTHON_BIN"
  "CACHE_DIR=$CACHE_DIR"
  "BATCH_SIZE=128"
  "EVAL_BATCH_SIZE=64"
  "MONITOR_METRIC=ndcg@10"
  "EARLY_STOPPING_PATIENCE=6"
  "LR_PATIENCE=2"
  "DIM=128"
  "LORA_RANK=8"
  "LORA_ALPHA=16.0"
  "LORA_DROPOUT=0.05"
  "SHORT_WINDOW=10"
  "USE_GRAPH_EMBEDDINGS=1"
  "MAX_HISTORY=100"
  "MAMBA_ENCODE_BATCH_SIZE=32"
  "MAMBA_MAX_TOKENS=16"
  "GENERATE_REASONS=0"
  "SAVE_MODEL_WEIGHTS=0"
)

run_subset() {
  local subset_name="$1"
  local dataset="$2"
  local validate_every_steps="$3"
  local max_transitions="$4"
  local candidates="$5"
  local specialist_epochs="$6"
  local coordinator_epochs="$7"
  local rl_epochs="$8"
  local specialist_lr="$9"
  local coordinator_lr="${10}"
  local joint_lr="${11}"
  local entropy_coef="${12}"
  local supervised_coef="${13}"
  local specialization_coef="${14}"
  local popularity_alpha="${15}"
  local transition_beta="${16}"
  local timestamp run_dir

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/rl_$timestamp"

  env "${COMMON_ENV[@]}" bash "$PROJECT_ROOT/run_mamba_rl.sh" "$dataset" "$GPU_IDS" \
    --output-run-dir "$run_dir" \
    --score-file "$run_dir/${subset_name}_scores.json" \
    --validate-every-steps "$validate_every_steps" \
    --validation-user-limit 0 \
    --periodic-test-user-limit 0 \
    --max-transitions "$max_transitions" \
    --candidates "$candidates" \
    --specialist-epochs "$specialist_epochs" \
    --coordinator-epochs "$coordinator_epochs" \
    --rl-epochs "$rl_epochs" \
    --specialist-lr "$specialist_lr" \
    --coordinator-lr "$coordinator_lr" \
    --joint-lr "$joint_lr" \
    --entropy-coef "$entropy_coef" \
    --supervised-coef "$supervised_coef" \
    --specialization-coef "$specialization_coef" \
    --popularity-alpha "$popularity_alpha" \
    --transition-beta "$transition_beta" \
    --experiment-note "amazons_full subset=$subset_name; common architecture; periodic validation/test"
}

# One command per requested subset. Set REPEATS=5 to repeat the whole suite five times.
REPEATS="${REPEATS:-1}"
for ((run_number = 1; run_number <= REPEATS; run_number++)); do
  # name dataset validation_steps max_samples candidates specialist coord joint
  # specialist_lr coord_lr joint_lr entropy supervised specialization popularity transition
  # run_subset "Beauty" "amazon-all-beauty" 250 500000 64 3 2 20 2e-4 2e-4 5e-5 0.01 0.1 0.01 -0.25 4.0
  # run_subset "Sports" "amazon-sports-and-outdoors" 8000 1000000 256 3 2 12 2e-5 1e-4 2e-5 0.003 0.5 0.005 0.35 0.5
  # run_subset "Games" "amazon-games" 4000 1000000 128 3 2 15 5e-5 1e-4 2e-5 0.005 0.4 0.005 0.20 0.5
  # run_subset "Books" "amazon-books" 12000 2000000 256 3 2 12 2e-5 1e-4 2e-5 0.003 0.5 0.005 0.35 0.5
  run_subset "Toys" "amazon-toys" 6000 1500000 192 3 2 15 3e-5 1e-4 2e-5 0.004 0.5 0.005 0.30 0.5
  run_subset "Video_Games" "amazon-video-games" 4000 1000000 128 3 2 15 5e-5 1e-4 2e-5 0.005 0.4 0.005 0.20 0.5
  run_subset "Clothing" "amazon-clothing-shoes-and-jewelry" 12000 2000000 256 3 2 12 2e-5 1e-4 2e-5 0.003 0.5 0.005 0.35 0.5
done
