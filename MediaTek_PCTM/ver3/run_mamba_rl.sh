#!/usr/bin/env bash
# Usage: bash run_mamba_rl.sh DATASET [CUDA_IDS] [DATA_PATH] [TRAINING_OPTIONS...]
# Examples:
#   bash run_mamba_rl.sh movielens-1m 0
#   bash run_mamba_rl.sh amazon-all-beauty 0 --validate-every-steps 250
#   bash run_mamba_rl.sh amazon-sports-and-outdoors 0,1
#   bash run_mamba_rl.sh yelp19 0 /data/yelp-2019
#   bash run_mamba_rl.sh yelp23 0 /data/yelp-2023/yelp_academic_dataset_review.json
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
DATASET="${1:-movielens-1m}"
CUDA_IDS="${2:-0}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
DATA_PATH=""
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  DATA_PATH="$1"
  shift
fi
TRAINING_OPTIONS=("$@")
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_EVENTS="${MAX_EVENTS:-}"
SPECIALIST_EPOCHS="${SPECIALIST_EPOCHS:-3}"
COORDINATOR_EPOCHS="${COORDINATOR_EPOCHS:-2}"
RL_EPOCHS="${RL_EPOCHS:-20}"
VALIDATE_EVERY_STEPS="${VALIDATE_EVERY_STEPS:-1000}"
MONITOR_METRIC="${MONITOR_METRIC:-ndcg@10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
LR_PATIENCE="${LR_PATIENCE:-3}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-500000}"
SEED="${SEED:-25252}"
POPULARITY_ALPHA="${POPULARITY_ALPHA:-0.0}"
TRANSITION_BETA="${TRANSITION_BETA:-0.0}"
TARGET_RECALL_AT_10="${TARGET_RECALL_AT_10:-0.15}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs_mamba_rl}"
OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-}"
SCORE_FILE="${SCORE_FILE:-}"
GENERATE_REASONS="${GENERATE_REASONS:-0}"
SAVE_MODEL_WEIGHTS="${SAVE_MODEL_WEIGHTS:-1}"
CANDIDATES="${CANDIDATES:-64}"
DIM="${DIM:-128}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SHORT_WINDOW="${SHORT_WINDOW:-10}"
USE_GRAPH_EMBEDDINGS="${USE_GRAPH_EMBEDDINGS:-1}"
MAX_HISTORY="${MAX_HISTORY:-100}"
SPECIALIST_LR="${SPECIALIST_LR:-2e-4}"
COORDINATOR_LR="${COORDINATOR_LR:-2e-4}"
JOINT_LR="${JOINT_LR:-5e-5}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
SUPERVISED_COEF="${SUPERVISED_COEF:-0.1}"
SPECIALIZATION_COEF="${SPECIALIZATION_COEF:-0.01}"
FULL_CATALOG_SUPERVISED="${FULL_CATALOG_SUPERVISED:-0}"
MAMBA_ENCODE_BATCH_SIZE="${MAMBA_ENCODE_BATCH_SIZE:-4}"
MAMBA_MAX_TOKENS="${MAMBA_MAX_TOKENS:-48}"
VALIDATION_USER_LIMIT="${VALIDATION_USER_LIMIT:-0}"
PERIODIC_TEST_USER_LIMIT="${PERIODIC_TEST_USER_LIMIT:--1}"

if [[ ! "$CUDA_IDS" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "Usage: bash run_mamba_rl.sh DATASET [CUDA_IDS] [DATA_PATH] [TRAINING_OPTIONS...]" >&2
  exit 2
fi

case "$DATASET" in
  movielens-100k|movielens-1m|movielens-20m|movielens-25m|movielens-32m|movielens-latest-small|amazon-*|amazon:*|yelp19|yelp-2019|yelp23|yelp-2023|synthetic) ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    echo "Use movielens-{100k,1m,20m,25m,32m,latest-small}, amazon-CATEGORY, yelp19, or yelp23." >&2
    exit 2
    ;;
esac
if [[ "$DATASET" == yelp* && -z "$DATA_PATH" ]]; then
  echo "Yelp snapshots require DATA_PATH pointing to the licensed local dataset." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_IDS"
MAIN_DEVICE="cuda:0"
GRAPH_DEVICE="cuda:0"
if [[ "$CUDA_IDS" == *,* ]]; then
  GRAPH_DEVICE="cuda:1"
fi
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
cd "$PROJECT_ROOT"

ARGS=(
  --dataset "$DATASET"
  --cache-dir "$CACHE_DIR"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --specialist-epochs "$SPECIALIST_EPOCHS"
  --coordinator-epochs "$COORDINATOR_EPOCHS"
  --rl-epochs "$RL_EPOCHS"
  --validate-every-steps "$VALIDATE_EVERY_STEPS"
  --monitor-metric "$MONITOR_METRIC"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --lr-patience "$LR_PATIENCE"
  --max-transitions "$MAX_TRANSITIONS"
  --candidates "$CANDIDATES"
  --dim "$DIM"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --short-window "$SHORT_WINDOW"
  --max-history "$MAX_HISTORY"
  --specialist-lr "$SPECIALIST_LR"
  --coordinator-lr "$COORDINATOR_LR"
  --joint-lr "$JOINT_LR"
  --entropy-coef "$ENTROPY_COEF"
  --supervised-coef "$SUPERVISED_COEF"
  --specialization-coef "$SPECIALIZATION_COEF"
  --mamba-encode-batch-size "$MAMBA_ENCODE_BATCH_SIZE"
  --mamba-max-tokens "$MAMBA_MAX_TOKENS"
  --validation-user-limit "$VALIDATION_USER_LIMIT"
  --periodic-test-user-limit "$PERIODIC_TEST_USER_LIMIT"
  --seed "$SEED"
  --popularity-alpha "$POPULARITY_ALPHA"
  --transition-beta "$TRANSITION_BETA"
  --target-recall-at-10 "$TARGET_RECALL_AT_10"
  --experiment-note "$EXPERIMENT_NOTE"
  --output-dir "$OUTPUT_DIR"
  --device "$MAIN_DEVICE"
  --graph-device "$GRAPH_DEVICE"
)
if [[ "$USE_GRAPH_EMBEDDINGS" == "1" ]]; then
  ARGS+=(--use-graph-embeddings)
else
  ARGS+=(--no-use-graph-embeddings)
fi
if [[ -n "$MAX_EVENTS" ]]; then
  ARGS+=(--max-events "$MAX_EVENTS")
fi
if [[ "$GENERATE_REASONS" == "1" ]]; then
  ARGS+=(--generate-reasons)
else
  ARGS+=(--no-generate-reasons)
fi
if [[ "$FULL_CATALOG_SUPERVISED" == "1" ]]; then
  ARGS+=(--full-catalog-supervised)
else
  ARGS+=(--no-full-catalog-supervised)
fi
if [[ "$SAVE_MODEL_WEIGHTS" == "1" ]]; then
  ARGS+=(--save-model-weights)
else
  ARGS+=(--no-save-model-weights)
fi
if [[ -n "$OUTPUT_RUN_DIR" ]]; then
  ARGS+=(--output-run-dir "$OUTPUT_RUN_DIR")
fi
if [[ -n "$SCORE_FILE" ]]; then
  ARGS+=(--score-file "$SCORE_FILE")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "$DATA_PATH" ]]; then
  ARGS+=(--data-path "$DATA_PATH")
fi

# Explicit options supplied by a suite launcher come last and therefore
# override the environment-backed defaults above.
ARGS+=("${TRAINING_OPTIONS[@]}")

"$PYTHON_BIN" -m src.train_mamba_rl "${ARGS[@]}"
