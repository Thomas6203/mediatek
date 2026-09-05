#!/usr/bin/env bash
# Run or resume one matched-protocol MediaTek experiment.
#
# Usage:
#   bash run_pctm_protocol.sh DATASET HOURS [SEED] [CUDA_ID] [DATA_ROOT]
#
# Example (run at most six hours this invocation):
#   bash run_pctm_protocol.sh amazon-beauty-pctm 6 25252 0
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"

DATASET="${1:-}"
RUN_HOURS="${2:-}"
SEED="${3:-25252}"
CUDA_ID="${4:-0}"
DATA_ROOT="${5:-$WORKSPACE_ROOT/sequential-capacity-probes}"

if [[ -z "$DATASET" || -z "$RUN_HOURS" ]]; then
  echo "Usage: bash run_pctm_protocol.sh DATASET HOURS [SEED] [CUDA_ID] [DATA_ROOT]" >&2
  exit 2
fi
case "$DATASET" in
  amazon-beauty-pctm|amazon-sports-pctm|amazon-toys-pctm|movielens-1m-pctm|movielens-20m-pctm|amazon-all-beauty-2023-pctm|amazon-all-beauty-2023-unfiltered-pctm) ;;
  *)
    echo "Unsupported matched-protocol DATASET:" >&2
    echo "  amazon-beauty-pctm amazon-sports-pctm amazon-toys-pctm movielens-1m-pctm movielens-20m-pctm" >&2
    echo "  amazon-all-beauty-2023-pctm (custom frozen extension)" >&2
    echo "  amazon-all-beauty-2023-unfiltered-pctm (custom unfiltered extension)" >&2
    exit 2
    ;;
esac
if [[ ! "$RUN_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "HOURS must be a non-negative number; 0 means run until completion." >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$CUDA_ID" =~ ^[0-9]+$ ]]; then
  echo "CUDA_ID must name one physical GPU, for example 0." >&2
  exit 2
fi
if [[ "$DATASET" == "amazon-all-beauty-2023-pctm" || "$DATASET" == "amazon-all-beauty-2023-unfiltered-pctm" ]]; then
  if [[ "$DATASET" == "amazon-all-beauty-2023-pctm" ]]; then
    CUSTOM_DIRECTORY=amazon_2023_all_beauty_onepass5
  else
    CUSTOM_DIRECTORY=amazon_2023_all_beauty_unfiltered
  fi
  EXPECTED_SPLIT="$DATA_ROOT/$CUSTOM_DIRECTORY/leave_one_out"
  if [[ ! -f "$EXPECTED_SPLIT/train.csv" || ! -f "$EXPECTED_SPLIT/holdout.csv" ]]; then
    echo "Frozen All_Beauty split was not found under: $EXPECTED_SPLIT" >&2
    echo "Run prepare_all_beauty_2023_protocol.py and freeze-custom first." >&2
    exit 2
  fi
elif [[ ! -d "$DATA_ROOT/data/processed" ]]; then
  echo "Official processed data was not found under: $DATA_ROOT" >&2
  echo "Run the data-preparation and verification steps first." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/.venvs/mediatek-ver4/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_ROOT/experiments/pctm-official-four}"
CONTRACT_ROOT="${CONTRACT_ROOT:-$WORKSPACE_ROOT/contracts/pctm-official-four}"
CHECKPOINT_EVERY_MINUTES="${CHECKPOINT_EVERY_MINUTES:-30}"
RUN_DIR="$RESULT_ROOT/$DATASET/seed_$SEED"
CHECKPOINT="$RUN_DIR/run_checkpoint.pt"
STATUS_FILE="$RUN_DIR/run_status.json"
METRICS_FILE="$RUN_DIR/metrics.json"
REFERENCE_CONTRACT="$CONTRACT_ROOT/$DATASET.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -f "$STATUS_FILE" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"completed"' "$STATUS_FILE"; then
  echo "Already completed: $RUN_DIR"
  if [[ -f "$REFERENCE_CONTRACT" && -f "$METRICS_FILE" ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/verify_data_contract.py" compare \
      "$REFERENCE_CONTRACT" "$METRICS_FILE"
  fi
  exit 0
fi

mkdir -p "$RUN_DIR"
RESUME_ENV=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ENV+=("RESUME_CHECKPOINT=$CHECKPOINT")
  echo "Resuming checkpoint: $CHECKPOINT"
else
  echo "Starting new run: $RUN_DIR"
fi

env \
  "${RESUME_ENV[@]}" \
  "PYTHON_BIN=$PYTHON_BIN" \
  "CACHE_DIR=$CACHE_DIR" \
  "OUTPUT_RUN_DIR=$RUN_DIR" \
  "SCORE_FILE=$RUN_DIR/${DATASET}_scores.json" \
  "RUN_HOURS=$RUN_HOURS" \
  "CHECKPOINT_EVERY_MINUTES=$CHECKPOINT_EVERY_MINUTES" \
  "SEED=$SEED" \
  "MAX_TRANSITIONS=0" \
  "PCTM_VERIFY_SPLIT=1" \
  "PCTM_ITEM_TEXT_MODE=id" \
  "REFIT_OUTER_TRAIN=1" \
  "PERIODIC_TEST_USER_LIMIT=-1" \
  "GENERATE_REASONS=0" \
  "SAVE_MODEL_WEIGHTS=${SAVE_MODEL_WEIGHTS:-1}" \
  "EXPERIMENT_NOTE=matched full-catalogue protocol; frozen split; Ours-ID; test once after outer refit" \
  bash "$PROJECT_ROOT/run_mamba_rl.sh" "$DATASET" "$CUDA_ID" "$DATA_ROOT"

if [[ -f "$STATUS_FILE" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"completed"' "$STATUS_FILE"; then
  echo "Completed final test: $METRICS_FILE"
  if [[ ! -f "$REFERENCE_CONTRACT" ]]; then
    echo "Missing reference contract: $REFERENCE_CONTRACT" >&2
    echo "Run verify_data_contract.py verify before accepting this result." >&2
    exit 1
  fi
  "$PYTHON_BIN" "$PROJECT_ROOT/verify_data_contract.py" compare \
    "$REFERENCE_CONTRACT" "$METRICS_FILE"
else
  echo "Safely stopped before completion. Re-run the same command to resume."
fi
