#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../.venvs/mediatek-ver4/bin/python}"
STAGE="${1:-}"
DATASET="${2:-all}"
MAX_HOURS="${MAX_HOURS:-0}"
CHECKPOINT_INTERVAL_HOURS="${CHECKPOINT_INTERVAL_HOURS:-1}"
SHUTDOWN_BUFFER_MINUTES="${SHUTDOWN_BUFFER_MINUTES:-2}"
RESUME="${RESUME:-}"

if [[ "$STAGE" != "prepare" && "$STAGE" != "sasrec" && "$STAGE" != "llmsrec" && "$STAGE" != "verify" ]]; then
  echo "Usage: bash run_ver4_protocol.sh {prepare|sasrec|llmsrec|verify} [all|dataset_key]" >&2
  exit 2
fi

datasets=(all_beauty baby_products sports_and_outdoors toys_and_games)
if [[ "$DATASET" != "all" ]]; then
  datasets=("$DATASET")
fi

if [[ -n "$RESUME" && "$DATASET" == "all" ]]; then
  echo "RESUME requires one explicit dataset_key, not 'all'." >&2
  exit 2
fi

train_args=(
  --device "${DEVICE:-cuda:0}"
  --max-hours "$MAX_HOURS"
  --checkpoint-interval-hours "$CHECKPOINT_INTERVAL_HOURS"
  --shutdown-buffer-minutes "$SHUTDOWN_BUFFER_MINUTES"
)
if [[ -n "$RESUME" ]]; then
  train_args+=(--resume "$RESUME")
fi

cd "$ROOT"
case "$STAGE" in
  prepare)
    "$PYTHON_BIN" prepare_ver4_data.py "$DATASET"
    ;;
  sasrec)
    for dataset in "${datasets[@]}"; do
      "$PYTHON_BIN" train_sasrec_ver4.py "$dataset" "${train_args[@]}"
    done
    ;;
  llmsrec)
    for dataset in "${datasets[@]}"; do
      "$PYTHON_BIN" train_llmsrec_ver4.py "$dataset" "${train_args[@]}"
    done
    ;;
  verify)
    "$PYTHON_BIN" verify_ver4_setup.py ${ALLOW_INCOMPLETE:+--allow-incomplete}
    ;;
esac
