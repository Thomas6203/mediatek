#!/usr/bin/env bash
# Run from anywhere: bash /workspace/P78123011/MediaTek_2026_project/run_amazons_full_rl.sh
# Every subset uses the same model architecture. Training-only hyperparameters
# scale with catalog size; every validation and test evaluates all eligible users.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/dataspace/P78123011/miniconda3/envs/py31014/bin/python}"
CACHE_DIR="${CACHE_DIR:-$WORKSPACE_ROOT/cache}"
# 手動設定要使用的實體 GPU："0"、"1"、"0,1" 或 "1,0"
# 使用兩張卡時，第一張供主模型使用，第二張供 graph/preference 模型使用。
GPU_IDS="0"

# True: use SASRec's one-pass 5-core user/item filtering before leave-two-out.
# False: preserve ver4's original user-only filtering.
USE_SASREC_FILTERING="${USE_SASREC_FILTERING:-True}"
case "$USE_SASREC_FILTERING" in
  True) SASREC_FILTER_OPTION="--sasrec-filtering" ;;
  False) SASREC_FILTER_OPTION="--no-sasrec-filtering" ;;
  *)
    echo "USE_SASREC_FILTERING must be True or False, got: $USE_SASREC_FILTERING" >&2
    exit 2
    ;;
esac

# ================= 手動調整：訓練模式與三階段超參數 =================
# 1: LoRA；0: full-rank finetuning（推薦模型內的 dense adaptation）
ENABLE_LORA=1
if [[ "$ENABLE_LORA" == "1" ]]; then
  OUTPUT_GROUP="amazons_lora"
else
  OUTPUT_GROUP="amazons_full"
fi
OUTPUT_ROOT="$PROJECT_ROOT/outputs_mamba_rl/$OUTPUT_GROUP"
SPECIALISTS_EPOCH=4
COORDINATOR_EPOCH=4
JOINT_EPOCH=10
SPECIALISTS_LR=1e-4
COORDINATOR_LR=1e-4
JOINT_LR=5e-5

COMMON_ENV=(
  "PYTHON_BIN=$PYTHON_BIN"
  "CACHE_DIR=$CACHE_DIR"
  "ENABLE_LORA=$ENABLE_LORA"
  "SPECIALIST_EPOCHS=$SPECIALISTS_EPOCH"
  "COORDINATOR_EPOCHS=$COORDINATOR_EPOCH"
  "JOINT_EPOCHS=$JOINT_EPOCH"
  "SPECIALIST_LR=$SPECIALISTS_LR"
  "COORDINATOR_LR=$COORDINATOR_LR"
  "JOINT_LR=$JOINT_LR"
  "BATCH_SIZE=128"
  "EVAL_BATCH_SIZE=64"
  "MONITOR_METRIC=ndcg@10"
  "EARLY_STOPPING_PATIENCE=6"
  "LR_PATIENCE=2"
  "DIM=128"
  "LORA_RANK=16"
  "LORA_ALPHA=32.0"
  "LORA_DROPOUT=0.05"
  "SHORT_WINDOW=10"
  "PREFERENCE_COUNT=64"
  "PREFERENCE_HIDDEN=128"
  "PREFERENCE_TEMPERATURE=0.2"
  "PREFERENCE_SCORE_WEIGHT=0.2"
  "PREFERENCE_COEF=0.2"
  "PREFERENCE_TRANSITION_COEF=0.1"
  "PREFERENCE_BALANCE_COEF=0.01"
  "PREFERENCE_SEPARATION_COEF=0.01"
  "FUTURE_HORIZON=3"
  "FUTURE_DECAY=0.5"
  "HARD_NEGATIVE_POOL_MULTIPLIER=4"
  "PREFERENCE_CONTRASTIVE_COEF=0.05"
  "USE_GRAPH_EMBEDDINGS=1"
  "MAX_HISTORY=100"
  "MAMBA_ENCODE_BATCH_SIZE=32"
  "MAMBA_MAX_TOKENS=32"
  "ITEM_PROMPT_PREFIX=Preference-aware product representation: "
  "GENERATE_REASONS=0"
  "SAVE_MODEL_WEIGHTS=0"
)

run_subset() {
  local subset_name="$1"
  local dataset="$2"
  local validate_every_steps="$3"
  local max_transitions="$4"
  local candidates="$5"
  local popularity_alpha="$6"
  local transition_beta="$7"
  local timestamp run_dir

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  run_dir="$OUTPUT_ROOT/$subset_name/rl_$timestamp"

  env "${COMMON_ENV[@]}" bash "$PROJECT_ROOT/run_mamba_rl.sh" "$dataset" "$GPU_IDS" \
    --output-run-dir "$run_dir" \
    --score-file "$run_dir/${subset_name}_scores.json" \
    --validate-every-steps "$validate_every_steps" \
    --validation-user-limit 0 \
    --periodic-test-user-limit 0 \
    "$SASREC_FILTER_OPTION" \
    --max-transitions "$max_transitions" \
    --candidates "$candidates" \
    --specialist-epochs "$SPECIALISTS_EPOCH" \
    --coordinator-epochs "$COORDINATOR_EPOCH" \
    --joint-epochs "$JOINT_EPOCH" \
    --specialist-lr "$SPECIALISTS_LR" \
    --coordinator-lr "$COORDINATOR_LR" \
    --joint-lr "$JOINT_LR" \
    --popularity-alpha "$popularity_alpha" \
    --transition-beta "$transition_beta" \
    --experiment-note "$OUTPUT_GROUP subset=$subset_name; sasrec_filtering=$USE_SASREC_FILTERING; common architecture; periodic validation/test"
}

# One command per requested subset. Set REPEATS=5 to repeat the whole suite five times.
REPEATS="${REPEATS:-1}"
for ((run_number = 1; run_number <= REPEATS; run_number++)); do
  # name dataset validation_steps max_samples candidates popularity transition
  # run_subset "Full_Beauty" "amazon-all-beauty" 250 500000 64 -0.25 4.0
  # run_subset "Baby_Products" "amazon:Baby_Products" 6000 1500000 192 0.30 0.5
  # run_subset "Sports_and_Outdoors" "amazon-sports-and-outdoors" 8000 1000000 256 0.35 0.5
  # run_subset "Books" "amazon-books" 12000 2000000 256 0.35 0.5
  run_subset "Toys_and_Games" "amazon-toys" 6000 1500000 192 0.30 0.5
  run_subset "Video_Games" "amazon-video-games" 4000 1000000 128 0.20 0.5
  run_subset "Clothing_Shoes_and_Jewelry" "amazon-clothing-shoes-and-jewelry" 12000 2000000 256 0.35 0.5
  run_subset "Beauty_and_Personal_Care" "amazon:Beauty_and_Personal_Care" 12000 2000000 256 0.35 0.5
done
