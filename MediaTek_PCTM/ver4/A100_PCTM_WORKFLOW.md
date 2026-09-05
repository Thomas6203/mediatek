# RTX PRO 6000 Blackwell：官方四資料集從零到 final test

> 檔名為早期 A100 規劃時保留的名稱；目前內容已依實際 server 的兩張
> NVIDIA RTX PRO 6000 Blackwell、Driver 580.173.02 與 CUDA 13.0 driver
> compatibility 更新。MediaTek 使用 PyTorch CUDA 12.8 wheel。

> **不要再使用舊版 preflight。** 舊版包含 `set -euo pipefail` 與
> `python3.10 --version`；目前 server 沒有 system `python3.10`，因此舊版必定以
> exit code 127 關閉 interactive terminal。本文件目前版本使用 `set +e` 與
> `python3 --version`，Python 3.10.12 由既有 Miniconda 在下一步建立。

這份流程使用下列固定母資料夾：

```text
/workspace/S114065701/mediatek/
├── MediaTek_2026_project-0831/ver4
├── sequential-capacity-probes
├── .venvs/mediatek-ver4
├── cache
├── contracts/pctm-official-four
└── experiments/pctm-official-four
```

「官方四資料集」在本文是 Beauty、Sports、Toys、MovieLens-1M。Video
Games 是自訂第五組，不可標成 PCTM 論文官方資料。官方下載器目前一次處理
Beauty、Sports、Toys、ML-1M、ML-20M 五組；建議保留官方程式不改，只在實驗時
選前四組。這會多下載並驗證 ML-20M，但不會把它放進本次四資料集結果。

除了專案目錄外，其餘五個路徑一開始不存在是正常狀態：第 0 步會建立
`.venvs`、`cache`、`contracts`、`experiments`，第 4 步的 `git clone` 會建立
`sequential-capacity-probes`。不需要先手動建立空的 dataset 目錄。

## 0. 登入 RTX PRO 6000 Server 後設定路徑

以下區塊可以逐段貼入 Bash。專案應先放在指定的 `PROJECT_ROOT`。

```bash
# 這一段是貼在 interactive terminal 的 preflight，因此先不要啟用 `set -e`；
# 即使某個 command 不存在，也要保留 terminal 以便看完整診斷。
set +e
set -u
set -o pipefail

ROOT=/workspace/S114065701/mediatek
PROJECT_DIR=$ROOT/MediaTek_2026_project-0831
PROJECT_ROOT=$PROJECT_DIR/ver4
PCTM_ROOT=$ROOT/sequential-capacity-probes
VENV=$ROOT/.venvs/mediatek-ver4
CACHE_DIR=$ROOT/cache
CONTRACT_ROOT=$ROOT/contracts/pctm-official-four
RESULT_ROOT=$ROOT/experiments/pctm-official-four

mkdir -p "$ROOT"
if [ ! -d "$PROJECT_ROOT" ]; then
  echo "找不到專案：$PROJECT_ROOT" >&2
  echo "請先把 MediaTek 專案上傳、複製或重新命名成：$PROJECT_DIR" >&2
else
  echo "找到專案：$PROJECT_ROOT"
fi
mkdir -p "$ROOT/.venvs" "$CACHE_DIR" "$CONTRACT_ROOT" "$RESULT_ROOT"
df -h "$ROOT"
command -v nvidia-smi || echo "找不到 nvidia-smi"
command -v python3 || echo "找不到系統 python3"
nvidia-smi
python3 --version
```

若專案檢查、`nvidia-smi` 或 `python3 --version` 失敗，terminal 仍會保持開啟。
先修正專案實際位置或 GPU environment；不要在 preflight 尚未通過時繼續。系統沒有
`python3.10` 不構成問題，第 1 步會在母資料夾內建立 Python 3.10.12。正式寫入
`.sh` script 後，才建議在檔案開頭使用 `set -euo pipefail`。

## 1. 使用既有 Miniconda 建立 MediaTek Python 3.10.12 Environment

不要變更 Ubuntu 的 system Python，也不需要 `sudo apt install`。優先使用 server
既有的 Miniconda，但建立新的 dedicated Conda prefix，避免污染 base 或其他現有
environments。官方 PCTM repo 之後仍會建立自己的 `.venv`，兩者不能混用。

```bash
CONDA_BIN=""

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  CONDA_BIN=$CONDA_BASE/bin/conda
else
  for CANDIDATE in \
    "$ROOT/miniconda3/bin/conda" \
    "/workspace/S114065701/miniconda3/bin/conda" \
    "${HOME:-}/miniconda3/bin/conda"
  do
    if [ -x "$CANDIDATE" ]; then
      CONDA_BIN=$CANDIDATE
      break
    fi
  done
fi

if [ -z "$CONDA_BIN" ]; then
  echo "找不到既有 Miniconda；請先執行下列 find 命令定位 conda：" >&2
  echo "find /workspace/S114065701 -maxdepth 5 -type f -path '*/bin/conda' -print" >&2
else
  "$CONDA_BIN" --version
  "$CONDA_BIN" info --base

  if [ ! -x "$VENV/bin/python" ]; then
    "$CONDA_BIN" create --yes --prefix "$VENV" python=3.10.12 pip
  fi

  "$VENV/bin/python" --version
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

  export CUDA_VISIBLE_DEVICES=0
  "$VENV/bin/python" -c "import sys, torch; print('python=', sys.version); print('torch=', torch.__version__); print('torch CUDA=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('visible GPU count=', torch.cuda.device_count()); print('visible GPU 0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(); assert torch.version.cuda == '12.8'; assert torch.cuda.device_count() == 1"
fi
```

`nvidia-smi` 顯示的 `CUDA Version: 13.0` 是 Driver 580.173.02 可支援的最高 CUDA
runtime，並不要求 PyTorch wheel 也必須是 CUDA 13.0。此專案在 Blackwell 上使用
`torch==2.7.1+cu128`。本實驗只授權使用實體 GPU 0，因此設定
`CUDA_VISIBLE_DEVICES=0` 後，PyTorch 顯示一張可見的 RTX PRO 6000 才是正確結果。
若驗證沒有顯示 `torch CUDA=12.8`、一張可見 GPU 或 RTX PRO 6000，先停止，不要在
同一 environment 混裝其他 CUDA wheels。

## 2. 專案靜態測試與 unit test

```bash
cd "$PROJECT_ROOT"
export PYTHON_BIN="$VENV/bin/python"
export CACHE_DIR

bash -n run_mamba_rl.sh
bash -n run_pctm_protocol.sh
bash -n run_amazons_full_rl.sh
"$PYTHON_BIN" -m compileall -q src verify_data_contract.py summarize_protocol_results.py
"$PYTHON_BIN" -m unittest discover -s tests -v
```

任何一項失敗都先修好，不要直接下載資料或跑正式實驗。

## 3. Synthetic end-to-end smoke test

這一步不用正式 dataset，也不下載 Mamba 權重；目的是確認 GPU、training、validation、
final test、輸出檔與 checkpoint 路徑都能走完。

```bash
cd "$PROJECT_ROOT"
SMOKE_DIR=$ROOT/experiments/smoke/synthetic_$(date +%Y%m%d_%H%M%S)

PYTHON_BIN="$VENV/bin/python" \
CACHE_DIR="$CACHE_DIR" \
OUTPUT_RUN_DIR="$SMOKE_DIR" \
SPECIALIST_EPOCHS=1 \
COORDINATOR_EPOCHS=1 \
JOINT_EPOCHS=1 \
BATCH_SIZE=8 \
EVAL_BATCH_SIZE=8 \
MAX_TRANSITIONS=32 \
VALIDATE_EVERY_STEPS=1 \
CANDIDATES=4 \
DIM=16 \
LORA_RANK=4 \
PREFERENCE_COUNT=4 \
PREFERENCE_HIDDEN=16 \
MAX_HISTORY=5 \
SHORT_WINDOW=3 \
FUTURE_HORIZON=1 \
USE_GRAPH_EMBEDDINGS=0 \
GENERATE_REASONS=0 \
SAVE_MODEL_WEIGHTS=1 \
bash run_mamba_rl.sh synthetic 0 --skip-mamba

grep -q '"status": "completed"' "$SMOKE_DIR/run_status.json"
test -s "$SMOKE_DIR/metrics.json"
test -s "$SMOKE_DIR/multi_agent_lora.pt"
echo "SMOKE TEST PASSED: $SMOKE_DIR"
```

## 4. 安裝官方 PCTM reproduction

官方 repo 鎖定 Python 3.10.12 與自己的完整相依套件，所以由它自己的 `make
setup` 建立 `$PCTM_ROOT/.venv`。

```bash
cd "$ROOT"
if [ ! -d "$PCTM_ROOT/.git" ]; then
  git clone https://github.com/spotify-research/sequential-capacity-probes.git "$PCTM_ROOT"
fi

git -C "$PCTM_ROOT" remote get-url origin
git -C "$PCTM_ROOT" rev-parse HEAD
cd "$PCTM_ROOT"
make setup PYTHON="$VENV/bin/python"
make dependency
make test
```

不要對既有 checkout 無條件 `git pull`；先記錄 commit，避免同一份實驗中途換版本。

## 5. 下載、處理並驗證官方資料

`make data-public` 會從 S3Rec/GroupLens 公開來源下載，呼叫官方 pin 的 eSASRec
builder，再驗證 raw 與 processed hashes。不要以 Amazon Reviews 2023、Kaggle
Amazon 或自行重切的 CSV 代替。

```bash
cd "$PCTM_ROOT"
make data-public
make verify-data
```

本次使用的四個 split 應位於：

```text
data/processed/s3_beauty/leave_one_out/{train.csv,holdout.csv}
data/processed/s3_sports/leave_one_out/{train.csv,holdout.csv}
data/processed/s3_toys/leave_one_out/{train.csv,holdout.csv}
data/processed/ml_1m/leave_one_out/{train.csv,holdout.csv}
```

再由 MediaTek loader 做第二道獨立驗證，並保存 exact test-set contracts：

```bash
cd "$PROJECT_ROOT"
for DATASET in \
  amazon-beauty-pctm \
  amazon-sports-pctm \
  amazon-toys-pctm \
  movielens-1m-pctm
do
  "$VENV/bin/python" verify_data_contract.py verify \
    --dataset "$DATASET" \
    --data-path "$PCTM_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --output "$CONTRACT_ROOT/$DATASET.json"
done
```

預期資料列數（不含 CSV header）與 SHA-256：

| Dataset | File | Rows | SHA-256 |
|---|---|---:|---|
| Beauty | train | 176139 | `956357e585c8c4ddf50ff4dd4331d7e046c1b42b8eaa13cc21c8b9dccee89795` |
| Beauty | holdout | 22311 | `38dbedf2dedc6951be1d8a3d4f6b2d93aa7129ce5d4f0fcf0d17bee0e315f43a` |
| Sports | train | 260739 | `e90df6401241237709a9386d91fd9c2f24eb132665e99e74d6f5af5e12b8b9a6` |
| Sports | holdout | 35539 | `2fb358ca824c8df1e655275ba13d106b1c1b2dc1bb131810cc8519919abc1139` |
| Toys | train | 148185 | `ca9191ea48da1bc4b9ed812354ae3b58a7b93f2417d4794142c9c9dc0230b356` |
| Toys | holdout | 19365 | `ff18781f3e2784b1d4438bdd5a54be7fb75aa0f2d7d08f552271f135bf1ee7b8` |
| ML-1M | train | 994169 | `74afd9fcafba3e694195fe26c8ea486ec12dd16f22f0aaaa88cbcd66c960176a` |
| ML-1M | holdout | 6038 | `83bf1681a6dc94a5e567fcca9bd5de44618ff4097f9a2172df840bdbcb15f478` |

任一筆不符時，loader 只會報錯並拒絕實驗，不會修改 dataset。不要關閉
`PCTM_VERIFY_SPLIT` 來繞過錯誤。

## 6. 先復現官方 PCTM baseline

PCTM 是 deterministic，可用 CPU；四組應精確重現論文四位小數。

```bash
cd "$PCTM_ROOT"
PCTM_EVAL_JOBS=8 .venv/bin/python scripts/run_all.py \
  --models pctm \
  --datasets beauty,sports,toys,ml1m \
  --device cpu \
  --resume
```

預期 PCTM NDCG@10：Beauty `0.0635`、Sports `0.0368`、Toys `0.0738`、
ML-1M `0.1815`。若不一致，先停在 baseline/data/environment 調查，不要開始宣稱
與論文做 matched comparison。官方 `make verify` 是完整 35-cell Table 1 的 verifier；
只跑 PCTM 四組時不要把「缺少其他模型結果」誤判為本次四組失敗。

## 7. 正式 MediaTek 訓練、resume 與 final test

每次執行只需要輸入四個值：dataset、此次最多幾小時、seed、GPU。`HOURS=0`
表示不中途限時。例：Beauty 跑六小時：

```bash
cd "$PROJECT_ROOT"
PYTHON_BIN="$VENV/bin/python" \
CACHE_DIR="$CACHE_DIR" \
CONTRACT_ROOT="$CONTRACT_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
CHECKPOINT_EVERY_MINUTES=30 \
bash run_pctm_protocol.sh amazon-beauty-pctm 6 25252 0
```

六小時後若尚未完成，會在安全 batch boundary 寫入 `run_checkpoint.pt` 和
`run_status.json`。之後原封不動重貼同一命令，它會自動 resume；此次又有新的六小時
額度。受目前 batch 或已開始的 full-catalogue validation 影響，實際停止可能稍晚。

四個 dataset 名稱：

```text
amazon-beauty-pctm
amazon-sports-pctm
amazon-toys-pctm
movielens-1m-pctm
```

完成一組後，包裝器會看到 `status=completed`，自動把該 run 的 `metrics.json`
與第 5 步的 contract 比較，必須印出 `MATCH`。正式 protocol 固定：

- 官方 frozen train/holdout；
- inner validation 做 early stopping；
- 選定 horizon 後重新初始化，使用完整 outer train refit；
- Ours-ID、full catalogue、seen-item masking；
- periodic test 關閉；
- outer holdout 只在 refit 完成後做一次 final test。

因此「test」不是另一支要手動執行的程式。只有 `run_status.json` 是
`completed` 且已產生 `metrics.json`，才代表 final test 做完；`interrupted` 只代表可
resume，不能拿來報結果。

## 8. 重複 seeds

先用 seed `25252` 把四組各完成一次並檢查 pipeline；正式報告建議固定設定做五個
seeds：

```text
25252 25253 25254 25255 25256
```

例如 Sports 的第二個 seed：

```bash
cd "$PROJECT_ROOT"
PYTHON_BIN="$VENV/bin/python" \
CACHE_DIR="$CACHE_DIR" \
CONTRACT_ROOT="$CONTRACT_ROOT" \
RESULT_ROOT="$RESULT_ROOT" \
bash run_pctm_protocol.sh amazon-sports-pctm 6 25253 0
```

每一個 `(dataset, seed)` 都有獨立目錄，不會覆寫別的 seed。

## 9. Test 後的必要工作

Test 後不能直接挑最好 seed 報告，也不能看了 test 再改超參數並把同一 test 當公正
結果。依序做：

1. 確認每個 `run_status.json` 都是 `completed`，並保存 log、checkpoint、
   `metrics.json`、score JSON、recommendations、contract 與環境版本。
2. 再跑 contract compare；test targets、histories、training catalogue 任一 fingerprint
   不同就不合併。
3. 固定相同訓練設定彙整所有 seeds；工具會拒絕混合 training config 或混合 contract：

   ```bash
   cd "$PROJECT_ROOT"
   "$VENV/bin/python" summarize_protocol_results.py "$RESULT_ROOT" \
     --require-runs 5 \
     --output "$RESULT_ROOT/summary.csv"
   ```

4. 報 `NDCG@10 mean ± sample std`、`Recall@10 mean ± sample std`、相對 PCTM 的
   NDCG 差值；同時標明 MediaTek 是多-seed neural result，PCTM 是 deterministic。
5. 若要調參，只能回 inner validation；先定義新 experiment/seed，再做一次新的 final
   test。不要用 outer test 選模型。
6. 記錄 MediaTek commit、官方 PCTM commit、`pip freeze`、`nvidia-smi`、啟動命令與
   wall-clock/peak VRAM，然後把 artifacts 備份到 server 以外的位置。

只有完成以上步驟，才適合把結果放進 PCTM vs Ours 的正式比較表。
