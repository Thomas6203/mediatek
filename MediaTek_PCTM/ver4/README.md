# ver4：Preference-Transition Multi-Agent Mamba Ranker

`ver4` 是從 `ver3` 分離出的獨立 training system，不會匯入 `ver1`、`ver2`
或 `ver3` 的程式碼、logs 與 outputs。共用的 datasets、Hugging Face artifacts 與
pretrained model files 存放於：

```text
/workspace/S114065701/mediatek/cache
```

目前 `ver4` 的主要用途，是在 PCTM/eSASRec matched protocol 下執行
Preference-Transition Multi-Agent Mamba experiments，並以 frozen data contract
保證不同方法使用完全相同的 test set。

## Server 路徑與 Environment

專案固定位置：

```text
/workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4
```

兩種 GPU 使用獨立的 PyTorch/CUDA requirements：

- NVIDIA A100、CUDA 12.6：`requirements-a100.txt`
- NVIDIA RTX PRO 6000 Blackwell、CUDA 12.8：`requirements.txt`

不要在同一 virtual environment 混裝兩套 CUDA wheels。A100 從環境建立、smoke
test、官方 dataset 下載處理，到 final test 與結果彙整的完整流程，請依照
[`A100_PCTM_WORKFLOW.md`](A100_PCTM_WORKFLOW.md)。

## 快速執行

一般 Amazon Reviews 2023 suite：

```bash
cd /workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4
bash run_amazons_full_rl.sh
```

`run_amazons_full_rl.sh` 的 subset 組合延續自建立 `ver4` 時的 `ver3` 設定。
如需調整，編輯檔案末端的 `run_subset` 呼叫。輸出只會寫入
`ver4/outputs_mamba_rl`。

正式 PCTM matched-protocol experiment 建議使用包裝器：

```bash
cd /workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4

bash run_pctm_protocol.sh DATASET HOURS SEED CUDA_ID
```

例如 Beauty、此次最多六小時、seed `25252`、GPU `0`：

```bash
bash run_pctm_protocol.sh amazon-beauty-pctm 6 25252 0
```

`HOURS=0` 表示不中途限時。重新執行相同命令時，若發現
`run_checkpoint.pt`，包裝器會自動 resume；若 `run_status.json` 已是
`completed`，則不會重跑 final test。

## PCTM/eSASRec Protocol Datasets

Spotify Research 官方 `sequential-capacity-probes` 會重建並驗證 PCTM/eSASRec
released splits。本專案支援以下 frozen modes：

| Dataset mode | 官方目錄 | Max history | 身分 |
|---|---|---:|---|
| `amazon-beauty-pctm` | `s3_beauty` | 50 | PCTM paper dataset |
| `amazon-sports-pctm` | `s3_sports` | 50 | PCTM paper dataset |
| `amazon-toys-pctm` | `s3_toys` | 50 | PCTM paper dataset |
| `movielens-1m-pctm` | `ml_1m` | 200 | PCTM paper dataset |
| `movielens-20m-pctm` | `ml_20m` | 200 | PCTM paper dataset |
| `amazon-video-games-pctm` | `s3_video_games` | 50 | Custom frozen extension |

Video Games 不是 PCTM 論文官方 dataset，不得標示成 Table 1 matched comparison。

直接啟動 MovieLens 20M protocol mode：

```bash
cd /workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4

bash run_mamba_rl.sh movielens-20m-pctm 0 \
  /workspace/S114065701/mediatek/sequential-capacity-probes
```

Protocol modes 與一般 MovieLens/Amazon adapters 完全分開，並固定執行下列規則：

- 在讀取 interaction cache 前，驗證官方 `train.csv`／`holdout.csv` 的 row
  counts 與 SHA-256。
- 將 ratings 視為 implicit interactions，不套用一般 `min_rating` threshold。
- 不套用 Amazon/SASRec core filtering。
- 從 outer train 建立經 cold-user、cold-item 與 already-seen target filtering 的
  inner Leave-One-Out validation split。
- 僅使用 inner validation 選擇 training horizon 與 early stopping 狀態。
- 選定 horizon 後重新初始化 model，使用完整 outer train refit。
- Outer holdout 僅在 refit 完成後做一次 final test。
- Evaluation 使用 full-catalogue ranking、seen-item masking，不使用 sampled test
  negatives。
- Protocol datasets 預設 `MAX_TRANSITIONS=0`，代表使用全部可用 transitions。

`PCTM_VERIFY_SPLIT=0` 與 `REFIT_OUTER_TRAIN=0` 只供 development fixtures 或
ablations 使用；使用這些設定產生的結果不是 matched-protocol comparison。

## Data Contract

每次 protocol run 會在載入資料後，對以下 external-ID semantics 建立
fingerprints：

- final test targets；
- 每位 test user 的完整 histories；
- training catalogue。

產生的 `evaluation_contract_sha256` 會綁定至：

- interaction cache identity；
- resumable checkpoint 與 final checkpoint；
- `data_contract.json`；
- `metrics.json`；
- optional score JSON；
- `run_status.json`。

Resume 或結果彙整時若 contract 不一致，程式會 fail closed，不會繼續 training，也
不會修改 dataset。

### Training 前驗證 split

```bash
python verify_data_contract.py verify \
  --dataset amazon-beauty-pctm \
  --data-path /workspace/S114065701/mediatek/sequential-capacity-probes \
  --cache-dir /workspace/S114065701/mediatek/cache \
  --output /workspace/S114065701/mediatek/contracts/pctm-official-four/amazon-beauty-pctm.json
```

### 比對 experiment 與 reference contract

```bash
python verify_data_contract.py compare \
  /workspace/S114065701/mediatek/contracts/pctm-official-four/amazon-beauty-pctm.json \
  /path/to/mediatek-run/metrics.json
```

只有顯示 `MATCH` 的結果才能視為使用相同 evaluation population。

### Freeze Custom Video Games Split

Video Games split 必須先由使用者審查，再以 non-overwriting sidecar 固定 bytes：

```bash
python verify_data_contract.py freeze-custom \
  --dataset amazon-video-games-pctm \
  --data-path /path/to/s3_video_games/leave_one_out \
  --output /path/to/s3_video_games/leave_one_out/data_contract.json
```

若缺少 sidecar，或既有 sidecar 與 split bytes 不符，程式會在 training 前拒絕該
dataset。工具不會覆寫既有 contract。

## Ours-ID 與 Ours-Text

預設 `PCTM_ITEM_TEXT_MODE=id` 是與 PCTM、SASRec+、eSASRec 進行較公平比較的
Ours-ID：item representation 不使用額外 title/genre side information。

Ours-Text 是 side-information ablation。以 MovieLens 20M 為例：

```bash
PCTM_ITEM_TEXT_MODE=metadata \
PCTM_METADATA_PATH=/workspace/S114065701/mediatek/sequential-capacity-probes/.external/transformer_benchmark/data/raw/ml_20m \
bash run_mamba_rl.sh movielens-20m-pctm 0 \
  /workspace/S114065701/mediatek/sequential-capacity-probes
```

報告結果時應將 Ours-ID 與 Ours-Text 分開，不得把 side information 帶來的提升全部
歸因於 sequence modelling。

## 限時執行與 Exact Resume

`run_pctm_protocol.sh` 是正式 protocol 的建議入口。若要直接操作底層 launcher，
可指定本次 wall-clock budget 與 periodic checkpoint interval：

```bash
RUN_HOURS=6 CHECKPOINT_EVERY_MINUTES=30 \
bash run_mamba_rl.sh movielens-20m-pctm 0 \
  /workspace/S114065701/mediatek/sequential-capacity-probes
```

程式會在安全 batch boundary atomically 寫入 `run_checkpoint.pt` 與
`run_status.json`。Checkpoint 內容包括：

- model、optimizer、learning-rate scheduler 與 AMP scaler state；
- Python、NumPy、PyTorch 與 CUDA RNG state；
- phase、stage、epoch 與 next-batch cursor；
- losses 與 validation history；
- best inner model 與 early-stopping counters；
- outer-refit training horizon；
- data contract。

`SIGINT` 與 `SIGTERM` 也會要求在目前 batch 完成後保存相同 checkpoint。

手動 resume：

```bash
RUN_HOURS=8 CHECKPOINT_EVERY_MINUTES=30 \
RESUME_CHECKPOINT=/path/to/the/run/run_checkpoint.pt \
bash run_mamba_rl.sh movielens-20m-pctm 0 \
  /workspace/S114065701/mediatek/sequential-capacity-probes
```

Resume 時所有 training-critical options 必須與 checkpoint 一致。只有
`RUN_HOURS`、checkpoint interval、output/status paths 與 reason-generation
options 可以改變。若 full-catalogue validation 已經開始，controlled stop 會等該次
validation 結束，因此實際時間可能略超過設定值。

## Preference Agent

每個 item representation 會以 soft assignment 對應到一組共享、可學習的
preference prototypes。GRU 讀取 preference sequence，輸出：

- current preference distribution；
- predicted next-preference distribution；
- 下一個 interaction 發生 preference transition 的 probability；
- 融入 Coordinator ranking 的 preference-based candidate scores。

Encoder 另有一個與 GRU 平行的 tiny selective-state Mamba branch。其
input-dependent decay 以 linear-time 方式保留或遺忘 latent preference evidence。
Residual gate 初始值接近零，避免一開始破壞 GRU baseline，之後可由 training 學習
提高 Mamba branch 的貢獻。

Coordinator 會依 transition probability 與 prediction confidence，為每位 user
動態調整 preference ranking weight。

Frozen Mamba item encoding 使用簡短的 preference-aware representation prefix。
Prompt tokens 不納入 mean pooling，但 prompt 會參與 cache fingerprint，避免新 prompt
錯用舊 item vectors。

Preference prototypes 與 transition model 在所有 Amazon subsets 間使用同一套
architecture。四個 auxiliary objectives 分別訓練：

- next-preference prediction；
- transition detection；
- balanced prototype utilization；
- prototype separation。

## Outputs

每個 completed run 會保存：

- training log；
- `metrics.json`；
- optional score JSON；
- `recommendations.json`；
- `preference_analysis.json`；
- `data_contract.json`（protocol datasets）；
- `run_status.json`；
- `loss.png`；
- `multi_agent_lora.pt`（`SAVE_MODEL_WEIGHTS=1`）。

`preference_analysis.json` 包含 aggregate transition/entropy statistics，以及
部分 users 的 current/predicted preference IDs。這些 IDs 是 latent components，
不是人工定義的語意 labels。

`run_amazons_full_rl.sh` 預設 `SAVE_MODEL_WEIGHTS=0`，因此 Amazon suite 只保留
logs、scores、recommendations、preference analysis、metrics 與 loss curve。正式
PCTM protocol wrapper 預設保存 model weights。

## 主要 Hyperparameters

可在執行 launcher 前以 environment variables 覆寫：

| Variable | Default | 說明 |
|---|---:|---|
| `PREFERENCE_COUNT` | 64 | preference prototypes 數量 |
| `PREFERENCE_HIDDEN` | 128 | Preference Agent hidden dimension |
| `PREFERENCE_TEMPERATURE` | 0.2 | prototype soft-assignment temperature |
| `PREFERENCE_SCORE_WEIGHT` | 0.2 | preference score 初始融合權重 |
| `PREFERENCE_COEF` | 0.2 | next-preference objective coefficient |
| `PREFERENCE_TRANSITION_COEF` | 0.1 | transition objective coefficient |
| `PREFERENCE_BALANCE_COEF` | 0.01 | prototype balance coefficient |
| `PREFERENCE_SEPARATION_COEF` | 0.01 | prototype separation coefficient |
| `ITEM_PROMPT_PREFIX` | `Preference-aware product representation: ` | Mamba item prompt |

Protocol dataset 的 `MAX_HISTORY` 與 `MAX_TRANSITIONS` 已由 launcher 提供安全
defaults。若要做 matched comparison，不建議任意覆寫。

## 多個 Seeds 的結果彙整

正式 neural experiment 建議使用至少五個固定 seeds，並報告 mean ± sample
standard deviation：

```bash
python summarize_protocol_results.py \
  /workspace/S114065701/mediatek/experiments/pctm-official-four \
  --require-runs 5 \
  --output /workspace/S114065701/mediatek/experiments/pctm-official-four/summary.csv
```

彙整器會拒絕：

- non-completed runs；
- evaluation data contracts 不一致的 runs；
- training configurations 不一致的 runs；
- completed seed 數少於 `--require-runs` 的 datasets。

不得挑選最佳 seed，也不得查看 outer test 後再用同一 test set 調整
hyperparameters。

## Verification

```bash
cd /workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4

python -m unittest discover -s tests -v
python -m compileall -q src verify_data_contract.py summarize_protocol_results.py
bash -n run_mamba_rl.sh
bash -n run_pctm_protocol.sh
bash -n run_amazons_full_rl.sh
```

Synthetic end-to-end smoke test 與正式 A100 GPU verification 指令請見
[`A100_PCTM_WORKFLOW.md`](A100_PCTM_WORKFLOW.md)。

## LaTeX Report

方法與 protocol-aware literature benchmark report 位於：

```text
latex/preference_transition_mamba_rl.tex
```

從 `latex/` 執行：

```bash
bash build_latex.sh
```

需要標準 TeX Live，以及 XeLaTeX 或 pdfLaTeX、TikZ/PGF、AMSMath、booktabs、
longtable、tabularx、listings、hyperref、microtype 與 fancyhdr。若環境缺少 TikZ/PGF，
report 無法完整編譯。
