# MediaTek 2026 Sequential Recommendation Project

本專案研究以 Mamba、LightGCN、LoRA 與多代理（Multi-Agent）架構進行
Sequential Recommendation。主要實驗入口是 `ver4/`；舊版本各自保留獨立程式碼、
設定與輸出，不會互相匯入。

目前的論文比較目標是使用 Spotify Research 官方
`sequential-capacity-probes` 所重建的 PCTM/eSASRec frozen splits，在相同的
full-catalogue evaluation、seen-item masking 與 Leave-One-Out protocol 下，比較
PCTM、eSASRec 與本專案的 Ours-ID/Ours-Text。

## Server 目錄

目前的 dual NVIDIA RTX PRO 6000 Blackwell server 使用下列固定結構；本實驗只使用
實體 GPU 0：

```text
/workspace/S114065701/mediatek/
├── MediaTek_2026_project-0831/
│   ├── ver1/
│   ├── ver2/
│   ├── ver3/
│   ├── ver4/
│   └── Reproduce/
├── sequential-capacity-probes/
├── .venvs/mediatek-ver4/
├── cache/
├── contracts/pctm-official-four/
└── experiments/pctm-official-four/
```

除專案本身外，其餘資料夾可依
[`ver4/A100_PCTM_WORKFLOW.md`](ver4/A100_PCTM_WORKFLOW.md) 從零建立；不需要預先
手動建立 dataset 內容。

## 版本結構

| 目錄 | 方法 | 主要入口 |
|---|---|---|
| `ver1/` | Supervised Deep Learning recommender | `bash ver1/run.sh` |
| `ver2/` | REINFORCE + LoRA Mamba | `bash ver2/run_rl.sh` |
| `ver3/` | Multi-Agent Mamba-RL | `bash ver3/run_amazons_full_rl.sh` |
| `ver4/` | Preference-Transition Multi-Agent Mamba Ranker | `bash ver4/run_mamba_rl.sh ...` |
| `Reproduce/` | 獨立 baseline reproduction | 依各子目錄 README 執行 |

`ver4` 是目前進行 PCTM/eSASRec matched-protocol 實驗的版本。詳細架構、dataset
mode、checkpoint 與 data contract 說明見 [`ver4/README.md`](ver4/README.md)。

## 從零開始跑 RTX PRO 6000 實驗

完整流程包含：

1. 以 server 既有 Miniconda 建立 dedicated Python 3.10.12 與 RTX PRO
   6000/CUDA 12.8 environment。
2. 執行 syntax check、unit test 與 synthetic end-to-end smoke test。
3. Clone 官方 `sequential-capacity-probes`。
4. 下載、處理並以 SHA-256 驗證官方 splits。
5. 復現 PCTM baseline。
6. 執行 MediaTek inner validation、outer-train refit 與一次 final test。
7. 以 checkpoint 安全中斷／resume，最後彙整多個 seeds。

請直接依照：

```text
ver4/A100_PCTM_WORKFLOW.md
```

正式執行一組實驗的介面為：

```bash
cd /workspace/S114065701/mediatek/MediaTek_2026_project-0831/ver4

bash run_pctm_protocol.sh DATASET HOURS SEED CUDA_ID
```

例如 Beauty、最多六小時、seed `25252`、GPU `0`：

```bash
bash run_pctm_protocol.sh amazon-beauty-pctm 6 25252 0
```

`HOURS=0` 表示不中途限時。若在時限後安全停止，重新執行相同命令會從
`run_checkpoint.pt` 接續。

## Dataset 與 protocol

PCTM 論文的五個官方 benchmark datasets 為：

- Beauty
- Sports and Outdoors
- Toys and Games
- MovieLens 1M
- MovieLens 20M

本次「官方四資料集」實驗使用前三個 Amazon datasets 與 MovieLens 1M。
Video Games 是本專案的 custom frozen extension，不是 PCTM 論文官方 dataset；
不得把它的結果標示成 PCTM Table 1 matched comparison。

官方 protocol 使用：

- 官方 frozen `train.csv`／`holdout.csv`；
- inner Leave-One-Out validation；
- 選定 training horizon 後，以完整 outer train 重新初始化並 refit；
- full-catalogue ranking；
- seen-item masking；
- 不使用 sampled test negatives；
- outer holdout 僅在最後評估一次。

Amazon Reviews 2023、Kaggle Amazon、S3Rec processed sequences 是不同資料來源與
population，不能只因名稱相似就共用 preprocessing 後直接比較論文數字。

## Environment

兩種 GPU 使用不同 PyTorch/CUDA wheels：

- NVIDIA A100：`ver4/requirements-a100.txt`，PyTorch CUDA 12.6。
- NVIDIA RTX PRO 6000 Blackwell：`ver4/requirements.txt`，PyTorch CUDA 12.8。

不要在同一 virtual environment 混裝 CUDA 12.6 與 CUDA 12.8 的 PyTorch、
TorchVision、TorchAudio 或 Triton。

`datasets==3.6.0` 是刻意固定的版本，因為 Amazon Reviews 2023 仍使用
Hugging Face dataset loading script；`datasets` 4.x 不再支援該載入方式。

## Evaluation 與可重現性

主要指標為 NDCG@10 與 Recall@10。正式 neural experiment 應固定相同
hyperparameters 執行多個 seeds，報告 mean ± sample standard deviation；不得挑選
最佳 seed，也不得查看 outer test 後再以同一 test set 調參。

`ver4` 會在讀取 interaction cache 前驗證 split bytes，並對 external-ID test
targets、完整 test histories 與 training catalogue 建立
`evaluation_contract_sha256`。Data contract mismatch 時，training、resume 或結果
彙整會 fail closed，不會自動修改 dataset。

完成後可用：

```bash
python ver4/summarize_protocol_results.py \
  /workspace/S114065701/mediatek/experiments/pctm-official-four \
  --require-runs 5 \
  --output /workspace/S114065701/mediatek/experiments/pctm-official-four/summary.csv
```

## Data 安全原則

- 程式可以指出 dataset 缺檔、row count 或 SHA-256 mismatch。
- Dataset 不符合 contract 時，程式會拒絕實驗。
- 未經使用者明確允許，不得修改、重切、覆寫或刪除 dataset。
- 歷史 `scores.json`、logs 與 checkpoints 內的舊絕對路徑屬於 experiment
  provenance，不應為了美觀而改寫。

## 文件索引

- [`ver4/README.md`](ver4/README.md)：`ver4` 架構、protocol 與執行參數。
- [`ver4/A100_PCTM_WORKFLOW.md`](ver4/A100_PCTM_WORKFLOW.md)：目前 RTX PRO 6000
  server 從零到 final test 的逐步指令；檔名為早期 A100 規劃所保留。
- [`VERSIONS.md`](VERSIONS.md)：各版本的隔離關係。
- [`ver4/latex/preference_transition_mamba_rl.tex`](ver4/latex/preference_transition_mamba_rl.tex)：
  方法與 benchmark report 的 LaTeX source。
