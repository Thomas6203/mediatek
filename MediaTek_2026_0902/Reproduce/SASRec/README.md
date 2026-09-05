# SASRec Reproduction

這個資料夾提供 PyTorch SASRec baseline，模型實作獨立放在此處，但直接復用
`ver4/src/data.py` 與 `ver4/src/data_mamba_rl.py`，因此 Amazon Reviews 2023
載入、subset mapping、cache、時間排序及 train/validation/test split 均與 ver4
一致。

## 執行

在任何目錄均可執行：

```bash
bash /workspace/P78123011/MediaTek_2026_project/Reproduce/SASRec/run.sh
```

預設使用實體 GPU 0，並依序執行與 ver4 相同的八個 Amazon subsets。

指定單張 GPU：

```bash
bash run.sh 1
```

指定多張 GPU（自動使用 PyTorch `DataParallel`）：

```bash
bash run.sh 0,1
```

也可以透過環境變數設定：

```bash
GPU_IDS=0,1 bash run.sh
```

只跑部分 subsets：

```bash
SUBSETS=Full_Beauty,Video_Games bash run.sh 0
```

可用的 subset 名稱為：

- `Full_Beauty`
- `Beauty_and_Personal_Care`
- `Baby_Products`
- `Sports_and_Outdoors`
- `Books`
- `Toys_and_Games`
- `Video_Games`
- `Clothing_Shoes_and_Jewelry`

## 資料處理

`USE_SASREC_FILTERING=True`（預設）使用 SASRec 原始 preprocessing：

1. 在未過濾互動上計算 user/item frequency。
2. 一次性保留互動數至少 5 的 user 及 item（不是 iterative k-core）。
3. 依 timestamp 排序。
4. 最後兩筆分別作為 validation/test。
5. 過濾後少於三筆的 user 僅留在 training。

若要使用 ver4 原本只有 user threshold 的方法：

```bash
USE_SASREC_FILTERING=False bash run.sh 0
```

兩種模式會使用不同的 cache key。

## 常用超參數

所有設定皆可在呼叫 `run.sh` 時以環境變數覆寫：

```bash
EPOCHS=100 \
BATCH_SIZE=256 \
EVAL_BATCH_SIZE=64 \
LEARNING_RATE=1e-3 \
MAXLEN=100 \
HIDDEN_UNITS=128 \
NUM_BLOCKS=2 \
NUM_HEADS=1 \
DROPOUT_RATE=0.2 \
EARLY_STOPPING_PATIENCE=20 \
bash run.sh 0,1
```

`MAX_BATCHES_PER_EPOCH` 可用於快速 smoke run；`MAX_EVENTS` 可限制原始資料量。
正式比較時兩者都應維持預設的 `0`/空值。

結果寫入 `outputs/<subset>/sasrec_<timestamp>_r<repeat>/`：

- `train_*.log`
- `config.json`
- `metrics.json`

程式沒有模型權重輸出選項，不會在磁碟保存任何模型權重；最佳 epoch 僅暫存在
記憶體中供最終 test 使用。

## 驗證

```bash
bash -n run.sh
PYTHONPATH=. python -m unittest discover -s tests
```
