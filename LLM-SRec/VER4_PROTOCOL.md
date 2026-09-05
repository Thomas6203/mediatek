# 在 MediaTek ver4 實驗協定下重現 LLM-SRec

此目錄固定使用 LLM-SRec upstream commit
`b81019ca655fb759cee895924b8b6c7cc0f0cce9`，並以獨立 runner 取代 upstream 的
Amazon `5core_last_out_*` split、10,000-user sampling 與 100-candidate evaluation。

## 資料契約

`prepare_ver4_data.py` 會先驗證 `SIGMA/data/ver4_protocol` 中的 canonical artifact
與 SHA-256 fingerprint，再匯出 LLM-SRec 所需的 1-based ID 檔案與 item metadata。
資料處理固定為：raw review、user/item 原始互動數各至少 5 的單次篩選、依
`timestamp` stable sort，以及 chronological leave-two-out。篩選後少於 3 筆的
使用者只保留在 training set。

| Dataset key | Amazon Reviews 2023 config | `num_users` | `evaluation_users` | `num_items` | `train_interactions` |
|---|---|---:|---:|---:|---:|
| `all_beauty` | `raw_review_All_Beauty` | 1,603 | 1,457 | 4,090 | 8,445 |
| `baby_products` | `raw_review_Baby_Products` | 184,834 | 184,584 | 77,324 | 1,121,805 |
| `sports_and_outdoors` | `raw_review_Sports_and_Outdoors` | 631,803 | 625,563 | 401,572 | 3,678,023 |
| `toys_and_games` | `raw_review_Toys_and_Games` | 568,262 | 564,666 | 333,160 | 3,734,374 |

`toys_and_games` 是完整 subset，不是既有 ver4 launcher 中以 metadata keyword
切出的 `amazon-toys` 小型子群。

## 實驗契約

- seed：25252。
- training/evaluation batch size：128／64。
- sequence length：50。
- 每個 epoch 重新執行 coverage-first reservoir sampling。
- transition cap：500,000／1,500,000／1,000,000／1,500,000。
- 最多 18 epochs，gradient clipping 1.0。
- 依 validation NDCG@10 選擇 checkpoint，early stopping patience 為 6。
- validation/test 使用所有 eligible users、full catalog、seen-item masking。
- 回報 Recall/Hit Rate@5、Recall/Hit Rate@10、NDCG@5 與 NDCG@10。

LLM-SRec 專屬方法仍保留 released SASRec teacher、LLaMA-3.2-3B-Instruct、
item/user projection、recommendation loss 與 representation matching loss。SASRec
teacher 與 LLM distillation 是兩個必要階段，兩者各自保存依 validation NDCG@10
選出的 checkpoint。

## 準備與檢查

```bash
cd /workspace/S114065701/mediatek/LLM-SRec
bash run_ver4_protocol.sh prepare all
bash run_ver4_protocol.sh verify
```

只檢查 runner 參數而不載入模型：

```bash
../.venvs/mediatek-ver4/bin/python train_sasrec_ver4.py all_beauty --dry-run
../.venvs/mediatek-ver4/bin/python train_llmsrec_ver4.py all_beauty --dry-run
```

## 後續實驗命令

以下命令會實際執行 neural network；資料準備階段不應執行：

```bash
DEVICE=cuda:0 bash run_ver4_protocol.sh sasrec all_beauty
DEVICE=cuda:0 bash run_ver4_protocol.sh llmsrec all_beauty
```

### 限時執行與續跑

`MAX_HOURS` 是本次程序可使用的訓練時數；設為 `0` 代表不限制。以下範例讓
SASRec stage 最多執行 6 小時，並使用預設的每 1 小時 checkpoint：

```bash
MAX_HOURS=6 DEVICE=cuda:0 bash run_ver4_protocol.sh sasrec all_beauty
```

可另外設定 checkpoint 間隔與預留安全關閉時間：

```bash
MAX_HOURS=6 CHECKPOINT_INTERVAL_HOURS=1 SHUTDOWN_BUFFER_MINUTES=2 \
  DEVICE=cuda:0 bash run_ver4_protocol.sh llmsrec all_beauty
```

進度 checkpoint 位於該次 run 的 `checkpoints/`，檔名例如
`checkpoint_hour_0001.pt`。到達時間上限時，runner 會在完成目前 optimizer step
後以 atomic write（原子寫入）保存模型可訓練參數、optimizer、scheduler（LLM-SRec
stage）、epoch/batch 位置、early-stopping 狀態與 RNG state，並在 `status.json`
記錄可續跑的 checkpoint。validation 與 full-catalog encoding 不會在中途截斷；
若整點落在這些操作內，checkpoint 會延後到下一個安全邊界保存。

以同一 stage 與同一 dataset 指定 `RESUME` 即可續跑：

```bash
RESUME=/absolute/path/to/checkpoint_hour_0006.pt MAX_HOURS=4 \
  DEVICE=cuda:0 bash run_ver4_protocol.sh sasrec all_beauty
```

續跑時的 `MAX_HOURS=4` 代表本次再執行最多 4 小時，不是累計上限；checkpoint
檔名的 hour 編號則沿用累計訓練時間。續跑必須明確指定單一 dataset，不能搭配
`all`。兩個 stage 皆支援相同環境變數：

- `MAX_HOURS`：本次時數上限，預設 `0`。
- `CHECKPOINT_INTERVAL_HOURS`：進度 checkpoint 間隔，預設 `1`。
- `SHUTDOWN_BUFFER_MINUTES`：時間上限前的安全保存預留，預設 `2`。
- `RESUME`：要載入的進度 checkpoint 絕對或相對路徑。

完整四個 subset 將 `all_beauty` 改為 `all`。LLM stage 開始前必須具備：

1. `SeqRec/data_<dataset>/ver4_manifest.json` 與非 placeholder metadata。
2. `SeqRec/sasrec/<dataset>/` 中恰好一個 `.pth` teacher checkpoint。
3. Hugging Face 對 `meta-llama/Llama-3.2-3B-Instruct` 的授權與本機 token。

`verify_ver4_setup.py` 只做檔案、統計與 fingerprint 檢查，不會載入或執行任何
LLM、SASRec 或其他 neural network。
