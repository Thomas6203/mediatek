# 在 MediaTek ver4 實驗協定下重現 SIGMA

本目錄重現複製之 SIGMA 程式庫在 commit
`a3f63751503eca869338fc9c8b24a05fa128c6ef` 的模型架構，並以不可變更的
MediaTek `MediaTek_2026_0902/ver4` 實驗協定取代 RecBole 原有的資料切分與評估方式。

## 資料處理方式

針對每個 Amazon Reviews 2023 subset：

1. 從完整的 `raw_review_*` configuration 讀取 `user_id`、`parent_asin`
   與 `timestamp`。
2. 使用原始互動資料統計使用者與品項的互動次數。
3. 套用 ver4/SASRec 的單次篩選：保留原始互動數至少為 5 的使用者，並只保留
   原始互動數至少為 5 的品項。此方式不是 iterative k-core。
4. 依照 `timestamp` 對每位保留使用者的互動進行 stable sort。
5. 依使用者字串排序建立使用者 ID mapping；依品項第一次出現的順序建立品項
   ID mapping。
6. 篩選後歷史長度至少為 3 的使用者，最後一筆互動作為 test target、倒數第二筆
   作為 validation target，其餘作為 train。篩選後歷史少於 3 筆者只保留在
   training set，不納入 evaluation。

四個稽核欄位的定義如下：

- `num_users`：通過單次篩選後保留的全部使用者數。
- `evaluation_users`：具有 validation/test target 的使用者數。
- `num_items`：通過單次篩選後出現的全部品項數。
- `train_interactions`：所有使用者 training history 長度的總和。

產生的 manifest 亦會對所有整數化 train sequence 與 validation/test target
記錄 SHA-256 fingerprint，防止實驗在未被察覺的情況下使用不同的資料切分。

| Dataset | `num_users` | `evaluation_users` | `num_items` | `train_interactions` | 與已完成 ver4 run 的關係 |
|---|---:|---:|---:|---:|---|
| All_Beauty | 1,603 | 1,457 | 4,090 | 8,445 | 完全一致 |
| Baby_Products | 184,834 | 184,584 | 77,324 | 1,121,805 | 完全一致 |
| Sports_and_Outdoors | 631,803 | 625,563 | 401,572 | 3,678,023 | 完全一致 |
| Toys_and_Games（完整 subset） | 568,262 | 564,666 | 333,160 | 3,734,374 | 沒有有效的已完成參考 run |

現有 ver4 launcher 將經過 keyword 篩選的 `amazon-toys` 子群標記為
`Toys_and_Games`。該已完成 artifact 只有 59 位使用者、43 位 evaluation
users、149 個品項與 127 筆 training interactions；它不是完整的 Amazon Reviews
2023 `raw_review_Toys_and_Games` subset，因此不得與上表最後一列比較。
完整 subset 的 ver4 dataset argument 應為 `amazon:Toys_and_Games`。

## 實驗方式

與已完成 ver4 實驗共用的設定：

- random seed 為 25252。
- 以 coverage-first reservoir sampling 建立 training prefix，並於每個 epoch
  重新取樣。
- 四個 dataset 的 transition cap 依序為
  500,000／1,500,000／1,000,000／1,500,000。
- training batch size 為 128，evaluation batch size 為 64，
  gradient clipping 為 1.0。
- 最多執行 18 個 epochs，即 ver4 的 4 個 specialist epochs、4 個 coordinator
  epochs 與 10 個 joint epochs。
- early stopping patience 為 6，依 validation NDCG@10 選擇最佳模型。
- 對所有符合資格的 validation 與 test users 執行 full-catalog evaluation，
  並遮蔽 training history 中已看過的品項。
- 回報 Recall/Hit Rate@5、Recall/Hit Rate@10、NDCG@5 與 NDCG@10；
  SIGMA 另外回報 MRR@10。

為忠實重現方法而保留的 SIGMA 專屬設定：

- PF-Mamba、DS Gate、FE-GRU、可訓練的三路 mixing、projection 與
  point-wise feed-forward layer。
- 1 個 SIGMA layer、hidden size 64、sequence length 50，以及 Mamba
  `d_state=32`、`d_conv=4`、`expand=2`。
- full-catalog cross-entropy、Adam optimizer 與 learning rate 0.001。
- 採用論文所述的 Amazon dropout 0.3。

載入資料後刻意不使用 RecBole 的資料處理流程，因為再次套用其 filter 或
splitter 會破壞已驗證的 ver4 資料切分。

## 執行方式

```bash
bash SIGMA/run_amazon2023_ver4_protocol.sh
```

執行單一 subset：

```bash
PYTHONPATH=SIGMA/vendor:SIGMA CUDA_VISIBLE_DEVICES=0 \
  .venvs/mediatek-ver4/bin/python SIGMA/run_ver4_protocol.py all_beauty \
  --device cuda:0 --max-transitions 500000 --save-model-weights
```

Manifest 位於 `SIGMA/data/ver4_protocol`。每個完成的 run 會將
`train.log` 與 `metrics.json` 寫入 `SIGMA/outputs_ver4_protocol`。

## 此處的 main table 是什麼

ver4 的資料處理程式中沒有名為 `main-table` 的資料物件或處理階段。
SIGMA 論文的 Table 1 是 dataset statistics table；Table 2 才是整體實驗結果的
main table。

本次重現中的 `main_table.csv` 是最後的跨模型實驗結果表。只有在 raw subset、
filter、split、catalog、eligible users 與 evaluator 全部一致時，該列結果才具有
可比較性；前述四個稽核欄位是寫入 main table 前的最低必要檢查。
