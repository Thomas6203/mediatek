# 暫停檢查點（checkpoint）— 2026-09-04 01:36:22 CST

狀態：**已暫停**。目前沒有 `run_ver4_protocol.py` process 正在執行；在收到明確
的恢復指示前，不應啟動新的實驗。

## 可恢復的狀態

四個不可變更的資料 artifact 與 manifest 均已完成：

| Dataset | `num_users` | `evaluation_users` | `num_items` | `train_interactions` | split SHA-256 |
|---|---:|---:|---:|---:|---|
| all_beauty | 1,603 | 1,457 | 4,090 | 8,445 | `0a0349ae56194bbfa44cee2d21feee3a0983ef83ea8bbc06147033295ac14ad4` |
| baby_products | 184,834 | 184,584 | 77,324 | 1,121,805 | `64e1d90529a440fe0f3754577c94ea6f1676dae52ebfbb9e10c9adf236df6ca0` |
| sports_and_outdoors | 631,803 | 625,563 | 401,572 | 3,678,023 | `0871495b5f1c9ed60f2a3485b38a0bbce98d7371027a4eff05ac40dbf9eb4cc9` |
| toys_and_games | 568,262 | 564,666 | 333,160 | 3,734,374 | `2c046271d2f3366438f4ac30e74cbb1020554c42e10df30cca2bc88ded1d4810` |

目前暫停的 source code 已通過 syntax check，並已準備好執行下一次 clean run：

- 使用完整 FP32 訓練 SIGMA，以符合原始 RecBole default，並避免已觀察到的
  large-catalog mixed-precision instability。
- 採用 cloned repository 的 cross-entropy semantics：將 padding embedding
  納入 training negative classes，並將 positive target ID 位移 1。
- Evaluation 排除 padding、遮蔽使用者看過的真實品項，並檢查 target score
  是否為 finite value。

Source code 的 SHA-256：

- `run_ver4_protocol.py`：`f250ac1f30ea4605429053114bcb467cd290a3743a98c132621a74be7b6caf14`
- `model/sigma_ver4_protocol.py`：`66d3e8cc4b12b6093b75e0755c74f710f8b8849b135e1fda8c91ca812417ace5`
- `prepare_ver4_data.py`：`63c7a50fb14e4fe9fc7266cfb31f35bb8e7a37c8e096bb7cc8587c59f53b36bf`

## 模型 checkpoint 的限制

被中止的 Baby／Sports／Toys processes 沒有可續訓的 `.pt` model checkpoint。
這些 processes 當時以 `--no-save-model-weights`（當時的 default）啟動，
SIGMA 僅將最佳 state 保存在 process memory 中。Processes 已經停止，因此無法
在事後復原該 state。

先前 diagnostic runs 完成的 `metrics.json` 仍保留在磁碟中，但它們產生於最後
一次 clone-faithful cross-entropy 修正之前，不得作為最終 main table results：

- `outputs_ver4_protocol/all_beauty/20260904_012545/metrics.json`
- `outputs_ver4_protocol/baby_products/20260904_010718/metrics.json`

為了 audit，已刻意保留被中止的 logs。這些 runs 沒有 `metrics.json`，因此
main table builder 不會將它們納入結果。

## 恢復位置

恢復時應從 deterministic clean runs（random seed 25252）重新開始，並加入
`--save-model-weights`，使每個成功完成的 run 都會寫入
`best_validation.pt`。資料 artifacts 不需要重新建立。

```bash
PYTHONPATH=SIGMA/vendor:SIGMA CUDA_VISIBLE_DEVICES=0 \
  .venvs/mediatek-ver4/bin/python SIGMA/run_ver4_protocol.py all_beauty \
  --device cuda:0 --max-transitions 500000 --save-model-weights
```

其餘 datasets 使用相同命令：`baby_products` 的 transition cap 為 1,500,000，
`sports_and_outdoors` 為 1,000,000，`toys_and_games` 為 1,500,000。
