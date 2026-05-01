# CAFA5 實驗流程與結果總整理

更新日期：2026-04-26

這份文件整理目前 CAFA5 專案從資料取得、前處理、graph cache 建立、baseline 訓練、調參、到 the label-aware scorer/the sequence+structure fusion 後續實驗的完整脈絡。重點不是只列結果，而是說明每一步為什麼要做、得到什麼結果、以及我們根據結果做了什麼下一步決策。

主要結論：

- 我們已完成 graph training 的主流程，包含 full graph baseline、normalized feature run、tuned run、the single-factor hyperparameter sweep ablation、the architecture-and-objective experiments significant-improvement batch、the label-aware scorer formal confirmation、the label-aware scorer stability run、ontology regularization follow-up、以及 MFO 的 the sequence+structure fusion sequence-graph fusion。
- `BPO` full graph training 在目前記憶體預算下不穩定，所以正式 graph 實驗目前聚焦 `CCO` 與 `MFO`。
- `label-aware scorer` label-aware scorer 是目前第一個明確打贏 raw graph baseline 的方向，尤其在 `CCO` 上效果穩定。
- `MFO` 目前沒有出現像 `CCO` 那樣的大幅突破；the weighted-BCE variant weighted BCE、the label-aware scorer、the sequence+structure fusion fusion 都只有小幅改善或接近持平。

## 1. 結果來源

本整理主要根據以下 artifact：

| 類型 | 路徑 / 文件 | 用途 |
| --- | --- | --- |
| 原始計畫 | `docs/planning/experiment_plan.md` | 早期資料、cohort、split、label policy、model track 設計 |
| Graph 進度 | `docs/reports/graph_training_progress_report.md` | baseline、normalized、tuned、the architecture-and-objective experiments 初步結果脈絡 |
| the architecture-and-objective experiments 計畫 | `docs/planning/graph_significant_improvement_exploration_plan.md` | the architecture-and-objective experiments 實驗動機與 follow-up rule |
| Graph full runs | `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs` | 主要訓練結果 |
| Normalized graph run | `/global/scratch/users/bensonli/cafa5_outputs/graph_cache_normalized_features/training_runs` | normalized feature full-data 結果 |
| Sequence / fusion | `/global/scratch/users/bensonli/cafa5_outputs/sequence_runs`、`/global/scratch/users/bensonli/cafa5_outputs/n4_fusion` | the sequence+structure fusion sequence branch 與 late fusion 結果 |

注意：部分早期 `results_summary.json` 只存 final epoch 指標；本文件對 raw baseline 與 tuned run 使用各 aspect `summary.json` 中 history 的 best validation Fmax epoch 作為正式比較值。

## 2. 整體實驗流程

| 階段 | 目的 | 主要產出 | 結果判斷 | 下一步 |
| --- | --- | --- | --- | --- |
| 資料與 notebook 探索 | 確認 CAFA5 序列、label、AlphaFold coverage | raw sequence/label 統計、早期 k-mer/ESM baseline 概念 | label space 很大且 long-tailed，需要 per-aspect 和 frequency threshold | 建立可重現 pipeline |
| Graph preprocessing | 建立 structure-available graph cache | graph cache、aspect splits、vocab | `CCO/MFO/BPO` split 都能產生，但 BPO 訓練不穩 | 正式 graph training 聚焦 CCO/MFO |
| Small smoke runs | 驗證訓練 loop、metrics、checkpoint | local small experiment summaries | normalized 在 tiny MFO sample 有明顯訊號 | 跑 full normalized graph |
| Full graph baseline | 建立 raw graph reference | raw baseline, tuned, normalized runs | raw baseline 很強；normalized/tuned 都未超越 | 做更乾淨的 the single-factor hyperparameter sweep ablation |
| the single-factor hyperparameter sweep ablation | 測 learning rate、capacity、weighted BCE | the single-factor hyperparameter sweep results | 只有 MFO 小幅上升，沒有 CCO 大突破 | 停止 local tuning，改測高槓桿方向 |
| this batch of experiments | 測 loss、calibration、label-aware head | the focal-BCE attempt focal BCE、the logit-adjustment attempt logit adjustment、the label-aware scorer label-dot | the label-aware scorer 在 CCO 明顯贏；the focal-BCE attempt/the logit-adjustment attempt 不值得主推 | the label-aware scorer 正式化與穩定性驗證 |
| label-aware scorer formalization | 確認 the label-aware scorer 是否可重現 | confirm_long、seed2027、ontology_reg | CCO gain 可重現；MFO 仍小幅或持平 | the label-aware scorer 可作 CCO 新主線；MFO 需新方向 |
| the sequence+structure fusion fusion | 測 sequence + graph complementarity | sequence branch、graph-vocab bundle、late fusion | MFO fusion 只有非常小的 validation gain | 暫不視為 decisive win |

## 3. 資料取得與早期探索

### 3.1 Sequence 與 GO label

早期 notebook 與 `docs/planning/experiment_plan.md` 記錄的資料狀態：

| 項目 | 數值 / 狀態 |
| --- | ---: |
| Raw CAFA5 training proteins | `142,246` |
| Clean aligned proteins with valid sequence and labels | `140,569` |
| GO label matrix | `140,569 x 31,454` |
| Mean labels per protein | 約 `37.8` |
| GO terms with frequency `< 10` | `16,588` |
| GO terms with frequency `>= 50` | `6,398` |

解讀：

- label space 非常大，而且長尾嚴重。
- 因此後續不直接訓練所有 GO term，而是拆成 `BPO`、`CCO`、`MFO`，並採用 `min_term_frequency`。
- 正式 graph run 主要使用 `min_term_frequency = 20`。

### 3.2 早期 sequence baseline

| Track | 特徵 | 狀態 | 用途 |
| --- | --- | --- | --- |
| k-mer baseline | `k=2`, vocab size `400` | notebook 探索完成，但 workspace 中未形成完整 reproducible artifact | 最便宜 sequence baseline |
| protein-level ESM2 | `facebook/esm2_t6_8M_UR50D`, `320-d` mean pooled embedding | notebook artifact 存在，但 ID/split 不完整 | sequence-only baseline |
| graph modality ESM2 | `facebook/esm2_t30_150M_UR50D`, residue-level `640-d` | 用於 graph cache / matched cohort sequence branch | 不可與 320-d notebook ESM 混用 |

後續處理：

- 因為 notebook sequence baseline 的 split 與 graph split 沒完全 frozen，後續公平比較改成 matched structure-available cohort。
- the sequence+structure fusion 使用 graph cache 對齊的 sequence artifact，避免 sequence-only 和 graph-only cohort 不一致。

## 4. AlphaFold / Graph 前處理

早期 AlphaFold / graph pipeline 狀態：

| 項目 | 結果 |
| --- | ---: |
| Full training index size | `142,246` |
| AlphaFold ok | 約 `122,925` |
| Structural coverage | 約 `86.4%` |
| Graph cache preview size | 約 `122,924` proteins |
| Node feature width | `682` |
| Edge feature width | `6` |
| Graph feature width | `13` |

正式 graph split 狀態：

| Aspect | Entry count | Vocab size | Train | Val | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BPO` | `80,741` | `7,665` | `64,593` | `8,074` | `8,074` |
| `CCO` | `82,610` | `1,025` | `66,088` | `8,261` | `8,261` |
| `MFO` | `73,021` | `1,521` | `58,417` | `7,302` | `7,302` |

決策：

- Graph cache 與 split 已足夠支援正式訓練。
- `BPO` label 數與記憶體需求太高，後續正式 graph reporting 暫時排除。
- `CCO` 與 `MFO` 成為正式 graph 實驗主軸。

## 5. Small Smoke / Debug Experiments

本地 small experiments 的目的不是產出正式結論，而是確認訓練 loop、summary writing、Fmax sweep、normalization pipeline 是否可用。

| Run | Aspect | Normalized | Val Fmax | Test Fmax | Val/Test graphs | 解讀 |
| --- | --- | --- | ---: | ---: | --- | --- |
| `small_experiment_mfo_mtf20_n120` | MFO | no | `0.0630` | `0.0608` | `12 / 12` | tiny sample，僅 smoke |
| `small_experiment_mfo_mtf20_cap1200_n120` | MFO | no | `0.0391` | `0.0462` | `12 / 12` | tiny sample，僅 smoke |
| `small_experiment_mfo_mtf50_cap1200_n120` | MFO | no | `0.0828` | `0.0821` | `12 / 12` | tiny sample，僅 smoke |
| `small_experiment_mfo_mtf20_n120_normalized` | MFO | yes | `0.2136` | `0.1203` | `12 / 12` | normalization 在小樣本有訊號 |

後續處理：

- 因為 normalized small run 明顯高於其他 tiny runs，我們接著跑 full normalized-feature graph run。
- 但 tiny sample 結果不當作正式模型比較。

## 6. Full Graph Baseline、Normalized、Tuned

共同設定：

- structure-available graph cohort
- one model per aspect
- `min_term_frequency = 20`
- split seed `2026`
- primary metric：validation-selected Fmax

### 6.1 Full-run 比較

| Run | Aspect | Best Val Fmax | Best Test Fmax | 結論 |
| --- | --- | ---: | ---: | --- |
| raw baseline `full_graph_pyg_mtf20_33234089` | `CCO` | `0.5635` | `0.5647` | 初始強 baseline |
| raw baseline `full_graph_pyg_mtf20_33234089` | `MFO` | `0.4522` | `0.4574` | 初始強 baseline |
| raw baseline `full_graph_pyg_mtf20_33234089` | `BPO` | `0.2647` | `0.2655` | run failed after epoch 1，不列正式結果 |
| normalized `cco_mfo_parallel_20260415_1906` | `CCO` | `0.5605` | `0.5623` | 未超越 raw |
| normalized `cco_mfo_parallel_20260415_1906` | `MFO` | `0.4491` | `0.4544` | 未超越 raw |
| tuned `full_graph_tuned_pyg_mtf20_33275343` | `CCO` | `0.5567` | `0.5584` | 未超越 raw |
| tuned `full_graph_tuned_pyg_mtf20_33275343` | `MFO` | `0.4450` | `0.4513` | 未超越 raw |

### 6.2 解讀與下一步

Normalized run 的結果表示：

- feature scale 不是 full-data graph model 的主要瓶頸。
- 小樣本 normalization gain 沒有延伸到 full data。

Tuned run 的設定包含：

- `weighted_bce`
- `hidden_dim = 256`
- `dropout = 0.3`
- `lr = 0.0003`
- `weight_decay = 0.0005`
- plateau scheduler

但 tuned run 沒有提升 Fmax。這代表混合多個 tuning 改動後難以歸因，也沒有產生有效增益。

後續處理：

- 不再做大雜燴式 tuning。
- 改成 the single-factor hyperparameter sweep：一次只改一個主要因素，測 local hyperparameter 是否真的有 headroom。

## 7. Hyperparameter Sweep

目的：固定 raw baseline recipe，測幾個局部變化是否能穩定提升 Fmax。

| ID | 主要變化 |
| --- | --- |
| `control rerun` | control rerun |
| `lr=7e-4 variant` | lower LR `0.0007` |
| `lr=5e-4 variant` | lower LR `0.0005` |
| `hidden=192 variant` | `hidden_dim = 192` |
| `hidden=256 variant` | `hidden_dim = 256` |
| `weighted-BCE variant` | `weighted_bce` only |

結果：

| Run | Aspect | Status | Best Val Fmax | Best Test Fmax | 結論 |
| --- | --- | --- | ---: | ---: | --- |
| `control rerun` | `CCO` | success | `0.5641` | `0.5654` | 與 raw baseline 幾乎持平 |
| `control rerun` | `MFO` | success | `0.4537` | `0.4595` | 比原始 raw MFO 略高 |
| `lr=7e-4 variant` | `CCO` | failed | `0.5643` | `0.5660` | run 不乾淨，不作正式結論 |
| `lr=7e-4 variant` | `MFO` | success | `0.4531` | `0.4597` | 小幅改善 |
| `lr=5e-4 variant` | `CCO` | success | `0.5643` | `0.5655` | 幾乎持平 |
| `lr=5e-4 variant` | `MFO` | success | `0.4544` | `0.4603` | 小幅改善 |
| `hidden=192 variant` | `CCO` | success | `0.5631` | `0.5645` | 無改善 |
| `hidden=192 variant` | `MFO` | success | `0.4529` | `0.4586` | 幾乎持平 |
| `hidden=256 variant` | `CCO` | success | `0.5633` | `0.5651` | 無改善 |
| `hidden=256 variant` | `MFO` | success | `0.4519` | `0.4573` | 幾乎持平 |
| `weighted-BCE variant` | `CCO` | success | `0.5640` | `0.5656` | 幾乎持平 |
| `weighted-BCE variant` | `MFO` | success | `0.4551` | `0.4605` | MFO 此輪最佳，但提升仍小 |

解讀：

- `CCO` 對 local tuning 幾乎不敏感，沒有 meaningful gain。
- `MFO` 在 lower LR / weighted BCE 有小幅上升，但提升幅度不夠大。
- 這輪證明主要瓶頸不太像 learning rate、hidden dim、或單純 BCE weighting。

後續處理：

- 停止把主力放在 local hyperparameter sweep。
- 改測更高槓桿方向：objective、calibration、label-aware scoring、sequence-graph fusion，也就是 the architecture-and-objective experiments。

## 8. Architecture-and-Objective Experiments

固定 reference setup：

- `hidden_dim = 128`
- `dropout = 0.20`
- `lr = 0.001`
- `weight_decay = 0.0001`
- `EPOCHS = 5`
- checkpoint metric：`val_fmax`
- raw features，不做 normalization
- seed `2026`

| Direction | 核心改動 | 問題 |
| --- | --- | --- |
| `focal-BCE attempt` | `LOSS_FUNCTION=focal_bce` | 是否 loss/objective 才是 long-tail bottleneck？ |
| `logit-adjustment attempt` | `LOGIT_ADJUSTMENT=train_prior` | 是否 logits calibration 才是 Fmax bottleneck？ |
| `label-aware scorer` | `MODEL_HEAD=label_dot` | 是否 flat classifier head 是 bottleneck？ |

結果：

| Direction | Aspect | Best Val Fmax | Best Test Fmax | vs raw baseline | 結論 |
| --- | --- | ---: | ---: | --- | --- |
| raw baseline | `CCO` | `0.5635` | `0.5647` | reference | anchor |
| `focal-BCE attempt` | `CCO` | `0.5591` | `0.5609` | worse | 不繼續 |
| `logit-adjustment attempt` | `CCO` | `0.5637` | `0.5652` | tied | 保留工具，不主推 |
| `label-aware scorer` | `CCO` | `0.5822` | `0.5843` | clear win | 主線候選 |
| raw baseline | `MFO` | `0.4522` | `0.4574` | reference | anchor |
| `focal-BCE attempt` | `MFO` | `0.4452` | `0.4513` | worse | 不繼續 |
| `logit-adjustment attempt` | `MFO` | `0.4505` | `0.4558` | slightly worse | 不主推 |
| `label-aware scorer` | `MFO` | `0.4517` | `0.4580` | near tied | 可保留但不是突破 |

解讀：

- `focal-BCE attempt` 直接淘汰。
- `logit-adjustment attempt` 沒有證明 calibration 是主瓶頸。
- `label-aware scorer` 是第一個在 `CCO` 上帶來明確 step change 的方法。

後續處理：

- 對 the label-aware scorer 做正式化：longer confirmation、second seed stability、ontology regularization follow-up。
- the sequence+structure fusion fusion 也繼續，因為它回答 sequence 和 graph 是否互補，而不是 graph head 設計。

## 9. Label-Aware Scorer Confirmation and Stability

label-aware scorer formalization 相關 runs：

| Run | Slurm job | 目的 |
| --- | ---: | --- |
| `sigimp_n3_label_dot_20260422_172916` | `33706654` | 第一個 the label-aware scorer label-dot full run |
| `sigimp_n3_confirm_long_20260423_104340` | `33712068` | same-seed longer confirmation |
| `sigimp_n3_confirm_long_20260425_n3_confirm` | `33741105` | 最新 same-seed formal confirmation |
| `sigimp_n3_stability_seed2027_20260423_104340` | `33712069` | second-seed stability |
| `sigimp_n3_ontology_reg_20260423_105633` | `33712624` | ontology regularization follow-up |

結果：

| Run | Aspect | Best Val Fmax | Best Test Fmax | Best epoch | 解讀 |
| --- | --- | ---: | ---: | ---: | --- |
| first `label-aware scorer` | `CCO` | `0.5822` | `0.5843` | 4 | 已明顯贏 raw |
| first `label-aware scorer` | `MFO` | `0.4517` | `0.4580` | 3 | 接近持平 |
| `label-aware confirm run_long 20260423` | `CCO` | `0.5842` | `0.5865` | 6 | CCO gain 更強 |
| `label-aware confirm run_long 20260423` | `MFO` | `0.4501` | `0.4566` | 4 | MFO 無突破 |
| `label-aware confirm run_long 20260425` | `CCO` | `0.5855` | `0.5875` | 6 | 目前最佳 CCO |
| `label-aware confirm run_long 20260425` | `MFO` | `0.4514` | `0.4575` | 4 | 仍約持平 |
| `the label-aware scorer seed2027 stability` | `CCO` | `0.5806` | `0.5816` | 4 | 證明方向可重現，但有 seed variance |
| `the label-aware scorer seed2027 stability` | `MFO` | `0.4502` | `0.4562` | 5 | MFO 無突破 |
| `label-aware + ontology reg` | `CCO` | `0.5825` | `0.5846` | 4 | 與 the label-aware scorer 同級，未超越 latest confirm |
| `label-aware + ontology reg` | `MFO` | `0.4518` | `0.4576` | 3 | 接近持平 |

最新正式化 run：

- run：`sigimp_n3_confirm_long_20260425_n3_confirm`
- job：`33741105`
- status：`COMPLETED`
- time：2026-04-25 15:49:34 到 2026-04-26 04:02:58

解讀：

- `CCO`：the label-aware scorer 是目前最強 graph-side recipe。相較 raw baseline，test Fmax 約 `0.5647 -> 0.5875`，提升約 `+0.0228`。
- `MFO`：the label-aware scorer 沒有帶來 decisive gain。表現約在 raw / the single-factor hyperparameter sweep / fusion 的同一區間。

後續處理：

- 報告中可以把 the label-aware scorer 作為 `CCO` 的新主線結果。
- `MFO` 不能直接宣稱 the label-aware scorer 解決問題；需要新的 MFO-specific strategy，例如更好的 sequence branch、loss/threshold calibration、或不同 label-aware design。

## 10. Sequence + Graph Late Fusion

the sequence+structure fusion 的目的不是改 graph head，而是測 sequence 和 graph score 是否互補。

Sequence branch：

| Run | Aspect | Model | Best Val Fmax | Best epoch | Cohort |
| --- | --- | --- | ---: | ---: | --- |
| `sigimp_n4_seq_graph_vocab_mfo_mlp_20260424_222500` | `MFO` | MLP on graph-vocab sequence features | `0.4513` | 3 | matched structure cohort |

Fusion 設定：

- branch：the label-aware scorer MFO graph predictions + sequence predictions
- score space：logits
- weights：`g0p0_s1p0` 到 `g1p0_s0p0`
- validation Fmax 選 weight

the label-aware scorer graph + sequence fusion 結果：

| Weight | Val Fmax | Test Fmax | 解讀 |
| --- | ---: | ---: | --- |
| sequence only `g0p0_s1p0` | `0.4513` | `0.4557` | sequence branch 本身接近 graph |
| the label-aware scorer graph only `g1p0_s0p0` | `0.4517` | `0.4580` | graph-only slightly better test |
| val-selected fusion `g0p8_s0p2` | `0.4529` | `0.4579` | validation 小幅提升，test 幾乎持平 |
| test-best diagnostic `g0p9_s0p1` | `0.4523` on val | `0.4582` | 不作正式選擇，只是 diagnostic |

解讀：

- sequence 和 graph 有一點 complementarity，但目前很弱。
- the sequence+structure fusion 不是目前的 decisive win，尤其 test-side 沒有明顯改善。

後續處理：

- 保留 fusion pipeline，因為它已經能產生 matched-cohort score-level comparison。
- 但短期主力不應是大規模 fusion grid；更需要改善 sequence branch 或 MFO-specific graph model。

## 11. 目前最佳結果整理

| Aspect | 目前最佳正式 graph-side 結果 | Val Fmax | Test Fmax | 備註 |
| --- | --- | ---: | ---: | --- |
| `CCO` | `label-aware confirm run_long 20260425` | `0.5855` | `0.5875` | 目前最明確成功 |
| `MFO` | `weighted-BCE variant` by Fmax table | `0.4551` | `0.4605` | 小幅勝過 raw/the label-aware scorer，但不是大突破 |
| `BPO` | no formal graph result | `0.2647` | `0.2655` | raw run failed after epoch 1，不列正式比較 |

若只看 the label-aware scorer branch：

| Aspect | Best the label-aware scorer run | Val Fmax | Test Fmax | 結論 |
| --- | --- | ---: | ---: | --- |
| `CCO` | `label-aware confirm run_long 20260425` | `0.5855` | `0.5875` | the label-aware scorer 成立 |
| `MFO` | `the label-aware scorer label_dot / ontology_reg` level | 約 `0.4517-0.4518` | 約 `0.4580` | 接近 raw，不是突破 |

## 12. 實驗決策脈絡總結

| 觀察 | 判斷 | 後續處理 |
| --- | --- | --- |
| label space 大且 long-tailed | 不能直接全 GO term 混訓 | per-aspect training + `min_term_frequency=20` |
| AlphaFold coverage 約 86% | graph model 有足夠資料可跑 | 建立 structure-available graph cohort |
| BPO full graph 不穩 | 記憶體 / label size 是阻礙 | 正式 graph 結果先排除 BPO |
| tiny normalized MFO 有訊號 | normalization 值得 full-data 驗證 | 跑 normalized full graph |
| normalized full graph 未超 raw | scale 不是主瓶頸 | 不主推 normalization |
| mixed tuned run 輸 raw | 多改動 tuning 不可靠 | 改 the single-factor hyperparameter sweep single-factor ablation |
| the single-factor hyperparameter sweep 沒有 CCO 大突破 | local tuning headroom 不大 | 轉向 the architecture-and-objective experiments 高槓桿方向 |
| the focal-BCE attempt focal BCE 輸 | objective-only focal 不是解方 | 停止 the focal-BCE attempt |
| the logit-adjustment attempt logit adjustment 持平 | calibration 不是主要瓶頸 | 保留工具，不主推 |
| the label-aware scorer CCO 大幅提升 | flat head 可能是 CCO bottleneck | label-aware scorer formalization、stability、ontology follow-up |
| the label-aware scorer MFO 無大幅提升 | MFO bottleneck 不同 | 需要 MFO-specific 方法 |
| the sequence+structure fusion fusion MFO 微幅 gain | sequence/graph complementarity 弱但存在 | 保留 pipeline，先改善 branch |

## 13. 建議下一步

短期優先順序：

1. 更新舊進度文件：`docs/reports/graph_training_progress_report.md` 和 checklist 仍把 `33741105` 寫成 `RUNNING`，需要改為 `COMPLETED` 並加入最新 label-aware scorer formalization 結果。
2. 產出一個正式 comparison artifact：把 raw、normalized、tuned、the single-factor hyperparameter sweep、the architecture-and-objective experiments、the label-aware scorer follow-up 全部整理成 machine-readable table，避免未來手動抄表。
3. 對 final report 採用 aspect-specific 結論：
   - `CCO`：主推 the label-aware scorer label-aware scorer。
   - `MFO`：不能主推 the label-aware scorer 為 breakthrough；目前最佳只是小幅 local-tuning / fusion gain。
   - `BPO`：列為 memory-limited / out-of-scope for formal graph result。
4. 若還有時間跑實驗，MFO 應優先：
   - 更強 sequence branch；
   - MFO-specific threshold/calibration；
   - label-aware head 的不同初始化或 regularization；
   - 更乾淨的 sequence + graph fusion，而不是單純擴大 local graph tuning。

## 14. 最終一句話

我們目前已經從資料取得與前處理走到正式 graph model improvement 階段。最重要的科學結果是：`label-aware scorer` label-aware protein-to-GO scorer 在 `CCO` 上明確超越 raw graph baseline，並且經過 longer confirmation 與 second-seed stability 檢查；但 `MFO` 還沒有同等級突破，後續應把 MFO 當作獨立問題處理，而不是假設 CCO 的 the label-aware scorer 解法會直接泛化。
