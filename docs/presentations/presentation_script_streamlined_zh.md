# AlphaFold Structure Extension — 精簡中文講稿

> 比 `presentation_script_structure_report.md` 更精簡的版本，砍掉重複的段落，把故事壓成一條線：問題 → 怎麼把結構接進來 → 訓練 → 兩個關鍵結果 → 解讀與限制 → 一句話收尾。預計講 6–7 分鐘。

---

## 1. 開場：我這部分在問什麼問題

大家好，我這部分要回答的問題很單純：

**在原本 sequence-only 的 protein function prediction 之上，加入 AlphaFold 預測的 3D 結構資訊，到底有沒有幫助？**

任務本身是給一條 amino acid sequence，預測它對應的 Gene Ontology terms。它是 **multi-label**——一個蛋白同時可以有很多 GO 標註；label space 很大；而且 GO term frequency 是嚴重 **long-tail**——少數 term 很常見，大量 term 非常稀有。所以後面我們都用 **Fmax** 而不是 accuracy。

**搭配圖表：** `../../figures/dataset_label_space_summary.png`

---

## 2. 怎麼把 structure 接進來

**搭配圖表：** `../../figures/preprocessing_pipeline_diagram.png`

對於有 AlphaFold 結構的 protein，我們把每個蛋白變成一張 graph：**node 是 residue，edge 是 residue 之間的接觸**。

**搭配圖表：** `../../figures/node_edge_graph_data_comparison.png`

graph 上有三種資料：

- **Node（per-residue）**：amino acid one-hot、pLDDT 與分桶、PAE row 統計、contact degree、序列相對位置 `i / L`。
- **Edge（per-contact）**：Cα 距離、sequence separation、PAE 值、是否是 strict 3D contact。
- **Graph（per-protein）**：一個 13 維向量，把整顆蛋白總結成 length、fragment 數、各 pLDDT 區段比例、平均 PAE、edge density、radius of gyration 等等。

這裡有一個重點：**graph-level vector 不是放在 node 或 edge 裡，因為它本來就是 protein-level 的資訊**——後面在模型架構裡，它會繞過 GNN，直接餵給分類器。

**搭配圖表：** `../../figures/model_architecture_diagram.png`

模型很乾淨：node features 經過 2 層 GCNConv 做 message passing，再 global mean pool 成 protein 向量；graph_feat 走另一條 shortcut（Linear + ReLU）；兩者 concat 後接分類頭，輸出每個 GO term 的機率。

---

## 3. Training setup（30 秒帶過）

**搭配圖表：** `../../figures/training_setup_table.png`

訓練設定我只強調三個跟結果關係最大的決定：

1. **三個 aspect（BPO / CCO / MFO）分開訓練、平行跑。** 各佔一張 1080Ti，總共四卡同時跑，互不干擾。
2. **`min_term_frequency = 20`**，把極稀有 GO term 直接從 label space 移掉，避免稀釋 BCE 訊號。
3. **Checkpoint 用 val Fmax，不是 val loss。** 因為 CAFA 的官方指標是 Fmax，loss 最低的 epoch 不一定 Fmax 最高。

其他都是常規：Adam、5 epochs、batch size 8、`weighted BCE` 處理 class imbalance。

---

## 4. 結果一：CCO 上，label-aware head 帶來最明顯改善

**搭配圖表：** `../../figures/per_ontology_graph_comparison.png`

第一個比較是 **graph-side 內部的架構 ablation**。Baseline 是標準的 `flat_linear` 分類頭——`Linear(256 → K)`，每個 GO term 對應一列權重。我們把它換成 **label-aware（`label_dot`）head**：

> 每一個 GO term 各自有一個 256 維可學習 embedding，蛋白向量跟它做內積就是 logit。
> `logits = fused @ E^T`，其中 `E ∈ R^{K × 256}` 是 label embedding 表。

其他所有超參數（hidden_dim、loss、lr、epoch）完全一樣，是一個乾淨的 ablation。

結果：**CCO test Fmax 從 0.5647 → 0.5875（+2.28 pp）**，是整個 graph-side 實驗最明顯的提升。

為什麼 CCO 特別有效？因為 **CCO 的 label 之間有很強的語意鄰近性**。Cellular Component 大量是「同一個 organelle 的不同子位置」，例如 `nucleus → nucleolus → nuclear envelope`。把 label 變成 embedding 之後，optimizer 會自動把這些語意相近的 GO term 推到向量空間裡相近的位置，所以稀有 label 也能借到鄰居的訊號。MFO 的 label 結構比較零散，同樣的改動在 MFO 上效果就小很多。

---

## 5. 結果二：MFO matched，sequence vs structure 差距很小

**搭配圖表：** `../../figures/overall_mfo_sequence_structure_comparison.png`

第二個比較是最接近 **controlled sequence-vs-structure** 的實驗。在 MFO 上，我們控制 protein cohort、train/val/test split、label vocabulary 完全相同，比較三個模型：

| 模型 | Test Fmax |
|---|---|
| Sequence-only ESM2 MLP | 0.4557 |
| AlphaFold structure GNN | **0.4580** |
| Late fusion (graph + sequence) | 0.4579 |

Structure 比 sequence 多 **+0.23 pp**，late fusion 沒有再贏。

這個改善很小，所以這裡的結論要保守：在 MFO 上，AlphaFold 結構 **有提供一點額外訊號，但程度不顯著、也沒有跟 sequence 形成可觀的互補性**。

---

## 6. 把兩個結果串起來：ontology-dependent

把兩個結果放在一起就是這份報告的核心 narrative：

- **CCO 上 graph-side 改架構就能 +2.28 pp**——CCO 描述「protein 在哪裡」，跟 3D 結構直接相關。
- **MFO 上 structure 比 sequence 只多 +0.23 pp**——MFO 描述「protein 做什麼」，更依賴 active site 的局部 chemistry，目前的 graph representation 不一定能抓到。

所以一個清楚的 take-away 是：

> **AlphaFold 結構資訊對 protein function prediction 是有幫助的，但這個幫助是 ontology-dependent**——對 **localization 類**（CCO）的 GO term 幫助比較大，對 **molecular function 類**（MFO）的 GO term 幫助比較小。

---

## 7. 限制與下一步（誠實點出）

我必須誠實點出一個 caveat：

**CCO 的 +2.28 pp 是 graph-side 內部比較**（raw graph baseline → label-aware graph），**沒有像 MFO 那樣直接跟 sequence-only baseline 比過**。所以嚴格來說，目前我還不能斷言 CCO 上 structure 也勝過 sequence。下一步要補的就是：在 CCO 上跑一個 matched ESM2-only MLP，跟 label-aware structure GNN 做完全對照的 sequence-vs-structure 比較。

如果之後還有時間，方向是更 structure-aware 的 GNN（例如 SE(3)-equivariant）、protein language model 跟 structure 的早期 fusion，以及把 GO 階層直接寫進 loss。

---

## 8. 一句話結論

如果用一句話收尾：

> **我們在 graph-side 把分類頭換成 label-aware embedding 之後，CCO test Fmax 提升了 2.28 pp；在 MFO matched 對照下，structure 比 sequence 多 0.23 pp。整體上，AlphaFold 結構訊息對 GO prediction 是 *有幫助但 ontology-dependent*——對 localization 類的 GO term 幫助明顯，對 molecular function 類的 GO term 幫助有限。**
