# AlphaFold Structure Extension 中文報告講稿

## 開場

大家好，我這部分主要負責的是在原本的 sequence-based protein function prediction baseline 上，加入 AlphaFold-derived structural information，並測試結構資訊是否能幫助 GO term prediction。

我們的整體任務是給定一條 protein amino acid sequence，預測它對應的 Gene Ontology terms。這是一個 multi-label classification problem，因為一個 protein 通常不只對應一個 GO term，而是會同時有多個 biological process、molecular function 或 cellular component annotations。

這個問題有幾個主要挑戰。第一，protein sequence 長度差異很大；第二，GO label space 非常大；第三，GO term frequency 呈現很嚴重的 long-tail distribution，也就是少數 GO terms 很常見，但大量 GO terms 非常稀有；第四，GO 本身還有 hierarchical structure，所以很多 label 之間不是完全獨立的。

**搭配圖表：** `../../figures/dataset_label_space_summary.png`

從這張圖可以看到，protein sequence length 和 labels per protein 都有很大的變異。同時 GO term frequency 呈現明顯 long-tail，代表如果只看 accuracy 會非常不合理，因為模型很容易被大量 negative labels 主導。因此我們後面主要使用 Fmax、micro-F1、precision-recall 這類更適合 multi-label prediction 的 metrics。

## Preprocessing Pipeline

接下來是我們的 preprocessing pipeline。

**搭配圖表：** `../../figures/preprocessing_pipeline_diagram.png`

我們的 pipeline 從 CAFA5 的 training sequences、GO annotations 和 taxonomy data 開始。sequence-based baseline 會把 protein sequence 轉成 k-mer features 或 ESM embeddings。我的 extension 則是針對有 AlphaFold structure 的 proteins，下載或讀取 AlphaFold PDB 和 PAE files，接著把蛋白質結構轉成 graph representation。

在 graph 裡面，node 代表 residue，edge 代表 residue-residue contact。node features 包含 amino acid identity、AlphaFold confidence score，也就是 pLDDT，以及額外的 ESM2 residue embedding 和 structure-derived features。edge features 則包含距離、contact strength 和 PAE-related information。最後模型會把整個 protein graph 做 graph neural network encoding，再輸出每個 GO term 的 probability。

這裡很重要的是，我們不是把 sequence model 和 structure model 當成兩個完全無關的模型，而是把它們放在同一個 protein function prediction system 裡比較。核心問題是：在相同資料切分和相同 label vocabulary 下，加入 structure information 是否能提供 sequence-only model 之外的額外訊號？

### Feature Extraction 階段：資料處理完長什麼樣子

為了讓大家看到 feature extraction 階段「資料變成什麼」，我用 cache 裡一個真實 protein（entry `Q8IXT2`，367 個 residue）當例子，分三張圖呈現。

**搭配圖表 1：** `../../figures/feature_extraction_node_features.png`（Node side）

第一張圖在講 **每一個 residue 變成什麼**。最上面那一條是「一個 residue 的 32 維 base feature vector」放大版：
- 前 21 維是 amino acid 的 one-hot（在這個例子裡是 'V'）
- 第 22 維是 pLDDT 分數（98.8，AlphaFold 對這個 residue 的位置很有信心）
- 接下來 4 維是 pLDDT 分桶（very_low / low / confident / very_high 的 indicator）
- 再來 3 維是 PAE row 統計（mean / min / p90）
- 接下來 2 維是 contact degree（這個 residue 連到幾個 3D 鄰居）
- 最後 1 維是 position fraction（i / sequence_length）

中間那條彩色長條是「全部 367 個 residue 各自的 amino acid identity」，每個顏色代表一個 AA。下面的 heatmap 把幾個重要連續特徵沿著 residue 軸畫出來，可以看到 pLDDT 在 residue 80–150 那段特別高，PAE 也比較低，代表那一段是 AlphaFold 預測得最有信心的 well-folded core。中間 residue 200 左右看起來是低 pLDDT 的 disordered 區段。

**搭配圖表 2：** `../../figures/feature_extraction_contact_map.png`（Contact map）

第二張圖在講 **edges 長什麼樣子**。這是 residue × residue 的 contact map，紅色是 strict 3D contact，灰色是 sequential / weak contact。對角線一條紅線是 backbone 上的相鄰 residue（i 和 i+1 一定靠得很近）。重點是 **off-diagonal 的紅點** — 那些是 sequence 上不相鄰、但 3D 結構上摺疊在一起的 residue 對，例如左下角 residue 40–100 那一塊明顯有「結構上摺回來」的 long-range contacts。這就是 sequence-only model 看不到、但 structure graph 可以拿到的訊號。

整個 protein 共有 1285 條 undirected edge，其中 907 條是 strict 3D contact、366 條是純 sequential。

**搭配圖表 3：** `../../figures/feature_extraction_edges_and_graph.png`（Edge / graph 級特徵）

第三張圖把 edge 的 6 維特徵和 protein 整體的 13 維 graph-level feature vector 攤開來：
- 左上：C-alpha 距離分布。Sequential edges 集中在 ~3.8 Å（standard backbone bond length），non-seq edges 散在 4–10 Å 之間。
- 中上：sequence separation。大部分 edges 是 |i−j|=1（backbone neighbors），但有不少 long-range edges 達到 |i−j|>50，這些就是結構摺疊提供的資訊。
- 右上：每條 edge 上的 PAE 值。Strict contacts 明顯集中在低 PAE（AlphaFold 對這個 pair 的相對位置很有信心），這支持「我們真的在抓對的接觸」。
- 下面那條紫色長條是這個 protein 的 graph-level summary vector，13 個數字包括 residue 數、fragment 數、各 pLDDT 區段比例、edge density、平均 PAE、radius of gyration 等等，會跟 node features 一起送進 GNN。

所以 feature extraction 階段的結果，就是把每個 protein 變成一個 (`x`: N×682 node features, `edge_index`: 2×E, `edge_attr`: E×6, `graph_feat`: 13) 的 tuple，可以直接餵進 PyG 或 DGL 的 GNN。

## Training Setup

下面這張表整理了我們 structure GNN 在 Savio 上的訓練設定，跟講稿後面的 results 對應。Baseline 對應 `scripts/savio_train_full_graph.sh` 的 default，Tuned 對應 `scripts/savio_train_full_graph_tuned.sh`，也是 CCO label-aware scorer 那個 +2.28 pp 結果跑出來的那組。

| 類別 | 設定 | Baseline | Tuned (CCO 結果用的) |
|---|---|---|---|
| **Data** | dataset | CAFA5 train (AlphaFold-matched cohort) | 同左 |
| | aspect | MFO / CCO | MFO / CCO |
| | min term frequency | 20 | 20 |
| | split | train / val / test (固定 seed) | 同左 |
| **Model** | architecture | 2-layer GCN + global mean pool + graph-feat MLP | 同左 |
| | hidden dim | 128 | **256** |
| | dropout | 0.2 | **0.3** |
| | classifier head | `flat_linear` | `flat_linear` 或 `label_dot` (label-aware) |
| | node feature dim | 682 (32 base + DSSP/ESM2 slots zero-filled) | 同左 |
| **Optimizer** | algorithm | Adam | Adam |
| | learning rate | 1e-3 | **3e-4** |
| | weight decay | 1e-4 | **5e-4** |
| | LR scheduler | none | **ReduceLROnPlateau** (factor 0.5, patience 1, min 1e-6) |
| **Loss** | function | BCE-with-logits | **weighted BCE** (`pos_weight_power=0.5`, `max_pos_weight=20`) |
| | logit adjustment | none | none |
| **Training schedule** | epochs | 5 | 5 |
| | batch size | 8 | 8 |
| | early stopping | off | **patience=2, min_delta=5e-4** |
| | checkpoint metric | `val_loss` | **`val_fmax`** |
| | seed | 2026 | 2026 |
| **Hardware / runtime** | partition | Savio `savio2_1080ti`, 1 node, 4× GTX 1080Ti | 同左 |
| | parallelism | 1 GPU per aspect (BPO / CCO / MFO 各自獨立 train) | 同左 |
| | wall-time cap | 72h | 72h |
| | framework | PyTorch 2.3.1 + PyG | 同左 |

在 training setup 上，我只強調五個跟結果最相關的決定。

第一，**三個 aspect 分開訓練、平行跑**。BPO、CCO、MFO 各自獨立模型、各佔一張 GTX 1080Ti，總共四張卡同時跑。這樣不只縮短 wall-clock 時間，也避免 aspect 之間在 loss 上互相干擾。

第二，**rare label 過濾**。`min_term_frequency = 20`，出現次數少於 20 的 GO term 直接從 label space 移除，因為這些幾乎學不起來，留著只會稀釋 BCE 的訊號。

第三，**處理 class imbalance**。loss 用 weighted BCE，搭配 `pos_weight_power = 0.5`、`max_pos_weight = 20`。也就是用 1/√freq 的方式給稀有 label 加權，再封頂在 20 倍，避免極稀有 label 把 gradient 撐爆。

第四，**checkpoint 由 val Fmax 決定，不是 val loss**。因為 CAFA 的官方指標是 Fmax，loss 最低的 epoch 不一定 Fmax 最高，所以我直接用任務指標來選 best model。搭配 `early stopping patience = 2`、`min_delta = 5e-4`，避免過擬合。

第五，**optimizer 與 LR schedule**。Adam，`lr = 3e-4`、`weight_decay = 5e-4`、`dropout = 0.3`、`hidden_dim = 256`；學習率用 `ReduceLROnPlateau`，val 指標停滯時砍半，下限 `1e-6`。`batch_size = 8`、訓練 5 個 epoch。

整套設定相較 baseline 主要調了三件事：**hidden dim 加大、loss 改成 weighted BCE、checkpoint 改用 val Fmax**——這也是後面 CCO label-aware scorer 那個 +2.28 pp 結果的主要來源。

> 時間真的非常緊時可以只留一句版本：「三個 aspect 分開、四卡平行；loss 用 weighted BCE 處理 imbalance、稀有 label 過濾掉 frequency < 20；checkpoint 用 val Fmax 而不是 val loss——這三點是相對 baseline 主要的改動。」

## Team Baselines

在比較我的結果之前，先簡單說一下 team 前面的 baseline。

**搭配圖表：** `../../figures/team_progress_sequence_baselines.png`

從過去 progress reports 可以看到，team 先建立了 sequence-only baselines。最早的是 2-mer k-mer MLP，也就是把 protein sequence 轉成 400 維的 k-mer frequency vector。後來使用 ESM protein language model embeddings，效果明顯比 k-mer 好。

在 all-GO setting 中，k-mer MLP 的 micro-F1 大約是 0.150，而 ESM MLP 提升到大約 0.201。checkpoint 4 裡面的 GO-split analysis 也顯示，ESM 在 BPO、CCO、MFO 三個 ontology 上都比 k-mer 有明顯提升。

這說明了一件事：sequence representation 本身很重要，ESM embedding 是比簡單 k-mer 更強的 baseline。不過這些舊的 ESM/k-mer 結果不能直接拿來和我的 AlphaFold graph model 做完全公平比較，因為它們的 protein cohort、label space、data split 和 metric 都不完全一樣。所以在 report 裡，我們把它們定位成 historical teammate baselines，用來說明 project progression，而不是最終 controlled comparison。

## Structure Graph Results

接下來是我的 structure-enhanced model 的正式結果。

**搭配圖表：** `../../figures/per_ontology_graph_comparison.png`

這張圖比較的是 graph-side experiments，也就是都使用 AlphaFold structure graph，只是模型設計不同。可以看到在 CCO，也就是 Cellular Component ontology 上，label-aware graph model 相比 raw graph baseline 有明顯提升。

Raw graph baseline 的 CCO test Fmax 是 0.5647，而 label-aware scorer 的 CCO test Fmax 提升到 0.5875，大約增加 2.28 percentage points。這是目前 structure graph experiment 裡最明顯的 improvement。

這個結果也有 biological intuition。Cellular Component prediction 和 protein localization、complex membership、surface exposure、domain organization 可能比較相關，而這些訊號有機會透過 structure graph 被模型捕捉到。

相對地，在 MFO，也就是 Molecular Function ontology 上，graph-side 的 improvement 就比較小。Weighted BCE 有一點提升，但 label-aware scorer 沒有像 CCO 那樣帶來明顯突破。這代表 structure information 的幫助可能是 ontology-dependent，不是所有 GO namespace 都一樣受益。

## Matched MFO Sequence vs Structure Comparison

接著看最接近 controlled comparison 的結果。

**搭配圖表：** `../../figures/overall_mfo_sequence_structure_comparison.png`

這張圖比較的是 matched MFO experiment。這裡的 sequence-only ESM2 MLP 和 structure-enhanced graph model 使用相同的 MFO protein cohort、相同 train/validation/test split，以及相同 GO label vocabulary。所以這是目前最公平的 sequence vs structure comparison。

結果顯示，sequence-only ESM2 MLP 的 test Fmax 是 0.4557，而 AlphaFold structure-enhanced graph model 的 test Fmax 是 0.4580。差距大約是 +0.23 percentage points。這個 improvement 很小，所以我們不能說 structure dramatically improves MFO prediction。但是它確實顯示，在 controlled MFO setting 下，structure graph 有略高於 sequence-only ESM2 的表現。

晚期 fusion 的結果也差不多，test Fmax 是 0.4579，沒有明顯超過 graph-only。這代表在目前模型設計下，sequence 和 structure 之間的 complementarity 存在，但還不夠強。

## Precision-Recall Analysis

**搭配圖表：** `../../figures/mfo_precision_recall_curve.png`

Precision-recall curve 也支持類似結論。Structure graph 的 micro AUPRC 約 0.367，sequence-only 約 0.359，structure graph 略高一點。右邊的 difference plot 顯示，在某些 recall range 裡，graph model 的 precision 比 sequence-only 稍微好一些，但整體差距仍然不大。

所以對 MFO，我們的結論要保守：AlphaFold structure information provides a small additional signal beyond sequence-only ESM2, but the improvement is not decisive.

## Frequency-Bin Analysis

再看 label frequency 的分析。

**搭配圖表：** `../../figures/performance_vs_go_term_frequency.png`

這張圖把 GO terms 按照 training frequency 分 bin，觀察 sequence-only 和 graph model 在不同 frequency range 的 per-term F1。由於 rare GO terms 的 positive examples 很少，per-term F1 本身會很 noisy，所以這張圖主要是 diagnostic。

目前結果比較明顯的是，模型對高頻 GO terms 的表現比較穩定，而低頻 terms 幾乎都很難預測。這反映出 class imbalance 仍然是主要 bottleneck。structure information 不是單獨就能解決 rare label problem，未來可能需要 hierarchical loss、label-aware modeling 或更好的 calibration。

## Contribution Summary

最後總結我的 contribution。

我的部分主要做了三件事。第一，建立 AlphaFold structure integration pipeline，把 PDB、PAE 和 residue contact information 轉成 graph features。第二，訓練 structure-enhanced graph model，並和 sequence-only baseline 做 controlled comparison。第三，分析不同 ontology 和不同 label frequency 下，structure information 是否有幫助。

整體結論是：

AlphaFold-derived structural information can help protein function prediction, but the benefit is ontology-dependent.

在 CCO 上，structure graph 加上 label-aware scorer 有最明顯提升，test Fmax 相比 raw graph baseline 增加約 2.28 percentage points。在 MFO 的 matched sequence-vs-structure comparison 中，structure graph 只比 sequence-only ESM2 MLP 稍微好一點，test Fmax 增加約 0.23 percentage points。

所以我們不能誇大說 AlphaFold structure 全面大幅提升所有 GO prediction，但可以合理地說：結構資訊確實提供了額外訊號，尤其在某些 ontology，例如 Cellular Component，效果更明顯。

未來如果要讓這個結論更強，我們下一步應該補上 matched CCO sequence-only ESM2 baseline，讓 CCO 也能像 MFO 一樣做完全 controlled sequence-vs-structure comparison。另外也可以嘗試更 structure-aware 的 GNN architecture、protein language model 和 structure fusion、以及 GO hierarchy-aware loss function。

## 一句話結論

如果要用一句話總結我這部分：

**我們發現 ESM sequence features 已經是很強的 baseline，而 AlphaFold structural information 在 matched MFO experiment 中提供小幅額外提升，並且在 CCO graph-side experiment 中帶來最明顯的 improvement，表示 structure information 對 protein function prediction 的幫助是存在但 ontology-dependent。**
