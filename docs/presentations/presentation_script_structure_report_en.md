# AlphaFold Structure Extension Presentation Script

## Opening

Hi everyone. In this part of the project, my contribution was to extend the original sequence-based protein function prediction baseline by incorporating AlphaFold-derived structural information. The main question I wanted to test was whether structural information can improve Gene Ontology, or GO term, prediction beyond sequence-only features.

Our overall task is to take a protein amino acid sequence as input and predict its associated GO terms. This is a multi-label classification problem because one protein can have many GO annotations at the same time, across biological process, molecular function, and cellular component categories.

This task is challenging for several reasons. First, protein sequences can vary greatly in length. Second, the GO label space is very large. Third, GO term frequencies are highly long-tailed: a small number of GO terms are very common, while many GO terms are rare. Finally, GO terms are also hierarchical, so the labels are not completely independent.

**Use figure:** `../../figures/dataset_label_space_summary.png`

This figure shows the dataset and label-space challenges. We can see variation in protein sequence lengths and in the number of labels per protein. We also see the long-tail distribution of GO term frequency. Because of this imbalance, accuracy is not a very informative metric. A model could obtain high accuracy by predicting mostly negatives. For this reason, we focus on metrics such as Fmax, micro-F1, and precision-recall behavior.

## Preprocessing Pipeline

Next, I will briefly describe the preprocessing pipeline.

**Use figure:** `../../figures/preprocessing_pipeline_diagram.png`

The pipeline starts from the CAFA5 training sequences, GO annotations, and taxonomy information. The sequence-based baseline represents proteins using k-mer features or ESM embeddings. My extension uses the subset of proteins with available AlphaFold structures. For those proteins, we use AlphaFold PDB files and PAE files, then convert each protein structure into a graph representation.

In this graph, nodes represent residues, and edges represent residue-residue contacts. Node features include amino acid identity, AlphaFold confidence information such as pLDDT, residue-level ESM2 embeddings, and additional structure-derived features. Edge features include distance, contact strength, and PAE-related information. The graph neural network then encodes the protein graph and outputs a probability for each GO term.

The important point is that we are not treating the sequence model and the structure model as two unrelated systems. Instead, this is one protein function prediction system with a structural extension. The controlled research question is: if we keep the dataset split and label vocabulary fixed, does adding structure provide additional signal beyond sequence-only features?

## Training Setup

The table below summarizes the structure GNN training configuration on Savio. Baseline corresponds to the defaults in `scripts/savio_train_full_graph.sh`, and Tuned corresponds to `scripts/savio_train_full_graph_tuned.sh`, which is the configuration used for the CCO label-aware scorer that produced the +2.28 pp improvement.

| Category | Setting | Baseline | Tuned (used for CCO result) |
|---|---|---|---|
| **Data** | dataset | CAFA5 train (AlphaFold-matched cohort) | same |
| | aspect | MFO / CCO | MFO / CCO |
| | min term frequency | 20 | 20 |
| | split | train / val / test (fixed seed) | same |
| **Model** | architecture | 2-layer GCN + global mean pool + graph-feat MLP | same |
| | hidden dim | 128 | **256** |
| | dropout | 0.2 | **0.3** |
| | classifier head | `flat_linear` | `flat_linear` or `label_dot` (label-aware) |
| | node feature dim | 682 (32 base + DSSP/ESM2 slots zero-filled) | same |
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
| **Hardware / runtime** | partition | Savio `savio2_1080ti`, 1 node, 4× GTX 1080Ti | same |
| | parallelism | 1 GPU per aspect (BPO / CCO / MFO trained independently) | same |
| | wall-time cap | 72h | 72h |
| | framework | PyTorch 2.3.1 + PyG | same |

For the talk, I will only highlight the five decisions that matter most for the results.

First, **the three aspects are trained separately and in parallel**. BPO, CCO, and MFO each get their own model and their own GTX 1080Ti, so all four GPUs are utilized at the same time. This not only shortens wall-clock time, it also prevents losses from different aspects from interfering with each other.

Second, **rare-label filtering**. We use `min_term_frequency = 20`, so any GO term seen fewer than 20 times in training is removed from the label space. These terms are essentially unlearnable and only dilute the BCE signal.

Third, **handling class imbalance**. We use weighted BCE with `pos_weight_power = 0.5` and `max_pos_weight = 20`. This means rare labels are upweighted by `1 / sqrt(freq)`, but the weight is capped at 20× to prevent extremely rare labels from blowing up the gradient.

Fourth, **checkpointing on val Fmax instead of val loss**. The official CAFA metric is Fmax, and the epoch with the lowest loss is not necessarily the epoch with the highest Fmax. So we select the best model directly by the task metric, combined with `early stopping patience = 2` and `min_delta = 5e-4` to avoid overfitting.

Fifth, **optimizer and learning-rate schedule**. Adam with `lr = 3e-4`, `weight_decay = 5e-4`, `dropout = 0.3`, and `hidden_dim = 256`. The learning rate is controlled by `ReduceLROnPlateau`: when the validation metric plateaus, the LR is halved, with a floor of `1e-6`. We use `batch_size = 8` and train for 5 epochs.

Compared with the baseline, the three changes that matter most are: **a larger hidden dim, weighted BCE for the loss, and val_fmax for checkpoint selection**. These are the changes responsible for the +2.28 pp CCO improvement reported later.

> If time is very tight, the one-sentence version is: "The three aspects are trained independently across four GPUs; weighted BCE handles label imbalance and rare labels with frequency below 20 are filtered out; and we checkpoint on val Fmax instead of val loss — these are the three main changes from the baseline."

## Team Baselines

Before discussing my structure results, it is useful to summarize the team baselines from the earlier progress reports.

**Use figure:** `../../figures/team_progress_sequence_baselines.png`

The team first developed sequence-only baselines. The earliest baseline used 2-mer frequency features, which represent each protein sequence as a 400-dimensional k-mer vector. Later, the team used ESM protein language model embeddings, which performed substantially better than k-mer features.

In the all-GO setting, the k-mer MLP achieved a micro-F1 of about 0.150, while the ESM MLP improved to about 0.201. The checkpoint 4 GO-split analysis also showed that ESM outperformed k-mer across BPO, CCO, and MFO.

The takeaway is that sequence representation matters. ESM embeddings are a much stronger sequence baseline than simple k-mer counts. However, these older ESM and k-mer results should not be used as a fully direct comparison against my AlphaFold graph model, because they use different protein cohorts, label spaces, data splits, and metrics. In the report, we present them as historical teammate baselines that explain the project progression, not as the final controlled sequence-versus-structure comparison.

## Structure Graph Results

Now I will describe the structure-enhanced model results.

**Use figure:** `../../figures/per_ontology_graph_comparison.png`

This figure compares graph-side experiments. All of these models use AlphaFold structure graphs, but they differ in model design. The clearest improvement appears in CCO, or Cellular Component. The label-aware graph model substantially improves over the raw graph baseline.

The raw graph baseline achieved a CCO test Fmax of 0.5647. The label-aware scorer increased CCO test Fmax to 0.5875, which is an improvement of about 2.28 percentage points. This is the strongest improvement we observed in the structure graph experiments.

There is also a biological intuition for this result. Cellular Component prediction is related to localization, protein complexes, surface exposure, and domain organization. These are properties that may be more connected to structural information, so the graph representation has a plausible way to help.

In contrast, the improvement for MFO, or Molecular Function, is much smaller. Weighted BCE gives a small improvement, but the label-aware scorer does not produce the same type of breakthrough as it does for CCO. This suggests that the usefulness of structure is ontology-dependent. Structural information does not help every GO namespace equally.

## Matched MFO Sequence vs Structure Comparison

Next, I will discuss the most controlled sequence-versus-structure comparison currently available.

**Use figure:** `../../figures/overall_mfo_sequence_structure_comparison.png`

This figure shows the matched MFO experiment. Here, the sequence-only ESM2 MLP and the structure-enhanced graph model use the same MFO protein cohort, the same train-validation-test split, and the same GO label vocabulary. This makes it the cleanest comparison we currently have between sequence-only and structure-enhanced modeling.

The sequence-only ESM2 MLP achieved a test Fmax of 0.4557. The AlphaFold structure-enhanced graph model achieved a test Fmax of 0.4580. The difference is about +0.23 percentage points. This is a small improvement, so we should not claim that structure dramatically improves MFO prediction. However, it does show that, under a controlled MFO setup, the structure graph model is slightly better than the matched sequence-only ESM2 baseline.

The late-fusion result is similar, with a test Fmax of 0.4579. It does not clearly outperform the graph-only model. This suggests that there may be some complementarity between sequence and structure, but under the current model design, that complementarity is weak.

## Precision-Recall Analysis

**Use figure:** `../../figures/mfo_precision_recall_curve.png`

The precision-recall curve supports the same interpretation. The structure graph model has a micro AUPRC of about 0.367, while the sequence-only model has a micro AUPRC of about 0.359. The structure model is slightly higher. The difference plot on the right shows that, in some recall ranges, the graph model has slightly better precision than the sequence-only model. However, the overall difference is still small.

So for MFO, the conclusion should be cautious: AlphaFold structural information provides a small additional signal beyond sequence-only ESM2, but the improvement is not decisive.

## Frequency-Bin Analysis

Next, we look at performance across GO term frequency bins.

**Use figure:** `../../figures/performance_vs_go_term_frequency.png`

This figure groups GO terms by their training frequency and compares mean per-term F1 for the sequence-only and graph models. Because rare GO terms have very few positive examples, per-term F1 can be noisy. So this plot should be interpreted as a diagnostic rather than as the main result.

The main pattern is that high-frequency GO terms are more stable, while rare GO terms remain very difficult to predict. This shows that class imbalance is still a major bottleneck. Structural information alone does not solve the rare-label problem. Future work may need hierarchical losses, stronger label-aware modeling, or better calibration methods.

## Contribution Summary

To summarize my contribution, I did three main things.

First, I built the AlphaFold structure integration pipeline, converting PDB, PAE, and residue-contact information into graph features. Second, I trained structure-enhanced graph models and compared them against sequence-only baselines where matched artifacts were available. Third, I analyzed whether structural information helps differently across ontologies and across GO term frequency ranges.

The overall conclusion is:

AlphaFold-derived structural information can help protein function prediction, but the benefit is ontology-dependent.

For CCO, the structure graph model with a label-aware scorer gives the clearest improvement, increasing test Fmax by about 2.28 percentage points over the raw graph baseline. For MFO, the matched sequence-versus-structure comparison shows only a small improvement: the structure graph model improves test Fmax by about 0.23 percentage points over the sequence-only ESM2 MLP.

Therefore, we should not overstate the result by saying that AlphaFold structure dramatically improves all GO prediction. A more accurate conclusion is that structural information provides additional signal, especially for some ontologies such as Cellular Component, but the size of the benefit depends on the prediction target.

To make this conclusion stronger in future work, the next step would be to train a matched CCO sequence-only ESM2 baseline, so that CCO can also be evaluated with the same controlled sequence-versus-structure setup as MFO. We could also explore more structure-aware GNN architectures, better sequence-structure fusion, and GO hierarchy-aware loss functions.

## One-Sentence Conclusion

If I had to summarize my part in one sentence:

**We found that ESM sequence features are already a strong baseline, and AlphaFold structural information provides additional but ontology-dependent signal: it gives a small improvement in the matched MFO experiment and the clearest graph-side gain in CCO.**
