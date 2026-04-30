# CAFA5 Experiment Plan

Date: 2026-04-02

## 1. Scope

This document turns the current notebook-based preprocessing status into a concrete experiment plan for model training.

The plan covers:

- what has already been done
- what is still missing before formal training
- how to organize the experiments
- what outputs must be saved for reproducibility

The immediate goal is not to jump straight to a final model. The immediate goal is to make the training pipeline reproducible, comparable across model families, and ready for a clean final report.

## 2. Current Status

### 2.1 Sequence preprocessing status

From the notebooks under [output/jupyter-notebook](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/output/jupyter-notebook):

- Raw CAFA5 training proteins: `142,246`
- Clean aligned proteins with valid sequence and labels: `140,569`
- GO label matrix size: `140,569 x 31,454`
- Mean labels per protein: about `37.8`
- GO terms with frequency `< 10`: `16,588`
- GO terms with frequency `>= 50`: `6,398`

This means the label space is very large and heavily long-tailed. A frequency threshold will be required in formal training.

### 2.2 K-mer baseline status

The notebook [Data Preprocessing Kmer.ipynb](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/output/jupyter-notebook/Data%20Preprocessing%20Kmer.ipynb) and the integrated notebook [CAFA5 Data Preparation Exploration.ipynb](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/output/jupyter-notebook/CAFA5%20Data%20Preparation%20Exploration.ipynb) show:

- `k = 2`
- vocabulary size: `400`
- feature matrix shape: `140,569 x 400`
- random 80/20 train/val split
- `StandardScaler` fit on train and applied to val
- outputs were intended to be saved as:
  - `X_train.npy`
  - `X_val.npy`
  - `Y_train.npz`
  - `Y_val.npz`
  - `train_idx.npy`
  - `val_idx.npy`
  - `meta.pkl`

Current gap:

- the current workspace does not contain a local `data_processed_kmer/` folder, so this artifact is not yet reproducibly available in the repo workspace

### 2.3 Protein-level ESM2 baseline status

The notebook [ESM2_embedding.ipynb](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/output/jupyter-notebook/ESM2_embedding.ipynb) and the integrated notebook show:

- model: `facebook/esm2_t6_8M_UR50D`
- output feature shape: `140,569 x 320`
- this is a protein-level mean pooled embedding, not a residue-level embedding

Current local artifacts in [data_processed_esm](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/data_processed_esm):

- `X.npy`: `(140569, 320)` `float32`
- `Y.npy`: `(140569, 31454)` `int8`
- `esm_dataset.npz`

Current gap:

- the notebook mentions `protein_ids.pkl`, but it is not present in the current local folder
- split files for the sequence ESM baseline are not frozen as formal train/val/test manifests

### 2.4 AlphaFold and graph preprocessing status

The AlphaFold and graph pipeline is much more structured than the notebook baselines. Relevant files include:

- [cafa5_alphafold_pipeline.py](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/cafa5_alphafold_pipeline.py)
- [alphafold_feature_extractor.py](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/alphafold_feature_extractor.py)
- [cafa_graph_dataset.py](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/cafa_graph_dataset.py)
- [cafa_graph_dataloaders.py](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/cafa_graph_dataloaders.py)
- [train_minimal_graph_model.py](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/train_minimal_graph_model.py)

Current sample run in [outputs/cafa5_af_100](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/outputs/cafa5_af_100):

- 100 proteins indexed
- 87 proteins with graph cache entries
- graph node feature width: `682`
- edge feature width: `6`
- graph feature width: `13`
- split manifests exist for `BPO`, `CCO`, `MFO`
- dataloader export verification exists
- benchmark summary exists
- ESM2 modality cache exists for 87 entries
- structure cache exists for 93 fragments

Notebook-observed full-pipeline status from [CAFA5 Data Preparation Exploration.ipynb](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/output/jupyter-notebook/CAFA5%20Data%20Preparation%20Exploration.ipynb):

- full training index size: `142,246`
- AlphaFold `ok`: about `122,925`
- structural coverage: about `86.4%`
- graph cache preview size in notebook: `122,924` proteins

This means the graph line is already a real training candidate, not just a concept.

### 2.5 Important current mismatch

There are currently two different ESM usages:

- sequence notebook baseline: `320-d` protein-level mean pooled ESM2
- graph multimodal cache: `640-d` residue-level ESM2 from `facebook/esm2_t30_150M_UR50D`

These are not interchangeable and must be treated as different model inputs in the experiment design.

## 3. Main Problems That Must Be Solved

Before formal experiments, the project still needs the following:

1. Sequence baseline preprocessing must be converted from notebooks into scripts.
2. Sequence and graph experiments must use explicit and frozen split manifests.
3. Label vocabulary policy must be fixed per experiment.
4. Cohort definitions must be made explicit so comparisons are fair.
5. Evaluation must go beyond training loss and one fixed-threshold micro-F1.
6. The AlphaFold PAE gzip parsing issue must be fixed before rebuilding the full structure cohort.
7. A formal training smoke test must be added before large runs.

## 4. Experimental Principles

The project should follow these rules:

- Train `BPO`, `CCO`, and `MFO` separately.
- Compare sequence-only and graph-based models on a matched structure-available cohort.
- Keep one broader full-sequence cohort for sequence baseline reference.
- Freeze split files and label vocab files for every official experiment.
- Tune thresholds on validation only.
- Treat the 100-entry graph sample as a debug set, not as a scientific result.

## 5. Dataset Definitions

### 5.1 Cohort A: Full-sequence cohort

Use all clean aligned proteins with valid sequence and labels.

Purpose:

- train sequence-only baselines
- estimate full-data sequence performance
- provide a non-structure upper baseline for coverage

Expected content:

- around `140,569` proteins
- sequence-based features only

### 5.2 Cohort B: Structure-available cohort

Use only proteins that successfully materialize into graph cache entries.

Purpose:

- graph model training
- multimodal training
- fair sequence-vs-graph comparison on the same proteins

Expected content:

- sample debug set: `outputs/cafa5_af_100/graph_cache`
- full run target: structure-available subset after full graph rebuild

### 5.3 Matched comparison cohort

For fair ablation:

- train a sequence baseline on exactly the same entry IDs as the graph cohort
- train graph and multimodal models on those same IDs

This should be the main comparison table in the final report.

## 6. Label Policy

### 6.1 Training namespace

Train one model per aspect:

- `BPO`
- `CCO`
- `MFO`

Do not merge all GO labels into one model for the main report.

### 6.2 Frequency threshold

Use a staged policy:

- debug stage on small sample: `min_term_frequency = 1`
- first stable full-data run: `min_term_frequency = 20`
- sensitivity study: compare `10`, `20`, and `50`

Rationale:

- frequency `1` is acceptable only for smoke tests
- full experiments need reduced tail noise
- the threshold itself should be treated as a hyperparameter study, not hidden preprocessing

### 6.3 GO propagation

Current recommendation:

- do not add true-path propagation in the first round of formal experiments

Reason:

- current preprocessing already preserves raw annotation
- propagation changes label semantics and should be introduced only as a clearly separated ablation

## 7. Split Policy

### 7.1 Official split format

Every official experiment should save:

- `train.txt`
- `val.txt`
- `test.txt`
- `summary.json`

The graph pipeline already does this under [outputs/cafa5_af_100/graph_cache/splits](/Users/bensonli/Documents/Project_2026/c242/Final%20Project/outputs/cafa5_af_100/graph_cache/splits).

### 7.2 Split strategy

Recommended official strategy:

- fixed random split for the first formal round
- seed: `2026`
- ratios: `0.8 / 0.1 / 0.1`

Recommended follow-up:

- repeat final selected models with 3 seeds: `2026`, `2027`, `2028`

### 7.3 Comparison rule

For any comparison table:

- same cohort
- same aspect
- same split
- same label frequency threshold

If any of these differ, the result should be marked as not directly comparable.

## 8. Model Tracks

### 8.1 Priority order

The experiments should be run in this order:

1. sequence k-mer baseline
2. sequence protein-level ESM2 baseline
3. graph structural baseline
4. graph + residue ESM2
5. graph + DSSP/SASA
6. graph full multimodal

### 8.2 Experiment matrix

| Priority | Track | Cohort | Input | Model family | Purpose |
| --- | --- | --- | --- | --- | --- |
| P0 | K-mer Linear | Full-sequence and matched structure subset | 400-d k-mer | linear / one-vs-rest baseline | establish simplest sequence baseline |
| P1 | K-mer MLP | Full-sequence and matched structure subset | 400-d k-mer | 2-layer MLP | stronger cheap sequence baseline |
| P2 | Protein ESM Linear | Full-sequence and matched structure subset | 320-d protein ESM2 | linear head | establish pretrained sequence baseline |
| P3 | Protein ESM MLP | Full-sequence and matched structure subset | 320-d protein ESM2 | MLP | main sequence baseline |
| P4 | Graph Base | Structure-available cohort | base graph features only | GNN | structural baseline |
| P5 | Graph + Residue ESM2 | Structure-available cohort | graph + 640-d residue ESM2 | multimodal GNN | main graph-sequence fusion model |
| P6 | Graph + DSSP/SASA | Structure-available cohort | graph + structure cache | multimodal GNN | isolate structure cache gain |
| P7 | Graph Full Multimodal | Structure-available cohort | graph + residue ESM2 + DSSP/SASA | multimodal GNN | final strongest candidate |

### 8.3 Optional ablations

After the main matrix is stable:

- graph without graph-level summary features
- graph without edge attributes
- graph without modality mask
- threshold sensitivity study
- raw labels vs propagated labels

## 9. Evaluation Plan

### 9.1 Metrics to report

Every official experiment should report:

- training loss
- validation loss
- micro-F1
- macro-F1
- micro-AUPR
- Fmax
- number of labels in vocab
- number of proteins in train/val/test

### 9.2 Thresholding

Do not keep a fixed `0.5` threshold as the final evaluation policy.

Recommended:

- sweep thresholds on validation
- select the threshold that maximizes validation Fmax
- freeze that threshold
- apply once to test

### 9.3 Reporting format

For each aspect:

- one table for sequence baselines
- one table for graph and multimodal models
- one final matched-cohort comparison table

For the overall final summary:

- one concise table with the best validation-selected configuration for each track

## 10. Missing Work Before Formal Training

### 10.1 Must-do engineering tasks

| Priority | Task | Why it matters |
| --- | --- | --- |
| High | Convert k-mer notebook into `prepare_kmer_features.py` | current artifact is not reproducible in workspace |
| High | Convert protein ESM notebook into `prepare_sequence_esm_baseline.py` | baseline artifacts and IDs are incomplete |
| High | Freeze split manifests for sequence experiments | sequence and graph results are not yet directly comparable |
| High | Implement a formal evaluation script | current graph training only tracks loss and micro-F1 |
| High | Add training smoke test for `.venv311` graph environment | formal training should not begin before environment validation |
| High | Fix gzip-aware PAE JSON reading in AlphaFold pipeline | current parse failures reduce structure coverage |
| Medium | Rebuild full AlphaFold manifests and graph cache after gzip fix | needed for final structure cohort |
| Medium | Save run config, label vocab, threshold, and metrics for every experiment | required for reproducibility |
| Medium | Add matched-cohort export for sequence baselines | needed for fair model-family comparison |

### 10.2 Specific known blocker

The current pipeline still reads JSON with standard UTF-8 text loading. The notebook failure analysis indicates that some PAE files are gzip-compressed. That bug should be fixed before any full graph rebuild.

## 11. Execution Phases

### Phase 0: Environment and artifact smoke checks

Tasks:

- verify `.venv` sequence environment
- verify `.venv311` graph environment
- run one tiny sequence training smoke test
- run one tiny graph training smoke test
- confirm checkpoints and metrics files are written

Exit criteria:

- one sequence run finishes end-to-end
- one graph run finishes end-to-end
- output files are readable

### Phase 1: Freeze data interfaces

Tasks:

- create scriptized sequence preprocessing
- export official split manifests
- freeze aspect-specific label vocab files
- define matched structure cohort entry lists

Deliverables:

- `data_processed_kmer/`
- `data_processed_esm/`
- `splits/<aspect>/`
- `vocabs/<aspect>.json`

### Phase 2: Sequence baselines

Tasks:

- train k-mer linear baseline
- train k-mer MLP baseline
- train protein-level ESM linear baseline
- train protein-level ESM MLP baseline

Run first on:

- full-sequence cohort
- matched structure-available cohort

Deliverables:

- per-aspect metrics
- validation thresholds
- best checkpoints

### Phase 3: Graph baseline

Tasks:

- train graph base-only model
- verify the effect of `min_term_frequency`
- verify train speed and memory footprint on full graph cohort

Deliverables:

- graph baseline table
- training curves
- debug notes on bottlenecks

### Phase 4: Multimodal graph runs

Tasks:

- graph + residue ESM2
- graph + DSSP/SASA
- graph + all modalities

Deliverables:

- ablation table
- comparison against sequence baseline on matched cohort

### Phase 5: Final reporting

Tasks:

- choose final best model per aspect
- repeat final runs with 3 seeds
- aggregate mean and standard deviation
- prepare final figures and tables

Deliverables:

- final report-ready tables
- final figures
- final checkpoints and configs

## 12. Run Output Standard

Every training run should save the following:

- `config.json`
- `split_summary.json`
- `label_vocab.json`
- `history.json`
- `best_checkpoint.pt`
- `val_metrics.json`
- `test_metrics.json`
- `selected_threshold.json`

Recommended folder layout:

- `experiments/<date>/<track>/<aspect>/<run_name>/`

Example:

- `experiments/2026-04-02/protein_esm_mlp/mfo/seed2026_freq20/`

## 13. Proposed Immediate Next Steps

The next four actions should be:

1. create `prepare_kmer_features.py`
2. create `prepare_sequence_esm_baseline.py`
3. fix gzip-aware PAE parsing in the AlphaFold pipeline
4. add a formal evaluation script and one-epoch smoke test command for graph training

These four steps will convert the current project from notebook-heavy preprocessing into an experiment-ready training pipeline.

## 14. Success Criteria

This project is ready for formal experimentation only when:

- sequence baselines are scriptized and reproducible
- graph training completes end-to-end on at least one official split
- sequence and graph comparisons use a matched cohort
- evaluation includes threshold-tuned test metrics
- every official run saves config, split, vocab, metrics, and checkpoint

Until then, the project should be treated as preprocessing-complete but experiment-incomplete.
