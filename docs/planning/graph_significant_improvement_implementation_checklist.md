# Graph Significant Improvement Implementation Checklist

Date: 2026-04-22

This checklist turns `docs/planning/graph_significant_improvement_exploration_plan.md`
into concrete implementation tasks.

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` completed

## 1. Shared Infrastructure

- [~] Extend the training CLI so new exploration variants can be configured
  without forking the whole training script.
- [ ] Ensure new run configs are written into `run_config.json`,
  `summary.json`, and `results_summary.json`.
- [~] Add one comparison helper for the new exploration runs so `N1`-`N4` can
  be summarized consistently.
- [x] Decide one stable naming convention for the new round, for example
  `sigimp_N1_*`, `sigimp_N2_*`, and so on.

## 2. `N1` Better Multi-Label Loss

- [x] Add a new loss option for `ASL` or focal-style multi-label training in
  `train_minimal_graph_model.py`.
- [x] Add CLI arguments for the new loss hyperparameters.
- [x] Write the selected loss configuration into the run summary.
- [x] Add one Savio submission path for the `N1` run.
- [x] Run one smoke test on a small split before any full run.

Implementation target:

- primary file: `train_minimal_graph_model.py`
- secondary file: `scripts/savio_train_full_graph.sh`

## 3. `N2` Logit Adjustment and Calibration

- [x] Add train-prior-based logit adjustment as an optional evaluation transform.
- [x] Add temperature scaling as part of the same logit transform path.
- [x] Save the label-prior summary and transform configuration into the run
  summary.
- [x] Add submission-time environment variables for the new transform options.
- [ ] Run one controlled comparison against the raw baseline on `MFO`, then
  `CCO`.

Implementation target:

- primary file: `train_minimal_graph_model.py`
- secondary file: `scripts/savio_train_full_graph.sh`

## 4. `N3` Label-Aware Protein-to-GO Scoring

- [x] Add a new model head option that scores protein embeddings against GO term
  embeddings instead of using a single flat linear classifier.
- [x] Decide the first GO embedding initialization strategy.
- [x] Preserve the current graph encoder so the first `N3` test isolates the
  head change.
- [x] Add config and checkpoint serialization for the new head type.
- [x] Run one smoke test before any full run.

Implementation target:

- primary file: `train_minimal_graph_model.py`
- possible helper file: new label-embedding utility if needed

## 5. `N4` Sequence + Graph Late Fusion

- [~] Identify the current best reusable sequence branch for the matched
  structure-available cohort.
- [x] Define the fusion interface at score level so graph and sequence branches
  can be trained independently.
- [x] Add a utility that loads per-protein predictions from both branches and
  produces fused scores.
- [x] Define one simple fusion rule for the first run, such as weighted average
  over logits or probabilities.
- [x] Add one Savio3 small-scale submission path for the first sequence-side
  matched-cohort run.
- [~] Build one matched-cohort evaluation path for the fused model.

Implementation target:

- likely new helper script plus evaluation glue
- should not block `N1`-`N3`

## 6. Recommended Build Order

- [~] Step 1: land `N2` infrastructure first because it is the lowest-risk
  extension to the current pipeline.
- [x] Step 2: add `N1` once the shared CLI/config path is stable.
- [x] Step 3: add `N3` after the objective-side changes are runnable.
- [~] Step 4: add `N4` once the sequence branch interface is clear.

## 7. Immediate Next Action

- [x] Implement `N2` in the training script and Savio launcher.
- [x] Run one small smoke comparison to confirm `N2` changes are reflected in
  `summary.json` and `run_config.json`.
- [x] Start `N3` by replacing the flat classifier with a label-aware scorer.
- [x] Submit the first Savio3 graph-side `N1`/`N2`/`N3` training batch using
  the fixed reference setup on `CCO` and `MFO`.
- [x] Start `N4` by defining the reusable sequence-side prediction interface.
- [x] Identify the first concrete sequence-side producer for the new fusion
  bundle format.
- [x] Run the first real matched-cohort sequence-plus-graph fusion experiment on
  `MFO`.

## 8. Active Run Tracking

| Direction | Job ID | Status | Run Name | Next Check |
| --- | ---: | --- | --- | --- |
| `N1` | `33706651` | `COMPLETED` | `sigimp_n1_focal_bce_20260422_172916` | result collected; baseline loser on `CCO` and `MFO` |
| `N2` | `33706652` | `COMPLETED` | `sigimp_n2_logit_adjust_20260422_172916` | result collected; only tiny `CCO` movement |
| `N3` | `33706654` | `COMPLETED` | `sigimp_n3_label_dot_20260422_172916` | result collected; clear `CCO` winner |
| `N4 sequence` | `33706500` | `COMPLETED` | `sigimp_n4_seq_small_mfo_mlp_20260422_172448` | superseded by graph-vocab sequence run below |
| `N4 sequence` | n/a | `COMPLETED` | `sigimp_n4_seq_graph_vocab_mfo_mlp_20260424_222500` | reusable MFO sequence bundle for score fusion |
| `N4 fusion` | n/a | `COMPLETED` | `raw_mfo_x_seq_graph_vocab_20260424_222500` | first fusion pass; graph branch used raw available `best.pt`, whose exported graph-only val/test is `0.4499 / 0.4571` |
| `N4 fusion` | `33740526` | `COMPLETED` | `n3_mfo_x_seq_graph_vocab_20260424_222500` | second fusion pass; graph branch used N3 MFO `best.pt` from epoch 3 |
| `N3 confirm` | `33741105` | `RUNNING` | `sigimp_n3_confirm_long_20260425_n3_confirm` | same seed confirmation; `EPOCHS=8`, `MODEL_HEAD=label_dot`, `CCO MFO` |

## 9. After Results Arrive

- [x] Collect `best_val_fmax`, `best_test_fmax`, threshold, micro-F1, macro-F1,
  and best epoch for every `sigimp_*` run.
- [x] Compare each direction against the raw baseline:
  `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`.
- [ ] Record one compact `N1`-`N4` comparison artifact from the run directories
  instead of doing future comparisons manually.
- [ ] Do not continue `N1`; the focal-style variant lost cleanly to baseline.
- [ ] Do not promote `N2` to the main branch; only keep calibration utilities if
  they help later `N3`-style runs.
- [ ] Extend `N3` with GO initialization or ontology-aware regularization.
- [ ] Run one longer `N3` confirmation job with the same seed.
- [ ] Run one second-seed `N3` stability check before adopting it as the new
  default graph recipe.
- [ ] If `N2` logic remains useful, add a reusable post-hoc calibration and threshold-selection
  utility.
- [x] Implement graph-side prediction bundle export and run score-level
  sequence-plus-graph fusion for `MFO`.

Observed first-batch outcome summary:

- `N1`: `CCO val/test = 0.5591 / 0.5609`, `MFO val/test = 0.4452 / 0.4513`
- `N2`: `CCO val/test = 0.5637 / 0.5652`, `MFO val/test = 0.4505 / 0.4558`
- `N3`: `CCO val/test = 0.5822 / 0.5843`, `MFO val/test = 0.4517 / 0.4580`
- raw baseline: `CCO val/test = 0.5635 / 0.5647`, `MFO val/test = 0.4522 / 0.4574`

Observed `N4` MFO fusion outcome summary:

- sequence-only graph-vocab branch: `val/test = 0.4513 / 0.4557`
- raw-checkpoint fusion, val-selected best: `g0p2_s0p8`, `val/test = 0.4518 / 0.4566`
- raw-checkpoint fusion, test-best diagnostic: `g0p9_s0p1`, `test = 0.4575`
- N3 MFO graph-only export: `val/test = 0.4517 / 0.4580`
- N3 MFO fusion, val-selected best: `g0p8_s0p2`, `val/test = 0.4529 / 0.4579`
- N3 MFO fusion, test-best diagnostic: `g0p9_s0p1`, `test = 0.4582`
- interpretation: N3+sequence fusion gives a small validation gain over N3
  graph-only, but the test gain is negligible; treat this as weak
  complementarity rather than a major `N4` win.
