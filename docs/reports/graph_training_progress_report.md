# Graph Training Progress Report

Date: 2026-04-22

## 1. Scope

This report summarizes the graph-model work completed so far for the CAFA5 project.

This document focuses only on the graph training line owned by our team:

- aspects in scope: `CCO` and `MFO`
- aspect out of scope for formal graph experiments: `BPO`
- current exclusion reason for `BPO`: the full graph run is not stable under the available memory budget

The goal of this report is to record:

- what graph experiments have already been run
- which full-training results are currently the strongest
- what we learned from the unsuccessful improvement attempts
- which next-step directions are now prioritized

This report does not propose the final next-step plan yet. It is intended to establish a clean shared record before deciding the next experiments.

## 2. Experimental Setup Summary

Across the completed full graph runs, the common setup is:

- graph-based training on the CAFA5 structure-available cohort
- one model per GO aspect
- full split manifests prepared in the graph cache pipeline
- `min_term_frequency = 20`
- main metrics tracked per epoch: loss, micro-F1, macro-F1, and Fmax

The main full-training runs discussed in this report are:

1. raw full graph baseline
2. tuned full graph run
3. normalized-feature full graph run

## 3. Completed Experiments

### 3.1 Raw Full Graph Baseline

Run directory:

- `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`

Main configuration:

- graph features: raw
- loss: BCE-style default
- `hidden_dim = 128`
- `dropout = 0.2`
- `lr = 0.001`
- `weight_decay = 0.0001`

Observed outcome:

- `CCO`: success
- `MFO`: success
- `BPO`: failed after the first epoch

Best full-run results from this baseline:

| Aspect | Best Validation Fmax | Best Test Fmax | Notes |
| --- | ---: | ---: | --- |
| `CCO` | `0.5635` | `0.5647` | strongest full-run result so far |
| `MFO` | `0.4522` | `0.4574` | strongest full-run result so far |
| `BPO` | `0.2647` | `0.2655` | run failed; not used as a formal result |

Interpretation:

- This run remains the strongest full graph baseline for both `CCO` and `MFO`.
- Any new graph experiment should be compared against this run first.

### 3.2 Tuned Full Graph Run

Run directory:

- `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_tuned_pyg_mtf20_33275343`

Main configuration changes relative to the raw baseline:

- loss changed to `weighted_bce`
- `hidden_dim = 256`
- `dropout = 0.3`
- `lr = 0.0003`
- `weight_decay = 0.0005`
- `lr_scheduler = plateau`
- checkpointing explicitly tied to `val_fmax`

Observed outcome:

- `CCO`: success
- `MFO`: success
- `BPO`: failed before a usable `summary.json` was produced

Best full-run results from this tuned run:

| Aspect | Best Validation Fmax | Best Test Fmax | Notes |
| --- | ---: | ---: | --- |
| `CCO` | `0.5567` | `0.5584` | lower than the raw baseline |
| `MFO` | `0.4450` | `0.4513` | lower than the raw baseline |
| `BPO` | n/a | n/a | failed |

Interpretation:

- The tuned recipe did not improve Fmax for either `CCO` or `MFO`.
- Notebook analysis indicates that this run improved macro-F1 more than it improved Fmax, suggesting that the tuning choices pushed the model toward better rare-label balance but not better validation-selected Fmax.

### 3.3 Normalized-Feature Full Graph Run

Run directory:

- `/global/scratch/users/bensonli/cafa5_outputs/graph_cache_normalized_features/training_runs/cco_mfo_parallel_20260415_1906`

Main configuration:

- graph features: normalized structural features
- loss: `bce`
- `hidden_dim = 128`
- `dropout = 0.2`
- `lr = 0.001`
- `weight_decay = 0.0001`
- `lr_scheduler = none`
- aspects run in parallel: `CCO`, `MFO`

Observed outcome:

- `CCO`: success
- `MFO`: success

Best full-run results from the normalized run:

| Aspect | Best Validation Fmax | Best Test Fmax | Notes |
| --- | ---: | ---: | --- |
| `CCO` | `0.5605` | `0.5623` | slightly below the raw baseline |
| `MFO` | `0.4491` | `0.4544` | slightly below the raw baseline |

Interpretation:

- Feature normalization did not beat the raw full graph baseline.
- The gap is small, so normalization is not clearly harmful, but it is also not a clear win under the current recipe.

### 3.4 Small-Scale Debug and Smoke Experiments

Local small runs were also completed under `output/small_experiment_*`.

These experiments were useful for:

- checking whether the training loop worked end-to-end
- testing normalization cheaply
- verifying that summary writing and metric tracking behaved as expected

Important limitation:

- these runs are small debug experiments and should not replace the full-training comparison table

The strongest local small experiment was:

- `output/small_experiment_mfo_mtf20_n120_normalized`
- `val_fmax = 0.2136`
- `test_fmax = 0.1203`

This result suggested that normalization might help, which motivated the full normalized run. However, that improvement did not carry over to the full-data setting.

## 4. Full-Run Comparison Table

The main full graph comparison for the two in-scope aspects is:

| Run | Aspect | Best Validation Fmax | Best Test Fmax | Outcome |
| --- | --- | ---: | ---: | --- |
| raw baseline | `CCO` | `0.5635` | `0.5647` | best overall so far |
| normalized | `CCO` | `0.5605` | `0.5623` | slightly worse than raw |
| tuned | `CCO` | `0.5567` | `0.5584` | worse than raw |
| raw baseline | `MFO` | `0.4522` | `0.4574` | best overall so far |
| normalized | `MFO` | `0.4491` | `0.4544` | slightly worse than raw |
| tuned | `MFO` | `0.4450` | `0.4513` | worse than raw |

Current best full-run checkpoints:

- `CCO`: raw baseline
- `MFO`: raw baseline

## 5. Completed Targeted Ablation Round (`E0`-`E5`)

After the baseline, normalized, and tuned full runs, the team completed a
focused ablation round around the raw baseline on `CCO` and `MFO`.

This round tested:

- `E0`: baseline control rerun
- `E1`: lower learning rate `0.0007`
- `E2`: lower learning rate `0.0005`
- `E3`: `hidden_dim = 192`
- `E4`: `hidden_dim = 256`
- `E5`: `weighted_bce`

Main findings from that round:

- `CCO` was essentially flat across the clean completed variants, with no
  meaningful gain over the raw control.
- `MFO` showed only small movement across variants.
- `weighted_bce` produced a positive signal on `MFO`, but the gain was not large
  enough to justify continuing that line as the main next-step strategy.
- `E1/CCO` was interrupted by a dataloader/cache-related failure, so that result
  is not treated as a clean final comparison.

Interpretation:

- the obvious local raw-style hyperparameter changes have now been tested
- none of them produced a convincing step change in `Fmax`
- the next round should move away from narrow local tuning and toward more
  structural changes in objective, architecture, and modality use

## 6. What We Learned So Far

### 6.1 The Raw Baseline Is Stronger Than Expected

The raw full graph run is not a placeholder baseline. It is already a competitive full-data result for both `CCO` and `MFO`.

This matters because later experiments are not improving a weak starting point. They are trying to beat a baseline that already performs well.

### 6.2 Normalization Did Not Solve the Main Bottleneck

Normalization improved some small debug runs, but the same pattern did not hold in the full-data setting.

The most likely interpretation is:

- raw feature scale was not the main limitation of the full model
- normalization changed feature magnitudes without adding new signal
- any training-stability gain from normalization was too small to outweigh the loss of the original raw-feature geometry

### 6.3 The Tuned Recipe Improved the Wrong Objective

The tuned run changed many things at once:

- weighted BCE
- larger hidden dimension
- stronger dropout
- lower learning rate
- stronger weight decay
- plateau scheduling

This recipe appears to have shifted the model toward better rare-label treatment and slightly better macro behavior, but not better validation-selected Fmax.

From the notebook analysis:

- `CCO`: tuned test Fmax decreased by about `0.0096`
- `MFO`: tuned test Fmax decreased by about `0.0095`
- macro-F1 improved, but Fmax did not

This suggests that the tuned recipe was not aligned with the primary selection metric used for the graph study.

### 6.4 BPO Is Not Currently a Reliable Full-Run Target

Both full graph attempts failed on `BPO`.

For the current project stage, this means:

- `BPO` should not be used in the main graph results table
- `CCO` and `MFO` should remain the formal graph focus
- any future `BPO` work would require a separate memory- or systems-focused effort

### 6.5 Local Raw-Style Tuning Has Limited Headroom

The completed `E0`-`E5` round strongly suggests that the current bottleneck is
not well described by small changes in:

- learning rate
- hidden dimension
- a single BCE reweighting variant

The most likely interpretation is that the graph line is now closer to a
representation, label-modeling, or objective-design bottleneck than to a simple
optimizer bottleneck.

## 7. Current Status

At the moment, the graph line is in the following state:

- full graph baseline exists and is usable
- the normalized and tuned full-data alternatives have been completed
- the focused `E0`-`E5` ablation round has also been completed
- none of the completed alternatives has produced a convincing improvement over
  the raw full graph baseline on `Fmax`
- the current best full graph results are already known for `CCO` and `MFO`
- the first `N1`/`N2`/`N3` significant-improvement graph batch has completed
- `N3` is now the first graph-side direction in this project to beat the raw
  baseline clearly on `CCO`
- the first `N4` sequence-side matched-cohort run has completed and produced
  graph-vocab prediction bundles for `MFO`
- two `N4` MFO score-level fusion passes have completed: one against the raw
  available graph checkpoint and one against the N3 MFO graph checkpoint

In other words, the graph work is not blocked by lack of a baseline. The graph work is now at the stage of targeted improvement and better evaluation.

### 7.1 Active Savio Jobs

Current active or recently completed jobs as of the latest check:

| Direction | Job ID | Slurm Name | Status | Run Name | Notes |
| --- | ---: | --- | --- | --- | --- |
| `N1` | `33706651` | `cafa5_n1_focal_bce` | `COMPLETED` | `sigimp_n1_focal_bce_20260422_172916` | focal-style BCE objective; lost to baseline on both aspects |
| `N2` | `33706652` | `cafa5_n2_logit_adjust` | `COMPLETED` | `sigimp_n2_logit_adjust_20260422_172916` | train-prior logit adjustment; nearly neutral overall |
| `N3` | `33706654` | `cafa5_n3_label_dot` | `COMPLETED` | `sigimp_n3_label_dot_20260422_172916` | label-dot scorer; strong `CCO` win |
| `N4 sequence` | `33706500` | `cafa5_n4_mfo` | `COMPLETED` | `sigimp_n4_seq_small_mfo_mlp_20260422_172448` | first sequence-side MFO run; superseded by graph-vocab sequence bundle |
| `N4 fusion` | n/a | local/slurm fusion | `COMPLETED` | `raw_mfo_x_seq_graph_vocab_20260424_222500` | first score-fusion pass using raw available graph `best.pt` |
| `N4 fusion` | `33740526` | `cafa5_n4_fusion` | `COMPLETED` | `n3_mfo_x_seq_graph_vocab_20260424_222500` | second score-fusion pass using N3 MFO epoch-3 `best.pt` |
| `N3 confirm` | `33741105` | `cafa5_n3_confirm_long` | `RUNNING` | `sigimp_n3_confirm_long_20260425_n3_confirm` | same-seed longer confirmation; `EPOCHS=8`, `CCO MFO` |

The `N1`/`N2`/`N3` batch uses the fixed reference setup on `CCO` and `MFO`:

- `hidden_dim = 128`
- `dropout = 0.20`
- `lr = 0.001`
- `weight_decay = 0.0001`
- `EPOCHS = 5`
- `CHECKPOINT_METRIC = val_fmax`
- `NORMALIZE_FEATURES = 0`
- `seed = 2026`
- `2x A40` per job on `savio3_gpu`

The only intended changes in that batch are:

- `N1`: `LOSS_FUNCTION=focal_bce`
- `N2`: `LOGIT_ADJUSTMENT=train_prior`
- `N3`: `MODEL_HEAD=label_dot`

Observed outcomes against the raw baseline
`/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`:

| Direction | Aspect | Best Validation Fmax | Delta vs Raw | Best Test Fmax | Delta vs Raw | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| raw baseline | `CCO` | `0.5635` | `0.0000` | `0.5647` | `0.0000` | reference |
| `N1` | `CCO` | `0.5591` | `-0.0044` | `0.5609` | `-0.0038` | worse |
| `N2` | `CCO` | `0.5637` | `+0.0002` | `0.5652` | `+0.0005` | effectively tied |
| `N3` | `CCO` | `0.5822` | `+0.0187` | `0.5843` | `+0.0196` | clear win |
| raw baseline | `MFO` | `0.4522` | `0.0000` | `0.4574` | `0.0000` | reference |
| `N1` | `MFO` | `0.4452` | `-0.0070` | `0.4513` | `-0.0061` | worse |
| `N2` | `MFO` | `0.4505` | `-0.0017` | `0.4558` | `-0.0016` | slightly worse |
| `N3` | `MFO` | `0.4517` | `-0.0005` | `0.4580` | `+0.0006` | essentially tied; slight test gain |

Result summary:

- `N1` did not improve either aspect and should not be prioritized next.
- `N2` showed only tiny movement on `CCO` and small regression on `MFO`, so it
  is not strong enough to become the main branch by itself.
- `N3` is the first result in this significant-improvement round that clearly
  beats the raw graph baseline, and it does so on the higher-signal `CCO`
  branch with a substantial margin.

Observed `N4` MFO score-fusion outcomes:

| Fusion Run | Graph Branch | Val-Selected Weight | Validation Fmax | Test Fmax at Same Weight | Test-Best Diagnostic |
| --- | --- | --- | ---: | ---: | ---: |
| sequence only | n/a | `g0p0_s1p0` | `0.4513` | `0.4557` | `0.4557` |
| raw fusion | raw available `best.pt` | `g0p2_s0p8` | `0.4518` | `0.4566` | `0.4575` |
| N3 fusion | N3 MFO epoch-3 `best.pt` | `g0p8_s0p2` | `0.4529` | `0.4579` | `0.4582` |

Interpretation:

- The raw fusion pass is useful as a reference, but its graph branch used the
  available raw `best.pt` export, whose graph-only row is `0.4499 / 0.4571` in
  this matched-bundle evaluation.
- The N3 fusion pass is closer to the intended "best available graph +
  sequence" `N4` comparison because N3 MFO checkpoint selection is normal
  (`best_epoch = 3`).
- N3+sequence fusion improves validation Fmax slightly over N3 graph-only, but
  the test-side change is tiny; this is weak complementarity rather than a
  decisive multimodal gain.

## 8. Open Questions Before the Next Round

The key questions that remain open are:

1. How much of the `N3` gain comes from the label-aware scorer itself versus
   the exact label embedding parameterization and threshold choice?
2. Can GO hierarchy information be injected into the `N3` branch so the model
   learns a better label space rather than treating GO terms as independent IDs?
3. Does a cleaner sequence-plus-graph fusion strategy outperform the current
   graph-only baseline on the matched structure-available cohort, especially if
   the graph branch uses the `N3` head instead of the flat baseline head?
4. Is there any cheap post-hoc calibration utility worth keeping from `N2`, or
   was its effect too small to justify a dedicated branch right now?

## 9. Next Active Exploration Directions

The next round should focus on the following three directions:

1. Extend the winning `N3` label-aware scorer rather than continuing flat-head
   objective tuning.
2. Treat the completed N3 MFO `N4` fusion as the first real score-level
   sequence-plus-graph comparison, then decide whether a broader fusion search
   is worth the cost.
3. Add better evaluation and comparison utilities so future result collection is
   faster and less manual.

These directions are prioritized because the current evidence now favors
label-aware modeling and multimodal integration over more local tuning around
the flat raw baseline.

## 10. Direction After Current Jobs Finish

The next implementation direction should now be chosen from the observed
results rather than from another manual hyperparameter sweep.

Decision after the first `N1`-`N3` batch:

- `N1` should stop here; the focal-style loss variant is not competitive on the
  fixed full-data setup.
- `N2` should not become the main branch; at most, keep its logic available as
  an optional post-hoc utility if later `N3`-style runs show calibration drift.
- `N3` should become the primary graph-side branch. The next graph change
  should be GO-term initialization, ontology-aware regularization, or another
  stronger label-aware scorer built on the same idea.
- `N4` remains important because it tests modality complementarity rather than
  graph-head design; fusion work should continue in parallel with `N3`
  refinement.

Implementation tasks to queue after result collection:

- add a comparison helper that reads the `sigimp_*` run directories and produces
  one `N1`-`N4` table
- keep the graph-side prediction-bundle export path and consider a smaller
  follow-up fusion search only if calibration or weighting can be justified
- add one `N3` follow-up branch with GO-term initialization or ontology-aware
  label regularization
- run one longer `N3` confirmation job with the same seed, then a second-seed
  stability check before treating it as the official new graph recipe

## 11. Summary

The graph team has already completed the main baseline, the normalized and tuned
full-data alternatives, the focused `E0`-`E5` ablation round, and the first
`N1`-`N3` significant-improvement batch.

The main findings so far are:

- the raw full graph baseline remains the main reference, but `N3` is now the
  strongest observed `CCO` graph result
- the normalized full run did not beat the raw baseline
- the tuned full run also did not beat the raw baseline
- the focused `E0`-`E5` ablation round did not reveal a strong enough local
  tuning gain to justify continuing that direction
- `BPO` remains out of scope for formal graph reporting because the full graph runs were not stable enough under current memory constraints
- `N1` lost to the raw baseline on both aspects
- `N2` produced only negligible movement and did not establish calibration as
  the main bottleneck
- `N3` clearly improved `CCO` and is now the leading graph-side follow-up
- `N4` has completed one raw-checkpoint fusion pass and one N3 MFO fusion pass;
  the N3 fusion shows only a small validation-side gain

The next discussion should therefore focus on objective design, label-aware
modeling, sequence-structure integration, and `N3` confirmation, rather than
another round of local baseline tuning.
