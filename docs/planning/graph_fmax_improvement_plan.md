# Graph Fmax Improvement Plan

Status: completed and retained as a historical record. The next active plan is
documented in `docs/planning/graph_significant_improvement_exploration_plan.md`.

Date: 2026-04-20

## 1. Goal

This plan defines the next experiment round for improving graph-model Fmax on CAFA5.

The plan is intentionally narrow:

- scope: graph models only
- aspects in scope: `CCO` and `MFO`
- aspect excluded: `BPO`
- primary optimization target: validation-selected `Fmax`

The current strongest full-run reference is the raw full graph baseline:

- `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`

This plan treats that run as the comparison anchor.

## 2. Why This Plan Is Narrow

The completed full runs already established three important facts:

1. the raw full graph baseline is the strongest current result for both `CCO` and `MFO`
2. the normalized full graph run did not beat the raw baseline
3. the larger tuned recipe also did not beat the raw baseline

Because of that, the next step should not be another broad mixed-parameter tuning attempt.

Instead, the next round should:

- keep the baseline recipe intact
- change only one major variable at a time
- make it possible to attribute any Fmax gain to a specific cause

## 3. Baseline Recipe

All planned experiments in this round start from the same raw-style baseline recipe unless the row explicitly changes one field.

Baseline recipe:

- loss: `bce`
- `hidden_dim = 128`
- `dropout = 0.20`
- `lr = 0.001`
- `weight_decay = 0.0001`
- scheduler: `none`
- `min_term_frequency = 20`
- aspects: `CCO`, `MFO`
- split: official full split
- seed: `2026`
- selection metric: `val_fmax`

## 4. Planned Experiment Matrix

### 4.1 Round 1 Experiments

| ID | Purpose | Loss | Hidden Dim | Dropout | LR | Weight Decay | Scheduler | Main Question |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `E0` | control rerun | `bce` | 128 | 0.20 | 0.0010 | 0.0001 | `none` | do we reproduce the baseline cleanly in the current environment? |
| `E1` | lower-LR ablation | `bce` | 128 | 0.20 | 0.0007 | 0.0001 | `none` | does a slightly smaller learning rate improve Fmax? |
| `E2` | lower-LR ablation | `bce` | 128 | 0.20 | 0.0005 | 0.0001 | `none` | was the baseline learning rate too aggressive? |
| `E3` | capacity ablation | `bce` | 192 | 0.20 | 0.0010 | 0.0001 | `none` | does modest extra capacity help? |
| `E4` | capacity ablation | `bce` | 256 | 0.20 | 0.0010 | 0.0001 | `none` | does larger capacity help when isolated from the tuned recipe? |
| `E5` | loss-only ablation | `weighted_bce` | 128 | 0.20 | 0.0010 | 0.0001 | `none` | does weighted BCE help when it is the only major change? |

### 4.2 Why These Six Experiments

This set is designed to answer the highest-value questions first:

- whether the baseline is still reproducible
- whether learning rate is the main reason the tuned run underperformed
- whether model capacity helps on its own
- whether weighted BCE is useful when tested cleanly

This round avoids:

- re-running normalization immediately
- mixing scheduler, regularization, capacity, and loss changes in a single recipe
- spending compute on `BPO`

## 5. Execution Order

If compute is limited, the recommended execution order is:

1. `E0`
2. `E1`
3. `E3`
4. `E5`
5. `E2`
6. `E4`

Rationale:

- `E0` refreshes the comparison anchor
- `E1` tests the most plausible optimization change first
- `E3` checks whether capacity is a bottleneck
- `E5` isolates the weighted-BCE question
- `E2` and `E4` extend the same two directions if the earlier results justify them

## 6. Evaluation Rule

The decision rule for this round should be fixed before running the jobs.

Primary selection rule:

- choose by `best val_fmax`

Secondary confirmation rule:

- compare `best test_fmax`

Tie-break guidance:

- if validation Fmax differences are very small, prefer the run with more stable training behavior
- do not replace the main recipe only because macro-F1 is higher

Metrics to record for every run:

- best validation Fmax
- best test Fmax
- best validation threshold
- final validation Fmax
- final test Fmax
- micro-F1
- macro-F1
- per-epoch loss and Fmax curves

## 7. Interpretation Rules

After Round 1, interpret outcomes as follows.

### 7.1 If a lower-LR run wins

If `E1` or `E2` beats the raw baseline:

- treat learning rate as the current most promising direction
- keep the winning LR
- move to a smaller second-round search near the winning value

Possible follow-up:

- `0.0008`
- `0.0006`
- `0.0004`

### 7.2 If a capacity run wins

If `E3` or `E4` beats the raw baseline:

- treat model capacity as a real gain source
- keep the winning hidden dimension
- next test dropout around that winning configuration

Possible follow-up:

- `dropout = 0.15`
- `dropout = 0.20`
- `dropout = 0.25`

### 7.3 If weighted BCE wins

If `E5` beats the raw baseline:

- keep weighted BCE as a viable direction
- do not immediately reintroduce all tuned-run changes
- instead test weighted BCE with one additional variable at a time

Possible follow-up:

- weighted BCE + best LR
- weighted BCE + best hidden dim

### 7.4 If nothing wins

If none of `E1`–`E5` beats the raw baseline:

- keep the raw baseline as the official graph recipe
- conclude that the obvious raw-style hyperparameter changes did not improve Fmax
- shift the next discussion toward richer model inputs or evaluation improvements instead of more local tuning

## 8. What Not To Do In This Round

This round should explicitly avoid the following:

- another large mixed tuned recipe
- immediate re-run of normalized features under the same recipe
- any `BPO` full training run
- introducing multiple new changes without a clean attribution path

These are deferred because they either already underperformed or make interpretation difficult.

## 9. Deliverables

At the end of this round, the team should have:

- one clean comparison table for `E0`–`E5`
- one selected best recipe relative to the raw baseline
- one short interpretation note explaining which direction helped and which did not

If a new best recipe is found, the next round should be:

- targeted refinement around that recipe
- then multi-seed confirmation (`2026`, `2027`, `2028`)

## 10. Summary

This plan keeps the next Fmax-improvement round disciplined and interpretable.

The key idea is simple:

- treat the raw full graph baseline as the current best model
- test only a few targeted variations around it
- identify one real gain source before attempting broader model changes

The purpose of this round is not to try everything. The purpose is to learn exactly which change, if any, can actually beat the current best full graph baseline on `CCO` and `MFO`.

## 11. Outcome Summary

This round has now been completed.

Observed outcome summary:

- `E0` successfully refreshed the raw-style comparison anchor.
- `E1` produced an incomplete `CCO` run because of a dataloader/cache failure, so
  it is not treated as a clean final comparison for that aspect.
- `E2` showed only very small movement relative to baseline.
- `E3` and `E4` did not provide evidence that larger hidden dimensions improve
  `Fmax`.
- `E5` produced only a small positive signal on `MFO`, but not one strong enough
  to justify continuing `weighted_bce` as the main next-step direction.

Decision after this round:

- keep the raw baseline as the current reference recipe
- stop spending the next round on raw-style local tuning alone
- move the next exploration round toward objective design, label-aware modeling,
  and richer sequence-structure integration
