# Graph Significant Improvement Exploration Plan

Date: 2026-04-22

## 1. Goal

This plan defines the next graph-model exploration round after the completed
`E0`-`E5` ablation study.

The purpose of this round is different from the previous one:

- previous round goal: test whether simple local changes around the raw baseline
  can beat it
- current round goal: test higher-leverage changes that could produce a more
  meaningful improvement in `Fmax`

Scope:

- model family: graph-focused training for CAFA5
- aspects in scope: `CCO`, `MFO`
- aspect excluded from the formal plan: `BPO`
- primary optimization target: validation-selected `Fmax`

Reference baseline:

- `/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`

## 2. Why This Round Is Different

The completed evidence now says the following:

1. the raw graph baseline is already strong
2. normalized features did not beat it
3. the earlier mixed tuned recipe did not beat it
4. the focused `E0`-`E5` local ablation round did not produce a convincing gain
5. `weighted_bce` showed only a weak signal and is not prioritized as the main
   next-step direction

Because of that, the next round should not spend most of its budget on:

- another local learning-rate sweep
- another hidden-dimension sweep
- another `weighted_bce`-centered branch

Instead, the next round should test changes that target a different bottleneck:

- objective mismatch with `Fmax`
- flat label modeling
- missing GO-structure bias in the model
- underused complementarity between sequence and structure

## 3. Fixed Reference Setup

Unless a row explicitly changes one of these components, the default reference
setup remains:

- loss: `bce`
- `hidden_dim = 128`
- `dropout = 0.20`
- `lr = 0.001`
- `weight_decay = 0.0001`
- scheduler: `none`
- `min_term_frequency = 20`
- split: official full split
- seed: `2026`
- aspects: `CCO`, `MFO`
- selection metric: `val_fmax`

## 4. Planned Exploration Matrix

### 4.1 Direction `N1`: Better Multi-Label Loss

| ID | Direction | Core Change | Main Question |
| --- | --- | --- | --- |
| `N1` | `ASL` or focal-style multi-label loss | replace plain BCE with a loss designed for long-tail multi-label classification | is the main bottleneck an inadequate training objective for long-tail labels? |

Why this is included:

- it still addresses imbalance, but without continuing the weakly supported
  `weighted_bce` path
- it is a higher-information test than another raw BCE reweight sweep

Expected payoff:

- most likely to help `MFO`
- plausible upside for `CCO` if rare-label decision boundaries are the issue

### 4.2 Direction `N2`: Logit Adjustment and Calibration

| ID | Direction | Core Change | Main Question |
| --- | --- | --- | --- |
| `N2` | logit adjustment / calibration | apply prior-aware logit correction or post-hoc calibration targeted at `val_fmax` | are the current logits usable but poorly calibrated for validation-selected `Fmax`? |

Why this is included:

- it is low-cost compared with architecture changes
- it targets the mismatch between training loss and the metric actually used for
  model selection

Expected payoff:

- moderate upside with relatively low engineering risk
- useful even if it does not become the final strongest direction, because it
  clarifies whether the bottleneck is representational or calibration-related

### 4.3 Direction `N3`: Label-Aware Protein-to-GO Scoring

| ID | Direction | Core Change | Main Question |
| --- | --- | --- | --- |
| `N3` | protein embedding + GO embedding scorer | replace the flat output head with a label-aware scorer between protein embeddings and GO term embeddings | is the main bottleneck the flat label head rather than the graph encoder itself? |

Why this is included:

- it gives the model a stronger inductive bias for multi-label protein function
  prediction
- it creates a natural path toward ontology-aware label representations later

Expected payoff:

- higher upside than local tuning
- especially valuable if current head labels are learned reasonably well but
  medium-tail labels do not generalize

### 4.4 Direction `N4`: Sequence + Graph Late Fusion

| ID | Direction | Core Change | Main Question |
| --- | --- | --- | --- |
| `N4` | late fusion of sequence and graph models | train separate sequence and graph branches and combine them at score level | are sequence and structure signals complementary enough to beat graph-only performance when fused cleanly? |

Why this is included:

- it avoids the implementation complexity of early cross-attention fusion
- it is often the safest path to capture complementary modality signal

Expected payoff:

- one of the best chances for a meaningful step up if a usable sequence branch is
  available
- especially attractive if graph-only and sequence-only errors are not strongly
  aligned

## 5. Suggested Execution Order

If compute or engineering time is limited, the recommended order is:

1. `N2`
2. `N1`
3. `N3`
4. `N4`

Rationale:

- `N2` is the cheapest way to test whether calibration is the main issue
- `N1` is the cheapest way to test whether the core training objective is wrong
- `N3` is the first architecture-level test
- `N4` depends more on sequence branch availability and is the largest
  integration step

## 5.1 Active Execution Status

The first execution batch for this plan has been submitted.

| Direction | Job ID | Status | Run Name | Core Change |
| --- | ---: | --- | --- | --- |
| `N1` | `33706651` | `COMPLETED` | `sigimp_n1_focal_bce_20260422_172916` | `LOSS_FUNCTION=focal_bce` |
| `N2` | `33706652` | `COMPLETED` | `sigimp_n2_logit_adjust_20260422_172916` | `LOGIT_ADJUSTMENT=train_prior` |
| `N3` | `33706654` | `COMPLETED` | `sigimp_n3_label_dot_20260422_172916` | `MODEL_HEAD=label_dot` |
| `N4` | `33706500` | `PENDING` | `sigimp_n4_seq_small_mfo_mlp_20260422_172448` | sequence-side `MFO` matched-cohort run |

Current interpretation:

- `N1`/`N2`/`N3` were controlled graph-side tests against the fixed raw setup.
- `N4` is not yet full graph-plus-sequence fusion; it is the first sequence-side
  producer needed for the fusion path.
- `N2` is intentionally treated as a metric/logit transform test, not a new
  representation-learning method.
- `N3` is now the first clear graph-side winner from this branch because it
  improved `CCO` strongly while holding `MFO` roughly flat.

Observed first-batch results against raw baseline
`/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/full_graph_pyg_mtf20_33234089`:

| Direction | Aspect | Best Validation Fmax | Delta vs Raw | Best Test Fmax | Delta vs Raw |
| --- | --- | ---: | ---: | ---: | ---: |
| raw baseline | `CCO` | `0.5635` | `0.0000` | `0.5647` | `0.0000` |
| `N1` | `CCO` | `0.5591` | `-0.0044` | `0.5609` | `-0.0038` |
| `N2` | `CCO` | `0.5637` | `+0.0002` | `0.5652` | `+0.0005` |
| `N3` | `CCO` | `0.5822` | `+0.0187` | `0.5843` | `+0.0196` |
| raw baseline | `MFO` | `0.4522` | `0.0000` | `0.4574` | `0.0000` |
| `N1` | `MFO` | `0.4452` | `-0.0070` | `0.4513` | `-0.0061` |
| `N2` | `MFO` | `0.4505` | `-0.0017` | `0.4558` | `-0.0016` |
| `N3` | `MFO` | `0.4517` | `-0.0005` | `0.4580` | `+0.0006` |

## 6. Evaluation Rules

Primary rule:

- choose by `best val_fmax`

Secondary rule:

- compare `best test_fmax`

Required outputs for every run:

- best validation Fmax
- best test Fmax
- best validation threshold
- final validation Fmax
- final test Fmax
- micro-F1
- macro-F1
- per-epoch curves when training is iterative
- run config and explicit variant label

Additional interpretation guidance:

- if a method improves only macro-F1 but not `Fmax`, do not treat it as the new
  main direction
- if a method wins only on one aspect, keep aspect-specific conclusions rather
  than forcing one global recipe

## 7. Success Criteria

This round should be considered successful if it produces at least one of the
following:

- a clear `Fmax` gain that is larger than the very small movements seen in
  `E0`-`E5`
- evidence that the current bottleneck is calibration rather than representation
- evidence that a label-aware scorer outperforms the flat head
- evidence that sequence and graph signals are materially complementary

## 7.1 Result-Driven Follow-Up Rules

When the active jobs finish, choose the next implementation branch using these
rules:

| Result Pattern | Follow-Up Direction | Implementation Action |
| --- | --- | --- |
| `N1` beats raw baseline on `val_fmax` | objective-side improvement | run a small `focal_gamma`/`focal_alpha` sensitivity check and keep the graph encoder fixed |
| `N2` beats raw baseline on `val_fmax` | calibration-side improvement | add post-hoc temperature and threshold selection utilities before changing architecture |
| `N3` beats raw baseline on `val_fmax` | label-aware modeling | add GO-term initialization or ontology-aware regularization to the label scorer |
| `N4` sequence branch is competitive on matched `MFO` | multimodal fusion | implement graph-side prediction-bundle export and run real score-level fusion |
| no direction beats raw baseline | fusion and ontology should take priority | stop local graph tuning and focus on `N4` fusion plus ontology-aware labels |

After the first batch, the rule application is:

- `N1`: do not continue
- `N2`: keep optional, but do not treat as the primary branch
- `N3`: make this the main graph-side follow-up
- `N4`: continue in parallel because it answers a different multimodal question

Do not expand to `BPO` until at least one `CCO`/`MFO` direction shows a clear
gain or the memory issue is addressed separately.

## 8. What Not To Prioritize In This Round

This round explicitly deprioritizes:

- another hidden-dimension sweep
- another learning-rate sweep around the raw baseline
- another `weighted_bce` exploration branch
- immediate `BPO` expansion before the main `CCO`/`MFO` line improves

## 9. Deliverables

At the end of this round, the team should have:

- one comparison table for `N1`-`N4`
- one short note identifying whether the main gain source came from objective,
  calibration, label-aware modeling, or multimodal fusion
- one updated recommendation for the next implementation round

Near-term result-processing deliverables:

- one status/result table keyed by Slurm job ID and run name
- one comparison helper for `sigimp_*` directories
- one decision note selecting the next branch before launching any larger
  confirmation jobs

## 10. Summary

The previous graph rounds established that the raw baseline is strong and that
simple local tuning has limited headroom.

Therefore, the next serious improvement attempt should focus on four directions:

1. better multi-label loss (`N1`)
2. logit adjustment and calibration (`N2`)
3. label-aware protein-to-GO scoring (`N3`)
4. sequence-plus-graph late fusion (`N4`)

This round is designed to search for a real gain source rather than another
small local variation of the current baseline.
