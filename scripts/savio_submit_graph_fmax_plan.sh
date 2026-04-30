#!/bin/bash
# Submit the focused CCO/MFO Fmax-improvement plan on Savio.
#
# This script targets the E0-E5 experiments defined in
# docs/planning/graph_fmax_improvement_plan.md.
#
# Default mode is dry-run. Pass --submit to actually submit.

set -euo pipefail

MODE="dry-run"
if [[ "${1:-}" == "--submit" ]]; then
  MODE="submit"
elif [[ "${1:-}" == "--dry-run" || "${1:-}" == "" ]]; then
  MODE="dry-run"
else
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/savio_train_full_graph.sh}"
RUN_PREFIX="${RUN_PREFIX:-fmax_plan}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PLAN="${PLAN:-priority}"
ACCOUNT="${ACCOUNT:-ic_chem242}"

# Target the fastest FCA-accessible GPU by default.
# Override PARTITION/GPU_TYPE/QOS/GPUS_PER_JOB/CPUS_PER_TASK if needed.
PARTITION="${PARTITION:-savio3_gpu}"
if [[ -z "${GPU_TYPE+x}" ]]; then
  GPU_TYPE="A40"
else
  GPU_TYPE="${GPU_TYPE}"
fi
QOS="${QOS:-a40_gpu3_normal}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
REQUESTED_WALLTIME="${REQUESTED_WALLTIME:-72:00:00}"

GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-/global/scratch/users/$USER/cafa5_outputs/graph_cache}"
SPLIT_DIR="${SPLIT_DIR:-$GRAPH_CACHE_DIR/splits}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing training script: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! "$GPUS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$GPUS_PER_JOB" == "0" ]]; then
  echo "GPUS_PER_JOB must be a positive integer: $GPUS_PER_JOB" >&2
  exit 2
fi
if [[ ! "$CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [[ "$CPUS_PER_TASK" == "0" ]]; then
  echo "CPUS_PER_TASK must be a positive integer: $CPUS_PER_TASK" >&2
  exit 2
fi

build_gres() {
  if [[ -n "$GPU_TYPE" && "$GPU_TYPE" != "none" ]]; then
    printf 'gpu:%s:%s\n' "$GPU_TYPE" "$GPUS_PER_JOB"
  else
    printf 'gpu:%s\n' "$GPUS_PER_JOB"
  fi
}

submit_experiment() {
  local exp_id="$1"
  local loss_function="$2"
  local hidden_dim="$3"
  local dropout="$4"
  local lr="$5"
  local weight_decay="$6"
  local seed="$7"
  local run_name="${RUN_PREFIX}_${exp_id}_${RUN_STAMP}"
  local job_name="cafa5_${exp_id}"
  local gpu_ids
  local gres

  gpu_ids="$(seq -s ' ' 0 "$((GPUS_PER_JOB - 1))")"
  gres="$(build_gres)"

  local env_args=(
    "RUN_NAME=$run_name"
    "GRAPH_CACHE_DIR=$GRAPH_CACHE_DIR"
    "SPLIT_DIR=$SPLIT_DIR"
    "ASPECTS=CCO MFO"
    "EPOCHS=5"
    "BATCH_SIZE=8"
    "NUM_WORKERS=2"
    "MAX_PARALLEL=$GPUS_PER_JOB"
    "HIDDEN_DIM=$hidden_dim"
    "DROPOUT=$dropout"
    "LR=$lr"
    "WEIGHT_DECAY=$weight_decay"
    "LOSS_FUNCTION=$loss_function"
    "CHECKPOINT_METRIC=val_fmax"
    "EARLY_STOPPING_PATIENCE=2"
    "EARLY_STOPPING_MIN_DELTA=0.0003"
    "LR_SCHEDULER=none"
    "NORMALIZE_FEATURES=0"
    "PROGRESS_EVERY=250"
    "USE_SRUN=1"
    "SRUN_CPUS_PER_TASK=$CPUS_PER_TASK"
    "REQUESTED_WALLTIME=$REQUESTED_WALLTIME"
    "REQUESTED_GPU_TOTAL=$GPUS_PER_JOB"
    "REQUESTED_GPU_PER_NODE=$GPUS_PER_JOB"
    "GPU_IDS=$gpu_ids"
    "SEED=$seed"
  )

  if [[ "$loss_function" == "bce" ]]; then
    env_args+=(
      "POS_WEIGHT_POWER=1.0"
      "MAX_POS_WEIGHT="
    )
  else
    env_args+=(
      "POS_WEIGHT_POWER=0.25"
      "MAX_POS_WEIGHT=5"
    )
  fi

  local sbatch_args=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --job-name="$job_name"
    --ntasks-per-node="$GPUS_PER_JOB"
    --cpus-per-task="$CPUS_PER_TASK"
    --gres="$gres"
    --time="$REQUESTED_WALLTIME"
    --export=ALL
  )

  if [[ -n "$QOS" ]]; then
    sbatch_args+=(--qos="$QOS")
  fi
  sbatch_args+=("$TRAIN_SCRIPT")

  if [[ "$MODE" == "submit" ]]; then
    echo "submitting $exp_id as RUN_NAME=$run_name"
    env "${env_args[@]}" "${sbatch_args[@]}"
  else
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
  fi
}

case "$PLAN" in
  priority)
    submit_experiment "E0_control" "bce" "128" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E1_lr7e4" "bce" "128" "0.20" "0.0007" "0.0001" "2026"
    submit_experiment "E3_h192" "bce" "192" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E5_weighted_only" "weighted_bce" "128" "0.20" "0.0010" "0.0001" "2026"
    ;;
  priority_head)
    submit_experiment "E0_control" "bce" "128" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E1_lr7e4" "bce" "128" "0.20" "0.0007" "0.0001" "2026"
    ;;
  priority_tail)
    submit_experiment "E3_h192" "bce" "192" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E5_weighted_only" "weighted_bce" "128" "0.20" "0.0010" "0.0001" "2026"
    ;;
  full_tail)
    submit_experiment "E2_lr5e4" "bce" "128" "0.20" "0.0005" "0.0001" "2026"
    submit_experiment "E4_h256" "bce" "256" "0.20" "0.0010" "0.0001" "2026"
    ;;
  full)
    submit_experiment "E0_control" "bce" "128" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E1_lr7e4" "bce" "128" "0.20" "0.0007" "0.0001" "2026"
    submit_experiment "E2_lr5e4" "bce" "128" "0.20" "0.0005" "0.0001" "2026"
    submit_experiment "E3_h192" "bce" "192" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E4_h256" "bce" "256" "0.20" "0.0010" "0.0001" "2026"
    submit_experiment "E5_weighted_only" "weighted_bce" "128" "0.20" "0.0010" "0.0001" "2026"
    ;;
  *)
    echo "PLAN must be one of: priority, priority_head, priority_tail, full_tail, full" >&2
    exit 2
    ;;
esac
