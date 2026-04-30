#!/bin/bash
# Submit the N1/N2/N3 graph-side significant-improvement runs on Savio3 GPU.
#
# Default mode is dry-run. Pass --submit to actually submit.
# Default PLAN=all submits N1, N2, and N3 without dependencies.

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
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-$RUN_ROOT/graph_cache}"
SPLIT_DIR="${SPLIT_DIR:-$GRAPH_CACHE_DIR/splits}"
RUN_PREFIX="${RUN_PREFIX:-sigimp}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PLAN="${PLAN:-all}"
ACCOUNT="${ACCOUNT:-ic_chem242}"

PARTITION="${PARTITION:-savio3_gpu}"
if [[ -z "${GPU_TYPE+x}" ]]; then
  GPU_TYPE="A40"
else
  GPU_TYPE="${GPU_TYPE}"
fi
QOS="${QOS:-a40_gpu3_normal}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
REQUESTED_WALLTIME="${REQUESTED_WALLTIME:-24:00:00}"

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
  local name="$1"
  local loss_function="$2"
  local model_head="$3"
  local focal_gamma="$4"
  local focal_alpha="$5"
  local logit_adjustment="$6"
  local logit_adjustment_strength="$7"
  local logit_temperature="$8"
  local run_name="${RUN_PREFIX}_${name}_${RUN_STAMP}"
  local job_name="cafa5_${name}"
  local gpu_ids
  local gres

  gpu_ids="$(seq -s ' ' 0 "$((GPUS_PER_JOB - 1))")"
  gres="$(build_gres)"

  local env_args=(
    "REPO_ROOT=$REPO_ROOT"
    "RUN_ROOT=$RUN_ROOT"
    "GRAPH_CACHE_DIR=$GRAPH_CACHE_DIR"
    "SPLIT_DIR=$SPLIT_DIR"
    "RUN_NAME=$run_name"
    "ASPECTS=CCO MFO"
    "EPOCHS=5"
    "BATCH_SIZE=8"
    "NUM_WORKERS=2"
    "MAX_PARALLEL=$GPUS_PER_JOB"
    "HIDDEN_DIM=128"
    "DROPOUT=0.20"
    "MODEL_HEAD=$model_head"
    "LR=0.0010"
    "WEIGHT_DECAY=0.0001"
    "LOSS_FUNCTION=$loss_function"
    "POS_WEIGHT_POWER=1.0"
    "MAX_POS_WEIGHT="
    "FOCAL_GAMMA=$focal_gamma"
    "FOCAL_ALPHA=$focal_alpha"
    "LOGIT_ADJUSTMENT=$logit_adjustment"
    "LOGIT_ADJUSTMENT_STRENGTH=$logit_adjustment_strength"
    "LOGIT_TEMPERATURE=$logit_temperature"
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
    "SEED=2026"
  )

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
    echo "submitting $name as RUN_NAME=$run_name"
    env "${env_args[@]}" "${sbatch_args[@]}"
  else
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
  fi
}

case "$PLAN" in
  all)
    submit_experiment "n1_focal_bce" "focal_bce" "flat_linear" "2.0" "0.25" "none" "1.0" "1.0"
    submit_experiment "n2_logit_adjust" "bce" "flat_linear" "2.0" "0.25" "train_prior" "1.0" "1.0"
    submit_experiment "n3_label_dot" "bce" "label_dot" "2.0" "0.25" "none" "1.0" "1.0"
    ;;
  n1)
    submit_experiment "n1_focal_bce" "focal_bce" "flat_linear" "2.0" "0.25" "none" "1.0" "1.0"
    ;;
  n2)
    submit_experiment "n2_logit_adjust" "bce" "flat_linear" "2.0" "0.25" "train_prior" "1.0" "1.0"
    ;;
  n3)
    submit_experiment "n3_label_dot" "bce" "label_dot" "2.0" "0.25" "none" "1.0" "1.0"
    ;;
  *)
    echo "PLAN must be one of: all, n1, n2, n3" >&2
    exit 2
    ;;
esac
