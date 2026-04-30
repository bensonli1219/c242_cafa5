#!/bin/bash
# Submit N3 follow-up runs on Savio3 GPU.
#
# Default mode is dry-run. Pass --submit to actually submit.
# PLAN can be: all, confirm, stability, ontology.

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

CONFIRM_EPOCHS="${CONFIRM_EPOCHS:-8}"
STABILITY_EPOCHS="${STABILITY_EPOCHS:-8}"
ONTOLOGY_EPOCHS="${ONTOLOGY_EPOCHS:-5}"
CONFIRM_SEED="${CONFIRM_SEED:-2026}"
STABILITY_SEED="${STABILITY_SEED:-2027}"
ONTOLOGY_SEED="${ONTOLOGY_SEED:-2026}"
ONTOLOGY_REG_WEIGHT="${ONTOLOGY_REG_WEIGHT:-0.0005}"

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

submit_n3_run() {
  local name="$1"
  local epochs="$2"
  local seed="$3"
  local ontology_reg_weight="$4"
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
    "EPOCHS=$epochs"
    "BATCH_SIZE=8"
    "NUM_WORKERS=2"
    "MAX_PARALLEL=$GPUS_PER_JOB"
    "HIDDEN_DIM=128"
    "DROPOUT=0.20"
    "MODEL_HEAD=label_dot"
    "LR=0.0010"
    "WEIGHT_DECAY=0.0001"
    "LOSS_FUNCTION=bce"
    "POS_WEIGHT_POWER=1.0"
    "MAX_POS_WEIGHT="
    "FOCAL_GAMMA=2.0"
    "FOCAL_ALPHA=0.25"
    "LOGIT_ADJUSTMENT=none"
    "LOGIT_ADJUSTMENT_STRENGTH=1.0"
    "LOGIT_TEMPERATURE=1.0"
    "LABEL_ONTOLOGY_REG_WEIGHT=$ontology_reg_weight"
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
    submit_n3_run "n3_confirm_long" "$CONFIRM_EPOCHS" "$CONFIRM_SEED" "0.0"
    submit_n3_run "n3_stability_seed${STABILITY_SEED}" "$STABILITY_EPOCHS" "$STABILITY_SEED" "0.0"
    submit_n3_run "n3_ontology_reg" "$ONTOLOGY_EPOCHS" "$ONTOLOGY_SEED" "$ONTOLOGY_REG_WEIGHT"
    ;;
  confirm)
    submit_n3_run "n3_confirm_long" "$CONFIRM_EPOCHS" "$CONFIRM_SEED" "0.0"
    ;;
  stability)
    submit_n3_run "n3_stability_seed${STABILITY_SEED}" "$STABILITY_EPOCHS" "$STABILITY_SEED" "0.0"
    ;;
  ontology)
    submit_n3_run "n3_ontology_reg" "$ONTOLOGY_EPOCHS" "$ONTOLOGY_SEED" "$ONTOLOGY_REG_WEIGHT"
    ;;
  *)
    echo "PLAN must be one of: all, confirm, stability, ontology" >&2
    exit 2
    ;;
esac
