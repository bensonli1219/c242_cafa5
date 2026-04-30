#!/bin/bash
# Submit one staged CCO/MFO improvement experiment on Savio.
#
# Default mode is a dry run that prints the sbatch command. Pass --submit to
# launch exactly one experiment. Select the experiment with PLAN=...

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
RUN_PREFIX="${RUN_PREFIX:-cco_mfo_stage}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
PLAN="${PLAN:-weighted_bce_lr5e4_mild}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing training script: $TRAIN_SCRIPT" >&2
  exit 1
fi

submit_experiment() {
  local name="$1"
  local epochs="$2"
  local hidden_dim="$3"
  local dropout="$4"
  local lr="$5"
  local weight_decay="$6"
  local loss_function="$7"
  local pos_weight_power="$8"
  local max_pos_weight="$9"
  local lr_scheduler="${10}"
  local lr_plateau_factor="${11}"
  local lr_plateau_patience="${12}"
  local seed="${13}"
  local run_name="${RUN_PREFIX}_${name}_${RUN_STAMP}"
  local job_name="cafa5_${name}"
  local max_parallel="$GPUS_PER_JOB"
  local ntasks_per_node="$GPUS_PER_JOB"
  local gpu_ids

  if [[ ! "$GPUS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$GPUS_PER_JOB" == "0" ]]; then
    echo "GPUS_PER_JOB must be a positive integer: $GPUS_PER_JOB" >&2
    exit 2
  fi
  gpu_ids="$(seq -s ' ' 0 "$((GPUS_PER_JOB - 1))")"

  local env_args=(
    "RUN_NAME=$run_name"
    "GRAPH_CACHE_DIR=/global/scratch/users/$USER/cafa5_outputs/graph_cache"
    "SPLIT_DIR=/global/scratch/users/$USER/cafa5_outputs/graph_cache/splits"
    "ASPECTS=CCO MFO"
    "EPOCHS=$epochs"
    "BATCH_SIZE=8"
    "NUM_WORKERS=2"
    "MAX_PARALLEL=$max_parallel"
    "HIDDEN_DIM=$hidden_dim"
    "DROPOUT=$dropout"
    "LR=$lr"
    "WEIGHT_DECAY=$weight_decay"
    "LOSS_FUNCTION=$loss_function"
    "POS_WEIGHT_POWER=$pos_weight_power"
    "MAX_POS_WEIGHT=$max_pos_weight"
    "CHECKPOINT_METRIC=val_fmax"
    "EARLY_STOPPING_PATIENCE=3"
    "EARLY_STOPPING_MIN_DELTA=0.0003"
    "LR_SCHEDULER=$lr_scheduler"
    "LR_PLATEAU_FACTOR=$lr_plateau_factor"
    "LR_PLATEAU_PATIENCE=$lr_plateau_patience"
    "NORMALIZE_FEATURES=0"
    "PROGRESS_EVERY=250"
    "USE_SRUN=1"
    "REQUESTED_GPU_TOTAL=$GPUS_PER_JOB"
    "REQUESTED_GPU_PER_NODE=$GPUS_PER_JOB"
    "GPU_IDS=$gpu_ids"
    "SEED=$seed"
  )

  local sbatch_args=(
    sbatch
    --parsable
    --job-name="$job_name"
    --ntasks-per-node="$ntasks_per_node"
    --gres="gpu:$GPUS_PER_JOB"
    --cpus-per-task=2
    --export=ALL
    "$TRAIN_SCRIPT"
  )

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
  weighted_bce_lr5e4_mild)
    submit_experiment "$PLAN" "10" "128" "0.2" "0.0005" "0.0001" "weighted_bce" "0.25" "5" "none" "0.5" "1" "2026"
    ;;
  capacity256_lr5e4_bce)
    submit_experiment "$PLAN" "10" "256" "0.2" "0.0005" "0.0001" "bce" "1.0" "" "none" "0.5" "1" "2026"
    ;;
  bce_lr1e3_plateau)
    submit_experiment "$PLAN" "10" "128" "0.2" "0.001" "0.0001" "bce" "1.0" "" "plateau" "0.5" "1" "2026"
    ;;
  raw_bce_lr5e4_seed2027)
    submit_experiment "$PLAN" "10" "128" "0.2" "0.0005" "0.0001" "bce" "1.0" "" "none" "0.5" "1" "2027"
    ;;
  raw_bce_lr5e4_seed2028)
    submit_experiment "$PLAN" "10" "128" "0.2" "0.0005" "0.0001" "bce" "1.0" "" "none" "0.5" "1" "2028"
    ;;
  *)
    echo "PLAN must be one of:" >&2
    echo "  weighted_bce_lr5e4_mild" >&2
    echo "  capacity256_lr5e4_bce" >&2
    echo "  bce_lr1e3_plateau" >&2
    echo "  raw_bce_lr5e4_seed2027" >&2
    echo "  raw_bce_lr5e4_seed2028" >&2
    exit 2
    ;;
esac
