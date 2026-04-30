#!/bin/bash
# Submit focused CCO/MFO improvement experiments on Savio.
#
# Default mode is a dry run that prints sbatch commands. Pass --submit to submit.

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
RUN_PREFIX="${RUN_PREFIX:-cco_mfo_improve}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
PLAN="${PLAN:-focused}"
PREVIOUS_JOB_ID=""

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
  local seed="$7"
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
    "LOSS_FUNCTION=bce"
    "POS_WEIGHT_POWER=1.0"
    "MAX_POS_WEIGHT="
    "CHECKPOINT_METRIC=val_fmax"
    "EARLY_STOPPING_PATIENCE=3"
    "EARLY_STOPPING_MIN_DELTA=0.0003"
    "LR_SCHEDULER=none"
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
    if [[ -n "$PREVIOUS_JOB_ID" ]]; then
      sbatch_args=( "${sbatch_args[@]:0:1}" "--dependency=afterok:$PREVIOUS_JOB_ID" "${sbatch_args[@]:1}" )
    fi
    PREVIOUS_JOB_ID="$(env "${env_args[@]}" "${sbatch_args[@]}")"
    echo "$name job_id=$PREVIOUS_JOB_ID"
  else
    if [[ -n "$PREVIOUS_JOB_ID" ]]; then
      printf '# %s should be submitted after the previous experiment succeeds.\n' "$name"
    fi
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
    PREVIOUS_JOB_ID="DRY_RUN_PREVIOUS_JOB"
  fi
}

case "$PLAN" in
  focused)
    submit_experiment "raw_bce_10ep" "10" "128" "0.2" "0.001" "0.0001" "2026"
    submit_experiment "raw_bce_lr5e4_10ep" "10" "128" "0.2" "0.0005" "0.0001" "2026"
    ;;
  seeds)
    submit_experiment "raw_bce_seed2027_10ep" "10" "128" "0.2" "0.001" "0.0001" "2027"
    submit_experiment "raw_bce_seed2028_10ep" "10" "128" "0.2" "0.001" "0.0001" "2028"
    ;;
  capacity)
    submit_experiment "raw_bce_h192_8ep" "8" "192" "0.25" "0.001" "0.0001" "2026"
    submit_experiment "raw_bce_h256_8ep" "8" "256" "0.3" "0.001" "0.0003" "2026"
    ;;
  *)
    echo "PLAN must be one of: focused, seeds, capacity" >&2
    exit 2
    ;;
esac
