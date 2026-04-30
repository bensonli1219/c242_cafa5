#!/bin/bash
# Submit one small sequence-side N4 experiment on Savio3 GPU.
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
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/savio_train_sequence_esm_from_graph_cache.sh}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
RUN_PREFIX="${RUN_PREFIX:-sigimp_n4_seq_small}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ACCOUNT="${ACCOUNT:-ic_chem242}"

PARTITION="${PARTITION:-savio3_gpu}"
if [[ -z "${GPU_TYPE+x}" ]]; then
  GPU_TYPE="A40"
else
  GPU_TYPE="${GPU_TYPE}"
fi
QOS="${QOS:-a40_gpu3_normal}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
REQUESTED_WALLTIME="${REQUESTED_WALLTIME:-08:00:00}"

PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-pytorch-2.3.1/bin/python}"
PROTEIN_ESM_DIR="${PROTEIN_ESM_DIR:-$RUN_ROOT/sequence_artifacts/protein_esm2_t30_150m_640_from_graph_cache}"
MATCHED_SPLIT_DIR="${MATCHED_SPLIT_DIR:-$RUN_ROOT/sequence_artifacts/matched_structure_splits}"
GRAPH_ROOT="${GRAPH_ROOT:-$RUN_ROOT/graph_cache}"

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
  local aspect="$1"
  local model_type="$2"
  local epochs="$3"
  local batch_size="$4"
  local hidden_dim="$5"
  local dropout="$6"
  local lr="$7"
  local weight_decay="$8"
  local seed="$9"
  local run_name="${RUN_PREFIX}_${aspect,,}_${model_type}_${RUN_STAMP}"
  local job_name="cafa5_n4_${aspect,,}"
  local gres

  gres="$(build_gres)"

  local env_args=(
    "REPO_ROOT=$REPO_ROOT"
    "RUN_ROOT=$RUN_ROOT"
    "PYTHON_BIN=$PYTHON_BIN"
    "RUN_NAME=$run_name"
    "ASPECT=$aspect"
    "MODEL_TYPE=$model_type"
    "EPOCHS=$epochs"
    "BATCH_SIZE=$batch_size"
    "HIDDEN_DIM=$hidden_dim"
    "DROPOUT=$dropout"
    "LR=$lr"
    "WEIGHT_DECAY=$weight_decay"
    "SEED=$seed"
    "DEVICE=cuda"
    "PROTEIN_ESM_DIR=$PROTEIN_ESM_DIR"
    "MATCHED_SPLIT_DIR=$MATCHED_SPLIT_DIR"
    "GRAPH_ROOT=$GRAPH_ROOT"
    "CHECKPOINT_DIR=$RUN_ROOT/sequence_runs/$run_name"
  )

  local sbatch_args=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --job-name="$job_name"
    --nodes=1
    --ntasks=1
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
    echo "submitting $aspect/$model_type as RUN_NAME=$run_name"
    env "${env_args[@]}" "${sbatch_args[@]}"
  else
    printf 'env'
    printf ' %q' "${env_args[@]}"
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
  fi
}

submit_experiment "MFO" "mlp" "5" "256" "256" "0.20" "0.0010" "0.0001" "2026"
