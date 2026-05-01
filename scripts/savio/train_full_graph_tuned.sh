#!/bin/bash
#SBATCH --job-name=train_graph_full_tuned
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
# savio_normal max walltime is 72 hours.
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/%u/logs/train_graph_full_tuned_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/train_graph_full_tuned_%j.err

# Tuned full-graph training wrapper for Savio.
# Keeps the original batch orchestration but swaps in stronger defaults for
# class imbalance, checkpoint selection, and learning-rate control.

set -euo pipefail

export FRAMEWORK="${FRAMEWORK:-pyg}"
export ASPECTS="${ASPECTS:-CCO MFO}"
export MIN_TERM_FREQUENCY="${MIN_TERM_FREQUENCY:-20}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export DROPOUT="${DROPOUT:-0.3}"
export LR="${LR:-0.0003}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
export LOSS_FUNCTION="${LOSS_FUNCTION:-weighted_bce}"
export POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0.5}"
export MAX_POS_WEIGHT="${MAX_POS_WEIGHT:-20}"
export CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-val_fmax}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-2}"
export EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0005}"
export LR_SCHEDULER="${LR_SCHEDULER:-plateau}"
export LR_PLATEAU_FACTOR="${LR_PLATEAU_FACTOR:-0.5}"
export LR_PLATEAU_PATIENCE="${LR_PLATEAU_PATIENCE:-1}"
export MIN_LR="${MIN_LR:-1e-6}"

RUN_STAMP="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${RUN_NAME:-full_graph_tuned_${FRAMEWORK}_mtf${MIN_TERM_FREQUENCY}_${RUN_STAMP}}"

REPO_ROOT_FOR_WRAPPER="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
BASE_SCRIPT="${BASE_SCRIPT:-$REPO_ROOT_FOR_WRAPPER/scripts/savio/train_full_graph.sh}"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base training script not found: $BASE_SCRIPT" >&2
  echo "Set REPO_ROOT or BASE_SCRIPT to the checked-out repo on Savio before sbatch." >&2
  exit 1
fi

exec "$BASE_SCRIPT" "$@"
