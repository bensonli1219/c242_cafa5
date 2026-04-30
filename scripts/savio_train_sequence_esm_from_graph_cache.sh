#!/bin/bash
#SBATCH --job-name=cafa5_seq_esm
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=08:00:00
#SBATCH --output=/global/scratch/users/%u/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/%x_%j.err

set -euo pipefail

LOG_DIR="/global/scratch/users/$USER/logs"
REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-pytorch-2.3.1/bin/python}"

ASPECT="${ASPECT:-MFO}"
MODEL_TYPE="${MODEL_TYPE:-mlp}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.20}"
LR="${LR:-0.0010}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
METRIC_THRESHOLD="${METRIC_THRESHOLD:-0.50}"
FMAX_THRESHOLD_STEP="${FMAX_THRESHOLD_STEP:-0.01}"
EXPORT_SPLITS="${EXPORT_SPLITS:-val test}"
RUN_NAME="${RUN_NAME:-sigimp_n4_seq_small_${ASPECT,,}_$(date +%Y%m%d_%H%M%S)}"

PROTEIN_ESM_DIR="${PROTEIN_ESM_DIR:-$RUN_ROOT/sequence_artifacts/protein_esm2_t30_150m_640_from_graph_cache}"
MATCHED_SPLIT_DIR="${MATCHED_SPLIT_DIR:-$RUN_ROOT/sequence_artifacts/matched_structure_splits}"
GRAPH_ROOT="${GRAPH_ROOT:-$RUN_ROOT/graph_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/sequence_runs/$RUN_NAME}"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/train_sequence_esm_from_graph_cache.py" ]]; then
  echo "Missing script: $REPO_ROOT/train_sequence_esm_from_graph_cache.py" >&2
  exit 1
fi

if [[ ! -f "$PROTEIN_ESM_DIR/X.npy" ]]; then
  echo "Missing protein ESM matrix: $PROTEIN_ESM_DIR/X.npy" >&2
  exit 1
fi

if [[ ! -f "$MATCHED_SPLIT_DIR/${ASPECT,,}/train.txt" ]]; then
  echo "Missing matched split file: $MATCHED_SPLIT_DIR/${ASPECT,,}/train.txt" >&2
  exit 1
fi

if [[ ! -f "$GRAPH_ROOT/metadata/entries.json" ]]; then
  echo "Missing graph metadata: $GRAPH_ROOT/metadata/entries.json" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "PYTHON_BIN=$PYTHON_BIN"
echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "ASPECT=$ASPECT"
echo "MODEL_TYPE=$MODEL_TYPE"
echo "EPOCHS=$EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "HIDDEN_DIM=$HIDDEN_DIM"
echo "DROPOUT=$DROPOUT"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "SEED=$SEED"
echo "DEVICE=$DEVICE"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "PROTEIN_ESM_DIR=$PROTEIN_ESM_DIR"
echo "MATCHED_SPLIT_DIR=$MATCHED_SPLIT_DIR"
echo "GRAPH_ROOT=$GRAPH_ROOT"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  "$PYTHON_BIN" -u "$REPO_ROOT/train_sequence_esm_from_graph_cache.py" \
    --run-root "$RUN_ROOT" \
    --aspect "$ASPECT" \
    --model-type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --dropout "$DROPOUT" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --metric-threshold "$METRIC_THRESHOLD" \
    --fmax-threshold-step "$FMAX_THRESHOLD_STEP" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --protein-esm-dir "$PROTEIN_ESM_DIR" \
    --matched-split-dir "$MATCHED_SPLIT_DIR" \
    --graph-root "$GRAPH_ROOT" \
    --export-splits $EXPORT_SPLITS \
    "$@"
