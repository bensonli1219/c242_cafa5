#!/bin/bash
#SBATCH --job-name=export_seq_artifacts
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/global/scratch/users/%u/logs/export_seq_artifacts_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/export_seq_artifacts_%j.err

set -euo pipefail

LOG_DIR="/global/scratch/users/$USER/logs"
REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-notebook/bin/python}"
MIN_TERM_FREQUENCY="${MIN_TERM_FREQUENCY:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
PROGRESS_MODE="${PROGRESS_MODE:-log}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/export_sequence_artifacts_from_graph_cache.py" ]]; then
  echo "Missing script: $REPO_ROOT/export_sequence_artifacts_from_graph_cache.py" >&2
  echo "Set REPO_ROOT to your checked-out repo path before sbatch." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "PYTHON_BIN=$PYTHON_BIN"
echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "MIN_TERM_FREQUENCY=$MIN_TERM_FREQUENCY"
echo "PROGRESS_EVERY=$PROGRESS_EVERY"
echo "WORKERS=$WORKERS"
echo "PROGRESS_MODE=$PROGRESS_MODE"

"$PYTHON_BIN" -u "$REPO_ROOT/export_sequence_artifacts_from_graph_cache.py" \
  --run-root "$RUN_ROOT" \
  --min-term-frequency "$MIN_TERM_FREQUENCY" \
  --progress-every "$PROGRESS_EVERY" \
  --workers "$WORKERS" \
  --progress-mode "$PROGRESS_MODE" \
  "$@"
