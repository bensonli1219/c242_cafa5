#!/bin/bash
#SBATCH --job-name=build_esm2_cache
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/%u/logs/build_esm2_cache_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/build_esm2_cache_%j.err

set -euo pipefail

LOG_DIR="/global/scratch/users/$USER/logs"
REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-notebook/bin/python}"
HF_HOME="${HF_HOME:-/global/scratch/users/$USER/.cache/huggingface}"

mkdir -p "$LOG_DIR" "$HF_HOME"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/build_esm2_cache.py" ]]; then
  echo "Missing script: $REPO_ROOT/build_esm2_cache.py" >&2
  echo "Set REPO_ROOT to your checked-out repo path before sbatch." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME
export PYTHONUNBUFFERED=1

echo "PYTHON_BIN=$PYTHON_BIN"
echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "HF_HOME=$HF_HOME"

"$PYTHON_BIN" -u "$REPO_ROOT/build_esm2_cache.py" \
    --training-index "$RUN_ROOT/manifests/training_index.parquet" \
    --output-dir "$RUN_ROOT/graph_cache/modality_cache/esm2" \
    --resume
