#!/bin/bash
#SBATCH --job-name=build_graph_cache
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/global/scratch/users/%u/logs/build_graph_cache_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/build_graph_cache_%j.err

set -euo pipefail

LOG_DIR="/global/scratch/users/$USER/logs"
REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/.conda/envs/cafa5/bin/python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cafa5}"
CONDA_BASE="${CONDA_BASE:-}"
USE_CONDA="${USE_CONDA:-0}"
MIN_TERM_FREQUENCY="${MIN_TERM_FREQUENCY:-20}"
GRAPH_BATCH_SIZE="${GRAPH_BATCH_SIZE:-500}"
GRAPH_WORKERS="${GRAPH_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$REPO_ROOT/src/build_cafa_graph_cache.py" ]]; then
  echo "Missing script: $REPO_ROOT/src/build_cafa_graph_cache.py" >&2
  echo "Set REPO_ROOT to your checked-out repo path before sbatch." >&2
  exit 1
fi

if [[ "$USE_CONDA" == "1" ]]; then
  if [[ -z "$CONDA_BASE" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  fi
  if [[ -z "$CONDA_BASE" ]] && [[ -f /global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1/etc/profile.d/conda.sh ]]; then
    CONDA_BASE="/global/software/rocky-8.x86_64/manual/modules/langs/anaconda3/2024.02-1"
  fi
  if [[ -z "$CONDA_BASE" ]] || [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "Unable to locate conda.sh. Set CONDA_BASE or disable USE_CONDA." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
  PYTHON_BIN="$(command -v python)"
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "PYTHON_BIN=$PYTHON_BIN"
echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "USE_CONDA=$USE_CONDA"
if [[ "$USE_CONDA" == "1" ]]; then
  echo "CONDA_BASE=$CONDA_BASE"
  echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
  echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "MIN_TERM_FREQUENCY=$MIN_TERM_FREQUENCY"
echo "GRAPH_BATCH_SIZE=$GRAPH_BATCH_SIZE"
echo "GRAPH_WORKERS=$GRAPH_WORKERS"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

"$PYTHON_BIN" -c "import sys; print('python_executable=' + sys.executable)"

"$PYTHON_BIN" -u "$REPO_ROOT/src/build_cafa_graph_cache.py" \
    --training-index "$RUN_ROOT/manifests/training_index.parquet" \
    --fragment-features "$RUN_ROOT/features/fragment_features.parquet" \
    --residue-features "$RUN_ROOT/features/residue_features.parquet" \
    --edge-features "$RUN_ROOT/features/contact_graph_edges.parquet" \
    --output-dir "$RUN_ROOT/graph_cache" \
    --min-term-frequency "$MIN_TERM_FREQUENCY" \
    --workers "$GRAPH_WORKERS" \
    --batch-size "$GRAPH_BATCH_SIZE" \
    --resume
