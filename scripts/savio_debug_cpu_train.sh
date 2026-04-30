#!/bin/bash
#SBATCH --job-name=cafa5_cpu_train
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/%u/logs/cafa5_cpu_train_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/cafa5_cpu_train_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-$RUN_ROOT/graph_cache}"
BASE_SPLIT_DIR="${BASE_SPLIT_DIR:-$GRAPH_CACHE_DIR/preprocessing_splits_mtf20_cap1200_full/experiment_splits/full_mtf20_cap1200}"
DEBUG_ROOT="${DEBUG_ROOT:-$RUN_ROOT/debug_cpu_train/${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}}"
DEBUG_SPLIT_DIR="${DEBUG_SPLIT_DIR:-$DEBUG_ROOT/splits}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEBUG_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-pytorch-2.3.1/bin/python}"
PYTORCH_MODULE="${PYTORCH_MODULE:-ml/pytorch/2.3.1-py3.11.7}"

ASPECT="${ASPECT:-MFO}"
MIN_TERM_FREQUENCY="${MIN_TERM_FREQUENCY:-20}"
TRAIN_COUNT="${TRAIN_COUNT:-32}"
VAL_COUNT="${VAL_COUNT:-8}"
TEST_COUNT="${TEST_COUNT:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"

mkdir -p "$DEBUG_SPLIT_DIR" "$CHECKPOINT_DIR" "/global/scratch/users/$USER/logs"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

echo "started_at=$(date -Iseconds)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "partition=${SLURM_JOB_PARTITION:-savio2}"
echo "repo_root=$REPO_ROOT"
echo "graph_cache_dir=$GRAPH_CACHE_DIR"
echo "base_split_dir=$BASE_SPLIT_DIR"
echo "debug_root=$DEBUG_ROOT"

if command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null; then
  set +e
  set +u
  module load "$PYTORCH_MODULE"
  module_status=$?
  set -u
  set -e
  echo "module_load_status=$module_status"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "torch_geometric",
    "torch_geometric.loader",
    "numpy",
    "pyarrow",
    "cafa_graph_dataset",
    "cafa_graph_dataloaders",
    "train_minimal_graph_model",
]
print("python_executable=" + sys.executable)
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "ok")
    print(f"import_ok {name} {version}")

torch = importlib.import_module("torch")
print("cuda_available=" + str(torch.cuda.is_available()))
PY

"$PYTHON_BIN" - "$BASE_SPLIT_DIR" "$DEBUG_SPLIT_DIR" "$ASPECT" "$TRAIN_COUNT" "$VAL_COUNT" "$TEST_COUNT" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
out = Path(sys.argv[2])
aspect = sys.argv[3].upper()
counts = {
    "train": int(sys.argv[4]),
    "val": int(sys.argv[5]),
    "test": int(sys.argv[6]),
}
aspect_dir = out / aspect.lower()
aspect_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "source_split_dir": str(base),
    "aspects": {
        aspect: {
            "aspect": aspect,
            "counts": counts,
            "entry_ids": {},
        }
    },
}
for split_name, count in counts.items():
    source = base / aspect.lower() / f"{split_name}.txt"
    ids = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:count]
    (aspect_dir / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    summary["aspects"][aspect]["entry_ids"][split_name] = ids

(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("debug_split_dir=" + str(out))
print("debug_split_counts=" + json.dumps(counts, sort_keys=True))
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u train_minimal_graph_model.py \
  --root "$GRAPH_CACHE_DIR" \
  --split-dir "$DEBUG_SPLIT_DIR" \
  --framework pyg \
  --aspect "$ASPECT" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --hidden-dim "$HIDDEN_DIM" \
  --dropout 0.2 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --checkpoint-metric val_loss \
  --early-stopping-patience 0 \
  --lr-scheduler none \
  --progress-mode log \
  --progress-every 1 \
  --device cpu \
  --seed 2026 \
  --min-term-frequency "$MIN_TERM_FREQUENCY" \
  --disable-esm2 \
  --disable-dssp \
  --disable-sasa \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "finished_at=$(date -Iseconds)"
