#!/bin/bash
#SBATCH --job-name=cafa5_norm_cache
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
INPUT_ROOT="${INPUT_ROOT:-$RUN_ROOT/graph_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/graph_cache_normalized_features}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-notebook/bin/python}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
COPY_SPLITS="${COPY_SPLITS:-1}"
COPY_MODALITY_CACHE="${COPY_MODALITY_CACHE:-0}"
LINK_MODALITY_CACHE="${LINK_MODALITY_CACHE:-1}"

if [[ ! -f "$REPO_ROOT/scripts/materialize_normalized_graph_cache.py" ]]; then
  echo "Missing materialization script under REPO_ROOT=$REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "WORKERS=$WORKERS"
echo "COPY_SPLITS=$COPY_SPLITS"
echo "COPY_MODALITY_CACHE=$COPY_MODALITY_CACHE"
echo "LINK_MODALITY_CACHE=$LINK_MODALITY_CACHE"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_normalized_graph_cache.py"
  --input-root "$INPUT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --progress-every "$PROGRESS_EVERY"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "$COPY_SPLITS" == "1" ]]; then
  cmd+=(--copy-splits)
fi
if [[ "$COPY_MODALITY_CACHE" == "1" ]]; then
  cmd+=(--copy-modality-cache)
fi
if [[ "$LINK_MODALITY_CACHE" == "1" ]]; then
  cmd+=(--link-modality-cache)
fi

printf '%q ' "${cmd[@]}" > "$OUTPUT_ROOT/materialize_command.sh"
printf '\n' >> "$OUTPUT_ROOT/materialize_command.sh"
"${cmd[@]}" | tee "$OUTPUT_ROOT/materialize.log"
