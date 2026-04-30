#!/bin/bash
#SBATCH --job-name=cafa5_audit
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-$RUN_ROOT/graph_cache}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-notebook/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$GRAPH_CACHE_DIR/preprocessing_audit_savio_${SLURM_JOB_ID:-manual}}"
ASPECTS="${ASPECTS:-BPO CCO MFO}"
MIN_TERM_FREQUENCIES="${MIN_TERM_FREQUENCIES:-20}"
GRAPH_RESIDUE_CAPS="${GRAPH_RESIDUE_CAPS:-800 1200 1600 2000}"
TENSOR_SAMPLE_SIZE="${TENSOR_SAMPLE_SIZE:-1000}"
CHECK_GRAPH_FILES="${CHECK_GRAPH_FILES:-1}"
WRITE_EXPERIMENT_SPLITS="${WRITE_EXPERIMENT_SPLITS:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sample_mtf20_cap1200}"
EXPERIMENT_MIN_TERM_FREQUENCY="${EXPERIMENT_MIN_TERM_FREQUENCY:-20}"
EXPERIMENT_MAX_RESIDUES="${EXPERIMENT_MAX_RESIDUES:-1200}"
EXPERIMENT_SAMPLE_PER_ASPECT="${EXPERIMENT_SAMPLE_PER_ASPECT:-600}"
SEED="${SEED:-2026}"

if [[ ! -f "$REPO_ROOT/scripts/audit_graph_cache_preprocessing.py" ]]; then
  echo "Missing audit script under REPO_ROOT=$REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "REPO_ROOT=$REPO_ROOT"
echo "GRAPH_CACHE_DIR=$GRAPH_CACHE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "ASPECTS=$ASPECTS"
echo "MIN_TERM_FREQUENCIES=$MIN_TERM_FREQUENCIES"
echo "GRAPH_RESIDUE_CAPS=$GRAPH_RESIDUE_CAPS"

read -r -a ASPECT_ARRAY <<< "$ASPECTS"
read -r -a MIN_FREQ_ARRAY <<< "$MIN_TERM_FREQUENCIES"
read -r -a CAP_ARRAY <<< "$GRAPH_RESIDUE_CAPS"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/audit_graph_cache_preprocessing.py"
  --root "$GRAPH_CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --aspects "${ASPECT_ARRAY[@]}"
  --min-term-frequencies "${MIN_FREQ_ARRAY[@]}"
  --graph-residue-caps "${CAP_ARRAY[@]}"
  --tensor-sample-size "$TENSOR_SAMPLE_SIZE"
  --experiment-name "$EXPERIMENT_NAME"
  --experiment-min-term-frequency "$EXPERIMENT_MIN_TERM_FREQUENCY"
  --experiment-max-residues "$EXPERIMENT_MAX_RESIDUES"
  --experiment-sample-per-aspect "$EXPERIMENT_SAMPLE_PER_ASPECT"
  --seed "$SEED"
)

if [[ "$WRITE_EXPERIMENT_SPLITS" == "1" ]]; then
  cmd+=(--write-experiment-splits)
fi
if [[ "$CHECK_GRAPH_FILES" == "1" ]]; then
  cmd+=(--check-graph-files)
fi

printf '%q ' "${cmd[@]}" > "$OUTPUT_DIR/command.sh"
printf '\n' >> "$OUTPUT_DIR/command.sh"
"${cmd[@]}" | tee "$OUTPUT_DIR/audit.log"
