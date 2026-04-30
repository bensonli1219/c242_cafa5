#!/bin/bash
#SBATCH --job-name=cafa5_n4_fusion
#SBATCH --account=ic_chem242
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/%u/logs/%x_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-pytorch-2.3.1/bin/python}"

ASPECT="${ASPECT:-MFO}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$RUN_ROOT/graph_cache/training_runs/full_graph_pyg_mtf20_33234089/mfo/best.pt}"
SEQUENCE_RUN="${SEQUENCE_RUN:-$RUN_ROOT/sequence_runs/sigimp_n4_seq_graph_vocab_mfo_mlp_20260424_222500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/n4_fusion/raw_mfo_x_seq_graph_vocab_20260424_222500}"
GRAPH_BUNDLE_ROOT="${GRAPH_BUNDLE_ROOT:-$OUTPUT_ROOT/graph_bundles/raw_mfo_full_graph_pyg_mtf20_33234089}"
FUSED_ROOT="${FUSED_ROOT:-$OUTPUT_ROOT/fused_bundles}"
GRAPH_ROOT="${GRAPH_ROOT:-$RUN_ROOT/graph_cache}"
SCORE_SPACE="${SCORE_SPACE:-logits}"
WEIGHT_GRID="${WEIGHT_GRID:-1.0:0.0 0.9:0.1 0.8:0.2 0.7:0.3 0.6:0.4 0.5:0.5 0.4:0.6 0.3:0.7 0.2:0.8 0.1:0.9 0.0:1.0}"
GRAPH_EXPORT_BATCH_SIZE="${GRAPH_EXPORT_BATCH_SIZE:-128}"
GRAPH_EXPORT_NUM_WORKERS="${GRAPH_EXPORT_NUM_WORKERS:-4}"

mkdir -p "$OUTPUT_ROOT" "$GRAPH_BUNDLE_ROOT" "$FUSED_ROOT" "/global/scratch/users/$USER/logs"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$GRAPH_CHECKPOINT" ]]; then
  echo "Missing graph checkpoint: $GRAPH_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$SEQUENCE_RUN/prediction_bundles" ]]; then
  echo "Missing sequence prediction bundles: $SEQUENCE_RUN/prediction_bundles" >&2
  exit 1
fi

echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "ASPECT=$ASPECT"
echo "GRAPH_CHECKPOINT=$GRAPH_CHECKPOINT"
echo "SEQUENCE_RUN=$SEQUENCE_RUN"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SCORE_SPACE=$SCORE_SPACE"
echo "WEIGHT_GRID=$WEIGHT_GRID"
echo "GRAPH_EXPORT_BATCH_SIZE=$GRAPH_EXPORT_BATCH_SIZE"
echo "GRAPH_EXPORT_NUM_WORKERS=$GRAPH_EXPORT_NUM_WORKERS"

cd "$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u export_graph_prediction_bundles.py \
  --checkpoint-path "$GRAPH_CHECKPOINT" \
  --output-dir "$GRAPH_BUNDLE_ROOT" \
  --device auto \
  --batch-size "$GRAPH_EXPORT_BATCH_SIZE" \
  --num-workers "$GRAPH_EXPORT_NUM_WORKERS" \
  --export-splits val test

for weight_pair in $WEIGHT_GRID; do
  graph_weight="${weight_pair%%:*}"
  sequence_weight="${weight_pair##*:}"
  weight_label="g${graph_weight//./p}_s${sequence_weight//./p}"
  for split in val test; do
    "$PYTHON_BIN" -u fuse_prediction_scores.py \
      --graph-bundle "$GRAPH_BUNDLE_ROOT/$split" \
      --sequence-bundle "$SEQUENCE_RUN/prediction_bundles/$split" \
      --output-dir "$FUSED_ROOT/$weight_label/$split" \
      --graph-weight "$graph_weight" \
      --sequence-weight "$sequence_weight" \
      --score-space "$SCORE_SPACE" \
      --evaluate-with-graph-root "$GRAPH_ROOT" \
      --aspect "$ASPECT"
  done
done

"$PYTHON_BIN" - "$FUSED_ROOT" "$OUTPUT_ROOT/summary.tsv" <<'PY'
import json
import sys
from pathlib import Path

fused_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for meta_path in sorted(fused_root.glob("*/*/meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    evaluation = meta.get("evaluation") or {}
    rows.append(
        {
            "weight_label": meta_path.parent.parent.name,
            "split": meta_path.parent.name,
            "graph_weight": meta.get("graph_weight"),
            "sequence_weight": meta.get("sequence_weight"),
            "score_space": meta.get("score_space"),
            "fmax": evaluation.get("fmax"),
            "fmax_threshold": evaluation.get("fmax_threshold"),
            "micro_f1": evaluation.get("micro_f1"),
            "macro_f1": evaluation.get("macro_f1"),
            "entry_count": meta.get("entry_count"),
            "term_count": meta.get("term_count"),
        }
    )

fields = [
    "weight_label",
    "split",
    "graph_weight",
    "sequence_weight",
    "score_space",
    "fmax",
    "fmax_threshold",
    "micro_f1",
    "macro_f1",
    "entry_count",
    "term_count",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("\t".join(fields) + "\n")
    for row in rows:
        handle.write("\t".join("" if row[field] is None else str(row[field]) for field in fields) + "\n")
print(f"wrote {summary_path}")
PY

echo "N4 fusion complete: $OUTPUT_ROOT"
