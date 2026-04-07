#!/bin/bash
#SBATCH --job-name=train_graph_full
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
# savio_normal max walltime is 72 hours.
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/%u/logs/train_graph_full_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/train_graph_full_%j.err

set -euo pipefail

LOG_DIR="/global/scratch/users/$USER/logs"
REPO_ROOT="${REPO_ROOT:-/global/home/users/$USER/c242_cafa5}"
RUN_ROOT="${RUN_ROOT:-/global/scratch/users/$USER/cafa5_outputs}"
GRAPH_CACHE_DIR="${GRAPH_CACHE_DIR:-$RUN_ROOT/graph_cache}"
SPLIT_DIR="${SPLIT_DIR:-$GRAPH_CACHE_DIR/splits}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-notebook/bin/python}"

FRAMEWORK="${FRAMEWORK:-pyg}"
ASPECTS="${ASPECTS:-BPO CCO MFO}"
MIN_TERM_FREQUENCY="${MIN_TERM_FREQUENCY:-20}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DROPOUT="${DROPOUT:-0.2}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
METRIC_THRESHOLD="${METRIC_THRESHOLD:-0.5}"
FMAX_THRESHOLD_STEP="${FMAX_THRESHOLD_STEP:-0.01}"
PROGRESS_MODE="${PROGRESS_MODE:-log}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-2026}"
USE_ESM2="${USE_ESM2:-1}"
USE_DSSP="${USE_DSSP:-1}"
USE_SASA="${USE_SASA:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
USE_SRUN="${USE_SRUN:-1}"
SRUN_CPUS_PER_TASK="${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-2}}"
REQUESTED_WALLTIME="${REQUESTED_WALLTIME:-72:00:00}"
REQUESTED_GPU_TOTAL="${REQUESTED_GPU_TOTAL:-8}"
REQUESTED_GPU_PER_NODE="${REQUESTED_GPU_PER_NODE:-4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

EVAL_ROOT="${EVAL_ROOT:-$HOME/cafa5}"
IA_FILE="${IA_FILE:-$EVAL_ROOT/IA.txt}"
GO_OBO_FILE="${GO_OBO_FILE:-$EVAL_ROOT/go-basic.obo}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-$EVAL_ROOT/test_terms.tsv}"
CAFAEVAL_BIN="${CAFAEVAL_BIN:-cafaeval}"
REQUIRE_IA_FILE="${REQUIRE_IA_FILE:-1}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
RUN_NAME="${RUN_NAME:-full_graph_${FRAMEWORK}_mtf${MIN_TERM_FREQUENCY}_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-$GRAPH_CACHE_DIR/training_runs/$RUN_NAME}"

expand_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

EVAL_ROOT="$(expand_path "$EVAL_ROOT")"
IA_FILE="$(expand_path "$IA_FILE")"
GO_OBO_FILE="$(expand_path "$GO_OBO_FILE")"
GROUND_TRUTH_FILE="$(expand_path "$GROUND_TRUTH_FILE")"

mkdir -p "$LOG_DIR" "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

echo "started_at=$(date -Iseconds)"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "partition=${SLURM_JOB_PARTITION:-savio2_1080ti}"
echo "node_list=${SLURM_JOB_NODELIST:-local}"
echo "job_num_nodes=${SLURM_JOB_NUM_NODES:-1}"
echo "ntasks=${SLURM_NTASKS:-1}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-$SRUN_CPUS_PER_TASK}"
echo "requested_walltime=$REQUESTED_WALLTIME"
echo "requested_gpu_total=$REQUESTED_GPU_TOTAL"
echo "requested_gpu_per_node=$REQUESTED_GPU_PER_NODE"
echo "slurm_gpus=${SLURM_GPUS:-}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  echo "allocated_nodes:"
  scontrol show hostnames "$SLURM_JOB_NODELIST"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" - "$REQUESTED_WALLTIME" <<'PY'
import datetime as dt
import re
import sys

walltime = sys.argv[1]
match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", walltime)
if match:
    days = int(match.group(1) or 0)
    hours = int(match.group(2))
    minutes = int(match.group(3))
    seconds = int(match.group(4))
    finish = dt.datetime.now().astimezone() + dt.timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )
    print("walltime_limit_finish_by=" + finish.isoformat(timespec="seconds"))
else:
    print("walltime_limit_finish_by=unknown")
PY

if [[ ! -f "$REPO_ROOT/train_minimal_graph_model.py" ]]; then
  echo "Missing script: $REPO_ROOT/train_minimal_graph_model.py" >&2
  echo "Set REPO_ROOT to your checked-out repo path before sbatch." >&2
  exit 1
fi

if [[ ! -d "$GRAPH_CACHE_DIR" ]]; then
  echo "Missing graph cache directory: $GRAPH_CACHE_DIR" >&2
  exit 1
fi

if [[ ! -f "$SPLIT_DIR/summary.json" ]]; then
  echo "Missing split summary: $SPLIT_DIR/summary.json" >&2
  echo "Run export_graph_dataloaders.py before full training." >&2
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
echo "GRAPH_CACHE_DIR=$GRAPH_CACHE_DIR"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "FRAMEWORK=$FRAMEWORK"
echo "ASPECTS=$ASPECTS"
echo "MIN_TERM_FREQUENCY=$MIN_TERM_FREQUENCY"
echo "EPOCHS=$EPOCHS"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "HIDDEN_DIM=$HIDDEN_DIM"
echo "DROPOUT=$DROPOUT"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "METRIC_THRESHOLD=$METRIC_THRESHOLD"
echo "FMAX_THRESHOLD_STEP=$FMAX_THRESHOLD_STEP"
echo "PROGRESS_MODE=$PROGRESS_MODE"
echo "PROGRESS_EVERY=$PROGRESS_EVERY"
echo "DEVICE=$DEVICE"
echo "SEED=$SEED"
echo "USE_ESM2=$USE_ESM2"
echo "USE_DSSP=$USE_DSSP"
echo "USE_SASA=$USE_SASA"
echo "CONTINUE_ON_FAILURE=$CONTINUE_ON_FAILURE"
echo "USE_SRUN=$USE_SRUN"
echo "SRUN_CPUS_PER_TASK=$SRUN_CPUS_PER_TASK"
echo "REQUESTED_WALLTIME=$REQUESTED_WALLTIME"
echo "REQUESTED_GPU_TOTAL=$REQUESTED_GPU_TOTAL"
echo "REQUESTED_GPU_PER_NODE=$REQUESTED_GPU_PER_NODE"
echo "GPU_IDS=$GPU_IDS"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "IA_FILE=$IA_FILE"
echo "GO_OBO_FILE=$GO_OBO_FILE"
echo "GROUND_TRUTH_FILE=$GROUND_TRUTH_FILE"
echo "CAFAEVAL_BIN=$CAFAEVAL_BIN"
echo "REQUIRE_IA_FILE=$REQUIRE_IA_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD > "$RUN_DIR/git_branch.txt" 2>/dev/null || true
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || true
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt" 2>/dev/null || true

mkdir -p "$RUN_DIR/eval_inputs"
if [[ -f "$IA_FILE" ]]; then
  ln -sfn "$IA_FILE" "$RUN_DIR/eval_inputs/IA.txt"
else
  echo "Missing IA_FILE: $IA_FILE" >&2
  if [[ "$REQUIRE_IA_FILE" == "1" ]]; then
    exit 1
  fi
fi
if [[ -f "$GO_OBO_FILE" ]]; then
  ln -sfn "$GO_OBO_FILE" "$RUN_DIR/eval_inputs/go-basic.obo"
else
  echo "GO_OBO_FILE not found yet: $GO_OBO_FILE" >&2
fi
if [[ -f "$GROUND_TRUTH_FILE" ]]; then
  ln -sfn "$GROUND_TRUTH_FILE" "$RUN_DIR/eval_inputs/test_terms.tsv"
else
  echo "GROUND_TRUTH_FILE not found yet: $GROUND_TRUTH_FILE" >&2
fi
if command -v "$CAFAEVAL_BIN" >/dev/null 2>&1; then
  command -v "$CAFAEVAL_BIN" > "$RUN_DIR/eval_inputs/cafaeval_path.txt"
else
  echo "cafaeval command not found yet: $CAFAEVAL_BIN" >&2
fi
cat > "$RUN_DIR/eval_inputs/cafaeval_command_template.sh" <<EOF
$CAFAEVAL_BIN "$GO_OBO_FILE" <prediction_dir> "$GROUND_TRUTH_FILE" -ia "$IA_FILE" -prop fill -norm cafa -th_step 0.001 -max_terms 500
EOF
cat > "$RUN_DIR/eval_inputs/README.txt" <<EOF
These files are staged for CAFA-style evaluation after prediction TSV export.
Current train_minimal_graph_model.py writes checkpoints plus unweighted threshold metrics and Fmax summaries, but does not yet export cafaeval prediction TSV files.
EOF

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

print("python_executable=" + sys.executable)
torch = importlib.import_module("torch")
print("torch=" + getattr(torch, "__version__", "unknown"))
backend = importlib.import_module("torch_geometric")
print("torch_geometric=" + getattr(backend, "__version__", "unknown"))
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device_count=" + str(torch.cuda.device_count()))
    print("cuda_device_name=" + torch.cuda.get_device_name(0))
PY

export RUN_DIR REPO_ROOT RUN_ROOT GRAPH_CACHE_DIR SPLIT_DIR PYTHON_BIN FRAMEWORK
export ASPECTS MIN_TERM_FREQUENCY EPOCHS BATCH_SIZE NUM_WORKERS HIDDEN_DIM
export DROPOUT LR WEIGHT_DECAY METRIC_THRESHOLD FMAX_THRESHOLD_STEP PROGRESS_MODE PROGRESS_EVERY DEVICE SEED USE_ESM2 USE_DSSP USE_SASA
export USE_SRUN SRUN_CPUS_PER_TASK REQUESTED_WALLTIME REQUESTED_GPU_TOTAL REQUESTED_GPU_PER_NODE
export GPU_IDS MAX_PARALLEL EVAL_ROOT IA_FILE GO_OBO_FILE GROUND_TRUTH_FILE CAFAEVAL_BIN
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "REPO_ROOT",
    "RUN_ROOT",
    "GRAPH_CACHE_DIR",
    "SPLIT_DIR",
    "PYTHON_BIN",
    "FRAMEWORK",
    "ASPECTS",
    "MIN_TERM_FREQUENCY",
    "EPOCHS",
    "BATCH_SIZE",
    "NUM_WORKERS",
    "HIDDEN_DIM",
    "DROPOUT",
    "LR",
    "WEIGHT_DECAY",
    "METRIC_THRESHOLD",
    "FMAX_THRESHOLD_STEP",
    "PROGRESS_MODE",
    "PROGRESS_EVERY",
    "DEVICE",
    "SEED",
    "USE_ESM2",
    "USE_DSSP",
    "USE_SASA",
    "USE_SRUN",
    "SRUN_CPUS_PER_TASK",
    "REQUESTED_WALLTIME",
    "REQUESTED_GPU_TOTAL",
    "REQUESTED_GPU_PER_NODE",
    "GPU_IDS",
    "MAX_PARALLEL",
    "EVAL_ROOT",
    "IA_FILE",
    "GO_OBO_FILE",
    "GROUND_TRUTH_FILE",
    "CAFAEVAL_BIN",
]
payload = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "slurm_job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
    "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
    "slurm_gpus": os.environ.get("SLURM_GPUS"),
    "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "config": {key: os.environ.get(key) for key in keys},
    "eval_inputs": {
        "ia_file_exists": Path(os.environ["IA_FILE"]).exists(),
        "go_obo_file_exists": Path(os.environ["GO_OBO_FILE"]).exists(),
        "ground_truth_file_exists": Path(os.environ["GROUND_TRUTH_FILE"]).exists(),
    },
}
Path(os.environ["RUN_DIR"], "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

ASPECTS_NORMALIZED="${ASPECTS//,/ }"
read -r -a ASPECT_ARRAY <<< "$ASPECTS_NORMALIZED"
GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
read -r -a GPU_ID_ARRAY <<< "$GPU_IDS_NORMALIZED"
USE_SRUN_ACTIVE=0
if [[ "$USE_SRUN" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  USE_SRUN_ACTIVE=1
fi

if [[ "${#ASPECT_ARRAY[@]}" == "0" ]]; then
  echo "ASPECTS is empty." >&2
  exit 1
fi
if [[ "${#GPU_ID_ARRAY[@]}" == "0" ]]; then
  echo "GPU_IDS is empty." >&2
  exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" == "0" ]]; then
  echo "MAX_PARALLEL must be a positive integer: $MAX_PARALLEL" >&2
  exit 1
fi
if [[ ! "$SRUN_CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [[ "$SRUN_CPUS_PER_TASK" == "0" ]]; then
  echo "SRUN_CPUS_PER_TASK must be a positive integer: $SRUN_CPUS_PER_TASK" >&2
  exit 1
fi
if [[ "$USE_SRUN_ACTIVE" != "1" ]] && (( MAX_PARALLEL > ${#GPU_ID_ARRAY[@]} )); then
  MAX_PARALLEL="${#GPU_ID_ARRAY[@]}"
fi
echo "USE_SRUN_ACTIVE=$USE_SRUN_ACTIVE"
if [[ "$USE_SRUN_ACTIVE" == "1" && "${#ASPECT_ARRAY[@]}" -lt "$REQUESTED_GPU_TOTAL" ]]; then
  echo "NOTE: requested $REQUESTED_GPU_TOTAL GPUs, but ASPECTS has ${#ASPECT_ARRAY[@]} independent task(s)."
  echo "NOTE: this single-GPU-per-aspect training script can use at most one GPU per active aspect unless DDP is added."
fi

EXTRA_ARGS=("$@")
failures=0
batch_pids=()
batch_labels=()

run_aspect() {
  local aspect_upper="$1"
  local aspect_lower="$2"
  local gpu_id="$3"
  checkpoint_dir="$RUN_DIR/$aspect_lower"
  mkdir -p "$checkpoint_dir"
  start_time="$(date -Iseconds)"

  cmd=(
    "$PYTHON_BIN" -u "$REPO_ROOT/train_minimal_graph_model.py"
    --root "$GRAPH_CACHE_DIR"
    --split-dir "$SPLIT_DIR"
    --framework "$FRAMEWORK"
    --aspect "$aspect_upper"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --hidden-dim "$HIDDEN_DIM"
    --dropout "$DROPOUT"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --metric-threshold "$METRIC_THRESHOLD"
    --fmax-threshold-step "$FMAX_THRESHOLD_STEP"
    --progress-mode "$PROGRESS_MODE"
    --progress-every "$PROGRESS_EVERY"
    --device "$DEVICE"
    --seed "$SEED"
    --min-term-frequency "$MIN_TERM_FREQUENCY"
    --checkpoint-dir "$checkpoint_dir"
  )
  if [[ "$USE_ESM2" != "1" ]]; then
    cmd+=(--disable-esm2)
  fi
  if [[ "$USE_DSSP" != "1" ]]; then
    cmd+=(--disable-dssp)
  fi
  if [[ "$USE_SASA" != "1" ]]; then
    cmd+=(--disable-sasa)
  fi

  {
    if [[ "$USE_SRUN_ACTIVE" == "1" ]]; then
      printf 'srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=%q --gres=gpu:1 ' "$SRUN_CPUS_PER_TASK"
    else
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    fi
    printf '%q ' "${cmd[@]}"
    printf '%q ' "${EXTRA_ARGS[@]}"
    printf '\n'
  } > "$checkpoint_dir/command.sh"

  echo ""
  echo "[$aspect_upper] started_at=$start_time gpu_id=$gpu_id"
  echo "[$aspect_upper] command=$(cat "$checkpoint_dir/command.sh")"

  set +e
  if [[ "$USE_SRUN_ACTIVE" == "1" ]]; then
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --gres=gpu:1 \
      "${cmd[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$checkpoint_dir/train.log"
    status_code="${PIPESTATUS[0]}"
  else
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$checkpoint_dir/train.log"
    status_code="${PIPESTATUS[0]}"
  fi
  set -e

  end_time="$(date -Iseconds)"
  if [[ "$status_code" == "0" ]]; then
    status="success"
    echo "[$aspect_upper] finished successfully at $end_time"
  else
    status="failed"
    failures=$((failures + 1))
    echo "[$aspect_upper] failed with status $status_code at $end_time" >&2
  fi

  "$PYTHON_BIN" - "$checkpoint_dir" "$aspect_upper" "$status" "$status_code" "$start_time" "$end_time" "$gpu_id" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
summary_path = checkpoint_dir / "summary.json"
payload = {
    "aspect": sys.argv[2],
    "status": sys.argv[3],
    "status_code": int(sys.argv[4]),
    "started_at": sys.argv[5],
    "finished_at": sys.argv[6],
    "gpu_id": sys.argv[7],
    "checkpoint_dir": str(checkpoint_dir),
    "summary_path": str(summary_path),
    "summary_exists": summary_path.exists(),
}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary.get("history") or []
    payload["best_val_loss"] = summary.get("best_val_loss")
    payload["best_checkpoint_path"] = summary.get("best_checkpoint_path")
    if history:
        payload["final_epoch"] = history[-1]
Path(checkpoint_dir, "run_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  return "$status_code"
}

wait_for_batch() {
  local index
  local pid
  local label
  local status_code
  for index in "${!batch_pids[@]}"; do
    pid="${batch_pids[$index]}"
    label="${batch_labels[$index]}"
    if wait "$pid"; then
      echo "[$label] batch wait status=0"
    else
      status_code="$?"
      failures=$((failures + 1))
      echo "[$label] batch wait status=$status_code" >&2
    fi
  done
  batch_pids=()
  batch_labels=()
}

for raw_aspect in "${ASPECT_ARRAY[@]}"; do
  aspect_upper="$(printf '%s' "$raw_aspect" | tr '[:lower:]' '[:upper:]')"
  aspect_lower="$(printf '%s' "$aspect_upper" | tr '[:upper:]' '[:lower:]')"
  case "$aspect_upper" in
    BPO|CCO|MFO) ;;
    *)
      echo "Invalid aspect: $raw_aspect. Expected BPO, CCO, or MFO." >&2
      exit 1
      ;;
  esac

  aspect_split_dir="$SPLIT_DIR/$aspect_lower"
  for split_name in train val test; do
    if [[ ! -f "$aspect_split_dir/$split_name.txt" ]]; then
      echo "Missing split file: $aspect_split_dir/$split_name.txt" >&2
      exit 1
    fi
  done

  if [[ "$USE_SRUN_ACTIVE" == "1" ]]; then
    gpu_id="srun-auto"
  else
    gpu_id="${GPU_ID_ARRAY[${#batch_pids[@]}]}"
  fi
  run_aspect "$aspect_upper" "$aspect_lower" "$gpu_id" &
  batch_pids+=("$!")
  batch_labels+=("$aspect_upper")

  if (( ${#batch_pids[@]} >= MAX_PARALLEL )); then
    wait_for_batch
    if [[ "$failures" != "0" && "$CONTINUE_ON_FAILURE" != "1" ]]; then
      break
    fi
  fi
done

if (( ${#batch_pids[@]} > 0 )); then
  wait_for_batch
fi

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for result_path in sorted(run_dir.glob("*/run_result.json")):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    final_epoch = result.get("final_epoch") or {}
    row = {
        "aspect": result.get("aspect"),
        "status": result.get("status"),
        "status_code": result.get("status_code"),
        "gpu_id": result.get("gpu_id"),
        "checkpoint_dir": result.get("checkpoint_dir"),
        "summary_path": result.get("summary_path"),
        "best_val_loss": result.get("best_val_loss"),
        "best_checkpoint_path": result.get("best_checkpoint_path"),
        "final_epoch": final_epoch.get("epoch"),
        "final_epoch_seconds": final_epoch.get("epoch_seconds"),
        "average_epoch_seconds": final_epoch.get("average_epoch_seconds"),
        "estimated_remaining_seconds": final_epoch.get("estimated_remaining_seconds"),
        "estimated_finished_at": final_epoch.get("estimated_finished_at"),
    }
    for split_name in ("train", "val", "test"):
        metrics = final_epoch.get(split_name) or {}
        row[f"{split_name}_loss"] = metrics.get("loss")
        row[f"{split_name}_micro_precision"] = metrics.get("micro_precision")
        row[f"{split_name}_micro_recall"] = metrics.get("micro_recall")
        row[f"{split_name}_micro_f1"] = metrics.get("micro_f1")
        row[f"{split_name}_macro_precision"] = metrics.get("macro_precision")
        row[f"{split_name}_macro_recall"] = metrics.get("macro_recall")
        row[f"{split_name}_macro_f1"] = metrics.get("macro_f1")
        row[f"{split_name}_macro_f1_positive_labels"] = metrics.get("macro_f1_positive_labels")
        row[f"{split_name}_fmax"] = metrics.get("fmax")
        row[f"{split_name}_fmax_threshold"] = metrics.get("fmax_threshold")
        row[f"{split_name}_fmax_precision"] = metrics.get("fmax_precision")
        row[f"{split_name}_fmax_recall"] = metrics.get("fmax_recall")
        row[f"{split_name}_graphs"] = metrics.get("graphs")
    rows.append(row)

(run_dir / "results_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
base_columns = [
    "aspect",
    "status",
    "status_code",
    "gpu_id",
    "final_epoch",
    "final_epoch_seconds",
    "average_epoch_seconds",
    "estimated_remaining_seconds",
    "estimated_finished_at",
    "best_val_loss",
]
metric_columns = []
for split_name in ("train", "val", "test"):
    metric_columns.extend(
        [
            f"{split_name}_loss",
            f"{split_name}_micro_precision",
            f"{split_name}_micro_recall",
            f"{split_name}_micro_f1",
            f"{split_name}_macro_precision",
            f"{split_name}_macro_recall",
            f"{split_name}_macro_f1",
            f"{split_name}_macro_f1_positive_labels",
            f"{split_name}_fmax",
            f"{split_name}_fmax_threshold",
            f"{split_name}_fmax_precision",
            f"{split_name}_fmax_recall",
            f"{split_name}_graphs",
        ]
    )
columns = [
    *base_columns,
    *metric_columns,
    "checkpoint_dir",
    "summary_path",
    "best_checkpoint_path",
]
with (run_dir / "results_summary.tsv").open("w", encoding="utf-8") as handle:
    handle.write("\t".join(columns) + "\n")
    for row in rows:
        handle.write("\t".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + "\n")
print("wrote " + str(run_dir / "results_summary.json"))
print("wrote " + str(run_dir / "results_summary.tsv"))
PY

echo "finished_at=$(date -Iseconds)"
echo "results_dir=$RUN_DIR"

if [[ "$failures" != "0" ]]; then
  echo "Training finished with $failures failed aspect(s)." >&2
  exit 1
fi
