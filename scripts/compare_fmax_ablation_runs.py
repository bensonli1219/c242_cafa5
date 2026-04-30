#!/usr/bin/env python3
"""Compare CAFA graph training runs by best validation Fmax."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Run directories to compare. Defaults to fmax5_* runs under --training-runs-dir.",
    )
    parser.add_argument(
        "--training-runs-dir",
        type=Path,
        default=Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs"),
    )
    parser.add_argument("--glob", default="fmax5_*", help="Run glob used when no run directories are passed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional TSV output path.")
    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def config_value(run_dir: Path, key: str) -> Any:
    run_config = read_json(run_dir / "run_config.json", default={}) or {}
    return (run_config.get("config") or {}).get(key)


def best_epoch_record(summary: dict[str, Any]) -> dict[str, Any]:
    best_epoch = summary.get("best_epoch")
    history = summary.get("history") or []
    for record in history:
        if record.get("epoch") == best_epoch:
            return record
    return max(history, key=lambda record: (((record.get("val") or {}).get("fmax")) or -1.0), default={})


def final_epoch_record(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history") or []
    return history[-1] if history else {}


def metric(record: dict[str, Any], split: str, name: str) -> Any:
    return ((record.get(split) or {}).get(name)) if record else None


def rows_for_run(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    aspect_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and ((path / "run_result.json").exists() or (path / "summary.json").exists())
    ]
    for aspect_dir in sorted(aspect_dirs):
        run_result = read_json(aspect_dir / "run_result.json", default={}) or {}
        summary = read_json(aspect_dir / "summary.json", default={}) or {}
        best_record = best_epoch_record(summary)
        final_record = final_epoch_record(summary)
        row = {
            "run_name": run_dir.name,
            "aspect": (run_result.get("aspect") or aspect_dir.name).upper(),
            "status": run_result.get("status"),
            "status_code": run_result.get("status_code"),
            "loss_function": summary.get("loss_function") or config_value(run_dir, "LOSS_FUNCTION"),
            "hidden_dim": summary.get("hidden_dim") or config_value(run_dir, "HIDDEN_DIM"),
            "dropout": summary.get("dropout") or config_value(run_dir, "DROPOUT"),
            "lr": summary.get("lr") or config_value(run_dir, "LR"),
            "weight_decay": summary.get("weight_decay") or config_value(run_dir, "WEIGHT_DECAY"),
            "pos_weight_power": (summary.get("loss_config") or {}).get("pos_weight_power")
            or config_value(run_dir, "POS_WEIGHT_POWER"),
            "max_pos_weight": (summary.get("loss_config") or {}).get("max_pos_weight")
            or config_value(run_dir, "MAX_POS_WEIGHT"),
            "checkpoint_metric": summary.get("checkpoint_metric") or run_result.get("best_checkpoint_metric_name"),
            "best_epoch": summary.get("best_epoch") or run_result.get("best_epoch"),
            "final_epoch": final_record.get("epoch"),
            "best_val_fmax": metric(best_record, "val", "fmax"),
            "best_test_fmax": metric(best_record, "test", "fmax"),
            "best_val_micro_f1": metric(best_record, "val", "micro_f1"),
            "best_test_micro_f1": metric(best_record, "test", "micro_f1"),
            "best_val_macro_f1": metric(best_record, "val", "macro_f1"),
            "best_test_macro_f1": metric(best_record, "test", "macro_f1"),
            "final_val_fmax": metric(final_record, "val", "fmax"),
            "final_test_fmax": metric(final_record, "test", "fmax"),
            "checkpoint_dir": str(aspect_dir),
        }
        if row["best_test_fmax"] is not None and row["final_test_fmax"] is not None:
            row["best_minus_final_test_fmax"] = float(row["best_test_fmax"]) - float(row["final_test_fmax"])
        else:
            row["best_minus_final_test_fmax"] = None
        rows.append(row)
    return rows


def sort_key(row: dict[str, Any]) -> tuple[str, int, float]:
    status_rank = 0 if row.get("status") == "success" else 1
    value = row.get("best_val_fmax")
    return str(row.get("aspect")), status_rank, -(float(value) if value is not None else -1.0)


def write_rows(rows: list[dict[str, Any]], output: Path | None) -> None:
    if not rows:
        print("No runs found.")
        return
    columns = list(rows[0].keys())
    target = output.open("w", encoding="utf-8", newline="") if output else None
    try:
        handle = target if target is not None else None
        if handle is None:
            import sys

            handle = sys.stdout
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if target is not None:
            target.close()


def main() -> int:
    args = parse_args()
    run_dirs = args.runs or sorted(args.training_runs_dir.glob(args.glob), key=lambda path: path.stat().st_mtime)
    rows = []
    for run_dir in run_dirs:
        if run_dir.is_dir():
            rows.extend(rows_for_run(run_dir))
    write_rows(sorted(rows, key=sort_key), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
