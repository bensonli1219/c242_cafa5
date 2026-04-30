#!/usr/bin/env python3
"""Extract graph training results from train.log files for report notebooks.

This script intentionally parses the per-epoch metric lines in ``train.log``
instead of reading per-aspect ``summary.json`` files. It writes compact CSV
artifacts that the progress-report notebook can load without hard-coding
metric values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GRAPH_ROOT = Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache")
DEFAULT_NORMALIZED_ROOT = Path(
    "/global/scratch/users/bensonli/cafa5_outputs/graph_cache_normalized_features"
)

EPOCH_LINE_RE = re.compile(r"^epoch=(?P<epoch>\d+)\s+(?P<body>.*)$")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")


@dataclass(frozen=True)
class RunSpec:
    stage: str
    model_change: str
    root: Path
    run_dir: str
    include_in_local_tuning: bool = False
    include_in_followup: bool = False

    @property
    def path(self) -> Path:
        return self.root / "training_runs" / self.run_dir


def build_default_run_specs(graph_root: Path, normalized_root: Path) -> list[RunSpec]:
    return [
        RunSpec(
            "Initial comparison",
            "Raw graph baseline",
            graph_root,
            "full_graph_pyg_mtf20_33234089",
        ),
        RunSpec(
            "Initial comparison",
            "Normalized graph features",
            normalized_root,
            "cco_mfo_parallel_20260415_1906",
        ),
        RunSpec(
            "Initial comparison",
            "Broad tuned recipe",
            graph_root,
            "full_graph_tuned_pyg_mtf20_33275343",
        ),
        RunSpec(
            "Targeted local tuning",
            "Baseline rerun",
            graph_root,
            "fmax_plan_s2_E0_control_20260420_223454",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Targeted local tuning",
            "Lower learning rate, 0.0007",
            graph_root,
            "fmax_plan_s2_E1_lr7e4_20260420_223454",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Targeted local tuning",
            "Lower learning rate, 0.0005",
            graph_root,
            "fmax_plan_g2080_E2_lr5e4_20260421_094844",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Targeted local tuning",
            "Moderate hidden size",
            graph_root,
            "fmax_plan_g2080_E3_h192_20260420_223136",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Targeted local tuning",
            "Larger hidden size",
            graph_root,
            "fmax_plan_g2080_E4_h256_20260421_094844",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Targeted local tuning",
            "Weighted BCE",
            graph_root,
            "fmax_plan_g2080_E5_weighted_only_20260420_223136",
            include_in_local_tuning=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Focal loss",
            graph_root,
            "sigimp_n1_focal_bce_20260422_172916",
            include_in_followup=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Logit calibration",
            graph_root,
            "sigimp_n2_logit_adjust_20260422_172916",
            include_in_followup=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Label-aware scorer",
            graph_root,
            "sigimp_n3_label_dot_20260422_172916",
            include_in_followup=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Label-aware scorer, longer confirmation",
            graph_root,
            "sigimp_n3_confirm_long_20260425_n3_confirm",
            include_in_followup=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Label-aware scorer, second seed",
            graph_root,
            "sigimp_n3_stability_seed2027_20260423_104340",
            include_in_followup=True,
        ),
        RunSpec(
            "Follow-up model changes",
            "Ontology regularization",
            graph_root,
            "sigimp_n3_ontology_reg_20260423_105633",
            include_in_followup=True,
        ),
    ]


def parse_value(value: str) -> int | float | str:
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_epoch_metrics(log_path: Path) -> list[dict[str, int | float | str]]:
    records: list[dict[str, int | float | str]] = []
    if not log_path.exists():
        return records

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_LINE_RE.match(line.strip())
        if not match:
            continue
        record: dict[str, int | float | str] = {"epoch": int(match.group("epoch"))}
        for kv_match in KEY_VALUE_RE.finditer(match.group("body")):
            record[kv_match.group("key")] = parse_value(kv_match.group("value"))
        records.append(record)
    return records


def read_run_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def extract_training_rows(run_specs: list[RunSpec], aspects: list[str]) -> tuple[list[dict], list[dict]]:
    epoch_rows: list[dict] = []
    best_rows: list[dict] = []

    for spec in run_specs:
        run_config = read_run_config(spec.path)
        for aspect in aspects:
            log_path = spec.path / aspect.lower() / "train.log"
            records = parse_epoch_metrics(log_path)
            status = "parsed" if records else "missing_log_or_epoch_metrics"
            for record in records:
                epoch_rows.append(
                    {
                        "stage": spec.stage,
                        "model_change": spec.model_change,
                        "aspect": aspect,
                        "run_dir": spec.run_dir,
                        "include_in_local_tuning": spec.include_in_local_tuning,
                        "include_in_followup": spec.include_in_followup,
                        "status": status,
                        **record,
                    }
                )
            if not records:
                continue

            best = max(records, key=lambda item: float(item.get("val_fmax", float("-inf"))))
            best_rows.append(
                {
                    "stage": spec.stage,
                    "model_change": spec.model_change,
                    "aspect": aspect,
                    "run_dir": spec.run_dir,
                    "include_in_local_tuning": spec.include_in_local_tuning,
                    "include_in_followup": spec.include_in_followup,
                    "status": status,
                    "best_epoch": best.get("epoch"),
                    "best_val_fmax": best.get("val_fmax"),
                    "best_test_fmax": best.get("test_fmax"),
                    "best_val_loss": best.get("val_loss"),
                    "best_test_loss": best.get("test_loss"),
                    "epochs_completed": max(int(row["epoch"]) for row in records),
                    "loss_function": run_config.get("loss_function", ""),
                    "model_head": run_config.get("model_head", "flat_linear"),
                    "hidden_dim": run_config.get("hidden_dim", ""),
                    "dropout": run_config.get("dropout", ""),
                    "lr": run_config.get("lr", ""),
                    "weight_decay": run_config.get("weight_decay", ""),
                }
            )

    return epoch_rows, best_rows


def extract_split_rows(graph_root: Path, aspects: list[str]) -> list[dict]:
    rows: list[dict] = []
    split_root = graph_root / "splits"
    for aspect in aspects + ["BPO"]:
        aspect_lower = aspect.lower()
        split_counts = {}
        for split_name in ("train", "val", "test"):
            split_file = split_root / aspect_lower / f"{split_name}.txt"
            if split_file.exists():
                split_counts[split_name] = sum(1 for line in split_file.read_text().splitlines() if line.strip())
            else:
                split_counts[split_name] = None

        split_summary_path = split_root / aspect_lower / "summary.json"
        vocab_size = None
        if split_summary_path.exists():
            with split_summary_path.open(encoding="utf-8") as handle:
                split_summary = json.load(handle)
            vocab_size = split_summary.get("vocab_size")

        rows.append(
            {
                "aspect": aspect,
                "proteins": sum(count or 0 for count in split_counts.values()),
                "labels": vocab_size,
                "train": split_counts["train"],
                "validation": split_counts["val"],
                "test": split_counts["test"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--normalized-root", type=Path, default=DEFAULT_NORMALIZED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/jupyter-notebook/report_assets"))
    parser.add_argument("--aspects", nargs="+", default=["CCO", "MFO"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_default_run_specs(args.graph_root, args.normalized_root)
    epoch_rows, best_rows = extract_training_rows(specs, args.aspects)
    split_rows = extract_split_rows(args.graph_root, args.aspects)

    write_csv(args.output_dir / "graph_training_epoch_metrics.csv", epoch_rows)
    write_csv(args.output_dir / "graph_training_best_results.csv", best_rows)
    write_csv(args.output_dir / "graph_split_overview.csv", split_rows)

    print(f"wrote {args.output_dir / 'graph_training_epoch_metrics.csv'} ({len(epoch_rows)} rows)")
    print(f"wrote {args.output_dir / 'graph_training_best_results.csv'} ({len(best_rows)} rows)")
    print(f"wrote {args.output_dir / 'graph_split_overview.csv'} ({len(split_rows)} rows)")


if __name__ == "__main__":
    main()
