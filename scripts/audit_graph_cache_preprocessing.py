#!/usr/bin/env python3
"""Audit CAFA graph-cache preprocessing and create small filtered split sets.

The audit is metadata-first so it can run quickly on login nodes for a small
sample. Optional tensor checks load a bounded number of graph ``.pt`` files.
Use the Savio wrapper for full-cache tensor audits or expensive preprocessing
experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs


DEFAULT_ROOT = Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache")
DEFAULT_ASPECTS = ("BPO", "CCO", "MFO")
DEFAULT_MIN_FREQS = (20, 50, 100)
DEFAULT_GRAPH_CAPS = (800, 1200, 1600, 2000)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit graph-cache preprocessing quality and write optional filtered sample splits."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aspects", nargs="*", default=list(DEFAULT_ASPECTS))
    parser.add_argument("--min-term-frequencies", nargs="*", type=int, default=list(DEFAULT_MIN_FREQS))
    parser.add_argument("--graph-residue-caps", nargs="*", type=int, default=list(DEFAULT_GRAPH_CAPS))
    parser.add_argument("--seed", type=int, default=dataloaders.DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--check-graph-files",
        action="store_true",
        help="Stat every graph_path. This is useful but slow on scratch; prefer Savio for full checks.",
    )
    parser.add_argument("--tensor-sample-size", type=int, default=0)
    parser.add_argument("--write-experiment-splits", action="store_true")
    parser.add_argument("--experiment-name", default="sample_mtf20_cap1200")
    parser.add_argument("--experiment-min-term-frequency", type=int, default=20)
    parser.add_argument("--experiment-max-residues", type=int, default=1200)
    parser.add_argument("--experiment-sample-per-aspect", type=int, default=600)
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + ((upper_value - lower_value) * (position - lower))


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    parsed = sorted(float(value) for value in values if value is not None)
    if not parsed:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(parsed),
        "min": parsed[0],
        "p50": quantile(parsed, 0.50),
        "p90": quantile(parsed, 0.90),
        "p95": quantile(parsed, 0.95),
        "p99": quantile(parsed, 0.99),
        "max": parsed[-1],
        "mean": sum(parsed) / len(parsed),
    }


def normalize_aspects(values: Iterable[str]) -> list[str]:
    aspects = []
    for value in values:
        aspect = value.upper()
        if aspect not in graphs.ASPECT_TO_LABEL_KEY:
            raise ValueError(f"Unknown aspect: {value}")
        if aspect not in aspects:
            aspects.append(aspect)
    return aspects


def build_vocab(term_counts: dict[str, int], min_term_frequency: int) -> set[str]:
    return {term for term, count in term_counts.items() if int(count) >= min_term_frequency}


def labels_for(entry: dict[str, Any], aspect: str) -> set[str]:
    return {str(term) for term in (entry.get("labels") or {}).get(aspect, [])}


def eligible_entries(
    entries: list[dict[str, Any]],
    vocab: set[str],
    aspect: str,
    max_residues: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for entry in entries:
        if max_residues is not None and int(entry.get("residue_count") or 0) > max_residues:
            continue
        if labels_for(entry, aspect) & vocab:
            selected.append(entry)
    return selected


def split_ids(entry_ids: list[str], seed: int) -> dict[str, list[str]]:
    return dataloaders.split_entry_ids(
        entry_ids,
        train_ratio=dataloaders.DEFAULT_TRAIN_RATIO,
        val_ratio=dataloaders.DEFAULT_VAL_RATIO,
        test_ratio=dataloaders.DEFAULT_TEST_RATIO,
        seed=seed,
    )


def write_experiment_splits(
    output_dir: Path,
    name: str,
    entries: list[dict[str, Any]],
    term_counts: dict[str, dict[str, int]],
    aspects: list[str],
    min_term_frequency: int,
    max_residues: int,
    sample_per_aspect: int,
    seed: int,
) -> dict[str, Any]:
    split_root = output_dir / "experiment_splits" / name
    rng = random.Random(seed)
    summary: dict[str, Any] = {
        "name": name,
        "root": str(split_root.resolve()),
        "min_term_frequency": min_term_frequency,
        "max_residues": max_residues,
        "sample_per_aspect": sample_per_aspect,
        "seed": seed,
        "aspects": {},
    }
    for aspect in aspects:
        vocab = build_vocab(term_counts[aspect], min_term_frequency)
        selected = eligible_entries(entries, vocab, aspect, max_residues=max_residues)
        selected_ids = sorted(str(entry["entry_id"]) for entry in selected)
        if sample_per_aspect > 0 and len(selected_ids) > sample_per_aspect:
            selected_ids = sorted(rng.sample(selected_ids, sample_per_aspect))
        splits = split_ids(selected_ids, seed=seed)
        aspect_dir = split_root / aspect.lower()
        aspect_dir.mkdir(parents=True, exist_ok=True)
        for split_name, ids in splits.items():
            (aspect_dir / f"{split_name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        aspect_summary = {
            "aspect": aspect,
            "entry_count": len(selected_ids),
            "vocab_size": len(vocab),
            "counts": {split_name: len(ids) for split_name, ids in splits.items()},
            "entry_ids": splits,
        }
        write_json(aspect_dir / "summary.json", aspect_summary)
        summary["aspects"][aspect] = aspect_summary
    write_json(split_root / "summary.json", summary)
    return summary


def audit_tensor_sample(entries: list[dict[str, Any]], output_dir: Path, sample_size: int, seed: int) -> dict[str, Any]:
    if sample_size <= 0:
        return {"enabled": False}
    if graphs.torch is None:
        return {"enabled": True, "skipped": "torch is not available"}

    rng = random.Random(seed)
    sample = entries if len(entries) <= sample_size else rng.sample(entries, sample_size)
    rows = []
    failures = []
    torch_module = graphs.torch
    for entry in sample:
        graph_path = Path(entry.get("graph_path") or "")
        row: dict[str, Any] = {
            "entry_id": entry.get("entry_id"),
            "graph_path": str(graph_path),
            "exists": graph_path.exists(),
        }
        try:
            payload = torch_module.load(graph_path, map_location="cpu", weights_only=False)
            x = payload.get("x")
            pos = payload.get("pos")
            edge_index = payload.get("edge_index")
            edge_attr = payload.get("edge_attr")
            row.update(
                {
                    "nodes": int(x.shape[0]) if x is not None else None,
                    "node_features": int(x.shape[1]) if x is not None and x.ndim > 1 else None,
                    "edges": int(edge_index.shape[1]) if edge_index is not None and edge_index.ndim > 1 else None,
                    "edge_features": int(edge_attr.shape[1]) if edge_attr is not None and edge_attr.ndim > 1 else None,
                    "x_finite": bool(torch_module.isfinite(x).all().item()) if x is not None else False,
                    "pos_finite": bool(torch_module.isfinite(pos).all().item()) if pos is not None else False,
                    "edge_attr_finite": bool(torch_module.isfinite(edge_attr).all().item()) if edge_attr is not None else False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(row)
        rows.append(row)

    write_csv(output_dir / "tensor_sample_audit.csv", rows)
    return {
        "enabled": True,
        "sampled": len(rows),
        "failures": len(failures),
        "nonfinite_x": sum(1 for row in rows if row.get("x_finite") is False),
        "nonfinite_pos": sum(1 for row in rows if row.get("pos_finite") is False),
        "nonfinite_edge_attr": sum(1 for row in rows if row.get("edge_attr_finite") is False),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "preprocessing_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    aspects = normalize_aspects(args.aspects)

    entries = load_json(root / "metadata" / "entries.json")
    term_counts = load_json(root / "metadata" / "term_counts.json")

    missing_graph_count = None
    if args.check_graph_files:
        graph_paths = [Path(entry.get("graph_path") or "") for entry in entries]
        missing_graph_count = sum(1 for path in graph_paths if not path.exists())
    residue_counts = [int(entry.get("residue_count") or 0) for entry in entries]
    fragment_counts = [int(entry.get("fragment_count") or 0) for entry in entries]

    overview_rows = [
        {"metric": "entry_count", "value": len(entries)},
        {"metric": "graph_files_missing", "value": missing_graph_count},
        {"metric": "residue_count_zero", "value": sum(1 for value in residue_counts if value == 0)},
        {"metric": "fragment_count_zero", "value": sum(1 for value in fragment_counts if value == 0)},
    ]
    for prefix, values in (("residue_count", residue_counts), ("fragment_count", fragment_counts)):
        for key, value in numeric_summary(values).items():
            overview_rows.append({"metric": f"{prefix}_{key}", "value": value})
    write_csv(output_dir / "overview.csv", overview_rows, fieldnames=["metric", "value"])

    aspect_rows = []
    for aspect in aspects:
        label_counts = [len(labels_for(entry, aspect)) for entry in entries]
        labeled = [entry for entry in entries if labels_for(entry, aspect)]
        row = {
            "aspect": aspect,
            "entries_with_any_label": len(labeled),
            "entries_without_label": len(entries) - len(labeled),
            "unique_terms_raw": len(term_counts[aspect]),
        }
        for key, value in numeric_summary(label_counts).items():
            row[f"labels_per_entry_{key}"] = value
        aspect_rows.append(row)
    write_csv(output_dir / "aspect_label_overview.csv", aspect_rows)

    minfreq_rows = []
    for aspect in aspects:
        for min_freq in sorted(set(args.min_term_frequencies)):
            vocab = build_vocab(term_counts[aspect], min_freq)
            selected = eligible_entries(entries, vocab, aspect)
            split_counts = {
                split_name: len(ids)
                for split_name, ids in split_ids([str(entry["entry_id"]) for entry in selected], seed=args.seed).items()
            }
            minfreq_rows.append(
                {
                    "aspect": aspect,
                    "min_term_frequency": min_freq,
                    "vocab_size": len(vocab),
                    "eligible_entries": len(selected),
                    "train_count": split_counts["train"],
                    "val_count": split_counts["val"],
                    "test_count": split_counts["test"],
                }
            )
    write_csv(output_dir / "min_frequency_scenarios.csv", minfreq_rows)

    cap_rows = []
    for aspect in aspects:
        for min_freq in sorted(set(args.min_term_frequencies)):
            vocab = build_vocab(term_counts[aspect], min_freq)
            uncapped = eligible_entries(entries, vocab, aspect)
            for cap in sorted(set(args.graph_residue_caps)):
                capped = eligible_entries(entries, vocab, aspect, max_residues=cap)
                removed = len(uncapped) - len(capped)
                cap_rows.append(
                    {
                        "aspect": aspect,
                        "min_term_frequency": min_freq,
                        "max_residues": cap,
                        "vocab_size": len(vocab),
                        "eligible_entries_uncapped": len(uncapped),
                        "eligible_entries_capped": len(capped),
                        "entries_removed_by_cap": removed,
                        "removed_fraction": (removed / len(uncapped)) if uncapped else 0.0,
                    }
                )
    write_csv(output_dir / "graph_cap_scenarios.csv", cap_rows)

    top_terms_rows = []
    for aspect in aspects:
        for term, count in Counter(term_counts[aspect]).most_common(100):
            top_terms_rows.append({"aspect": aspect, "term": term, "count": count})
    write_csv(output_dir / "top_terms.csv", top_terms_rows)

    tensor_summary = audit_tensor_sample(entries, output_dir, args.tensor_sample_size, args.seed)

    experiment_summary = None
    if args.write_experiment_splits:
        experiment_summary = write_experiment_splits(
            output_dir=output_dir,
            name=args.experiment_name,
            entries=entries,
            term_counts=term_counts,
            aspects=aspects,
            min_term_frequency=args.experiment_min_term_frequency,
            max_residues=args.experiment_max_residues,
            sample_per_aspect=args.experiment_sample_per_aspect,
            seed=args.seed,
        )

    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "aspects": aspects,
        "min_term_frequencies": sorted(set(args.min_term_frequencies)),
        "graph_residue_caps": sorted(set(args.graph_residue_caps)),
        "check_graph_files": bool(args.check_graph_files),
        "overview_csv": str((output_dir / "overview.csv").resolve()),
        "min_frequency_scenarios_csv": str((output_dir / "min_frequency_scenarios.csv").resolve()),
        "graph_cap_scenarios_csv": str((output_dir / "graph_cap_scenarios.csv").resolve()),
        "tensor_summary": tensor_summary,
        "experiment_splits": experiment_summary,
    }
    write_json(output_dir / "summary.json", summary)
    console_summary = dict(summary)
    if console_summary.get("experiment_splits"):
        experiment = dict(console_summary["experiment_splits"])
        experiment["aspects"] = {
            aspect: {
                key: value
                for key, value in aspect_summary.items()
                if key != "entry_ids"
            }
            for aspect, aspect_summary in experiment.get("aspects", {}).items()
        }
        console_summary["experiment_splits"] = experiment
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
