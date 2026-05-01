from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_minimal_graph_model as graph_training


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse graph and sequence prediction bundles for late-fusion experiments. "
            "Each bundle directory must contain scores.npy, entry_ids.txt, and terms.txt."
        )
    )
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--sequence-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graph-weight", type=non_negative_float, default=0.5)
    parser.add_argument("--sequence-weight", type=non_negative_float, default=0.5)
    parser.add_argument("--score-space", choices=["probabilities", "logits"], default="probabilities")
    parser.add_argument(
        "--evaluate-with-graph-root",
        type=Path,
        default=None,
        help="Optional graph cache root. When set, compute multilabel metrics for the fused bundle.",
    )
    parser.add_argument("--aspect", choices=["BPO", "CCO", "MFO"], default=None)
    parser.add_argument("--metric-threshold", type=non_negative_float, default=0.5)
    parser.add_argument("--fmax-threshold-step", type=positive_float, default=0.01)
    return parser.parse_args(argv)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def load_bundle(bundle_dir: Path, score_space: str = "probabilities") -> dict[str, Any]:
    scores_path = bundle_dir / ("logits.npy" if score_space == "logits" and (bundle_dir / "logits.npy").exists() else "scores.npy")
    entry_ids_path = bundle_dir / "entry_ids.txt"
    terms_path = bundle_dir / "terms.txt"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores file: {scores_path}")
    if not entry_ids_path.exists():
        raise FileNotFoundError(f"Missing entry id file: {entry_ids_path}")
    if not terms_path.exists():
        raise FileNotFoundError(f"Missing terms file: {terms_path}")

    scores = np.load(scores_path)
    entry_ids = read_lines(entry_ids_path)
    terms = read_lines(terms_path)
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix in {scores_path}, got shape {scores.shape}")
    if scores.shape[0] != len(entry_ids):
        raise ValueError(
            f"Entry id count mismatch for {bundle_dir}: scores rows={scores.shape[0]} entry_ids={len(entry_ids)}"
        )
    if scores.shape[1] != len(terms):
        raise ValueError(
            f"Term count mismatch for {bundle_dir}: scores cols={scores.shape[1]} terms={len(terms)}"
        )

    meta_path = bundle_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "scores": scores.astype(np.float32, copy=False),
        "entry_ids": entry_ids,
        "terms": terms,
        "meta": meta,
    }


def validate_compatible_bundles(graph_bundle: dict[str, Any], sequence_bundle: dict[str, Any]) -> None:
    if graph_bundle["entry_ids"] != sequence_bundle["entry_ids"]:
        raise ValueError("Graph and sequence bundles do not share the same entry_ids ordering.")
    if graph_bundle["terms"] != sequence_bundle["terms"]:
        raise ValueError("Graph and sequence bundles do not share the same GO term ordering.")


def align_bundle_to_reference(reference_bundle: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    if reference_bundle["terms"] != bundle["terms"]:
        raise ValueError("Graph and sequence bundles do not share the same GO term ordering.")
    if reference_bundle["entry_ids"] == bundle["entry_ids"]:
        return bundle
    reference_ids = reference_bundle["entry_ids"]
    bundle_ids = bundle["entry_ids"]
    if set(reference_ids) != set(bundle_ids) or len(reference_ids) != len(bundle_ids):
        raise ValueError("Graph and sequence bundles do not contain the same entry_ids.")
    index_by_entry = {entry_id: index for index, entry_id in enumerate(bundle_ids)}
    order = [index_by_entry[entry_id] for entry_id in reference_ids]
    aligned = dict(bundle)
    aligned["scores"] = bundle["scores"][order]
    aligned["entry_ids"] = list(reference_ids)
    aligned["meta"] = dict(bundle.get("meta") or {})
    aligned["meta"]["reordered_to_match_reference"] = True
    return aligned


def fuse_scores(
    graph_scores: np.ndarray,
    sequence_scores: np.ndarray,
    graph_weight: float,
    sequence_weight: float,
    score_space: str,
) -> dict[str, np.ndarray]:
    total_weight = graph_weight + sequence_weight
    if total_weight <= 0:
        raise ValueError("graph_weight + sequence_weight must be positive.")
    norm_graph = graph_weight / total_weight
    norm_sequence = sequence_weight / total_weight

    if score_space == "probabilities":
        fused_scores = (norm_graph * graph_scores) + (norm_sequence * sequence_scores)
        return {"fused_scores": fused_scores.astype(np.float32, copy=False)}

    fused_logits = (norm_graph * graph_scores) + (norm_sequence * sequence_scores)
    return {
        "fused_logits": fused_logits.astype(np.float32, copy=False),
        "fused_scores": sigmoid(fused_logits).astype(np.float32, copy=False),
    }


def write_bundle(
    output_dir: Path,
    *,
    entry_ids: list[str],
    terms: list[str],
    fused_payload: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scores.npy", fused_payload["fused_scores"])
    if "fused_logits" in fused_payload:
        np.save(output_dir / "logits.npy", fused_payload["fused_logits"])
    (output_dir / "entry_ids.txt").write_text("\n".join(entry_ids) + "\n", encoding="utf-8")
    (output_dir / "terms.txt").write_text("\n".join(terms) + "\n", encoding="utf-8")
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_targets_from_graph_root(
    graph_root: Path,
    *,
    aspect: str,
    entry_ids: list[str],
    terms: list[str],
) -> np.ndarray:
    entries_path = graph_root / "metadata" / "entries.json"
    if not entries_path.exists():
        raise FileNotFoundError(f"Missing graph metadata entries: {entries_path}")
    entries = json.loads(entries_path.read_text(encoding="utf-8"))
    labels_by_entry = {
        str(entry["entry_id"]): set((entry.get("labels") or {}).get(aspect, []))
        for entry in entries
    }
    targets = np.zeros((len(entry_ids), len(terms)), dtype=np.float32)
    for row, entry_id in enumerate(entry_ids):
        labels = labels_by_entry.get(entry_id)
        if labels is None:
            raise ValueError(f"Entry {entry_id} was not found in {entries_path}")
        for col, term in enumerate(terms):
            if term in labels:
                targets[row, col] = 1.0
    return targets


def evaluate_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    metric_threshold: float,
    fmax_threshold_step: float,
) -> dict[str, float]:
    torch_module = graph_training.require_torch()
    logits = torch_module.logit(
        torch_module.clamp(torch_module.from_numpy(scores.astype(np.float32, copy=False)), min=1e-7, max=1.0 - 1e-7)
    )
    labels = torch_module.from_numpy(targets.astype(np.float32, copy=False))
    return graph_training.multilabel_metrics_from_logits(
        logits,
        labels,
        threshold=metric_threshold,
        fmax_threshold_step=fmax_threshold_step,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph_bundle = load_bundle(args.graph_bundle, score_space=args.score_space)
    sequence_bundle = load_bundle(args.sequence_bundle, score_space=args.score_space)
    sequence_bundle = align_bundle_to_reference(graph_bundle, sequence_bundle)
    fused_payload = fuse_scores(
        graph_scores=graph_bundle["scores"],
        sequence_scores=sequence_bundle["scores"],
        graph_weight=args.graph_weight,
        sequence_weight=args.sequence_weight,
        score_space=args.score_space,
    )
    meta = {
        "graph_bundle": graph_bundle["bundle_dir"],
        "sequence_bundle": sequence_bundle["bundle_dir"],
        "graph_weight": args.graph_weight,
        "sequence_weight": args.sequence_weight,
        "score_space": args.score_space,
        "entry_count": len(graph_bundle["entry_ids"]),
        "term_count": len(graph_bundle["terms"]),
        "bundle_format": {
            "scores": "scores.npy",
            "entry_ids": "entry_ids.txt",
            "terms": "terms.txt",
        },
    }
    if args.evaluate_with_graph_root is not None:
        aspect = args.aspect or graph_bundle["meta"].get("aspect") or sequence_bundle["meta"].get("aspect")
        if not aspect:
            raise ValueError("--aspect is required when bundle metadata does not include an aspect.")
        targets = build_targets_from_graph_root(
            args.evaluate_with_graph_root,
            aspect=str(aspect).upper(),
            entry_ids=graph_bundle["entry_ids"],
            terms=graph_bundle["terms"],
        )
        meta["evaluation"] = evaluate_scores(
            fused_payload["fused_scores"],
            targets,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )
        meta["evaluation"]["aspect"] = str(aspect).upper()
        meta["evaluation"]["graph_root"] = str(args.evaluate_with_graph_root.resolve())
    write_bundle(
        args.output_dir,
        entry_ids=graph_bundle["entry_ids"],
        terms=graph_bundle["terms"],
        fused_payload=fused_payload,
        meta=meta,
    )
    print(f"wrote {args.output_dir / 'scores.npy'}")
    print(f"wrote {args.output_dir / 'entry_ids.txt'}")
    print(f"wrote {args.output_dir / 'terms.txt'}")
    print(f"wrote {args.output_dir / 'meta.json'}")
    if "fused_logits" in fused_payload:
        print(f"wrote {args.output_dir / 'logits.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
