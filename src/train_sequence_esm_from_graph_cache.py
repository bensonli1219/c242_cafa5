#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None

import train_minimal_graph_model as graph_training

NNModuleBase = nn.Module if nn is not None else object


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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


def probability_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def require_torch():
    if torch is None or nn is None or F is None:
        raise RuntimeError("This script requires torch.")
    return torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a minimal sequence-side predictor from the exported protein-level "
            "ESM matrix derived from graph cache outputs, and write prediction bundles "
            "for late fusion."
        )
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--aspect", choices=["BPO", "CCO", "MFO"], required=True)
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--dropout", type=probability_float, default=0.2)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=non_negative_float, default=1e-4)
    parser.add_argument("--metric-threshold", type=probability_float, default=0.5)
    parser.add_argument("--fmax-threshold-step", type=positive_float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--protein-esm-dir", type=Path, default=None)
    parser.add_argument("--matched-split-dir", type=Path, default=None)
    parser.add_argument("--graph-root", type=Path, default=None)
    parser.add_argument("--export-splits", nargs="*", default=["val", "test"])
    return parser.parse_args(argv)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module = require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def resolve_device(device_name: str):
    torch_module = require_torch()
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def load_terms(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "terms" in payload:
        return list(payload["terms"])
    if isinstance(payload, dict):
        return list(payload.keys())
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unsupported vocab format in {path}")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def build_label_lookup(entries_json: Path, aspect: str, vocab: list[str]) -> dict[str, np.ndarray]:
    term_to_index = {term: idx for idx, term in enumerate(vocab)}
    entries = json.loads(entries_json.read_text(encoding="utf-8"))
    lookup: dict[str, np.ndarray] = {}
    for entry in entries:
        vector = np.zeros(len(vocab), dtype=np.float32)
        for term in (entry.get("labels") or {}).get(aspect, []):
            index = term_to_index.get(term)
            if index is not None:
                vector[index] = 1.0
        lookup[str(entry["entry_id"])] = vector
    return lookup


def load_split_arrays(
    feature_dir: Path,
    split_dir: Path,
    entries_json: Path,
    aspect: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], list[str]]:
    x_all = np.load(feature_dir / "X.npy")
    feature_entry_ids = read_lines(feature_dir / "entry_ids.txt")
    vocab = load_terms(split_dir / aspect.lower() / "vocab.json")
    label_lookup = build_label_lookup(entries_json, aspect=aspect, vocab=vocab)
    feature_index = {entry_id: index for index, entry_id in enumerate(feature_entry_ids)}

    split_arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split_name in ("train", "val", "test"):
        split_ids = read_lines(split_dir / aspect.lower() / f"{split_name}.txt")
        kept_ids = [entry_id for entry_id in split_ids if entry_id in feature_index and entry_id in label_lookup]
        indices = [feature_index[entry_id] for entry_id in kept_ids]
        x = x_all[indices].astype(np.float32, copy=False)
        y = np.stack([label_lookup[entry_id] for entry_id in kept_ids], axis=0) if kept_ids else np.zeros((0, len(vocab)), dtype=np.float32)
        split_arrays[split_name] = (x, y, kept_ids)
    return split_arrays, vocab


class SequenceLinearClassifier(NNModuleBase):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: Any):
        return self.classifier(x)


class SequenceMlpClassifier(NNModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, x: Any):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


def build_model(model_type: str, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
    if model_type == "linear":
        return SequenceLinearClassifier(input_dim=input_dim, output_dim=output_dim)
    if model_type == "mlp":
        return SequenceMlpClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    raise ValueError(f"Unknown model type: {model_type}")


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append((x[batch_indices], y[batch_indices]))
    return batches


def run_epoch(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: Any,
    optimizer: Any | None = None,
    metric_threshold: float = 0.5,
    fmax_threshold_step: float = 0.01,
) -> dict[str, Any]:
    torch_module = require_torch()
    is_training = optimizer is not None
    model.train(is_training)
    batches = iter_batches(x, y, batch_size=batch_size, shuffle=is_training)
    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_labels = []

    context = torch_module.enable_grad() if is_training else torch_module.inference_mode()
    with context:
        for x_batch, y_batch in batches:
            x_tensor = torch_module.from_numpy(x_batch).to(device)
            y_tensor = torch_module.from_numpy(y_batch).to(device)
            logits = model(x_tensor)
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * x_batch.shape[0]
            total_examples += x_batch.shape[0]
            all_logits.append(logits.detach().cpu())
            all_labels.append(y_tensor.detach().cpu())

    metrics = graph_training.empty_metrics(
        metric_threshold=metric_threshold,
        fmax_threshold_step=fmax_threshold_step,
    )
    if all_logits:
        metrics = graph_training.multilabel_metrics_from_logits(
            torch_module.cat(all_logits, dim=0),
            torch_module.cat(all_labels, dim=0),
            threshold=metric_threshold,
            fmax_threshold_step=fmax_threshold_step,
        )
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    metrics["graphs"] = float(total_examples)
    return metrics


def predict_scores(model: Any, x: np.ndarray, batch_size: int, device: Any) -> tuple[np.ndarray, np.ndarray]:
    torch_module = require_torch()
    model.eval()
    logits_parts = []
    with torch_module.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            x_tensor = torch_module.from_numpy(x[start : start + batch_size]).to(device)
            logits = model(x_tensor)
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
    if not logits_parts:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    logits = np.concatenate(logits_parts, axis=0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return logits, scores.astype(np.float32, copy=False)


def write_prediction_bundle(
    bundle_dir: Path,
    *,
    logits: np.ndarray,
    scores: np.ndarray,
    entry_ids: list[str],
    terms: list[str],
    meta: dict[str, Any],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    np.save(bundle_dir / "logits.npy", logits.astype(np.float32, copy=False))
    np.save(bundle_dir / "scores.npy", scores.astype(np.float32, copy=False))
    (bundle_dir / "entry_ids.txt").write_text("\n".join(entry_ids) + "\n", encoding="utf-8")
    (bundle_dir / "terms.txt").write_text("\n".join(terms) + "\n", encoding="utf-8")
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch_module = require_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    run_root = args.run_root.expanduser().resolve()
    protein_esm_dir = (
        args.protein_esm_dir.expanduser().resolve()
        if args.protein_esm_dir is not None
        else (run_root / "sequence_artifacts" / "protein_esm2_t30_150m_640_from_graph_cache").resolve()
    )
    matched_split_dir = (
        args.matched_split_dir.expanduser().resolve()
        if args.matched_split_dir is not None
        else (run_root / "sequence_artifacts" / "matched_structure_splits").resolve()
    )
    graph_root = (
        args.graph_root.expanduser().resolve()
        if args.graph_root is not None
        else (run_root / "graph_cache").resolve()
    )
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else (run_root / "sequence_runs" / f"esm_from_graph_cache_{args.model_type}_{args.aspect.lower()}").resolve()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    split_arrays, terms = load_split_arrays(
        feature_dir=protein_esm_dir,
        split_dir=matched_split_dir,
        entries_json=graph_root / "metadata" / "entries.json",
        aspect=args.aspect,
    )
    train_x, train_y, _ = split_arrays["train"]
    val_x, val_y, _ = split_arrays["val"]
    test_x, test_y, _ = split_arrays["test"]
    if train_x.shape[0] == 0:
        raise ValueError(f"No training rows are available for aspect {args.aspect}.")

    model = build_model(
        model_type=args.model_type,
        input_dim=int(train_x.shape[1]),
        hidden_dim=args.hidden_dim,
        output_dim=int(train_y.shape[1]),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch_module.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = 0
    best_val_fmax = -math.inf
    history: list[dict[str, Any]] = []
    best_checkpoint_path = checkpoint_dir / "best.pt"
    summary_path = checkpoint_dir / "summary.json"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            x=train_x,
            y=train_y,
            batch_size=args.batch_size,
            device=device,
            optimizer=optimizer,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )
        val_metrics = run_epoch(
            model=model,
            x=val_x,
            y=val_y,
            batch_size=args.batch_size,
            device=device,
            optimizer=None,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )
        test_metrics = run_epoch(
            model=model,
            x=test_x,
            y=test_y,
            batch_size=args.batch_size,
            device=device,
            optimizer=None,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_fmax={train_metrics['fmax']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_fmax={val_metrics['fmax']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_fmax={test_metrics['fmax']:.4f}",
            flush=True,
        )
        if val_metrics["fmax"] >= best_val_fmax:
            best_val_fmax = float(val_metrics["fmax"])
            best_epoch = epoch
            torch_module.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "aspect": args.aspect,
                    "model_type": args.model_type,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

        summary = {
            "aspect": args.aspect,
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "best_epoch": best_epoch,
            "best_val_fmax": best_val_fmax,
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "feature_dir": str(protein_esm_dir),
            "matched_split_dir": str(matched_split_dir),
            "history": history,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    checkpoint = torch_module.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    bundle_root = checkpoint_dir / "prediction_bundles"
    for split_name in args.export_splits:
        if split_name not in split_arrays:
            raise ValueError(f"Unknown split requested for export: {split_name}")
        split_x, _, split_entry_ids = split_arrays[split_name]
        logits, scores = predict_scores(model, split_x, batch_size=args.batch_size, device=device)
        write_prediction_bundle(
            bundle_root / split_name,
            logits=logits,
            scores=scores,
            entry_ids=split_entry_ids,
            terms=terms,
            meta={
                "aspect": args.aspect,
                "model_type": args.model_type,
                "source_checkpoint": str(best_checkpoint_path.resolve()),
                "split_name": split_name,
                "entry_count": len(split_entry_ids),
                "term_count": len(terms),
                "score_space": "logits_and_probabilities",
            },
        )
    print(f"wrote {summary_path}")
    print(f"wrote {bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
