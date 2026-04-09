#!/usr/bin/env python3
"""
Minimal graph classification training loop for the CAFA graph cache.

The goal is to provide a small but real end-to-end training path that can run
on top of the exported train/val/test splits and the existing multimodal graph
schema. The script supports both PyG and DGL backends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs

try:  # pragma: no cover - optional in the default py313 env
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None

try:  # pragma: no cover - optional
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:  # pragma: no cover
    GCNConv = None
    global_mean_pool = None

try:  # pragma: no cover - optional
    from dgl.nn import GraphConv
except ImportError:  # pragma: no cover
    GraphConv = None

NNModuleBase = nn.Module if nn is not None else object


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
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


def reduction_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def probability_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def require_torch():
    return graphs.require_torch()


def require_pyg_model_parts():
    require_torch()
    if GCNConv is None or global_mean_pool is None or nn is None:
        raise RuntimeError("PyG model dependencies are not installed in the current environment.")
    return GCNConv, global_mean_pool, nn


def require_dgl_model_parts():
    require_torch()
    graphs.require_dgl()
    if GraphConv is None or nn is None:
        raise RuntimeError("DGL model dependencies are not installed in the current environment.")
    return GraphConv, nn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal graph training loop on the CAFA graph cache.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--framework", choices=["pyg", "dgl"], default="pyg")
    parser.add_argument("--aspect", choices=list(dataloaders.DEFAULT_ASPECTS), default="MFO")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=positive_int, default=128)
    parser.add_argument("--dropout", type=positive_float, default=0.2)
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-function", choices=["bce", "weighted_bce"], default="bce")
    parser.add_argument("--pos-weight-power", type=positive_float, default=1.0)
    parser.add_argument("--max-pos-weight", type=positive_float, default=None)
    parser.add_argument("--metric-threshold", type=probability_float, default=0.5)
    parser.add_argument("--fmax-threshold-step", type=positive_float, default=0.01)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_loss", "val_micro_f1", "val_macro_f1", "val_fmax"],
        default="val_loss",
    )
    parser.add_argument("--early-stopping-patience", type=non_negative_int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=non_negative_float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--lr-plateau-factor", type=reduction_float, default=0.5)
    parser.add_argument("--lr-plateau-patience", type=non_negative_int, default=1)
    parser.add_argument("--min-lr", type=non_negative_float, default=1e-6)
    parser.add_argument("--progress-mode", choices=["auto", "tqdm", "log", "none"], default="auto")
    parser.add_argument("--progress-every", type=positive_int, default=25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=dataloaders.DEFAULT_SPLIT_SEED)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--min-term-frequency", type=positive_int, default=1)
    parser.add_argument("--disable-esm2", action="store_true")
    parser.add_argument("--disable-dssp", action="store_true")
    parser.add_argument("--disable-sasa", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device_name: str):
    torch_module = require_torch()
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch_module = require_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def synchronize_device(device: Any) -> None:
    torch_module = require_torch()
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.synchronize()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds_int = max(0, int(round(seconds)))
    days, remainder = divmod(seconds_int, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m{seconds_int:02d}s"
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_int:02d}s"
    if minutes:
        return f"{minutes}m{seconds_int:02d}s"
    return f"{seconds_int}s"


def timestamp_after(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    eta = dt.datetime.now().astimezone() + dt.timedelta(seconds=max(0.0, seconds))
    return eta.isoformat(timespec="seconds")


def metric_optimization_mode(metric_name: str) -> str:
    return "min" if metric_name.endswith("loss") else "max"


def metric_value_from_record(record: dict[str, Any], metric_name: str) -> float:
    split_name, _, metric_key = metric_name.partition("_")
    if split_name not in dataloaders.SPLIT_NAMES or not metric_key:
        raise ValueError(f"Unsupported metric name: {metric_name}")
    split_metrics = record.get(split_name) or {}
    if metric_key not in split_metrics:
        raise KeyError(f"Metric {metric_name} was not found in record.")
    return float(split_metrics[metric_key])


def metric_is_improved(candidate: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if not math.isfinite(candidate):
        return False
    if best is None or not math.isfinite(best):
        return True
    if mode == "min":
        return candidate < (best - min_delta)
    if mode == "max":
        return candidate > (best + min_delta)
    raise ValueError(f"Unknown optimization mode: {mode}")


def micro_f1_from_logits(logits: Any, labels: Any, threshold: float = 0.5) -> float:
    torch_module = require_torch()
    predictions = torch_module.sigmoid(logits) >= threshold
    targets = labels >= 0.5
    true_positive = torch_module.logical_and(predictions, targets).sum().item()
    false_positive = torch_module.logical_and(predictions, torch_module.logical_not(targets)).sum().item()
    false_negative = torch_module.logical_and(torch_module.logical_not(predictions), targets).sum().item()
    micro_precision = _safe_ratio(true_positive, true_positive + false_positive)
    micro_recall = _safe_ratio(true_positive, true_positive + false_negative)
    return _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _threshold_values(step: float) -> list[float]:
    if step <= 0:
        raise ValueError("fmax threshold step must be positive")
    values = [0.0]
    current = step
    while current < 1.0:
        values.append(float(current))
        current += step
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def fmax_from_scores(scores: Any, targets: Any, threshold_step: float = 0.01) -> dict[str, float]:
    torch_module = require_torch()
    best = {
        "fmax": 0.0,
        "fmax_threshold": 0.0,
        "fmax_precision": 0.0,
        "fmax_recall": 0.0,
    }
    for threshold in _threshold_values(threshold_step):
        predictions = scores >= threshold
        true_positive = torch_module.logical_and(predictions, targets).sum().item()
        false_positive = torch_module.logical_and(predictions, torch_module.logical_not(targets)).sum().item()
        false_negative = torch_module.logical_and(torch_module.logical_not(predictions), targets).sum().item()
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        if f1 > best["fmax"]:
            best = {
                "fmax": f1,
                "fmax_threshold": float(threshold),
                "fmax_precision": precision,
                "fmax_recall": recall,
            }
    return best


def multilabel_metrics_from_logits(
    logits: Any,
    labels: Any,
    threshold: float = 0.5,
    fmax_threshold_step: float = 0.01,
) -> dict[str, float]:
    torch_module = require_torch()
    scores = torch_module.sigmoid(logits)
    targets = labels >= 0.5
    predictions = scores >= threshold

    true_positive = torch_module.logical_and(predictions, targets).sum().item()
    false_positive = torch_module.logical_and(predictions, torch_module.logical_not(targets)).sum().item()
    false_negative = torch_module.logical_and(torch_module.logical_not(predictions), targets).sum().item()
    micro_precision = _safe_ratio(true_positive, true_positive + false_positive)
    micro_recall = _safe_ratio(true_positive, true_positive + false_negative)
    micro_f1 = _safe_ratio(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    true_positive_by_label = torch_module.logical_and(predictions, targets).sum(dim=0).float()
    false_positive_by_label = torch_module.logical_and(predictions, torch_module.logical_not(targets)).sum(dim=0).float()
    false_negative_by_label = torch_module.logical_and(torch_module.logical_not(predictions), targets).sum(dim=0).float()
    macro_precision_by_label = torch_module.where(
        (true_positive_by_label + false_positive_by_label) > 0,
        true_positive_by_label / (true_positive_by_label + false_positive_by_label),
        torch_module.zeros_like(true_positive_by_label),
    )
    macro_recall_by_label = torch_module.where(
        (true_positive_by_label + false_negative_by_label) > 0,
        true_positive_by_label / (true_positive_by_label + false_negative_by_label),
        torch_module.zeros_like(true_positive_by_label),
    )
    macro_f1_by_label = torch_module.where(
        ((2 * true_positive_by_label) + false_positive_by_label + false_negative_by_label) > 0,
        (2 * true_positive_by_label)
        / ((2 * true_positive_by_label) + false_positive_by_label + false_negative_by_label),
        torch_module.zeros_like(true_positive_by_label),
    )
    labels_with_positive = targets.sum(dim=0) > 0

    metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(macro_precision_by_label.mean().item()) if macro_precision_by_label.numel() else 0.0,
        "macro_recall": float(macro_recall_by_label.mean().item()) if macro_recall_by_label.numel() else 0.0,
        "macro_f1": float(macro_f1_by_label.mean().item()) if macro_f1_by_label.numel() else 0.0,
        "macro_f1_positive_labels": (
            float(macro_f1_by_label[labels_with_positive].mean().item()) if labels_with_positive.any().item() else 0.0
        ),
        "label_count": float(targets.shape[1]) if targets.ndim > 1 else 0.0,
        "label_count_with_positive": float(labels_with_positive.sum().item()),
        "metric_threshold": float(threshold),
        "fmax_threshold_step": float(fmax_threshold_step),
    }
    metrics.update(fmax_from_scores(scores=scores, targets=targets, threshold_step=fmax_threshold_step))
    return metrics


def empty_metrics(metric_threshold: float = 0.5, fmax_threshold_step: float = 0.01) -> dict[str, float]:
    return {
        "loss": 0.0,
        "micro_precision": 0.0,
        "micro_recall": 0.0,
        "micro_f1": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "macro_f1_positive_labels": 0.0,
        "fmax": 0.0,
        "fmax_threshold": 0.0,
        "fmax_precision": 0.0,
        "fmax_recall": 0.0,
        "label_count": 0.0,
        "label_count_with_positive": 0.0,
        "metric_threshold": float(metric_threshold),
        "fmax_threshold_step": float(fmax_threshold_step),
        "graphs": 0.0,
    }


def build_pos_weight_tensor(
    dataset: Any,
    power: float = 1.0,
    max_pos_weight: float | None = None,
):
    torch_module = require_torch()
    vocab = getattr(dataset, "vocab", None)
    term_to_index = getattr(dataset, "term_to_index", None)
    aspect = getattr(dataset, "aspect", None)
    entries = getattr(dataset, "entries", None)
    if vocab is None or term_to_index is None or aspect is None or entries is None:
        raise ValueError("Dataset does not expose the attributes required to build class weights.")

    positive_counts = torch_module.zeros(len(vocab), dtype=torch_module.float32)
    for entry in entries:
        labels = set((entry.get("labels") or {}).get(aspect, []))
        for term in labels:
            index = term_to_index.get(term)
            if index is not None:
                positive_counts[index] += 1.0

    total_graphs = float(len(entries))
    negative_counts = torch_module.clamp(torch_module.full_like(positive_counts, total_graphs) - positive_counts, min=0.0)
    safe_positive_counts = torch_module.where(
        positive_counts > 0,
        positive_counts,
        torch_module.ones_like(positive_counts),
    )
    pos_weight = negative_counts / safe_positive_counts
    pos_weight = torch_module.where(
        positive_counts > 0,
        pos_weight,
        torch_module.ones_like(pos_weight),
    )
    pos_weight = torch_module.clamp(pos_weight, min=1.0)
    if power != 1.0:
        pos_weight = torch_module.pow(pos_weight, float(power))
    if max_pos_weight is not None:
        pos_weight = torch_module.clamp(pos_weight, max=float(max_pos_weight))
    return pos_weight


def summarize_pos_weight_tensor(pos_weight: Any) -> dict[str, float]:
    if int(pos_weight.numel()) == 0:
        return {
            "label_count": 0.0,
            "weighted_label_count": 0.0,
            "min": 1.0,
            "max": 1.0,
            "mean": 1.0,
        }
    weighted = pos_weight[pos_weight > 1.0]
    return {
        "label_count": float(pos_weight.numel()),
        "weighted_label_count": float(weighted.numel()),
        "min": float(pos_weight.min().item()),
        "max": float(pos_weight.max().item()),
        "mean": float(pos_weight.mean().item()),
    }


def build_loss_function(
    args: argparse.Namespace,
    train_dataset: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    torch_module = require_torch()
    config: dict[str, Any] = {
        "loss_function": args.loss_function,
    }
    if args.loss_function == "bce":
        return torch_module.nn.BCEWithLogitsLoss(), config
    if args.loss_function == "weighted_bce":
        pos_weight = build_pos_weight_tensor(
            train_dataset,
            power=args.pos_weight_power,
            max_pos_weight=args.max_pos_weight,
        )
        config.update(
            {
                "pos_weight_power": args.pos_weight_power,
                "max_pos_weight": args.max_pos_weight,
                "pos_weight_summary": summarize_pos_weight_tensor(pos_weight),
            }
        )
        return torch_module.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)), config
    raise ValueError(f"Unknown loss function: {args.loss_function}")


def build_lr_scheduler(
    args: argparse.Namespace,
    optimizer: Any,
) -> tuple[Any | None, dict[str, Any]]:
    torch_module = require_torch()
    if args.lr_scheduler == "none":
        return None, {"lr_scheduler": "none"}
    if args.lr_scheduler == "plateau":
        mode = metric_optimization_mode(args.checkpoint_metric)
        scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.min_lr,
        )
        return scheduler, {
            "lr_scheduler": args.lr_scheduler,
            "mode": mode,
            "monitor": args.checkpoint_metric,
            "factor": args.lr_plateau_factor,
            "patience": args.lr_plateau_patience,
            "min_lr": args.min_lr,
        }
    raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")


class MinimalPygClassifier(NNModuleBase):
    def __init__(self, input_dim: int, graph_feat_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        GCN, _, base_nn = require_pyg_model_parts()
        super().__init__()
        self.input_proj = base_nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCN(hidden_dim, hidden_dim)
        self.conv2 = GCN(hidden_dim, hidden_dim)
        self.graph_mlp = base_nn.Linear(graph_feat_dim, hidden_dim)
        self.classifier = base_nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = float(dropout)

    def forward(self, batch: Any):
        _, pool_fn, _ = require_pyg_model_parts()
        x = F.relu(self.input_proj(batch.x))
        x = F.relu(self.conv1(x, batch.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, batch.edge_index))
        pooled = pool_fn(x, batch.batch)
        graph_feat = batch.graph_feat.float().view(pooled.shape[0], -1)
        fused = torch.cat([pooled, F.relu(self.graph_mlp(graph_feat))], dim=-1)
        return self.classifier(F.dropout(fused, p=self.dropout, training=self.training))


class MinimalDglClassifier(NNModuleBase):
    def __init__(self, input_dim: int, graph_feat_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        GraphConvLayer, base_nn = require_dgl_model_parts()
        super().__init__()
        self.input_proj = base_nn.Linear(input_dim, hidden_dim)
        self.conv1 = GraphConvLayer(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.graph_mlp = base_nn.Linear(graph_feat_dim, hidden_dim)
        self.classifier = base_nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = float(dropout)

    def forward(self, batch: Any):
        dgl_module = graphs.require_dgl()
        with batch.local_scope():
            x = F.relu(self.input_proj(batch.ndata["x"]))
            x = F.relu(self.conv1(batch, x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(batch, x))
            batch.ndata["_hidden"] = x
            pooled = dgl_module.mean_nodes(batch, "_hidden")
        graph_feat = batch.graph_feat.float().view(pooled.shape[0], -1)
        fused = torch.cat([pooled, F.relu(self.graph_mlp(graph_feat))], dim=-1)
        return self.classifier(F.dropout(fused, p=self.dropout, training=self.training))


def move_batch_to_device(batch: Any, framework: str, device: Any):
    if framework == "pyg":
        return batch.to(device) if device.type != "cpu" else batch
    if framework == "dgl":
        if device.type == "cpu":
            return batch
        moved = batch.to(device)
        moved.y = batch.y.to(device)
        moved.graph_feat = batch.graph_feat.to(device)
        return moved
    raise ValueError(f"Unknown framework: {framework}")


def extract_labels(batch: Any, framework: str):
    if framework == "pyg":
        return batch.y.float().view(batch.num_graphs, -1)
    if framework == "dgl":
        return batch.y.float().view(batch.y.shape[0], -1)
    raise ValueError(f"Unknown framework: {framework}")


def batch_graph_count(batch: Any, framework: str) -> int:
    if framework == "pyg":
        return int(batch.num_graphs)
    if framework == "dgl":
        return int(len(batch.batch_num_nodes()))
    raise ValueError(f"Unknown framework: {framework}")


def loader_batch_count(loader: Any) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def progress_iterable(loader: Any, label: str, progress_mode: str, total_batches: int | None):
    if progress_mode == "none":
        return loader, "none"
    if progress_mode == "tqdm" or (progress_mode == "auto" and sys.stderr.isatty()):
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print(f"[progress] tqdm is not installed; falling back to log mode for {label}", flush=True)
        else:
            return tqdm(loader, total=total_batches, desc=label, leave=False, dynamic_ncols=True), "tqdm"
    if progress_mode in {"auto", "log"}:
        return loader, "log"
    return loader, "none"


def run_epoch(
    model: Any,
    loader: Any,
    framework: str,
    device: Any,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
    metric_threshold: float = 0.5,
    fmax_threshold_step: float = 0.01,
    epoch: int | None = None,
    split_name: str = "split",
    progress_mode: str = "auto",
    progress_every: int = 25,
) -> dict[str, float]:
    torch_module = require_torch()
    is_training = optimizer is not None
    if is_training and loss_fn is None:
        raise ValueError("loss_fn is required during training")

    model.train(is_training)
    total_loss = 0.0
    total_graphs = 0
    logits_parts = []
    label_parts = []
    total_batches = loader_batch_count(loader)
    epoch_label = f"epoch={epoch} {split_name}" if epoch is not None else split_name
    batch_iterable, active_progress_mode = progress_iterable(
        loader,
        label=epoch_label,
        progress_mode=progress_mode,
        total_batches=total_batches,
    )
    split_start = time.perf_counter()

    context = torch_module.enable_grad() if is_training else torch_module.inference_mode()
    with context:
        for batch_index, batch in enumerate(batch_iterable, start=1):
            batch = move_batch_to_device(batch, framework=framework, device=device)
            labels = extract_labels(batch, framework=framework)
            logits = model(batch)
            loss = loss_fn(logits, labels)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * batch_graph_count(batch, framework=framework)
            total_graphs += batch_graph_count(batch, framework=framework)
            logits_parts.append(logits.detach().cpu())
            label_parts.append(labels.detach().cpu())
            if active_progress_mode == "log" and (
                batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches
            ):
                total_label = "?" if total_batches is None else str(total_batches)
                running_loss = total_loss / total_graphs if total_graphs else 0.0
                elapsed_seconds = time.perf_counter() - split_start
                remaining_seconds = None
                if total_batches is not None and batch_index:
                    estimated_total_seconds = elapsed_seconds * total_batches / batch_index
                    remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
                print(
                    f"[progress] {epoch_label} batch={batch_index}/{total_label} "
                    f"graphs={total_graphs} loss={running_loss:.4f} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"remaining={format_duration(remaining_seconds)} "
                    f"eta={timestamp_after(remaining_seconds)}",
                    flush=True,
                )
    synchronize_device(device)

    mean_loss = total_loss / total_graphs if total_graphs else 0.0
    if logits_parts:
        all_logits = torch_module.cat(logits_parts, dim=0)
        all_labels = torch_module.cat(label_parts, dim=0)
        metrics = multilabel_metrics_from_logits(
            all_logits,
            all_labels,
            threshold=metric_threshold,
            fmax_threshold_step=fmax_threshold_step,
        )
    else:
        metrics = empty_metrics(metric_threshold=metric_threshold, fmax_threshold_step=fmax_threshold_step)
    metrics.update({
        "loss": mean_loss,
        "graphs": float(total_graphs),
    })
    return metrics


def build_model(framework: str, output_dim: int, hidden_dim: int, dropout: float):
    if framework == "pyg":
        return MinimalPygClassifier(
            input_dim=graphs.NODE_FEATURE_DIM,
            graph_feat_dim=graphs.GRAPH_FEAT_DIM,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    if framework == "dgl":
        return MinimalDglClassifier(
            input_dim=graphs.NODE_FEATURE_DIM,
            graph_feat_dim=graphs.GRAPH_FEAT_DIM,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown framework: {framework}")


def build_training_objects(args: argparse.Namespace) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    split_dir = args.split_dir or (args.root / "splits")
    if not (split_dir / "summary.json").exists():
        dataloaders.export_split_manifests(
            root=args.root,
            output_dir=split_dir,
            aspects=[args.aspect],
            seed=args.seed,
            min_term_frequency=args.min_term_frequency,
        )

    datasets = dataloaders.build_split_datasets(
        framework=args.framework,
        root=args.root,
        aspect=args.aspect,
        split_dir=split_dir,
        min_term_frequency=args.min_term_frequency,
        use_esm2=not args.disable_esm2,
        use_dssp=not args.disable_dssp,
        use_sasa=not args.disable_sasa,
    )
    if args.framework == "pyg":
        loaders = {
            split_name: dataloaders.build_pyg_loader(
                dataset=datasets[split_name],
                batch_size=args.batch_size,
                shuffle=(split_name == "train"),
                num_workers=args.num_workers,
                seed=args.seed,
            )
            for split_name in dataloaders.SPLIT_NAMES
        }
    else:
        loaders = {
            split_name: dataloaders.build_dgl_loader(
                dataset=datasets[split_name],
                batch_size=args.batch_size,
                shuffle=(split_name == "train"),
                num_workers=args.num_workers,
                seed=args.seed,
            )
            for split_name in dataloaders.SPLIT_NAMES
        }

    output_dim = len(datasets["train"].vocab)
    if output_dim <= 0:
        raise ValueError(f"No labels available for aspect {args.aspect}")
    model = build_model(
        framework=args.framework,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    return loaders, model, datasets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    loaders, model, datasets = build_training_objects(args)

    checkpoint_dir = args.checkpoint_dir or (
        args.root / "training_runs" / f"minimal_{args.framework}_{args.aspect.lower()}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn, loss_config = build_loss_function(args, datasets["train"], device=device)
    scheduler, scheduler_config = build_lr_scheduler(args, optimizer)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_metric = None
    best_checkpoint_metric_mode = metric_optimization_mode(args.checkpoint_metric)
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    early_stopping_reason = ""
    best_checkpoint_path = checkpoint_dir / "best.pt"
    summary_path = checkpoint_dir / "summary.json"
    training_started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            framework=args.framework,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
            epoch=epoch,
            split_name=f"{args.aspect} train",
            progress_mode=args.progress_mode,
            progress_every=args.progress_every,
        )
        val_metrics = run_epoch(
            model=model,
            loader=loaders["val"],
            framework=args.framework,
            device=device,
            optimizer=None,
            loss_fn=loss_fn,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
            epoch=epoch,
            split_name=f"{args.aspect} val",
            progress_mode=args.progress_mode,
            progress_every=args.progress_every,
        ) if len(loaders["val"].dataset) > 0 else empty_metrics(
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )
        test_metrics = run_epoch(
            model=model,
            loader=loaders["test"],
            framework=args.framework,
            device=device,
            optimizer=None,
            loss_fn=loss_fn,
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
            epoch=epoch,
            split_name=f"{args.aspect} test",
            progress_mode=args.progress_mode,
            progress_every=args.progress_every,
        ) if len(loaders["test"].dataset) > 0 else empty_metrics(
            metric_threshold=args.metric_threshold,
            fmax_threshold_step=args.fmax_threshold_step,
        )

        epoch_seconds = time.perf_counter() - epoch_start
        total_elapsed_seconds = time.perf_counter() - training_start
        average_epoch_seconds = total_elapsed_seconds / epoch
        estimated_remaining_seconds = average_epoch_seconds * (args.epochs - epoch)
        estimated_finished_at = timestamp_after(estimated_remaining_seconds)
        record = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
            "average_epoch_seconds": average_epoch_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "estimated_finished_at": estimated_finished_at,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        checkpoint_metric_value = metric_value_from_record(record, args.checkpoint_metric)
        record["checkpoint_metric_name"] = args.checkpoint_metric
        record["checkpoint_metric_value"] = checkpoint_metric_value
        history.append(record)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_micro_f1={train_metrics['micro_f1']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} train_fmax={train_metrics['fmax']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_micro_f1={val_metrics['micro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_fmax={val_metrics['fmax']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_micro_f1={test_metrics['micro_f1']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f} test_fmax={test_metrics['fmax']:.4f} "
            f"{args.checkpoint_metric}={checkpoint_metric_value:.4f} "
            f"seconds={epoch_seconds:.2f} "
            f"avg_epoch_seconds={average_epoch_seconds:.2f} "
            f"estimated_remaining={format_duration(estimated_remaining_seconds)} "
            f"eta={estimated_finished_at}"
        )

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]

        if scheduler is not None:
            scheduler.step(checkpoint_metric_value)

        if metric_is_improved(
            checkpoint_metric_value,
            best_checkpoint_metric,
            mode=best_checkpoint_metric_mode,
            min_delta=args.early_stopping_min_delta,
        ):
            best_checkpoint_metric = checkpoint_metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "framework": args.framework,
                    "aspect": args.aspect,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_fmax": val_metrics["fmax"],
                    "val_fmax_threshold": val_metrics["fmax_threshold"],
                    "checkpoint_metric_name": args.checkpoint_metric,
                    "checkpoint_metric_value": checkpoint_metric_value,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            early_stopping_reason = (
                f"No {args.checkpoint_metric} improvement for {epochs_without_improvement} epoch(s)."
            )
            print(f"early_stopping epoch={epoch} reason={early_stopping_reason}", flush=True)

        summary = {
            "framework": args.framework,
            "aspect": args.aspect,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss_function": args.loss_function,
            "loss_config": loss_config,
            "metric_threshold": args.metric_threshold,
            "fmax_threshold_step": args.fmax_threshold_step,
            "checkpoint_metric": args.checkpoint_metric,
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_metric_mode": best_checkpoint_metric_mode,
            "best_epoch": best_epoch,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "epochs_completed": epoch,
            "stopped_early": stopped_early,
            "early_stopping_reason": early_stopping_reason,
            "epochs_without_improvement": epochs_without_improvement,
            "lr_scheduler": args.lr_scheduler,
            "scheduler_config": scheduler_config,
            "progress_mode": args.progress_mode,
            "progress_every": args.progress_every,
            "best_val_loss": best_val_loss,
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "training_started_at": training_started_at,
            "history": history,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if stopped_early:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
