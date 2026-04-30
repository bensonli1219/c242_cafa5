#!/usr/bin/env python3
"""
Minimal graph classification training loop for the CAFA graph cache.

The goal is to provide a small but real end-to-end training path that can run
on top of the exported train/val/test splits and the existing multimodal graph
schema. The script supports both PyG and DGL backends.
"""

from __future__ import annotations

import argparse
import json
import random
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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


def micro_f1_from_logits(logits: Any, labels: Any) -> float:
    torch_module = require_torch()
    predictions = torch_module.sigmoid(logits) >= 0.5
    targets = labels >= 0.5
    true_positive = torch_module.logical_and(predictions, targets).sum().item()
    false_positive = torch_module.logical_and(predictions, torch_module.logical_not(targets)).sum().item()
    false_negative = torch_module.logical_and(torch_module.logical_not(predictions), targets).sum().item()
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float((2 * true_positive) / denominator)


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


def run_epoch(
    model: Any,
    loader: Any,
    framework: str,
    device: Any,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
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

    context = torch_module.enable_grad() if is_training else torch_module.inference_mode()
    with context:
        for batch in loader:
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
    synchronize_device(device)

    mean_loss = total_loss / total_graphs if total_graphs else 0.0
    if logits_parts:
        all_logits = torch_module.cat(logits_parts, dim=0)
        all_labels = torch_module.cat(label_parts, dim=0)
        micro_f1 = micro_f1_from_logits(all_logits, all_labels)
    else:
        micro_f1 = 0.0
    return {
        "loss": mean_loss,
        "micro_f1": micro_f1,
        "graphs": float(total_graphs),
    }


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


def build_training_objects(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
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
    return loaders, model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    loaders, model = build_training_objects(args)

    checkpoint_dir = args.checkpoint_dir or (
        args.root / "training_runs" / f"minimal_{args.framework}_{args.aspect.lower()}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    summary_path = checkpoint_dir / "summary.json"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            framework=args.framework,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        val_metrics = run_epoch(
            model=model,
            loader=loaders["val"],
            framework=args.framework,
            device=device,
            optimizer=None,
            loss_fn=loss_fn,
        ) if len(loaders["val"].dataset) > 0 else {"loss": 0.0, "micro_f1": 0.0, "graphs": 0.0}
        test_metrics = run_epoch(
            model=model,
            loader=loaders["test"],
            framework=args.framework,
            device=device,
            optimizer=None,
            loss_fn=loss_fn,
        ) if len(loaders["test"].dataset) > 0 else {"loss": 0.0, "micro_f1": 0.0, "graphs": 0.0}

        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['micro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['micro_f1']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_f1={test_metrics['micro_f1']:.4f} "
            f"seconds={epoch_seconds:.2f}"
        )

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "framework": args.framework,
                    "aspect": args.aspect,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_micro_f1": val_metrics["micro_f1"],
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

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
            "best_val_loss": best_val_loss,
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "history": history,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
