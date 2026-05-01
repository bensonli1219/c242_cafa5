#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import cafa_graph_dataloaders as dataloaders
import train_minimal_graph_model as graph_training


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export graph model prediction bundles for late fusion. The output "
            "format matches train_sequence_esm_from_graph_cache.py: logits.npy, "
            "scores.npy, entry_ids.txt, terms.txt, and meta.json."
        )
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--aspect", choices=list(dataloaders.DEFAULT_ASPECTS), default=None)
    parser.add_argument("--framework", choices=["pyg", "dgl"], default=None)
    parser.add_argument("--batch-size", type=positive_int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=positive_int, default=None)
    parser.add_argument("--dropout", type=positive_float, default=None)
    parser.add_argument("--model-head", choices=["flat_linear", "label_dot"], default=None)
    parser.add_argument("--min-term-frequency", type=positive_int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--export-splits", nargs="*", default=["val", "test"])
    parser.add_argument("--disable-esm2", action="store_true")
    parser.add_argument("--disable-dssp", action="store_true")
    parser.add_argument("--disable-sasa", action="store_true")
    parser.add_argument("--normalize-features", action="store_true", default=None)
    return parser.parse_args(argv)


def checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("Checkpoint args payload is missing or invalid.")
    return args


def choose_value(cli_value: Any, checkpoint_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (checkpoint_value if checkpoint_value is not None else default)


def resolve_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    saved = checkpoint_args(checkpoint)
    root = choose_value(args.root, saved.get("root"))
    if root is None:
        raise ValueError("--root is required because it was not recorded in the checkpoint.")
    split_dir = choose_value(args.split_dir, saved.get("split_dir"))
    if split_dir is None:
        split_dir = Path(root) / "splits"

    return {
        "root": Path(root),
        "split_dir": Path(split_dir),
        "aspect": str(choose_value(args.aspect, checkpoint.get("aspect") or saved.get("aspect"))).upper(),
        "framework": choose_value(args.framework, checkpoint.get("framework") or saved.get("framework"), "pyg"),
        "batch_size": int(choose_value(args.batch_size, saved.get("batch_size"), 8)),
        "num_workers": int(args.num_workers),
        "hidden_dim": int(choose_value(args.hidden_dim, saved.get("hidden_dim"), 128)),
        "dropout": float(choose_value(args.dropout, saved.get("dropout"), 0.2)),
        "model_head": choose_value(args.model_head, saved.get("model_head"), "flat_linear"),
        "min_term_frequency": int(choose_value(args.min_term_frequency, saved.get("min_term_frequency"), 1)),
        "seed": int(choose_value(args.seed, saved.get("seed"), dataloaders.DEFAULT_SPLIT_SEED)),
        "normalize_features": bool(
            args.normalize_features if args.normalize_features is not None else saved.get("normalize_features", False)
        ),
        "use_esm2": not args.disable_esm2 and not bool(saved.get("disable_esm2", False)),
        "use_dssp": not args.disable_dssp and not bool(saved.get("disable_dssp", False)),
        "use_sasa": not args.disable_sasa and not bool(saved.get("disable_sasa", False)),
    }


def build_loader(dataset: Any, framework: str, batch_size: int, num_workers: int, seed: int):
    if framework == "pyg":
        return dataloaders.build_pyg_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
        )
    return dataloaders.build_dgl_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )


def batch_entry_ids(batch: Any, framework: str, fallback_ids: list[str], offset: int) -> list[str]:
    value = getattr(batch, "entry_id", None)
    if value is None:
        count = graph_training.batch_graph_count(batch, framework)
        return fallback_ids[offset : offset + count]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def predict_bundle(
    *,
    model: Any,
    loader: Any,
    dataset: Any,
    framework: str,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    torch_module = graph_training.require_torch()
    model.eval()
    logits_parts = []
    entry_ids: list[str] = []
    fallback_ids = [str(entry["entry_id"]) for entry in dataset.entries]
    offset = 0
    with torch_module.inference_mode():
        for batch in loader:
            ids = batch_entry_ids(batch, framework, fallback_ids=fallback_ids, offset=offset)
            offset += len(ids)
            batch = graph_training.move_batch_to_device(batch, framework=framework, device=device)
            logits = model(batch)
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            entry_ids.extend(ids)
    if not logits_parts:
        return np.zeros((0, len(dataset.vocab)), dtype=np.float32), np.zeros((0, len(dataset.vocab)), dtype=np.float32), []
    logits = np.concatenate(logits_parts, axis=0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return logits.astype(np.float32, copy=False), scores.astype(np.float32, copy=False), entry_ids


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
    torch_module = graph_training.require_torch()
    checkpoint = torch_module.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    config = resolve_config(args, checkpoint)
    device = graph_training.resolve_device(args.device)

    datasets = dataloaders.build_split_datasets(
        framework=config["framework"],
        root=config["root"],
        aspect=config["aspect"],
        split_dir=config["split_dir"],
        min_term_frequency=config["min_term_frequency"],
        use_esm2=config["use_esm2"],
        use_dssp=config["use_dssp"],
        use_sasa=config["use_sasa"],
        normalize_features=config["normalize_features"],
    )
    output_dim = len(datasets["train"].vocab)
    model = graph_training.build_model(
        framework=config["framework"],
        output_dim=output_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        model_head=config["model_head"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    for split_name in args.export_splits:
        if split_name not in datasets:
            raise ValueError(f"Unknown split requested for export: {split_name}")
        dataset = datasets[split_name]
        print(
            f"exporting split={split_name} entries={len(dataset)} terms={len(dataset.vocab)} "
            f"framework={config['framework']} device={device}",
            flush=True,
        )
        loader = build_loader(
            dataset,
            framework=config["framework"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        logits, scores, entry_ids = predict_bundle(
            model=model,
            loader=loader,
            dataset=dataset,
            framework=config["framework"],
            device=device,
        )
        meta = {
            "aspect": config["aspect"],
            "framework": config["framework"],
            "model_head": config["model_head"],
            "source_checkpoint": str(args.checkpoint_path.resolve()),
            "split_name": split_name,
            "entry_count": len(entry_ids),
            "term_count": len(dataset.vocab),
            "score_space": "logits_and_probabilities",
            "root": str(config["root"].resolve()),
            "split_dir": str(config["split_dir"].resolve()),
            "min_term_frequency": config["min_term_frequency"],
            "normalize_features": config["normalize_features"],
            "use_esm2": config["use_esm2"],
            "use_dssp": config["use_dssp"],
            "use_sasa": config["use_sasa"],
        }
        write_prediction_bundle(
            args.output_dir / split_name,
            logits=logits,
            scores=scores,
            entry_ids=entry_ids,
            terms=list(dataset.vocab),
            meta=meta,
        )
        print(f"wrote {args.output_dir / split_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
