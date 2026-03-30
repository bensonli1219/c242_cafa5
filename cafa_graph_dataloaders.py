#!/usr/bin/env python3
"""
Utilities for exporting deterministic train/val/test splits and building
training-ready PyG / DGL graph dataloaders from the protein graph cache.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import cafa5_alphafold_pipeline as pipeline
import cafa_graph_dataset as graphs

try:  # pragma: no cover - optional in the default py313 env
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional in the default py313 env
    from torch_geometric.loader import DataLoader as PygDataLoader
except ImportError:  # pragma: no cover
    PygDataLoader = None

try:  # pragma: no cover - optional in the default py313 env
    from dgl.dataloading import GraphDataLoader as DGLGraphDataLoader
except ImportError:  # pragma: no cover
    DGLGraphDataLoader = None


DEFAULT_ASPECTS = ("BPO", "CCO", "MFO")
DEFAULT_SPLIT_SEED = 2026
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
SPLIT_NAMES = ("train", "val", "test")


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


def parse_aspects(values: Iterable[str] | None) -> list[str]:
    parsed = []
    for value in values or DEFAULT_ASPECTS:
        aspect = str(value).strip().upper()
        if not aspect:
            continue
        if aspect not in graphs.ASPECT_TO_LABEL_KEY:
            raise ValueError(f"Unknown aspect: {aspect}")
        if aspect not in parsed:
            parsed.append(aspect)
    return parsed


def require_torch():
    return graphs.require_torch()


def require_pyg_loader():
    require_torch()
    if PygDataLoader is None:
        raise RuntimeError("torch_geometric is required for PyG dataloaders.")
    return PygDataLoader


def require_dgl_loader():
    require_torch()
    graphs.require_dgl()
    if DGLGraphDataLoader is None:
        raise RuntimeError("dgl is required for DGL dataloaders.")
    return DGLGraphDataLoader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic train/val/test splits and verify PyG/DGL dataloaders."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aspects", nargs="*", default=list(DEFAULT_ASPECTS))
    parser.add_argument("--train-ratio", type=positive_float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=positive_float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=positive_float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-term-frequency", type=positive_int, default=1)
    parser.add_argument(
        "--frameworks",
        nargs="*",
        choices=["pyg", "dgl"],
        default=["pyg", "dgl"],
    )
    parser.add_argument("--disable-esm2", action="store_true")
    parser.add_argument("--disable-dssp", action="store_true")
    parser.add_argument("--disable-sasa", action="store_true")
    return parser.parse_args(argv)


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return (train_ratio / total, val_ratio / total, test_ratio / total)


def allocate_split_counts(
    total_items: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    expected = [ratio * total_items for ratio in ratios]
    counts = [int(value) for value in expected]
    remainder = total_items - sum(counts)
    order = sorted(
        range(len(expected)),
        key=lambda index: (expected[index] - counts[index], ratios[index], -index),
        reverse=True,
    )
    for index in range(remainder):
        counts[order[index % len(order)]] += 1

    positive_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total_items >= len(positive_indices):
        for index in positive_indices:
            if counts[index] > 0:
                continue
            donor = max(
                (candidate for candidate in range(len(counts)) if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
                default=None,
            )
            if donor is None:
                break
            counts[donor] -= 1
            counts[index] += 1

    return counts[0], counts[1], counts[2]


def split_entry_ids(
    entry_ids: Iterable[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, list[str]]:
    unique_entry_ids = sorted({str(entry_id) for entry_id in entry_ids if str(entry_id).strip()})
    rng = random.Random(seed)
    rng.shuffle(unique_entry_ids)

    ratios = normalize_ratios(train_ratio, val_ratio, test_ratio)
    train_count, val_count, test_count = allocate_split_counts(len(unique_entry_ids), ratios)

    train_ids = unique_entry_ids[:train_count]
    val_ids = unique_entry_ids[train_count : train_count + val_count]
    test_ids = unique_entry_ids[train_count + val_count : train_count + val_count + test_count]
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def split_dir_for_aspect(output_dir: Path, aspect: str) -> Path:
    return output_dir / aspect.lower()


def write_split_ids(path: Path, entry_ids: Iterable[str]) -> None:
    pipeline.ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for entry_id in entry_ids:
            handle.write(f"{entry_id}\n")


def load_split_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_filtered_entries(
    root: str | Path,
    aspect: str,
    min_term_frequency: int = 1,
) -> tuple[list[dict[str, Any]], list[str]]:
    root_path = Path(root)
    entries = graphs.load_json(root_path / "metadata" / "entries.json")
    term_counts = graphs.load_json(root_path / "metadata" / "term_counts.json")
    vocab = graphs.build_vocab(term_counts[aspect], min_term_frequency=min_term_frequency)
    vocab_set = set(vocab)
    filtered_entries = []
    for entry in entries:
        labels = set(entry["labels"].get(aspect, []))
        if not labels & vocab_set:
            continue
        filtered_entries.append(entry)
    return filtered_entries, vocab


def export_split_manifest(
    root: str | Path,
    output_dir: str | Path,
    aspect: str,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    min_term_frequency: int = 1,
) -> dict[str, Any]:
    filtered_entries, vocab = load_filtered_entries(
        root=root,
        aspect=aspect,
        min_term_frequency=min_term_frequency,
    )
    entry_ids = [str(entry["entry_id"]) for entry in filtered_entries]
    splits = split_entry_ids(
        entry_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    aspect_dir = split_dir_for_aspect(Path(output_dir), aspect)
    for split_name in SPLIT_NAMES:
        write_split_ids(aspect_dir / f"{split_name}.txt", splits[split_name])

    summary = {
        "aspect": aspect,
        "entry_count": len(entry_ids),
        "vocab_size": len(vocab),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": {split_name: len(splits[split_name]) for split_name in SPLIT_NAMES},
        "entry_ids": splits,
    }
    with (aspect_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def export_split_manifests(
    root: str | Path,
    output_dir: str | Path,
    aspects: Iterable[str] | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    min_term_frequency: int = 1,
) -> dict[str, Any]:
    summaries = {}
    for aspect in parse_aspects(aspects):
        summaries[aspect] = export_split_manifest(
            root=root,
            output_dir=output_dir,
            aspect=aspect,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            min_term_frequency=min_term_frequency,
        )
    top_level = {
        "root": str(Path(root).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "aspects": summaries,
    }
    with (Path(output_dir) / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(top_level, handle, indent=2)
    return top_level


def build_split_dataset(
    framework: str,
    root: str | Path,
    aspect: str,
    split_dir: str | Path,
    split_name: str,
    min_term_frequency: int = 1,
    use_esm2: bool = True,
    use_dssp: bool = True,
    use_sasa: bool = True,
):
    entry_id_file = split_dir_for_aspect(Path(split_dir), aspect) / f"{split_name}.txt"
    if framework == "pyg":
        return graphs.CafaPyGDataset(
            root=root,
            aspect=aspect,
            entry_id_file=entry_id_file,
            min_term_frequency=min_term_frequency,
            use_esm2=use_esm2,
            use_dssp=use_dssp,
            use_sasa=use_sasa,
        )
    if framework == "dgl":
        return graphs.CafaDGLDataset(
            root=root,
            aspect=aspect,
            entry_id_file=entry_id_file,
            min_term_frequency=min_term_frequency,
            use_esm2=use_esm2,
            use_dssp=use_dssp,
            use_sasa=use_sasa,
        )
    raise ValueError(f"Unknown framework: {framework}")


def build_split_datasets(
    framework: str,
    root: str | Path,
    aspect: str,
    split_dir: str | Path,
    min_term_frequency: int = 1,
    use_esm2: bool = True,
    use_dssp: bool = True,
    use_sasa: bool = True,
) -> dict[str, Any]:
    return {
        split_name: build_split_dataset(
            framework=framework,
            root=root,
            aspect=aspect,
            split_dir=split_dir,
            split_name=split_name,
            min_term_frequency=min_term_frequency,
            use_esm2=use_esm2,
            use_dssp=use_dssp,
            use_sasa=use_sasa,
        )
        for split_name in SPLIT_NAMES
    }


class _PygGraphLevelWrapper:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]
        if hasattr(data, "clone"):
            data = data.clone()
        data.y = data.y.view(1, -1)
        data.graph_feat = data.graph_feat.view(1, -1)
        return data


def make_torch_generator(seed: int | None = None):
    torch_module = require_torch()
    generator = torch_module.Generator()
    generator.manual_seed(int(seed if seed is not None else DEFAULT_SPLIT_SEED))
    return generator


def build_pyg_loader(
    dataset: Any,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
):
    DataLoader = require_pyg_loader()
    wrapped_dataset = _PygGraphLevelWrapper(dataset)
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=bool(num_workers > 0),
        generator=make_torch_generator(seed),
    )


def dgl_collate_graphs(graphs_batch: list[Any]):
    dgl_module = graphs.require_dgl()
    torch_module = require_torch()
    batched_graph = dgl_module.batch(graphs_batch)
    batched_graph.y = torch_module.stack([graph.y for graph in graphs_batch], dim=0)
    batched_graph.graph_feat = torch_module.stack([graph.graph_feat for graph in graphs_batch], dim=0)
    batched_graph.entry_id = [graph.entry_id for graph in graphs_batch]
    batched_graph.taxonomy_id = [graph.taxonomy_id for graph in graphs_batch]
    batched_graph.fragment_ids = [graph.fragment_ids for graph in graphs_batch]
    return batched_graph


def build_dgl_loader(
    dataset: Any,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int | None = None,
):
    DataLoader = require_dgl_loader()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=make_torch_generator(seed),
        collate_fn=dgl_collate_graphs,
    )


def build_pyg_dataloaders(
    root: str | Path,
    aspect: str,
    split_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = DEFAULT_SPLIT_SEED,
    min_term_frequency: int = 1,
    use_esm2: bool = True,
    use_dssp: bool = True,
    use_sasa: bool = True,
) -> dict[str, Any]:
    datasets = build_split_datasets(
        framework="pyg",
        root=root,
        aspect=aspect,
        split_dir=split_dir,
        min_term_frequency=min_term_frequency,
        use_esm2=use_esm2,
        use_dssp=use_dssp,
        use_sasa=use_sasa,
    )
    return {
        split_name: build_pyg_loader(
            dataset=datasets[split_name],
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            seed=seed,
        )
        for split_name in SPLIT_NAMES
    }


def build_dgl_dataloaders(
    root: str | Path,
    aspect: str,
    split_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = DEFAULT_SPLIT_SEED,
    min_term_frequency: int = 1,
    use_esm2: bool = True,
    use_dssp: bool = True,
    use_sasa: bool = True,
) -> dict[str, Any]:
    datasets = build_split_datasets(
        framework="dgl",
        root=root,
        aspect=aspect,
        split_dir=split_dir,
        min_term_frequency=min_term_frequency,
        use_esm2=use_esm2,
        use_dssp=use_dssp,
        use_sasa=use_sasa,
    )
    return {
        split_name: build_dgl_loader(
            dataset=datasets[split_name],
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            seed=seed,
        )
        for split_name in SPLIT_NAMES
    }


def describe_batch(batch: Any, framework: str) -> dict[str, Any]:
    if framework == "pyg":
        return {
            "graphs": int(batch.num_graphs),
            "nodes": int(batch.num_nodes),
            "edges": int(batch.num_edges),
            "x_shape": tuple(batch.x.shape),
            "edge_attr_shape": tuple(batch.edge_attr.shape),
            "y_shape": tuple(batch.y.shape),
            "graph_feat_shape": tuple(batch.graph_feat.shape),
        }
    if framework == "dgl":
        return {
            "graphs": int(len(batch.batch_num_nodes())),
            "nodes": int(batch.num_nodes()),
            "edges": int(batch.num_edges()),
            "x_shape": tuple(batch.ndata["x"].shape),
            "edge_attr_shape": tuple(batch.edata["edge_attr"].shape),
            "y_shape": tuple(batch.y.shape),
            "graph_feat_shape": tuple(batch.graph_feat.shape),
        }
    raise ValueError(f"Unknown framework: {framework}")


def verify_loader_summary(
    framework: str,
    root: str | Path,
    aspect: str,
    split_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = DEFAULT_SPLIT_SEED,
    min_term_frequency: int = 1,
    use_esm2: bool = True,
    use_dssp: bool = True,
    use_sasa: bool = True,
) -> dict[str, Any]:
    if framework == "pyg":
        datasets = build_split_datasets(
            framework="pyg",
            root=root,
            aspect=aspect,
            split_dir=split_dir,
            min_term_frequency=min_term_frequency,
            use_esm2=use_esm2,
            use_dssp=use_dssp,
            use_sasa=use_sasa,
        )
        loaders = {
            split_name: build_pyg_loader(
                dataset=datasets[split_name],
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                seed=seed,
            )
            for split_name in SPLIT_NAMES
        }
    elif framework == "dgl":
        datasets = build_split_datasets(
            framework="dgl",
            root=root,
            aspect=aspect,
            split_dir=split_dir,
            min_term_frequency=min_term_frequency,
            use_esm2=use_esm2,
            use_dssp=use_dssp,
            use_sasa=use_sasa,
        )
        loaders = {
            split_name: build_dgl_loader(
                dataset=datasets[split_name],
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                seed=seed,
            )
            for split_name in SPLIT_NAMES
        }
    else:
        raise ValueError(f"Unknown framework: {framework}")

    summary = {
        "framework": framework,
        "aspect": aspect,
        "split_sizes": {split_name: len(datasets[split_name]) for split_name in SPLIT_NAMES},
    }
    for split_name in SPLIT_NAMES:
        if len(datasets[split_name]) == 0:
            summary[f"{split_name}_batch"] = None
            continue
        batch = next(iter(loaders[split_name]))
        summary[f"{split_name}_batch"] = describe_batch(batch, framework=framework)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aspects = parse_aspects(args.aspects)
    output_dir = args.output_dir or (args.root / "splits")

    split_summary = export_split_manifests(
        root=args.root,
        output_dir=output_dir,
        aspects=aspects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_term_frequency=args.min_term_frequency,
    )
    verification = {}
    for aspect in aspects:
        verification[aspect] = {}
        for framework in args.frameworks:
            verification[aspect][framework] = verify_loader_summary(
                framework=framework,
                root=args.root,
                aspect=aspect,
                split_dir=output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                min_term_frequency=args.min_term_frequency,
                use_esm2=not args.disable_esm2,
                use_dssp=not args.disable_dssp,
                use_sasa=not args.disable_sasa,
            )

    payload = {
        "split_summary": split_summary,
        "verification": verification,
    }
    summary_path = Path(output_dir) / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
