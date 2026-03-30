#!/usr/bin/env python3
"""
Benchmark PyG / DGL graph dataloaders on the exported protein graph cache.

The script focuses on the stages that matter on a remote training server:
dataset initialization, dataloader construction, and one measured epoch over
the chosen split. CPU metrics require psutil. NVIDIA GPU metrics are collected
through pynvml when available, otherwise via nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cafa_graph_dataloaders as dataloaders
import cafa_graph_dataset as graphs

try:  # pragma: no cover - optional in the default py313 env
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:  # pragma: no cover - optional
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyG / DGL graph dataloaders and record throughput plus system utilization."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--aspects", nargs="*", default=list(dataloaders.DEFAULT_ASPECTS))
    parser.add_argument(
        "--frameworks",
        nargs="*",
        choices=["pyg", "dgl"],
        default=["pyg", "dgl"],
    )
    parser.add_argument("--split", choices=list(dataloaders.SPLIT_NAMES), default="train")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-term-frequency", type=positive_int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--sample-interval", type=positive_float, default=0.2)
    parser.add_argument("--max-batches", type=non_negative_int, default=0)
    parser.add_argument("--warmup-batches", type=non_negative_int, default=1)
    parser.add_argument("--seed", type=int, default=dataloaders.DEFAULT_SPLIT_SEED)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--disable-esm2", action="store_true")
    parser.add_argument("--disable-dssp", action="store_true")
    parser.add_argument("--disable-sasa", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device_name: str):
    torch_module = graphs.require_torch()
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def synchronize_device(device: Any) -> None:
    torch_module = graphs.require_torch()
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.synchronize()


def touch_pyg_batch(batch: Any, device: Any) -> None:
    probe = batch
    if device.type != "cpu":
        probe = batch.to(device)
    scalar = probe.x.float().sum()
    scalar = scalar + probe.edge_attr.float().sum() + probe.y.float().sum() + probe.graph_feat.float().sum()
    scalar = scalar * 0.0 + 1.0
    float(scalar.detach().cpu().item())


def touch_dgl_batch(batch: Any, device: Any) -> None:
    probe = batch
    if device.type != "cpu":
        probe = batch.to(device)
        probe.y = batch.y.to(device)
        probe.graph_feat = batch.graph_feat.to(device)
    scalar = probe.ndata["x"].float().sum()
    scalar = scalar + probe.edata["edge_attr"].float().sum()
    scalar = scalar + probe.y.float().sum() + probe.graph_feat.float().sum()
    scalar = scalar * 0.0 + 1.0
    float(scalar.detach().cpu().item())


def describe_batch_counts(batch: Any, framework: str) -> tuple[int, int, int]:
    if framework == "pyg":
        return int(batch.num_graphs), int(batch.num_nodes), int(batch.num_edges)
    if framework == "dgl":
        return int(len(batch.batch_num_nodes())), int(batch.num_nodes()), int(batch.num_edges())
    raise ValueError(f"Unknown framework: {framework}")


class ResourceMonitor:
    def __init__(self, sample_interval: float, device: Any) -> None:
        self.sample_interval = float(sample_interval)
        self.device = device
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if psutil is not None else None
        self._nvml_handle = None
        if pynvml is not None and device.type == "cuda":
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    def _gpu_sample(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}

        if self._nvml_handle is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return {
                    "gpu_util_percent": float(utilization.gpu),
                    "gpu_mem_used_mb": float(memory.used) / (1024.0 * 1024.0),
                    "gpu_mem_total_mb": float(memory.total) / (1024.0 * 1024.0),
                }
            except Exception:
                return {}

        if shutil.which("nvidia-smi") is None:
            return {}
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            values = [chunk.strip() for chunk in completed.stdout.strip().split(",")]
            if len(values) != 3:
                return {}
            return {
                "gpu_util_percent": float(values[0]),
                "gpu_mem_used_mb": float(values[1]),
                "gpu_mem_total_mb": float(values[2]),
            }
        except Exception:
            return {}

    def _sample_once(self) -> None:
        sample: dict[str, float] = {}
        if self._process is not None:
            sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            sample["process_cpu_percent"] = float(self._process.cpu_percent(interval=None))
            sample["process_rss_mb"] = float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        sample.update(self._gpu_sample())
        if sample:
            self.samples.append(sample)

    def _run(self) -> None:
        if self._process is not None:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.sample_interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self.samples:
            return {
                "sample_count": 0,
                "cpu_backend": "unavailable" if psutil is None else "psutil",
                "gpu_backend": "unavailable",
            }

        keys = sorted({key for sample in self.samples for key in sample})
        averages = {}
        maxima = {}
        for key in keys:
            values = [sample[key] for sample in self.samples if key in sample]
            if not values:
                continue
            averages[key] = sum(values) / len(values)
            maxima[key] = max(values)
        return {
            "sample_count": len(self.samples),
            "cpu_backend": "psutil" if psutil is not None else "unavailable",
            "gpu_backend": (
                "pynvml"
                if self.device.type == "cuda" and self._nvml_handle is not None
                else "nvidia-smi"
                if self.device.type == "cuda" and shutil.which("nvidia-smi")
                else "unavailable"
            ),
            "avg": averages,
            "max": maxima,
        }


def benchmark_loader(
    loader: Any,
    framework: str,
    device: Any,
    warmup_batches: int,
    max_batches: int,
    sample_interval: float,
) -> dict[str, Any]:
    touch_fn = touch_pyg_batch if framework == "pyg" else touch_dgl_batch

    if warmup_batches > 0:
        for batch_index, batch in enumerate(loader):
            touch_fn(batch, device)
            synchronize_device(device)
            if batch_index + 1 >= warmup_batches:
                break

    iterator = iter(loader)
    monitor = ResourceMonitor(sample_interval=sample_interval, device=device)
    totals = {
        "batches": 0,
        "graphs": 0,
        "nodes": 0,
        "edges": 0,
    }
    first_batch_seconds = None
    monitor.start()
    start = time.perf_counter()
    while True:
        if max_batches > 0 and totals["batches"] >= max_batches:
            break
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        touch_fn(batch, device)
        synchronize_device(device)
        batch_elapsed = time.perf_counter() - batch_start
        if first_batch_seconds is None:
            first_batch_seconds = batch_elapsed
        graph_count, node_count, edge_count = describe_batch_counts(batch, framework)
        totals["batches"] += 1
        totals["graphs"] += graph_count
        totals["nodes"] += node_count
        totals["edges"] += edge_count
    elapsed = time.perf_counter() - start
    resource_summary = monitor.stop()

    throughput = {
        "batches_per_second": (totals["batches"] / elapsed) if elapsed > 0 else 0.0,
        "graphs_per_second": (totals["graphs"] / elapsed) if elapsed > 0 else 0.0,
        "nodes_per_second": (totals["nodes"] / elapsed) if elapsed > 0 else 0.0,
        "edges_per_second": (totals["edges"] / elapsed) if elapsed > 0 else 0.0,
    }
    return {
        "elapsed_seconds": elapsed,
        "first_batch_seconds": first_batch_seconds,
        "totals": totals,
        "throughput": throughput,
        "resource_usage": resource_summary,
    }


def benchmark_framework_aspect(
    root: str | Path,
    split_dir: str | Path,
    aspect: str,
    framework: str,
    split_name: str,
    batch_size: int,
    num_workers: int,
    device: Any,
    sample_interval: float,
    max_batches: int,
    warmup_batches: int,
    seed: int,
    min_term_frequency: int,
    use_esm2: bool,
    use_dssp: bool,
    use_sasa: bool,
) -> dict[str, Any]:
    dataset_start = time.perf_counter()
    dataset = dataloaders.build_split_dataset(
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
    dataset_seconds = time.perf_counter() - dataset_start

    loader_start = time.perf_counter()
    if framework == "pyg":
        loader = dataloaders.build_pyg_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
        )
    else:
        loader = dataloaders.build_dgl_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
        )
    loader_seconds = time.perf_counter() - loader_start

    iteration_summary = benchmark_loader(
        loader=loader,
        framework=framework,
        device=device,
        warmup_batches=warmup_batches,
        max_batches=max_batches,
        sample_interval=sample_interval,
    )
    return {
        "framework": framework,
        "aspect": aspect,
        "split": split_name,
        "device": str(device),
        "dataset_init_seconds": dataset_seconds,
        "dataloader_build_seconds": loader_seconds,
        "dataset_size": len(dataset),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "iteration": iteration_summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aspects = dataloaders.parse_aspects(args.aspects)
    split_dir = args.split_dir or (args.root / "splits")
    device = resolve_device(args.device)

    if not (Path(split_dir) / "summary.json").exists():
        dataloaders.export_split_manifests(
            root=args.root,
            output_dir=split_dir,
            aspects=aspects,
            seed=args.seed,
            min_term_frequency=args.min_term_frequency,
        )

    results = []
    for aspect in aspects:
        for framework in args.frameworks:
            results.append(
                benchmark_framework_aspect(
                    root=args.root,
                    split_dir=split_dir,
                    aspect=aspect,
                    framework=framework,
                    split_name=args.split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    sample_interval=args.sample_interval,
                    max_batches=args.max_batches,
                    warmup_batches=args.warmup_batches,
                    seed=args.seed,
                    min_term_frequency=args.min_term_frequency,
                    use_esm2=not args.disable_esm2,
                    use_dssp=not args.disable_dssp,
                    use_sasa=not args.disable_sasa,
                )
            )

    payload = {
        "root": str(args.root.resolve()),
        "split_dir": str(Path(split_dir).resolve()),
        "device": str(device),
        "results": results,
    }
    output_path = args.output_path or (Path(split_dir) / "benchmark_summary.json")
    pipeline = Path(output_path)
    pipeline.parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
