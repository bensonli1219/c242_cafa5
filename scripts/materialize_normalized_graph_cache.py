#!/usr/bin/env python3
"""Materialize a normalized CAFA graph cache without changing labels.

This copies graph tensors from an existing graph cache to a new cache root,
applies ``cafa_graph_dataset.normalize_structural_features`` to each tensor,
and rewrites ``metadata/entries.json`` so graph paths point at the new files.
All labels, term counts, vocab files, split IDs, and modality caches are
preserved. The script is resumable and intended for Savio for full-dataset
runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cafa_graph_dataset as graphs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a normalized copy of a CAFA graph cache without changing labels."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--copy-modality-cache", action="store_true")
    parser.add_argument("--link-modality-cache", action="store_true")
    parser.add_argument("--copy-splits", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def normalize_one(input_path: str, output_path: str, resume: bool) -> dict[str, Any]:
    torch_module = graphs.require_torch()
    src = Path(input_path)
    dst = Path(output_path)
    if resume and dst.exists():
        return {"status": "skipped", "input_path": input_path, "output_path": output_path}
    try:
        payload = torch_module.load(src, map_location="cpu", weights_only=False)
        graphs.normalize_structural_features(payload)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(payload, dst)
        return {"status": "ok", "input_path": input_path, "output_path": output_path}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "input_path": input_path,
            "output_path": output_path,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tasks(tasks: list[tuple[str, str]], workers: int, resume: bool, progress_every: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0

    def record(result: dict[str, Any]) -> None:
        nonlocal completed
        results.append(result)
        completed += 1
        if completed == 1 or completed % progress_every == 0 or completed == len(tasks):
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            failures = sum(1 for item in results if item["status"] == "failed")
            print(
                f"[progress] normalized={completed}/{len(tasks)} "
                f"failures={failures} rate={rate:.2f}/s elapsed={elapsed:.1f}s",
                flush=True,
            )

    if workers <= 1:
        for input_path, output_path in tasks:
            record(normalize_one(input_path, output_path, resume=resume))
        return results

    pending_iter = iter(tasks)
    in_flight: dict[Any, tuple[str, str]] = {}
    max_inflight = max(1, workers * 4)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        def submit_one() -> None:
            task = next(pending_iter, None)
            if task is None:
                return
            future = executor.submit(normalize_one, task[0], task[1], resume)
            in_flight[future] = task

        for _ in range(min(max_inflight, len(tasks))):
            submit_one()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future)
                record(future.result())
                submit_one()
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    input_metadata = input_root / "metadata"
    output_metadata = output_root / "metadata"
    output_graphs = output_root / "graphs"
    if args.copy_modality_cache and args.link_modality_cache:
        raise ValueError("Use only one of --copy-modality-cache or --link-modality-cache.")

    entries = read_json(input_metadata / "entries.json")
    if args.limit is not None:
        entries = entries[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    output_graphs.mkdir(parents=True, exist_ok=True)

    normalized_entries = []
    tasks = []
    for entry in entries:
        entry_id = str(entry["entry_id"])
        src_graph = Path(entry["graph_path"])
        dst_graph = output_graphs / f"{entry_id}.pt"
        updated = dict(entry)
        updated["graph_path"] = str(dst_graph.resolve())
        normalized_entries.append(updated)
        tasks.append((str(src_graph), str(dst_graph)))

    results = run_tasks(
        tasks=tasks,
        workers=max(1, int(args.workers)),
        resume=bool(args.resume),
        progress_every=max(1, int(args.progress_every)),
    )
    failures = [result for result in results if result["status"] == "failed"]

    output_metadata.mkdir(parents=True, exist_ok=True)
    write_json(output_metadata / "entries.json", normalized_entries)
    for name in ("term_counts.json", "schema.json"):
        shutil.copy2(input_metadata / name, output_metadata / name)
    copy_tree(input_metadata / "vocabs", output_metadata / "vocabs")

    if args.copy_splits:
        copy_tree(input_root / "splits", output_root / "splits")
    if args.link_modality_cache:
        link_path = output_root / "modality_cache"
        target = input_root / "modality_cache"
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                shutil.rmtree(link_path)
        link_path.symlink_to(target, target_is_directory=True)
    if args.copy_modality_cache:
        copy_tree(input_root / "modality_cache", output_root / "modality_cache")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "entries": len(normalized_entries),
        "ok": sum(1 for result in results if result["status"] == "ok"),
        "skipped": sum(1 for result in results if result["status"] == "skipped"),
        "failures": len(failures),
        "labels_preserved": True,
        "normalization": "structural_node_edge_graph_feature_scaling",
        "copy_splits": bool(args.copy_splits),
        "copy_modality_cache": bool(args.copy_modality_cache),
        "link_modality_cache": bool(args.link_modality_cache),
    }
    write_json(output_root / "normalization_summary.json", summary)
    if failures:
        write_json(output_root / "normalization_failures.json", failures)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
