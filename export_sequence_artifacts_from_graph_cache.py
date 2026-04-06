#!/usr/bin/env python3
"""
Mirror graph split manifests into sequence-side artifacts and export a
protein-level ESM2 matrix from the graph modality cache.

Expected standard layout under ``--run-root``:

  <run-root>/manifests/training_index.parquet
  <run-root>/graph_cache/splits/<aspect>/
  <run-root>/graph_cache/metadata/vocabs/<ASPECT>.json
  <run-root>/graph_cache/modality_cache/esm2/*.pt

By default the script writes:

  <run-root>/sequence_artifacts/matched_structure_splits/
  <run-root>/sequence_artifacts/protein_esm2_t30_150m_640_from_graph_cache/
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:  # pragma: no cover - optional outside the graph / notebook env
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional in lighter notebook envs
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_ASPECTS = ("BPO", "CCO", "MFO")
DEFAULT_MATCHED_SPLIT_DIRNAME = "matched_structure_splits"
DEFAULT_PROTEIN_ESM_DIRNAME = "protein_esm2_t30_150m_640_from_graph_cache"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_aspects(values: Iterable[str] | None) -> list[str]:
    parsed = []
    for value in values or DEFAULT_ASPECTS:
        aspect = str(value).strip().upper()
        if not aspect:
            continue
        if aspect not in DEFAULT_ASPECTS:
            raise ValueError(f"Unknown aspect: {aspect}")
        if aspect not in parsed:
            parsed.append(aspect)
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror graph split IDs into sequence-side folders and export a "
            "protein-level ESM2 matrix from graph cache modality outputs."
        )
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--aspects", nargs="*", default=list(DEFAULT_ASPECTS))
    parser.add_argument("--matched-split-dir", type=Path, default=None)
    parser.add_argument("--protein-esm-dir", type=Path, default=None)
    parser.add_argument("--min-term-frequency", type=positive_int, default=1)
    parser.add_argument("--skip-matched-splits", action="store_true")
    parser.add_argument("--skip-protein-esm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=500,
        help="Print progress every N ESM cache files during scan/write (default: 500).",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=positive_int,
        default=250_000,
        help="Rows per Parquet batch while counting ok training entries (default: 250000).",
    )
    parser.add_argument(
        "--parquet-progress-every-batches",
        type=positive_int,
        default=10,
        help="Print Parquet scan progress every N batches (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of worker threads for loading ESM cache .pt files (default: 1).",
    )
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "log", "tqdm", "none"),
        default="auto",
        help=(
            "Progress display mode. 'auto' uses tqdm for interactive terminals and "
            "periodic log lines elsewhere."
        ),
    )
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(message, flush=True)


def should_use_tqdm(progress_mode: str) -> bool:
    if progress_mode == "none":
        return False
    if progress_mode == "log":
        return False
    if progress_mode == "tqdm":
        if tqdm is None:
            raise RuntimeError("progress-mode=tqdm requires the 'tqdm' package to be installed.")
        return True
    return tqdm is not None and sys.stderr.isatty()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def classify_paths(paths: Iterable[Path]) -> str:
    existence = [path.exists() for path in paths]
    if all(existence):
        return "complete"
    if not any(existence):
        return "missing"
    return "partial"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def copy_text_file(src: Path, dst: Path, overwrite: bool = False) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    existed = dst.exists()
    if dst.exists() and not overwrite:
        log(f"[skip] already exists: {dst}")
        return "skipped"
    atomic_write_text(dst, src.read_text(encoding="utf-8"))
    action = "overwrite" if existed and overwrite else "write"
    log(f"[{action}] {dst}")
    return "written"


def write_json_artifact(path: Path, payload: Any, overwrite: bool = False) -> str:
    existed = path.exists()
    if path.exists() and not overwrite:
        log(f"[skip] JSON already exists: {path}")
        return "skipped"
    atomic_write_json(path, payload)
    action = "overwrite" if existed and overwrite else "write"
    log(f"[{action}] {path}")
    return "written"


def write_lines_artifact(path: Path, lines: Iterable[str], overwrite: bool = False) -> str:
    existed = path.exists()
    if path.exists() and not overwrite:
        log(f"[skip] text file already exists: {path}")
        return "skipped"
    atomic_write_lines(path, lines)
    action = "overwrite" if existed and overwrite else "write"
    log(f"[{action}] {path}")
    return "written"


def count_parquet_matches(
    path: Path,
    column: str,
    expected_value: str,
    label: str,
    batch_size: int = 250_000,
    progress_every_batches: int = 10,
) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    parquet_file = pq.ParquetFile(path)
    total_matches = 0
    total_rows = 0
    log(
        f"[progress] {label}: scanning {path} with {parquet_file.num_row_groups} row groups "
        f"(batch_size={batch_size:,})"
    )
    for batch_index, batch in enumerate(
        parquet_file.iter_batches(columns=[column], batch_size=batch_size),
        start=1,
    ):
        matches_scalar = pc.sum(pc.cast(pc.equal(batch.column(0), expected_value), pa.int64()))
        total_matches += int(matches_scalar.as_py() or 0)
        total_rows += batch.num_rows
        if batch_index == 1 or batch_index % progress_every_batches == 0:
            log(
                f"[progress] {label}: processed {total_rows:,} rows across {batch_index} batches; "
                f"matched {total_matches:,}"
            )
    log(
        f"[progress] {label}: finished {total_rows:,} rows; matched {total_matches:,}"
    )
    return total_matches


def count_ok_training_entries(
    training_manifest: Path,
    batch_size: int = 250_000,
    progress_every_batches: int = 10,
) -> int:
    return count_parquet_matches(
        training_manifest,
        column="af_status",
        expected_value="ok",
        label="training_index.af_status",
        batch_size=batch_size,
        progress_every_batches=progress_every_batches,
    )


def count_pt_files(cache_dir: Path, progress_every: int = 5_000) -> int:
    if not cache_dir.exists():
        return 0
    count = 0
    log(f"[progress] graph_esm_cache: scanning {cache_dir}")
    for item in cache_dir.iterdir():
        if item.is_file() and item.suffix == ".pt":
            count += 1
            if count % progress_every == 0:
                log(f"[progress] graph_esm_cache: counted {count:,} .pt files")
    log(f"[progress] graph_esm_cache: finished with {count:,} .pt files")
    return count


def modality_cache_state_from_count(pt_count: int, expected_count: int) -> tuple[str, dict[str, int]]:
    if expected_count <= 0:
        return ("complete" if pt_count > 0 else "missing"), {"pt_files": pt_count, "expected": expected_count}
    if pt_count == 0:
        return "missing", {"pt_files": 0, "expected": expected_count}
    if pt_count >= expected_count:
        return "complete", {"pt_files": pt_count, "expected": expected_count}
    return "partial", {"pt_files": pt_count, "expected": expected_count}


def modality_cache_state(cache_dir: Path, expected_count: int, progress_every: int = 5_000) -> tuple[str, dict[str, int]]:
    pt_count = count_pt_files(cache_dir, progress_every=progress_every)
    return modality_cache_state_from_count(pt_count=pt_count, expected_count=expected_count)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required to read graph ESM cache .pt files. "
            "Use a Python environment with torch installed."
        )
    return torch


def extract_embedding(payload: dict[str, Any], fallback_entry_id: str) -> tuple[str, np.ndarray | None]:
    entry_id = str(payload.get("entry_id") or fallback_entry_id)
    protein_embedding = payload.get("protein_embedding")
    if protein_embedding is None:
        return entry_id, None
    if hasattr(protein_embedding, "detach"):
        protein_embedding = protein_embedding.detach().cpu().numpy()
    array = np.asarray(protein_embedding, dtype=np.float32)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(
            f"Expected a 1D protein embedding for {entry_id}; got shape {tuple(array.shape)}"
        )
    return entry_id, array


def load_embedding_from_cache_path(cache_path: Path) -> tuple[Path, str, np.ndarray | None]:
    torch_module = require_torch()
    payload = torch_module.load(cache_path, map_location="cpu", weights_only=False)
    entry_id, embedding = extract_embedding(payload, fallback_entry_id=cache_path.stem)
    return cache_path, entry_id, embedding


def iter_loaded_embeddings(
    cache_paths: list[Path],
    workers: int = 1,
) -> Iterable[tuple[Path, str, np.ndarray | None]]:
    if workers <= 1:
        for cache_path in cache_paths:
            yield load_embedding_from_cache_path(cache_path)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(load_embedding_from_cache_path, cache_paths)


def mirror_graph_splits(
    graph_split_dir: Path,
    graph_vocab_dir: Path,
    matched_split_dir: Path,
    aspects: Iterable[str],
    min_term_frequency: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not (graph_split_dir / "summary.json").exists():
        raise FileNotFoundError(
            f"Graph split summary is missing: {graph_split_dir / 'summary.json'}"
        )
    aspects = parse_aspects(aspects)
    copied_files = 0
    skipped_files = 0
    matched_summary: dict[str, Any] = {
        "source_graph_splits": str(graph_split_dir.resolve()),
        "source_graph_vocab_dir": str(graph_vocab_dir.resolve()),
        "root": str(matched_split_dir.resolve()),
        "min_term_frequency": min_term_frequency,
        "aspects": {},
    }
    for aspect in aspects:
        aspect_lower = aspect.lower()
        src_dir = graph_split_dir / aspect_lower
        dst_dir = matched_split_dir / aspect_lower
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing graph split directory for {aspect}: {src_dir}")
        for name in ("train.txt", "val.txt", "test.txt", "summary.json"):
            result = copy_text_file(src_dir / name, dst_dir / name, overwrite=overwrite)
            copied_files += int(result == "written")
            skipped_files += int(result == "skipped")
        vocab_src = graph_vocab_dir / f"{aspect}.json"
        vocab_dst = dst_dir / "vocab.json"
        result = copy_text_file(vocab_src, vocab_dst, overwrite=overwrite)
        copied_files += int(result == "written")
        skipped_files += int(result == "skipped")
        matched_summary["aspects"][aspect] = {"source": str(src_dir.resolve())}
    summary_result = write_json_artifact(
        matched_split_dir / "summary.json",
        matched_summary,
        overwrite=overwrite,
    )
    copied_files += int(summary_result == "written")
    skipped_files += int(summary_result == "skipped")
    return {
        "root": str(matched_split_dir.resolve()),
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "aspects": aspects,
    }


def export_protein_esm_matrix(
    cache_paths: list[Path],
    esm_cache_dir: Path,
    output_dir: Path,
    expected_ok_entries: int,
    overwrite: bool = False,
    progress_every: int = 500,
    workers: int = 1,
    progress_mode: str = "auto",
) -> dict[str, Any]:
    x_path = output_dir / "X.npy"
    ids_path = output_dir / "entry_ids.txt"
    meta_path = output_dir / "meta.json"
    output_state = classify_paths([x_path, ids_path, meta_path])
    if output_state == "complete" and not overwrite:
        log(f"[skip] protein-level 640-d matrix already exists: {x_path}")
        matrix = np.load(x_path, mmap_mode="r")
        log(f"protein-level 640-d shape: {matrix.shape} dtype: {matrix.dtype}")
        return {
            "root": str(output_dir.resolve()),
            "entry_count": int(matrix.shape[0]),
            "embedding_dim": int(matrix.shape[1]),
            "state": "skipped_existing",
        }
    if output_state == "partial" and not overwrite:
        raise RuntimeError(
            f"Partial protein ESM outputs detected under {output_dir}. "
            "Rerun with --overwrite after checking the partial files."
        )

    if not cache_paths:
        raise FileNotFoundError(f"No .pt ESM cache files were found under {esm_cache_dir}")

    ensure_parent(x_path)
    tmp_x_path = x_path.with_name(f"{x_path.name}.tmp")
    tmp_ids_path = ids_path.with_name(f"{ids_path.name}.tmp")
    tmp_meta_path = meta_path.with_name(f"{meta_path.name}.tmp")
    use_tqdm = should_use_tqdm(progress_mode)
    total_files = len(cache_paths)
    log(
        f"[progress] protein_esm2_640: loading {total_files:,} cache files "
        f"(workers={workers}, progress_mode={'tqdm' if use_tqdm else progress_mode})"
    )
    iterator: Iterable[tuple[Path, str, np.ndarray | None]] = iter_loaded_embeddings(
        cache_paths,
        workers=workers,
    )
    if use_tqdm:
        iterator = tqdm(
            iterator,
            total=total_files,
            desc="protein_esm2_640",
            unit="file",
            mininterval=1.0,
        )

    entry_ids: list[str] = []
    matrix: np.ndarray | None = None
    embedding_dim: int | None = None
    valid_count = 0
    for file_index, (cache_path, entry_id, embedding) in enumerate(iterator, start=1):
        if embedding is None:
            if not use_tqdm and (file_index == 1 or file_index % progress_every == 0 or file_index == total_files):
                log(
                    f"[progress] protein_esm2_640: scanned {file_index:,}/{total_files:,}; "
                    f"valid embeddings={valid_count:,}"
                )
            continue
        current_dim = int(embedding.shape[0])
        if embedding_dim is None:
            embedding_dim = current_dim
            matrix = np.empty((total_files, embedding_dim), dtype=np.float32)
            log(
                f"[progress] protein_esm2_640: allocated in-memory matrix with capacity "
                f"({total_files:,}, {embedding_dim})"
            )
        elif current_dim != embedding_dim:
            raise ValueError(
                f"Inconsistent embedding dim for {cache_path}: "
                f"expected {embedding_dim}, got {current_dim}"
            )
        assert matrix is not None
        matrix[valid_count] = embedding
        entry_ids.append(entry_id)
        valid_count += 1
        if not use_tqdm and (file_index == 1 or file_index % progress_every == 0 or file_index == total_files):
            log(
                f"[progress] protein_esm2_640: scanned {file_index:,}/{total_files:,}; "
                f"valid embeddings={valid_count:,}"
            )

    if matrix is None or embedding_dim is None or valid_count == 0:
        raise RuntimeError(f"No protein_embedding payloads were found in {esm_cache_dir}")
    if expected_ok_entries > 0 and valid_count < expected_ok_entries:
        raise RuntimeError(
            f"Expected at least {expected_ok_entries:,} protein embeddings, "
            f"but found only {valid_count:,} valid cache payloads in {esm_cache_dir}"
        )
    if expected_ok_entries > 0 and valid_count > expected_ok_entries:
        log(
            f"[warn] protein_esm2_640: found {valid_count:,} valid embeddings for "
            f"{expected_ok_entries:,} expected ok entries; exporting all valid rows."
        )

    trimmed_matrix = matrix[:valid_count]
    with tmp_x_path.open("wb") as handle:
        np.save(handle, trimmed_matrix)
    atomic_write_lines(tmp_ids_path, entry_ids)
    atomic_write_json(
        tmp_meta_path,
        {
            "entry_count": len(entry_ids),
            "embedding_dim": embedding_dim,
            "source_cache_dir": str(esm_cache_dir.resolve()),
            "model_name": "facebook/esm2_t30_150M_UR50D",
            "notes": "Derived from existing graph ESM2 caches. No recomputation from raw sequence.",
        },
    )
    tmp_x_path.replace(x_path)
    tmp_ids_path.replace(ids_path)
    tmp_meta_path.replace(meta_path)
    log(f"[write] {x_path}")
    log(f"[write] {ids_path}")
    log(f"[write] {meta_path}")

    matrix = np.load(x_path, mmap_mode="r")
    log(f"protein-level 640-d shape: {matrix.shape} dtype: {matrix.dtype}")
    return {
        "root": str(output_dir.resolve()),
        "entry_count": len(entry_ids),
        "embedding_dim": embedding_dim,
        "state": "written",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    aspects = parse_aspects(args.aspects)
    if args.skip_matched_splits and args.skip_protein_esm:
        raise ValueError("Both export stages are disabled. Remove at least one --skip-* flag.")

    manifests_dir = run_root / "manifests"
    graph_cache_dir = run_root / "graph_cache"
    graph_split_dir = graph_cache_dir / "splits"
    graph_vocab_dir = graph_cache_dir / "metadata" / "vocabs"
    esm_cache_dir = graph_cache_dir / "modality_cache" / "esm2"
    sequence_artifacts_dir = run_root / "sequence_artifacts"
    matched_split_dir = (
        args.matched_split_dir.expanduser().resolve()
        if args.matched_split_dir is not None
        else (sequence_artifacts_dir / DEFAULT_MATCHED_SPLIT_DIRNAME).resolve()
    )
    protein_esm_dir = (
        args.protein_esm_dir.expanduser().resolve()
        if args.protein_esm_dir is not None
        else (sequence_artifacts_dir / DEFAULT_PROTEIN_ESM_DIRNAME).resolve()
    )

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "aspects": aspects,
    }

    if not args.skip_matched_splits:
        summary["matched_structure_splits"] = mirror_graph_splits(
            graph_split_dir=graph_split_dir,
            graph_vocab_dir=graph_vocab_dir,
            matched_split_dir=matched_split_dir,
            aspects=aspects,
            min_term_frequency=args.min_term_frequency,
            overwrite=args.overwrite,
        )

    if not args.skip_protein_esm:
        expected_ok_entries = count_ok_training_entries(
            manifests_dir / "training_index.parquet",
            batch_size=args.parquet_batch_size,
            progress_every_batches=args.parquet_progress_every_batches,
        )
        log(f"[progress] graph_esm_cache: listing cache files under {esm_cache_dir}")
        cache_paths = sorted(
            path for path in esm_cache_dir.iterdir()
            if path.is_file() and path.suffix == ".pt"
        ) if esm_cache_dir.exists() else []
        log(f"[progress] graph_esm_cache: found {len(cache_paths):,} .pt files")
        esm_state, esm_meta = modality_cache_state_from_count(
            pt_count=len(cache_paths),
            expected_count=expected_ok_entries,
        )
        log(
            f"[graph_esm_cache] state = {esm_state} "
            f"(pt_files={esm_meta['pt_files']:,}, expected={esm_meta['expected']:,})"
        )
        if esm_state != "complete":
            raise RuntimeError(
                f"Graph ESM cache is {esm_state}; refusing to export protein-level embeddings."
            )
        summary["protein_esm2_640_from_graph_cache"] = export_protein_esm_matrix(
            cache_paths=cache_paths,
            esm_cache_dir=esm_cache_dir,
            output_dir=protein_esm_dir,
            expected_ok_entries=expected_ok_entries,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            workers=args.workers,
            progress_mode=args.progress_mode,
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
