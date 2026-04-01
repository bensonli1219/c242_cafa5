#!/usr/bin/env python3
"""
Build protein-level graph caches from AlphaFold feature parquet tables and expose
PyG/DGL dataset loaders with a stable multimodal schema.

The local MVP only fills the "base" 32-dimensional structural block from the
existing parquet tables. Reserved DSSP/SASA (10 dims) and ESM2 (640 dims)
feature slots are zero-filled so that remote enrichment can overlay features
later without changing the dataset API.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Protocol

import pyarrow.parquet as pq

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

import cafa5_alphafold_pipeline as pipeline

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency in the default py313 env
    import torch
except ImportError:  # pragma: no cover - exercised when graph env is not installed
    torch = None

try:  # pragma: no cover - optional dependency in the default py313 env
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover
    class TorchDataset:  # type: ignore[no-redef]
        pass

try:  # pragma: no cover - optional dependency in the default py313 env
    from torch_geometric.data import Data as PygData
except ImportError:  # pragma: no cover
    PygData = None

try:  # pragma: no cover - optional dependency in the default py313 env
    import dgl
except ImportError:  # pragma: no cover
    dgl = None


ASPECT_TO_LABEL_KEY = {
    "BPO": "go_terms_bpo",
    "CCO": "go_terms_cco",
    "MFO": "go_terms_mfo",
}

AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X"]
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}

BASE_FEATURE_DIM = 32
DSSP_SASA_DIM = 10
ESM2_DIM = 640
NODE_FEATURE_DIM = BASE_FEATURE_DIM + DSSP_SASA_DIM + ESM2_DIM
EDGE_ATTR_DIM = 6
GRAPH_FEAT_DIM = 13
MODALITY_MASK_DIM = 3

BASE_SLICE = slice(0, BASE_FEATURE_DIM)
DSSP_SASA_SLICE = slice(BASE_FEATURE_DIM, BASE_FEATURE_DIM + DSSP_SASA_DIM)
ESM2_SLICE = slice(BASE_FEATURE_DIM + DSSP_SASA_DIM, NODE_FEATURE_DIM)


class DataProtocol(Protocol):
    x: Any
    pos: Any
    edge_index: Any
    edge_attr: Any
    y: Any


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required for graph cache serialization and dataset loading. "
            "Use the dedicated Python 3.11 graph environment."
        )
    return torch


def require_pyg():
    require_torch()
    if PygData is None:
        raise RuntimeError(
            "torch_geometric is required for CafaPyGDataset. "
            "Install it in the Python 3.11 graph environment."
        )
    return PygData


def require_dgl():
    require_torch()
    if dgl is None:
        raise RuntimeError(
            "dgl is required for CafaDGLDataset. Install it in the Python 3.11 graph environment."
        )
    return dgl


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build protein-level graph caches from AlphaFold feature parquet tables."
    )
    parser.add_argument("--training-index", required=True, type=Path)
    parser.add_argument("--fragment-features", required=True, type=Path)
    parser.add_argument("--residue-features", required=True, type=Path)
    parser.add_argument("--edge-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--entry-ids",
        nargs="*",
        default=None,
        help="Optional entry IDs. Accepts repeated values or comma-separated chunks.",
    )
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--min-term-frequency", type=positive_int, default=1)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of parallel worker processes (default: 1 = serial).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip entries whose .pt file already exists.",
    )
    return parser.parse_args(argv)


def parse_entry_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed: set[str] = set()
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                parsed.add(chunk)
    return parsed or None


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def read_parquet_grouped(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read a parquet file and group rows by entry_id without holding the full
    Python list in memory after grouping."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        for row in batch.to_pylist():
            eid = str(row.get("entry_id", "")).strip()
            if eid:
                grouped[eid].append(row)
    return grouped


def canonical_residue_name(value: str | None) -> str:
    aa = (value or "X").strip().upper()
    return aa if aa in AA_TO_INDEX else "X"


def quantile(values: Iterable[float], q: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("Cannot compute quantile for an empty sequence.")
    return pipeline.interpolate_quantile(sorted_values, q)


def mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        raise ValueError("Cannot compute mean for an empty sequence.")
    return total / count


def radius_of_gyration(positions: list[list[float]]) -> float:
    center_x = mean(position[0] for position in positions)
    center_y = mean(position[1] for position in positions)
    center_z = mean(position[2] for position in positions)
    squared = 0.0
    for x, y, z in positions:
        dx = x - center_x
        dy = y - center_y
        dz = z - center_z
        squared += dx * dx + dy * dy + dz * dz
    return math.sqrt(squared / len(positions))


def group_rows_by_entry(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        entry_id = str(row.get("entry_id", "")).strip()
        if entry_id:
            grouped[entry_id].append(row)
    return dict(grouped)


def residue_priority_key(row: dict[str, Any]) -> tuple[float, float, str]:
    plddt = float(row.get("plddt") or 0.0)
    pae_row_mean = float(row.get("pae_row_mean") or 0.0)
    model_entity_id = str(row.get("model_entity_id") or "")
    return (-plddt, pae_row_mean, model_entity_id)


def edge_priority_key(row: dict[str, Any]) -> tuple[float, float, str]:
    pae_mean_pair = float(row.get("pae_mean_pair") or 0.0)
    distance_ca = float(row.get("distance_ca") or 0.0)
    model_entity_id = str(row.get("model_entity_id") or "")
    return (pae_mean_pair, distance_ca, model_entity_id)


def make_base_feature(
    residue_row: dict[str, Any],
    sequence_length: int,
    contact_degree: int,
    strict_contact_degree: int,
) -> list[float]:
    feature = [0.0] * NODE_FEATURE_DIM

    aa = canonical_residue_name(str(residue_row.get("residue_name1") or "X"))
    feature[AA_TO_INDEX[aa]] = 1.0
    feature[21] = float(residue_row.get("plddt") or 0.0)
    feature[22] = 1.0 if residue_row.get("is_plddt_very_low") else 0.0
    feature[23] = 1.0 if residue_row.get("is_plddt_low") else 0.0
    feature[24] = 1.0 if residue_row.get("is_plddt_confident") else 0.0
    feature[25] = 1.0 if residue_row.get("is_plddt_very_high") else 0.0
    feature[26] = float(residue_row.get("pae_row_mean") or 0.0)
    feature[27] = float(residue_row.get("pae_row_min") or 0.0)
    feature[28] = float(residue_row.get("pae_row_p90") or 0.0)
    feature[29] = float(contact_degree)
    feature[30] = float(strict_contact_degree)

    cafa_index = residue_row.get("cafa_residue_index")
    if cafa_index is None or sequence_length <= 0:
        feature[31] = float(residue_row.get("sequence_position_fraction") or 0.0)
    else:
        feature[31] = float(cafa_index) / float(sequence_length)

    return feature


def build_term_counts(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    term_counts: dict[str, Counter[str]] = {aspect: Counter() for aspect in ASPECT_TO_LABEL_KEY}
    for row in entries:
        for aspect, key in ASPECT_TO_LABEL_KEY.items():
            for term in row.get(key) or []:
                term_counts[aspect][str(term)] += 1
    return {aspect: dict(counter) for aspect, counter in term_counts.items()}


def build_vocab(term_counts: dict[str, int], min_term_frequency: int = 1) -> list[str]:
    return sorted(term for term, count in term_counts.items() if int(count) >= min_term_frequency)


def write_json(path: Path, payload: Any) -> None:
    pipeline.ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _normalize_cafa_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_protein_graph_record(
    training_row: dict[str, Any],
    fragment_rows: list[dict[str, Any]],
    residue_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sequence_length = int(training_row.get("sequence_length") or len(training_row.get("sequence") or ""))
    selected_candidates: dict[int, dict[str, Any]] = {}

    for residue_row in residue_rows:
        cafa_index = _normalize_cafa_index(residue_row.get("cafa_residue_index"))
        if cafa_index is None or cafa_index <= 0:
            continue
        best = selected_candidates.get(cafa_index)
        if best is None or residue_priority_key(residue_row) < residue_priority_key(best):
            selected_candidates[cafa_index] = residue_row

    if not selected_candidates:
        raise ValueError(f"No resolved cafa_residue_index values for entry {training_row['entry_id']}")

    ordered_cafa_indices = sorted(selected_candidates)
    selected_rows = [selected_candidates[index] for index in ordered_cafa_indices]

    fragment_ids_lookup = sorted({str(row.get("model_entity_id") or "") for row in selected_rows})
    fragment_id_to_index = {value: index for index, value in enumerate(fragment_ids_lookup)}

    node_index_by_cafa = {cafa_index: index for index, cafa_index in enumerate(ordered_cafa_indices)}
    edge_candidates: dict[tuple[int, int], dict[str, Any]] = {}

    for edge_row in edge_rows:
        src_cafa = _normalize_cafa_index(edge_row.get("source_cafa_residue_index"))
        dst_cafa = _normalize_cafa_index(edge_row.get("target_cafa_residue_index"))
        if src_cafa is None or dst_cafa is None or src_cafa == dst_cafa:
            continue

        selected_src = selected_candidates.get(src_cafa)
        selected_dst = selected_candidates.get(dst_cafa)
        if selected_src is None or selected_dst is None:
            continue

        edge_model = str(edge_row.get("model_entity_id") or "")
        src_residue_index = int(edge_row.get("source_residue_index") or 0)
        dst_residue_index = int(edge_row.get("target_residue_index") or 0)

        if (
            str(selected_src.get("model_entity_id") or "") != edge_model
            or str(selected_dst.get("model_entity_id") or "") != edge_model
            or int(selected_src.get("residue_index") or 0) != src_residue_index
            or int(selected_dst.get("residue_index") or 0) != dst_residue_index
        ):
            continue

        edge_key = tuple(sorted((src_cafa, dst_cafa)))
        best_edge = edge_candidates.get(edge_key)
        if best_edge is None or edge_priority_key(edge_row) < edge_priority_key(best_edge):
            edge_candidates[edge_key] = edge_row

    undirected_edges = [edge_candidates[key] for key in sorted(edge_candidates)]
    contact_degrees = [0] * len(selected_rows)
    strict_contact_degrees = [0] * len(selected_rows)

    for edge_row in undirected_edges:
        left_index = node_index_by_cafa[int(edge_row["source_cafa_residue_index"])]
        right_index = node_index_by_cafa[int(edge_row["target_cafa_residue_index"])]
        contact_degrees[left_index] += 1
        contact_degrees[right_index] += 1
        if bool(edge_row.get("is_strict_contact")):
            strict_contact_degrees[left_index] += 1
            strict_contact_degrees[right_index] += 1

    x_rows: list[list[float]] = []
    pos_rows: list[list[float]] = []
    cafa_residue_index_rows: list[int] = []
    residue_index_rows: list[int] = []
    fragment_id_rows: list[int] = []
    modality_mask_rows: list[list[int]] = []

    for node_index, residue_row in enumerate(selected_rows):
        x_rows.append(
            make_base_feature(
                residue_row=residue_row,
                sequence_length=sequence_length,
                contact_degree=contact_degrees[node_index],
                strict_contact_degree=strict_contact_degrees[node_index],
            )
        )
        pos_rows.append(
            [
                float(residue_row.get("x") or 0.0),
                float(residue_row.get("y") or 0.0),
                float(residue_row.get("z") or 0.0),
            ]
        )
        cafa_residue_index_rows.append(int(residue_row["cafa_residue_index"]))
        residue_index_rows.append(int(residue_row["residue_index"]))
        fragment_id_rows.append(fragment_id_to_index[str(residue_row.get("model_entity_id") or "")])
        modality_mask_rows.append([0, 0, 0])

    directed_edge_index: list[list[int]] = [[], []]
    directed_edge_attr: list[list[float]] = []
    for edge_row in undirected_edges:
        left_index = node_index_by_cafa[int(edge_row["source_cafa_residue_index"])]
        right_index = node_index_by_cafa[int(edge_row["target_cafa_residue_index"])]
        attr = [
            float(edge_row.get("distance_ca") or 0.0),
            float(edge_row.get("seq_separation") or 0.0),
            float(edge_row.get("pae_mean_pair") or 0.0),
            1.0 if edge_row.get("is_sequential_neighbor") else 0.0,
            1.0 if edge_row.get("is_short_range_sequence") else 0.0,
            1.0 if edge_row.get("is_strict_contact") else 0.0,
        ]
        directed_edge_index[0].extend([left_index, right_index])
        directed_edge_index[1].extend([right_index, left_index])
        directed_edge_attr.extend([attr, attr.copy()])

    plddt_values = [float(row.get("plddt") or 0.0) for row in selected_rows]
    pae_row_means = [float(row.get("pae_row_mean") or 0.0) for row in selected_rows]
    pae_row_p90s = [float(row.get("pae_row_p90") or 0.0) for row in selected_rows]
    possible_pairs = len(selected_rows) * (len(selected_rows) - 1) / 2
    graph_feat = [
        float(len(selected_rows)),
        float(len(fragment_ids_lookup)),
        mean(plddt_values),
        quantile(plddt_values, 0.5),
        sum(1 for value in plddt_values if value < 50.0) / len(selected_rows),
        sum(1 for value in plddt_values if 50.0 <= value < 70.0) / len(selected_rows),
        sum(1 for value in plddt_values if 70.0 <= value < 90.0) / len(selected_rows),
        sum(1 for value in plddt_values if value >= 90.0) / len(selected_rows),
        mean(contact_degrees) if contact_degrees else 0.0,
        (len(undirected_edges) / possible_pairs) if possible_pairs else 0.0,
        mean(pae_row_means),
        quantile(pae_row_p90s, 0.9),
        radius_of_gyration(pos_rows),
    ]

    return {
        "entry_id": str(training_row["entry_id"]),
        "taxonomy_id": str(training_row.get("taxonomy_id") or ""),
        "fragment_ids": fragment_ids_lookup,
        "cafa_residue_index": cafa_residue_index_rows,
        "residue_index": residue_index_rows,
        "fragment_id": fragment_id_rows,
        "x": x_rows,
        "pos": pos_rows,
        "edge_index": directed_edge_index,
        "edge_attr": directed_edge_attr,
        "graph_feat": graph_feat,
        "node_modality_mask": modality_mask_rows,
        "labels": {
            "BPO": list(training_row.get("go_terms_bpo") or []),
            "CCO": list(training_row.get("go_terms_cco") or []),
            "MFO": list(training_row.get("go_terms_mfo") or []),
        },
        "selected_model_entity_id_by_residue": [
            str(row.get("model_entity_id") or "") for row in selected_rows
        ],
        "selected_residue_number_by_residue": [
            int(row.get("residue_number") or 0) for row in selected_rows
        ],
        "fragment_count_input": len(fragment_rows),
    }


def tensorize_graph_record(graph_record: dict[str, Any]) -> dict[str, Any]:
    torch_module = require_torch()
    return {
        "entry_id": graph_record["entry_id"],
        "taxonomy_id": graph_record["taxonomy_id"],
        "fragment_ids": graph_record["fragment_ids"],
        "cafa_residue_index": torch_module.tensor(graph_record["cafa_residue_index"], dtype=torch_module.long),
        "residue_index": torch_module.tensor(graph_record["residue_index"], dtype=torch_module.long),
        "fragment_id": torch_module.tensor(graph_record["fragment_id"], dtype=torch_module.long),
        "x": torch_module.tensor(graph_record["x"], dtype=torch_module.float32),
        "pos": torch_module.tensor(graph_record["pos"], dtype=torch_module.float32),
        "edge_index": torch_module.tensor(graph_record["edge_index"], dtype=torch_module.long),
        "edge_attr": torch_module.tensor(graph_record["edge_attr"], dtype=torch_module.float32),
        "graph_feat": torch_module.tensor(graph_record["graph_feat"], dtype=torch_module.float32),
        "node_modality_mask": torch_module.tensor(
            graph_record["node_modality_mask"], dtype=torch_module.bool
        ),
        "labels": graph_record["labels"],
    }


def _build_and_save_graph(
    training_row: dict[str, Any],
    fragment_rows: list[dict[str, Any]],
    residue_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    graph_path: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Module-level worker for ProcessPoolExecutor."""
    entry_id = str(training_row.get("entry_id") or "")
    try:
        graph_record = build_protein_graph_record(
            training_row=training_row,
            fragment_rows=fragment_rows,
            residue_rows=residue_rows,
            edge_rows=edge_rows,
        )
        tensor_payload = tensorize_graph_record(graph_record)
        require_torch().save(tensor_payload, Path(graph_path))
        entry_info = {
            "entry_id": entry_id,
            "taxonomy_id": graph_record["taxonomy_id"],
            "graph_path": graph_path,
            "fragment_count": len(graph_record["fragment_ids"]),
            "residue_count": len(graph_record["cafa_residue_index"]),
            "labels": graph_record["labels"],
        }
        return entry_id, entry_info, None
    except Exception as exc:  # noqa: BLE001
        return entry_id, None, f"{type(exc).__name__}: {exc}"


def build_graph_cache(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    entry_filter = parse_entry_filter(args.entry_ids)

    LOG.info("Loading parquet tables…")
    training_rows = read_parquet_rows(args.training_index)
    LOG.info("Loading fragment features…")
    fragment_rows_by_entry = read_parquet_grouped(args.fragment_features)
    LOG.info("Loading residue features…")
    residue_rows_by_entry = read_parquet_grouped(args.residue_features)
    LOG.info("Loading edge features…")
    edge_rows_by_entry = read_parquet_grouped(args.edge_features)
    LOG.info("All tables loaded.")

    output_dir = args.output_dir
    graphs_dir = output_dir / "graphs"
    metadata_dir = output_dir / "metadata"
    pipeline.ensure_parent(graphs_dir / "placeholder")
    pipeline.ensure_parent(metadata_dir / "placeholder")

    # Resume: load existing entries index
    entries_json_path = metadata_dir / "entries.json"
    existing_entries: dict[str, dict[str, Any]] = {}
    if args.resume and entries_json_path.exists():
        for entry in load_json(entries_json_path):
            existing_entries[entry["entry_id"]] = entry
        LOG.info("Resume: %d existing entries found", len(existing_entries))

    # Build task list
    training_rows_by_entry: dict[str, dict[str, Any]] = {}
    tasks: list[tuple] = []
    for training_row in training_rows:
        entry_id = str(training_row.get("entry_id") or "")
        if not entry_id:
            continue
        if entry_filter and entry_id not in entry_filter:
            continue
        if training_row.get("af_status") != "ok":
            continue
        if entry_id not in residue_rows_by_entry:
            continue
        training_rows_by_entry[entry_id] = training_row
        graph_path = str((graphs_dir / f"{entry_id}.pt").resolve())
        if args.resume and entry_id in existing_entries and Path(graph_path).exists():
            continue  # already done
        tasks.append((
            training_row,
            fragment_rows_by_entry.get(entry_id, []),
            residue_rows_by_entry[entry_id],
            edge_rows_by_entry.get(entry_id, []),
            graph_path,
        ))
        if args.limit is not None and len(tasks) + len(existing_entries) >= args.limit:
            break

    LOG.info("Pending: %d graphs to build (%d already done)", len(tasks), len(existing_entries))

    # Process
    graph_entries: list[dict[str, Any]] = list(existing_entries.values())
    failures: list[dict[str, str]] = []

    def _handle(entry_id: str, entry_info: dict | None, error: str | None) -> None:
        if error is not None:
            LOG.warning("FAILED %s: %s", entry_id, error)
            failures.append({"entry_id": entry_id, "error": error})
            return
        graph_entries.append(entry_info)  # type: ignore[arg-type]

    def _progress(it: Any, **kw: Any) -> Any:
        return tqdm(it, **kw) if HAVE_TQDM else it

    if args.workers > 1:
        max_inflight = args.workers * 32
        LOG.info("Parallel mode: %d workers, max %d tasks in-flight", args.workers, max_inflight)
        pending_iter = iter(tasks)
        in_flight: dict = {}
        bar = tqdm(total=len(tasks), desc="Building graphs", unit="entry", dynamic_ncols=True) if HAVE_TQDM else None

        def _submit_one() -> None:
            task = next(pending_iter, None)
            if task is None:
                return
            f = executor.submit(_build_and_save_graph, *task)
            in_flight[f] = str(task[0].get("entry_id") or "")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(min(max_inflight, len(tasks))):
                _submit_one()
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    eid, entry_info, error = future.result()
                    _handle(eid, entry_info, error)
                    if bar is not None:
                        bar.update(1)
                        if failures:
                            bar.set_postfix(failures=len(failures))
                    _submit_one()
        if bar is not None:
            bar.close()
    else:
        bar = _progress(tasks, total=len(tasks), desc="Building graphs", unit="entry", dynamic_ncols=True)
        for task in bar:
            eid, entry_info, error = _build_and_save_graph(*task)
            _handle(eid, entry_info, error)
            if HAVE_TQDM and failures:
                bar.set_postfix(failures=len(failures))  # type: ignore[union-attr]

    # Write failures CSV
    if failures:
        failures_path = output_dir / "graph_build_failures.csv"
        with failures_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["entry_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        LOG.warning("%d graphs failed — see %s", len(failures), failures_path)

    # Write metadata
    kept_training_rows = [
        training_rows_by_entry[e["entry_id"]]
        for e in graph_entries
        if e["entry_id"] in training_rows_by_entry
    ]
    term_counts = build_term_counts(kept_training_rows)
    vocab_dir = metadata_dir / "vocabs"
    pipeline.ensure_parent(vocab_dir / "placeholder")
    for aspect in ASPECT_TO_LABEL_KEY:
        vocab = build_vocab(term_counts[aspect], min_term_frequency=args.min_term_frequency)
        write_json(
            vocab_dir / f"{aspect}.json",
            {"aspect": aspect, "min_term_frequency": args.min_term_frequency, "terms": vocab},
        )

    write_json(entries_json_path, graph_entries)
    write_json(metadata_dir / "term_counts.json", term_counts)
    write_json(
        metadata_dir / "schema.json",
        {
            "node_feature_dim": NODE_FEATURE_DIM,
            "base_feature_dim": BASE_FEATURE_DIM,
            "dssp_sasa_dim": DSSP_SASA_DIM,
            "esm2_dim": ESM2_DIM,
            "edge_attr_dim": EDGE_ATTR_DIM,
            "graph_feat_dim": GRAPH_FEAT_DIM,
            "modality_mask_dim": MODALITY_MASK_DIM,
            "graph_feat_names": [
                "residue_count",
                "fragment_count",
                "mean_plddt",
                "median_plddt",
                "frac_very_low",
                "frac_low",
                "frac_confident",
                "frac_very_high",
                "mean_contact_degree",
                "contact_density",
                "pae_mean",
                "pae_p90",
                "radius_of_gyration",
            ],
        },
    )

    return {
        "entries": len(graph_entries),
        "failures": len(failures),
        "output_dir": str(output_dir.resolve()),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_esm2_cache_dir(root: Path) -> Path:
    return root / "modality_cache" / "esm2"


def _default_structure_cache_dir(root: Path) -> Path:
    return root / "modality_cache" / "structure"


class _BaseCafaGraphDataset(TorchDataset):
    def __init__(
        self,
        root: str | Path,
        aspect: str,
        entry_ids: Iterable[str] | None = None,
        entry_id_file: str | Path | None = None,
        min_term_frequency: int = 1,
        use_esm2: bool = True,
        use_dssp: bool = True,
        use_sasa: bool = True,
        esm2_cache_dir: str | Path | None = None,
        structure_cache_dir: str | Path | None = None,
    ) -> None:
        require_torch()
        self.root = Path(root)
        self.aspect = aspect.upper()
        if self.aspect not in ASPECT_TO_LABEL_KEY:
            raise ValueError(f"Unknown aspect: {aspect}")

        self.min_term_frequency = max(int(min_term_frequency), 1)
        self.use_esm2 = use_esm2
        self.use_dssp = use_dssp
        self.use_sasa = use_sasa
        self.esm2_cache_dir = Path(esm2_cache_dir) if esm2_cache_dir else _default_esm2_cache_dir(self.root)
        self.structure_cache_dir = (
            Path(structure_cache_dir) if structure_cache_dir else _default_structure_cache_dir(self.root)
        )

        selected_entry_ids = set(str(value) for value in entry_ids or [])
        if entry_id_file:
            with Path(entry_id_file).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        selected_entry_ids.add(line)

        self.entries = load_json(self.root / "metadata" / "entries.json")
        self.term_counts = load_json(self.root / "metadata" / "term_counts.json")
        self.vocab = build_vocab(self.term_counts[self.aspect], min_term_frequency=self.min_term_frequency)
        self.term_to_index = {term: index for index, term in enumerate(self.vocab)}

        filtered_entries = []
        vocab_set = set(self.vocab)
        for entry in self.entries:
            entry_id = str(entry["entry_id"])
            if selected_entry_ids and entry_id not in selected_entry_ids:
                continue
            labels = set(entry["labels"].get(self.aspect, []))
            if not labels & vocab_set:
                continue
            filtered_entries.append(entry)
        self.entries = filtered_entries

    def __len__(self) -> int:
        return len(self.entries)

    def _encode_labels(self, labels: list[str]):
        torch_module = require_torch()
        encoded = torch_module.zeros(len(self.vocab), dtype=torch_module.float32)
        for term in labels:
            index = self.term_to_index.get(term)
            if index is not None:
                encoded[index] = 1.0
        return encoded

    def _load_payload(self, entry: dict[str, Any]) -> dict[str, Any]:
        torch_module = require_torch()
        payload = torch_module.load(entry["graph_path"], map_location="cpu", weights_only=False)
        payload["x"] = payload["x"].clone()
        payload["node_modality_mask"] = payload["node_modality_mask"].clone()
        self._apply_structure_cache(payload)
        self._apply_esm2_cache(payload)
        payload["y"] = self._encode_labels(payload["labels"][self.aspect])
        return payload

    def _apply_esm2_cache(self, payload: dict[str, Any]) -> None:
        if not self.use_esm2:
            return
        cache_path = self.esm2_cache_dir / f"{payload['entry_id']}.pt"
        if not cache_path.exists():
            return

        torch_module = require_torch()
        cache = torch_module.load(cache_path, map_location="cpu", weights_only=False)
        residue_embedding = cache.get("residue_embedding")
        cafa_indices = cache.get("cafa_residue_index")
        if residue_embedding is None or cafa_indices is None:
            return
        if residue_embedding.shape[1] != ESM2_DIM:
            raise ValueError(f"Expected ESM2 embeddings of width {ESM2_DIM}, got {residue_embedding.shape[1]}")

        cache_index = {int(index): row for row, index in enumerate(cafa_indices.tolist())}
        for row, cafa_index in enumerate(payload["cafa_residue_index"].tolist()):
            cache_row = cache_index.get(int(cafa_index))
            if cache_row is None:
                continue
            payload["x"][row, ESM2_SLICE] = residue_embedding[cache_row]
            payload["node_modality_mask"][row, 2] = True

    def _apply_structure_cache(self, payload: dict[str, Any]) -> None:
        if not (self.use_dssp or self.use_sasa):
            return
        torch_module = require_torch()
        residue_indices = payload["residue_index"].tolist()
        fragment_ids = payload["fragment_id"].tolist()
        fragment_names = payload["fragment_ids"]

        for fragment_offset, fragment_name in enumerate(fragment_names):
            cache_path = self.structure_cache_dir / f"{fragment_name}.pt"
            if not cache_path.exists():
                continue

            cache = torch_module.load(cache_path, map_location="cpu", weights_only=False)
            residue_index_tensor = cache.get("residue_index")
            features_tensor = cache.get("features")
            if residue_index_tensor is None or features_tensor is None:
                continue
            if features_tensor.shape[1] != DSSP_SASA_DIM:
                raise ValueError(
                    f"Expected DSSP/SASA feature width {DSSP_SASA_DIM}, got {features_tensor.shape[1]}"
                )

            dssp_mask = cache.get("dssp_mask")
            sasa_mask = cache.get("sasa_mask")
            lookup = {int(index): row for row, index in enumerate(residue_index_tensor.tolist())}
            for node_index, (fragment_id, residue_index) in enumerate(zip(fragment_ids, residue_indices)):
                if int(fragment_id) != fragment_offset:
                    continue
                cache_row = lookup.get(int(residue_index))
                if cache_row is None:
                    continue
                payload["x"][node_index, DSSP_SASA_SLICE] = features_tensor[cache_row]
                if self.use_dssp:
                    payload["node_modality_mask"][node_index, 0] = (
                        bool(dssp_mask[cache_row].item()) if dssp_mask is not None else True
                    )
                if self.use_sasa:
                    payload["node_modality_mask"][node_index, 1] = (
                        bool(sasa_mask[cache_row].item()) if sasa_mask is not None else True
                    )


class CafaPyGDataset(_BaseCafaGraphDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_pyg()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        DataClass = require_pyg()
        payload = self._load_payload(self.entries[index])
        return DataClass(
            x=payload["x"],
            pos=payload["pos"],
            edge_index=payload["edge_index"],
            edge_attr=payload["edge_attr"],
            y=payload["y"],
            graph_feat=payload["graph_feat"],
            cafa_residue_index=payload["cafa_residue_index"],
            residue_index=payload["residue_index"],
            fragment_id=payload["fragment_id"],
            node_modality_mask=payload["node_modality_mask"],
            entry_id=payload["entry_id"],
            taxonomy_id=payload["taxonomy_id"],
            fragment_ids=payload["fragment_ids"],
        )


class CafaDGLDataset(_BaseCafaGraphDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        require_dgl()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        dgl_module = require_dgl()
        payload = self._load_payload(self.entries[index])
        graph = dgl_module.graph(
            (payload["edge_index"][0], payload["edge_index"][1]),
            num_nodes=payload["x"].shape[0],
        )
        graph.ndata["x"] = payload["x"]
        graph.ndata["pos"] = payload["pos"]
        graph.ndata["cafa_residue_index"] = payload["cafa_residue_index"]
        graph.ndata["residue_index"] = payload["residue_index"]
        graph.ndata["fragment_id"] = payload["fragment_id"]
        graph.ndata["node_modality_mask"] = payload["node_modality_mask"].to(payload["x"].dtype)
        graph.edata["edge_attr"] = payload["edge_attr"]
        graph.graph_feat = payload["graph_feat"]
        graph.y = payload["y"]
        graph.entry_id = payload["entry_id"]
        graph.taxonomy_id = payload["taxonomy_id"]
        graph.fragment_ids = payload["fragment_ids"]
        return graph


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    summary = build_graph_cache(args)
    LOG.info(
        "Done — %d entries built, %d failures → %s",
        summary["entries"],
        summary["failures"],
        summary["output_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
