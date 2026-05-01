#!/usr/bin/env python3
"""
Extract lightweight structure features from downloaded AlphaFold artifacts.

This script converts downloaded AlphaFold PDB/PAE files into three parquet
tables that are easier to feed into downstream ML pipelines:

  - fragment_features.parquet: one row per AlphaFold fragment
  - residue_features.parquet: one row per residue / graph node
  - contact_graph_edges.parquet: one row per residue-residue edge

The features intentionally avoid heavyweight dependencies such as numpy,
Biopython, or DSSP so they can run in the existing project environment.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

import cafa5_alphafold_pipeline as pipeline

LOG = logging.getLogger(__name__)

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "UNK": "X",
}


@dataclass(frozen=True)
class ResidueRecord:
    residue_index: int
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name3: str
    residue_name1: str
    x: float
    y: float
    z: float
    plddt: float


class ParquetBatchWriter:
    def __init__(self, path: Path, batch_size: int = 5000) -> None:
        self.path = path
        self.batch_size = max(batch_size, 1)
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    def extend_from_table(self, table: pa.Table) -> None:
        """Write an Arrow table directly without converting to Python dicts."""
        if table.num_rows == 0:
            return
        pipeline.ensure_parent(self.path)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        for batch in table.to_batches(max_chunksize=self.batch_size * 10):
            self.writer.write_batch(batch)

    def flush(self) -> None:
        if not self.rows:
            return

        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            pipeline.ensure_parent(self.path)
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract residue and graph features from downloaded AlphaFold artifacts."
    )
    parser.add_argument("--training-index", required=True, type=Path)
    parser.add_argument("--fragment-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--entry-ids",
        nargs="*",
        default=None,
        help="Optional UniProt/CAFA entry IDs to extract. Accepts repeated values or comma-separated chunks.",
    )
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--contact-threshold", type=positive_float, default=10.0)
    parser.add_argument("--strict-contact-threshold", type=positive_float, default=8.0)
    parser.add_argument("--batch-size", type=positive_int, default=5000)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of parallel worker processes (default: 1 = serial).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed fragments based on checkpoint file in output-dir.",
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


def load_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def parse_pdb_ca_residues(path: Path) -> list[ResidueRecord]:
    residues: list[ResidueRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        residue_name3 = line[17:20].strip().upper() or "UNK"
        chain_id = line[21].strip() or "A"
        residue_number = int(line[22:26].strip())
        insertion_code = line[26].strip()
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        plddt = float(line[60:66].strip())

        residues.append(
            ResidueRecord(
                residue_index=len(residues) + 1,
                chain_id=chain_id,
                residue_number=residue_number,
                insertion_code=insertion_code,
                residue_name3=residue_name3,
                residue_name1=AA3_TO_AA1.get(residue_name3, "X"),
                x=x,
                y=y,
                z=z,
                plddt=plddt,
            )
        )

    if not residues:
        raise ValueError(f"No CA atoms found in PDB file: {path}")
    return residues


def classify_plddt(plddt: float) -> str:
    if plddt < 50:
        return "very_low"
    if plddt < 70:
        return "low"
    if plddt < 90:
        return "confident"
    return "very_high"


def average(values: Iterable[float]) -> float | None:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        return None
    return total / count


def build_fragment_index(
    fragment_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in fragment_rows:
        entry_id = str(row.get("entry_id", ""))
        model_entity_id = str(row.get("model_entity_id", ""))
        if entry_id and model_entity_id:
            index[(entry_id, model_entity_id)] = row
    return index


def infer_cafa_positions(
    training_sequence: str,
    fragment_row: dict[str, Any],
    residues: list[ResidueRecord],
) -> tuple[list[int | None], str]:
    residue_sequence = "".join(residue.residue_name1 for residue in residues)
    sequence_start = int(fragment_row.get("sequence_start", 1) or 1)
    sequence_end = int(fragment_row.get("sequence_end", sequence_start + len(residues) - 1) or 0)
    expected = training_sequence[sequence_start - 1 : sequence_end]

    if residue_sequence == expected:
        return list(range(sequence_start, sequence_end + 1)), "exact_fragment_match"

    if residue_sequence == training_sequence:
        return list(range(1, len(training_sequence) + 1)), "exact_full_match"

    start_index = training_sequence.find(residue_sequence)
    if start_index >= 0:
        start_pos = start_index + 1
        end_pos = start_pos + len(residues) - 1
        return list(range(start_pos, end_pos + 1)), "substring_match"

    if len(residues) == max(sequence_end - sequence_start + 1, 0):
        return list(range(sequence_start, sequence_end + 1)), "range_length_match"

    return [None] * len(residues), "unresolved"


def row_statistics(row: list[float]) -> tuple[float, float, float]:
    values = sorted(float(value) for value in row)
    return (
        sum(values) / len(values),
        values[0],
        pipeline.interpolate_quantile(values, 0.9),
    )


def radius_of_gyration(residues: list[ResidueRecord]) -> float:
    center_x = sum(residue.x for residue in residues) / len(residues)
    center_y = sum(residue.y for residue in residues) / len(residues)
    center_z = sum(residue.z for residue in residues) / len(residues)

    squared = 0.0
    for residue in residues:
        dx = residue.x - center_x
        dy = residue.y - center_y
        dz = residue.z - center_z
        squared += dx * dx + dy * dy + dz * dz
    return math.sqrt(squared / len(residues))


def summarize_fragment_features(
    entry_id: str,
    taxonomy_id: str,
    model_entity_id: str,
    residues: list[ResidueRecord],
    cafa_positions: list[int | None],
    alignment_status: str,
    pae_matrix: list[list[float]],
    contact_counts: list[int],
    strict_contact_counts: list[int],
    edge_count: int,
    strict_edge_count: int,
    contact_threshold: float,
    strict_contact_threshold: float,
) -> dict[str, Any]:
    residue_count = len(residues)
    possible_pairs = residue_count * (residue_count - 1) / 2
    plddt_values = [residue.plddt for residue in residues]
    categories = [classify_plddt(value) for value in plddt_values]
    resolved_positions = [value for value in cafa_positions if value is not None]
    coverage_end = max(resolved_positions) if resolved_positions else None
    coverage_start = min(resolved_positions) if resolved_positions else None

    pae_values = [float(value) for row in pae_matrix for value in row]
    pae_values_sorted = sorted(pae_values)

    return {
        "entry_id": entry_id,
        "taxonomy_id": taxonomy_id,
        "model_entity_id": model_entity_id,
        "residue_count": residue_count,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "alignment_status": alignment_status,
        "mean_plddt": average(plddt_values),
        "median_plddt": pipeline.interpolate_quantile(sorted(plddt_values), 0.5),
        "fraction_plddt_very_low": categories.count("very_low") / residue_count,
        "fraction_plddt_low": categories.count("low") / residue_count,
        "fraction_plddt_confident": categories.count("confident") / residue_count,
        "fraction_plddt_very_high": categories.count("very_high") / residue_count,
        "mean_contact_degree": average(contact_counts),
        "max_contact_degree": max(contact_counts) if contact_counts else 0,
        "mean_strict_contact_degree": average(strict_contact_counts),
        "contact_edge_count": edge_count,
        "strict_contact_edge_count": strict_edge_count,
        "contact_density": (edge_count / possible_pairs) if possible_pairs else 0.0,
        "strict_contact_density": (strict_edge_count / possible_pairs) if possible_pairs else 0.0,
        "contact_threshold": contact_threshold,
        "strict_contact_threshold": strict_contact_threshold,
        "pae_mean": sum(pae_values) / len(pae_values),
        "pae_median": pipeline.interpolate_quantile(pae_values_sorted, 0.5),
        "pae_p90": pipeline.interpolate_quantile(pae_values_sorted, 0.9),
        "pae_max": pae_values_sorted[-1],
        "radius_of_gyration": radius_of_gyration(residues),
    }


def extract_fragment_features(
    training_row: dict[str, Any],
    fragment_row: dict[str, Any],
    contact_threshold: float,
    strict_contact_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entry_id = str(fragment_row["entry_id"])
    model_entity_id = str(fragment_row["model_entity_id"])
    residues = parse_pdb_ca_residues(Path(fragment_row["pdb_path"]))
    pae_matrix = pipeline.extract_pae_matrix(pipeline.read_json(Path(fragment_row["pae_path"])))

    if len(pae_matrix) != len(residues):
        raise ValueError(
            f"PAE matrix shape mismatch for {model_entity_id}: {len(pae_matrix)} rows vs {len(residues)} residues."
        )
    if any(len(row) != len(residues) for row in pae_matrix):
        raise ValueError(f"PAE matrix is not square for {model_entity_id}.")

    cafa_positions, alignment_status = infer_cafa_positions(
        training_sequence=str(training_row["sequence"]),
        fragment_row=fragment_row,
        residues=residues,
    )

    residue_row_stats = [row_statistics(row) for row in pae_matrix]
    contact_counts = [0] * len(residues)
    strict_contact_counts = [0] * len(residues)
    edge_rows: list[dict[str, Any]] = []
    edge_count = 0
    strict_edge_count = 0

    for left_index in range(len(residues)):
        left = residues[left_index]
        for right_index in range(left_index + 1, len(residues)):
            right = residues[right_index]
            dx = left.x - right.x
            dy = left.y - right.y
            dz = left.z - right.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance > contact_threshold:
                continue

            seq_separation = right.residue_index - left.residue_index
            pae_lr = float(pae_matrix[left_index][right_index])
            pae_rl = float(pae_matrix[right_index][left_index])
            pae_mean_pair = (pae_lr + pae_rl) / 2.0

            edge_rows.append(
                {
                    "entry_id": entry_id,
                    "taxonomy_id": str(fragment_row["taxonomy_id"]),
                    "model_entity_id": model_entity_id,
                    "source_residue_index": left.residue_index,
                    "target_residue_index": right.residue_index,
                    "source_cafa_residue_index": cafa_positions[left_index],
                    "target_cafa_residue_index": cafa_positions[right_index],
                    "distance_ca": distance,
                    "seq_separation": seq_separation,
                    "pae_lr": pae_lr,
                    "pae_rl": pae_rl,
                    "pae_mean_pair": pae_mean_pair,
                    "is_sequential_neighbor": seq_separation == 1,
                    "is_short_range_sequence": seq_separation <= 5,
                    "is_strict_contact": distance <= strict_contact_threshold,
                }
            )

            contact_counts[left_index] += 1
            contact_counts[right_index] += 1
            edge_count += 1
            if distance <= strict_contact_threshold:
                strict_contact_counts[left_index] += 1
                strict_contact_counts[right_index] += 1
                strict_edge_count += 1

    residue_rows: list[dict[str, Any]] = []
    total_residues = len(residues)
    for index, residue in enumerate(residues):
        pae_mean, pae_min, pae_p90 = residue_row_stats[index]
        plddt_bin = classify_plddt(residue.plddt)
        residue_rows.append(
            {
                "entry_id": entry_id,
                "taxonomy_id": str(fragment_row["taxonomy_id"]),
                "model_entity_id": model_entity_id,
                "residue_index": residue.residue_index,
                "cafa_residue_index": cafa_positions[index],
                "chain_id": residue.chain_id,
                "residue_number": residue.residue_number,
                "insertion_code": residue.insertion_code,
                "residue_name3": residue.residue_name3,
                "residue_name1": residue.residue_name1,
                "x": residue.x,
                "y": residue.y,
                "z": residue.z,
                "plddt": residue.plddt,
                "plddt_bin": plddt_bin,
                "is_plddt_very_low": plddt_bin == "very_low",
                "is_plddt_low": plddt_bin == "low",
                "is_plddt_confident": plddt_bin == "confident",
                "is_plddt_very_high": plddt_bin == "very_high",
                "pae_row_mean": pae_mean,
                "pae_row_min": pae_min,
                "pae_row_p90": pae_p90,
                "contact_degree": contact_counts[index],
                "strict_contact_degree": strict_contact_counts[index],
                "sequence_position_fraction": residue.residue_index / total_residues,
                "alignment_status": alignment_status,
            }
        )

    fragment_features = summarize_fragment_features(
        entry_id=entry_id,
        taxonomy_id=str(fragment_row["taxonomy_id"]),
        model_entity_id=model_entity_id,
        residues=residues,
        cafa_positions=cafa_positions,
        alignment_status=alignment_status,
        pae_matrix=pae_matrix,
        contact_counts=contact_counts,
        strict_contact_counts=strict_contact_counts,
        edge_count=edge_count,
        strict_edge_count=strict_edge_count,
        contact_threshold=contact_threshold,
        strict_contact_threshold=strict_contact_threshold,
    )
    return residue_rows, edge_rows, fragment_features


# ---------------------------------------------------------------------------
# Top-level worker function (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _extract_task(
    entry_id: str,
    model_entity_id: str,
    training_row: dict[str, Any],
    fragment_row: dict[str, Any],
    contact_threshold: float,
    strict_contact_threshold: float,
) -> tuple[str, str, list[dict[str, Any]] | None, list[dict[str, Any]] | None, dict[str, Any] | None, str | None]:
    """Thin wrapper around extract_fragment_features that catches all exceptions."""
    try:
        residue_rows, edge_rows, frag_feat = extract_fragment_features(
            training_row=training_row,
            fragment_row=fragment_row,
            contact_threshold=contact_threshold,
            strict_contact_threshold=strict_contact_threshold,
        )
        return entry_id, model_entity_id, residue_rows, edge_rows, frag_feat, None
    except Exception as exc:  # noqa: BLE001
        return entry_id, model_entity_id, None, None, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> set[str]:
    """Return set of 'entry_id\\tmodel_entity_id' strings that are already done."""
    if not path.exists():
        return set()
    return set(path.read_text(encoding="utf-8").splitlines())


def _append_checkpoint(path: Path, entry_id: str, model_entity_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{entry_id}\t{model_entity_id}\n")


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    training_rows = load_rows(args.training_index)
    fragment_rows = load_rows(args.fragment_manifest)
    entry_filter = parse_entry_filter(args.entry_ids)

    training_by_entry = {str(row["entry_id"]): row for row in training_rows}
    fragment_index = build_fragment_index(fragment_rows)

    # Build full task list (respects --limit on entries and --entry-ids filter)
    tasks: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    seen_entries: set[str] = set()
    for entry_id, training_row in training_by_entry.items():
        if entry_filter and entry_id not in entry_filter:
            continue
        if training_row.get("af_status") != "ok":
            continue
        model_entity_ids = list(training_row.get("af_model_entity_ids") or [])
        if not model_entity_ids:
            continue
        for model_entity_id in model_entity_ids:
            fragment_row = fragment_index.get((entry_id, model_entity_id))
            if fragment_row is None:
                LOG.warning("Missing fragment row for entry=%s model=%s — skipping", entry_id, model_entity_id)
                continue
            tasks.append((entry_id, model_entity_id, training_row, fragment_row))
        seen_entries.add(entry_id)
        if args.limit is not None and len(seen_entries) >= args.limit:
            break

    LOG.info("Task list built: %d fragments across %d entries", len(tasks), len(seen_entries))

    # Resume: filter out already-completed fragments
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / ".checkpoint"
    done_keys = _load_checkpoint(checkpoint_path) if args.resume else set()
    if done_keys:
        LOG.info("Resume mode: %d fragments already done, skipping them", len(done_keys))
    pending = [t for t in tasks if f"{t[0]}\t{t[1]}" not in done_keys]
    LOG.info("Pending: %d fragments to process", len(pending))

    if not pending:
        LOG.info("Nothing to do.")
        return {"entries": 0, "fragments": 0, "failures": 0, "output_dir": str(args.output_dir.resolve())}

    # Setup parquet writers
    residue_path   = args.output_dir / "residue_features.parquet"
    edge_path      = args.output_dir / "contact_graph_edges.parquet"
    fragment_path  = args.output_dir / "fragment_features.parquet"

    residue_writer  = ParquetBatchWriter(residue_path,  args.batch_size)
    edge_writer     = ParquetBatchWriter(edge_path,     max(args.batch_size // 2, 1))
    fragment_writer = ParquetBatchWriter(fragment_path, max(args.batch_size // 4, 1))

    # Resume: replay existing parquet rows into writers so they are preserved
    if args.resume:
        parquet_ok = True
        for path, writer, label in [
            (residue_path,  residue_writer,  "residue"),
            (edge_path,     edge_writer,     "edge"),
            (fragment_path, fragment_writer, "fragment"),
        ]:
            if not path.exists():
                continue
            try:
                existing_table = pq.read_table(path)
                writer.extend_from_table(existing_table)
                LOG.info("Resume: loaded %d existing %s rows", existing_table.num_rows, label)
                del existing_table
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "Existing %s parquet is corrupted (%s). "
                    "Clearing checkpoint — affected fragments will be reprocessed.",
                    label, exc,
                )
                parquet_ok = False

        if not parquet_ok and checkpoint_path.exists():
            checkpoint_path.unlink()
            done_keys = set()
            pending = tasks
            LOG.warning("Checkpoint cleared. All %d fragments will be reprocessed.", len(pending))

    failures: list[dict[str, str]] = []
    processed_fragments = 0
    processed_entries: set[str] = set()

    def _progress(iterable: Any, **kwargs: Any) -> Any:
        if HAVE_TQDM:
            return tqdm(iterable, **kwargs)
        return iterable

    def _handle_result(
        entry_id: str,
        model_entity_id: str,
        residue_rows: list[dict[str, Any]] | None,
        edge_rows: list[dict[str, Any]] | None,
        frag_feat: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        nonlocal processed_fragments
        if error is not None:
            LOG.warning("FAILED %s / %s: %s", entry_id, model_entity_id, error)
            failures.append({"entry_id": entry_id, "model_entity_id": model_entity_id, "error": error})
            return
        residue_writer.extend(residue_rows)  # type: ignore[arg-type]
        edge_writer.extend(edge_rows)  # type: ignore[arg-type]
        fragment_writer.add(frag_feat)  # type: ignore[arg-type]
        processed_fragments += 1
        processed_entries.add(entry_id)
        if args.resume:
            _append_checkpoint(checkpoint_path, entry_id, model_entity_id)

    try:
        if args.workers > 1:
            max_inflight = args.workers * 32
            LOG.info("Parallel mode: %d workers, max %d tasks in-flight", args.workers, max_inflight)
            pending_iter = iter(pending)
            in_flight: dict = {}
            bar = tqdm(total=len(pending), desc="Extracting", unit="frag", dynamic_ncols=True) if HAVE_TQDM else None

            def _submit_one() -> None:
                task = next(pending_iter, None)
                if task is None:
                    return
                eid, mid, tr, fr = task
                f = executor.submit(
                    _extract_task, eid, mid, tr, fr,
                    args.contact_threshold, args.strict_contact_threshold,
                )
                in_flight[f] = (eid, mid)

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for _ in range(min(max_inflight, len(pending))):
                    _submit_one()
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        in_flight.pop(future)
                        eid, mid, res_rows, edge_rows, frag_feat, error = future.result()
                        _handle_result(eid, mid, res_rows, edge_rows, frag_feat, error)
                        if bar is not None:
                            bar.update(1)
                            if failures:
                                bar.set_postfix(failures=len(failures))
                        _submit_one()
            if bar is not None:
                bar.close()
        else:
            bar = _progress(
                pending,
                total=len(pending),
                desc="Extracting",
                unit="frag",
                dynamic_ncols=True,
            )
            for eid, mid, tr, fr in bar:
                result = _extract_task(eid, mid, tr, fr, args.contact_threshold, args.strict_contact_threshold)
                _handle_result(*result)
                if HAVE_TQDM and failures:
                    bar.set_postfix(failures=len(failures))  # type: ignore[union-attr]
    finally:
        residue_writer.close()
        edge_writer.close()
        fragment_writer.close()

    # Write failures CSV
    if failures:
        failures_path = args.output_dir / "extraction_failures.csv"
        with failures_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["entry_id", "model_entity_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        LOG.warning("%d fragments failed — see %s", len(failures), failures_path)

    return {
        "entries": len(processed_entries),
        "fragments": processed_fragments,
        "failures": len(failures),
        "output_dir": str(args.output_dir.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    summary = run_extraction(args)
    LOG.info(
        "Done — %d entries / %d fragments extracted, %d failures → %s",
        summary["entries"],
        summary["fragments"],
        summary["failures"],
        summary["output_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
