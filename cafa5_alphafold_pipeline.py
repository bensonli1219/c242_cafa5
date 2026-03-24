#!/usr/bin/env python3
"""
CAFA5 AlphaFold downloader and training-index builder.

This script reads CAFA5 training inputs, queries the AlphaFold Protein
Database (APD) by UniProt accession, downloads selected artifacts, and writes
two parquet manifests plus a CSV failure report.

Required runtime dependencies:
  - requests
  - pyarrow
Optional:
  - tqdm

Example:
  ./.venv/bin/python cafa5_alphafold_pipeline.py \
    --train-taxonomy data/kaggle_cafa5/extracted/Train/train_taxonomy.tsv \
    --train-sequences data/kaggle_cafa5/extracted/Train/train_sequences.fasta \
    --train-terms data/kaggle_cafa5/extracted/Train/train_terms.tsv \
    --output-dir ./outputs/cafa5_af_output \
    --limit 100 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised at runtime, not tests
    raise SystemExit(
        "pyarrow is required for parquet output. Install with: pip install pyarrow"
    ) from exc

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


LOG = logging.getLogger("cafa5_alphafold_pipeline")
ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"

RETRY_STRATEGY = Retry(
    total=4,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
)


@dataclass(frozen=True)
class ProteinRecord:
    order: int
    entry_id: str
    taxonomy_id: str
    sequence: str
    go_terms_bpo: list[str]
    go_terms_cco: list[str]
    go_terms_mfo: list[str]


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    overwrite: bool
    resume: bool
    request_delay: float
    workers: int


@dataclass(frozen=True)
class FetchResult:
    status: str
    records: list[dict[str, Any]]
    error: str | None = None
    http_status: int | None = None


@dataclass
class EntryProcessingResult:
    order: int
    training_row: dict[str, Any]
    fragment_rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]


class AlphaFoldClientProtocol(Protocol):
    def fetch_exact_metadata(self, entry_id: str) -> FetchResult:
        ...

    def download_to_path(self, url: str, dest_path: Path) -> None:
        ...


class RequestThrottle:
    """Global throttle shared across workers."""

    def __init__(self, request_delay: float) -> None:
        self.request_delay = max(request_delay, 0.0)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.request_delay <= 0:
            return

        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
            start_at = max(now, self._next_allowed)
            self._next_allowed = start_at + self.request_delay
        if sleep_for > 0:
            time.sleep(sleep_for)


class AlphaFoldClient:
    def __init__(self, request_delay: float = 0.5) -> None:
        self._thread_local = threading.local()
        self._throttle = RequestThrottle(request_delay=request_delay)

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._thread_local.session = session
        return session

    def fetch_exact_metadata(self, entry_id: str) -> FetchResult:
        accession = entry_id.strip().upper()
        url = f"{ALPHAFOLD_API_BASE}/{accession}"
        self._throttle.wait()
        try:
            response = self._session().get(url, timeout=30)
        except requests.RequestException as exc:
            return FetchResult(status="http_error", records=[], error=str(exc))

        if response.status_code == 404:
            return FetchResult(
                status="not_found",
                records=[],
                error=f"UniProt accession '{accession}' not found in AlphaFold DB.",
                http_status=404,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return FetchResult(
                status="http_error",
                records=[],
                error=str(exc),
                http_status=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return FetchResult(
                status="metadata_parse_error",
                records=[],
                error=f"Failed to decode AlphaFold metadata JSON: {exc}",
                http_status=response.status_code,
            )

        if not payload:
            return FetchResult(
                status="empty_response",
                records=[],
                error=f"Empty AlphaFold metadata response for '{accession}'.",
                http_status=response.status_code,
            )

        exact_records = filter_exact_metadata_records(accession, payload)
        if not exact_records:
            return FetchResult(
                status="filtered_empty",
                records=[],
                error=f"AlphaFold returned records for '{accession}', but none matched the exact accession.",
                http_status=response.status_code,
            )

        return FetchResult(status="ok", records=exact_records, http_status=response.status_code)

    def download_to_path(self, url: str, dest_path: Path) -> None:
        self._throttle.wait()
        response = self._session().get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()

        ensure_parent(dest_path)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        try:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            tmp_path.replace(dest_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AlphaFold data for CAFA5 training entries and build manifests."
    )
    parser.add_argument("--train-taxonomy", required=True, type=Path)
    parser.add_argument("--train-sequences", required=True, type=Path)
    parser.add_argument("--train-terms", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--taxonomy-ids",
        nargs="*",
        default=None,
        help="Optional taxonomy IDs. Accepts repeated values or comma-separated chunks.",
    )
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--request-delay", type=non_negative_float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_taxonomy_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed: set[str] = set()
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if chunk:
                parsed.add(chunk)
    return parsed or None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def should_skip_existing(path: Path, overwrite: bool, resume: bool) -> bool:
    if overwrite:
        return False
    if not path.exists():
        return False
    return path.stat().st_size > 0 and (resume or not overwrite)


def load_selected_taxonomy_records(
    path: Path,
    taxonomy_filter: set[str] | None = None,
    limit: int | None = None,
) -> list[tuple[int, str, str]]:
    selected: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for order, row in enumerate(reader):
            entry_id = row["EntryID"].strip()
            taxonomy_id = row["taxonomyID"].strip()
            if taxonomy_filter and taxonomy_id not in taxonomy_filter:
                continue
            selected.append((order, entry_id, taxonomy_id))
            if limit is not None and len(selected) >= limit:
                break
    return selected


def load_fasta_sequences(path: Path, selected_ids: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_seq
        if current_id and current_id in selected_ids:
            sequences[current_id] = "".join(current_seq)
        current_id = None
        current_seq = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                current_id = header.split()[0]
                continue
            if current_id is not None:
                current_seq.append(line)
    flush()
    return sequences


def load_go_terms(path: Path, selected_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entry_id = row["EntryID"].strip()
            if entry_id not in selected_ids:
                continue
            aspect = row["aspect"].strip()
            term = row["term"].strip()
            bucket = grouped.setdefault(entry_id, {"BPO": [], "CCO": [], "MFO": []})
            if aspect not in bucket:
                bucket[aspect] = []
            bucket[aspect].append(term)
    return grouped


def build_protein_records(
    taxonomy_path: Path,
    sequences_path: Path,
    terms_path: Path,
    taxonomy_filter: set[str] | None = None,
    limit: int | None = None,
) -> list[ProteinRecord]:
    selected = load_selected_taxonomy_records(
        taxonomy_path,
        taxonomy_filter=taxonomy_filter,
        limit=limit,
    )
    selected_ids = {entry_id for _, entry_id, _ in selected}
    sequences = load_fasta_sequences(sequences_path, selected_ids)
    terms = load_go_terms(terms_path, selected_ids)

    missing_sequences = [entry_id for _, entry_id, _ in selected if entry_id not in sequences]
    if missing_sequences:
        preview = ", ".join(missing_sequences[:5])
        raise ValueError(
            f"Missing sequences for {len(missing_sequences)} selected EntryID values. First few: {preview}"
        )

    records: list[ProteinRecord] = []
    for order, entry_id, taxonomy_id in selected:
        grouped = terms.get(entry_id, {"BPO": [], "CCO": [], "MFO": []})
        records.append(
            ProteinRecord(
                order=order,
                entry_id=entry_id,
                taxonomy_id=taxonomy_id,
                sequence=sequences[entry_id],
                go_terms_bpo=list(grouped.get("BPO", [])),
                go_terms_cco=list(grouped.get("CCO", [])),
                go_terms_mfo=list(grouped.get("MFO", [])),
            )
        )
    return records


def filter_exact_metadata_records(entry_id: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in payload
        if str(record.get("uniprotAccession", "")).strip().upper() == entry_id.upper()
    ]


def extract_pae_matrix(payload: Any) -> list[list[float]]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("PAE payload list is empty")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("PAE payload must be a dict or single-element list")

    matrix = payload.get("predicted_aligned_error") or payload.get("pae")
    if not matrix:
        raise ValueError("PAE payload is missing a predicted_aligned_error/pae matrix")
    if not isinstance(matrix, list):
        raise ValueError("PAE matrix must be a list")
    return matrix


def interpolate_quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    weight = position - lower_index
    return lower + (upper - lower) * weight


def summarize_pae_payload(payload: Any) -> dict[str, float]:
    matrix = extract_pae_matrix(payload)
    values = [float(value) for row in matrix for value in row]
    if not values:
        raise ValueError("PAE matrix is empty")

    values.sort()
    total = float(sum(values))
    count = len(values)
    return {
        "pae_mean": total / count,
        "pae_median": interpolate_quantile(values, 0.5),
        "pae_p90": interpolate_quantile(values, 0.9),
        "pae_max": values[-1],
    }


def summarize_pae_json_file(path: Path) -> dict[str, float]:
    return summarize_pae_payload(read_json(path))


def materialize_metadata(
    client: AlphaFoldClientProtocol,
    entry_id: str,
    metadata_path: Path,
    overwrite: bool,
    resume: bool,
) -> tuple[FetchResult, Path | None]:
    if should_skip_existing(metadata_path, overwrite=overwrite, resume=resume):
        try:
            cached = read_json(metadata_path)
            exact_records = filter_exact_metadata_records(entry_id, cached)
            if exact_records:
                return FetchResult(status="ok", records=exact_records), metadata_path.resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            LOG.warning("Failed to reuse cached metadata for %s; refetching.", entry_id)

    fetch_result = client.fetch_exact_metadata(entry_id)
    if fetch_result.status != "ok":
        return fetch_result, None

    write_json_atomic(metadata_path, fetch_result.records)
    return fetch_result, metadata_path.resolve()


def materialize_artifact(
    client: AlphaFoldClientProtocol,
    url: str | None,
    dest_path: Path,
    overwrite: bool,
    resume: bool,
) -> Path:
    if not url:
        raise ValueError(f"Missing AlphaFold artifact URL for {dest_path.name}")
    if should_skip_existing(dest_path, overwrite=overwrite, resume=resume):
        return dest_path.resolve()
    client.download_to_path(url, dest_path)
    return dest_path.resolve()


def fragment_length_from_metadata(record: dict[str, Any]) -> int:
    start = int(record.get("sequenceStart", 1) or 1)
    end = int(record.get("sequenceEnd", start) or start)
    return max(end - start + 1, 1)


def combine_errors(errors: Iterable[str]) -> str | None:
    unique = []
    seen: set[str] = set()
    for error in errors:
        if error and error not in seen:
            unique.append(error)
            seen.add(error)
    return " | ".join(unique) if unique else None


def weighted_average(rows: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        weight = float(row.get("fragment_length", 0) or 0)
        if weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def max_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values)


def make_failure_row(
    protein: ProteinRecord,
    status: str,
    stage: str,
    error: str,
    model_entity_id: str | None = None,
    http_status: int | None = None,
) -> dict[str, Any]:
    return {
        "entry_id": protein.entry_id,
        "taxonomy_id": protein.taxonomy_id,
        "model_entity_id": model_entity_id,
        "stage": stage,
        "status": status,
        "http_status": http_status,
        "error": error,
    }


def process_fragment(
    protein: ProteinRecord,
    fragment_record: dict[str, Any],
    metadata_path: Path,
    client: AlphaFoldClientProtocol,
    config: PipelineConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry_dir = config.output_dir / "raw" / protein.entry_id
    model_entity_id = fragment_record.get("modelEntityId") or fragment_record.get("entryId")
    fragment_length = fragment_length_from_metadata(fragment_record)

    fragment_row = {
        "entry_id": protein.entry_id,
        "taxonomy_id": protein.taxonomy_id,
        "model_entity_id": model_entity_id,
        "uniprot_accession": fragment_record.get("uniprotAccession"),
        "sequence_start": int(fragment_record.get("sequenceStart", 1) or 1),
        "sequence_end": int(fragment_record.get("sequenceEnd", 1) or 1),
        "fragment_length": fragment_length,
        "latest_version": int(fragment_record.get("latestVersion", 0) or 0),
        "model_created_date": fragment_record.get("modelCreatedDate"),
        "global_metric_value": float(fragment_record.get("globalMetricValue"))
        if fragment_record.get("globalMetricValue") is not None
        else None,
        "fraction_plddt_very_low": float(fragment_record.get("fractionPlddtVeryLow"))
        if fragment_record.get("fractionPlddtVeryLow") is not None
        else None,
        "fraction_plddt_low": float(fragment_record.get("fractionPlddtLow"))
        if fragment_record.get("fractionPlddtLow") is not None
        else None,
        "fraction_plddt_confident": float(fragment_record.get("fractionPlddtConfident"))
        if fragment_record.get("fractionPlddtConfident") is not None
        else None,
        "fraction_plddt_very_high": float(fragment_record.get("fractionPlddtVeryHigh"))
        if fragment_record.get("fractionPlddtVeryHigh") is not None
        else None,
        "pdb_path": None,
        "pae_path": None,
        "metadata_path": str(metadata_path),
        "pae_mean": None,
        "pae_median": None,
        "pae_p90": None,
        "pae_max": None,
        "all_versions": [
            int(version) for version in fragment_record.get("allVersions", []) if version is not None
        ],
        "fragment_status": "ok",
        "download_error": None,
    }

    failures: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        pdb_url = fragment_record.get("pdbUrl")
        pdb_filename = Path(str(pdb_url).split("/")[-1]) if pdb_url else Path(f"{model_entity_id}.pdb")
        fragment_row["pdb_path"] = str(
            materialize_artifact(
                client,
                pdb_url,
                entry_dir / pdb_filename,
                overwrite=config.overwrite,
                resume=config.resume,
            )
        )
    except Exception as exc:
        message = f"Failed to download PDB: {exc}"
        errors.append(message)
        failures.append(
            make_failure_row(
                protein,
                status="download_error",
                stage="download_pdb",
                error=message,
                model_entity_id=model_entity_id,
            )
        )

    try:
        pae_url = fragment_record.get("paeDocUrl")
        pae_filename = Path(str(pae_url).split("/")[-1]) if pae_url else Path(f"{model_entity_id}_pae.json")
        pae_path = materialize_artifact(
            client,
            pae_url,
            entry_dir / pae_filename,
            overwrite=config.overwrite,
            resume=config.resume,
        )
        fragment_row["pae_path"] = str(pae_path)
        pae_summary = summarize_pae_json_file(pae_path)
        fragment_row.update(pae_summary)
    except json.JSONDecodeError as exc:
        message = f"Failed to decode PAE JSON: {exc}"
        errors.append(message)
        failures.append(
            make_failure_row(
                protein,
                status="parse_error",
                stage="parse_pae",
                error=message,
                model_entity_id=model_entity_id,
            )
        )
    except Exception as exc:
        status = "parse_error" if fragment_row.get("pae_path") else "download_error"
        stage = "parse_pae" if fragment_row.get("pae_path") else "download_pae"
        message = f"Failed to process PAE: {exc}"
        errors.append(message)
        failures.append(
            make_failure_row(
                protein,
                status=status,
                stage=stage,
                error=message,
                model_entity_id=model_entity_id,
            )
        )

    if errors:
        fragment_row["fragment_status"] = (
            "parse_error" if any(failure["status"] == "parse_error" for failure in failures) else "download_error"
        )
        fragment_row["download_error"] = combine_errors(errors)

    return fragment_row, failures


def build_training_row(
    protein: ProteinRecord,
    metadata_path: Path | None,
    fragment_rows: list[dict[str, Any]],
    fetch_result: FetchResult,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_message = combine_errors(failure["error"] for failure in failures)
    model_versions = sorted(
        {
            version
            for row in fragment_rows
            for version in row.get("all_versions", [])
            if version is not None
        }
    )

    if fetch_result.status != "ok":
        af_status = fetch_result.status
    elif failures:
        af_status = "partial_error"
    else:
        af_status = "ok"

    successful_pdbs = [row["pdb_path"] for row in fragment_rows if row.get("pdb_path")]
    successful_paes = [row["pae_path"] for row in fragment_rows if row.get("pae_path")]

    return {
        "entry_id": protein.entry_id,
        "taxonomy_id": protein.taxonomy_id,
        "sequence": protein.sequence,
        "sequence_length": len(protein.sequence),
        "go_terms_bpo": protein.go_terms_bpo,
        "go_terms_cco": protein.go_terms_cco,
        "go_terms_mfo": protein.go_terms_mfo,
        "af_status": af_status,
        "af_found": bool(fragment_rows),
        "af_fragment_count": len(fragment_rows),
        "af_model_versions": model_versions,
        "af_model_entity_ids": [row["model_entity_id"] for row in fragment_rows],
        "af_pdb_paths": successful_pdbs,
        "af_pae_paths": successful_paes,
        "af_metadata_path": str(metadata_path) if metadata_path else None,
        "af_mean_plddt": weighted_average(fragment_rows, "global_metric_value"),
        "af_fraction_plddt_very_low": weighted_average(fragment_rows, "fraction_plddt_very_low"),
        "af_fraction_plddt_low": weighted_average(fragment_rows, "fraction_plddt_low"),
        "af_fraction_plddt_confident": weighted_average(fragment_rows, "fraction_plddt_confident"),
        "af_fraction_plddt_very_high": weighted_average(fragment_rows, "fraction_plddt_very_high"),
        "af_pae_mean": weighted_average(fragment_rows, "pae_mean"),
        "af_pae_median": weighted_average(fragment_rows, "pae_median"),
        "af_pae_p90": weighted_average(fragment_rows, "pae_p90"),
        "af_pae_max": max_value(fragment_rows, "pae_max"),
        "download_error": failure_message or fetch_result.error,
    }


def process_entry(
    protein: ProteinRecord,
    client: AlphaFoldClientProtocol,
    config: PipelineConfig,
) -> EntryProcessingResult:
    entry_dir = config.output_dir / "raw" / protein.entry_id
    metadata_path = entry_dir / f"{protein.entry_id}_alphafold_metadata.json"
    fragment_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    fetch_result, resolved_metadata_path = materialize_metadata(
        client,
        protein.entry_id,
        metadata_path=metadata_path,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    if fetch_result.status != "ok":
        failures.append(
            make_failure_row(
                protein,
                status=fetch_result.status,
                stage="metadata",
                error=fetch_result.error or fetch_result.status,
                http_status=fetch_result.http_status,
            )
        )
        training_row = build_training_row(
            protein,
            metadata_path=None,
            fragment_rows=[],
            fetch_result=fetch_result,
            failures=failures,
        )
        return EntryProcessingResult(
            order=protein.order,
            training_row=training_row,
            fragment_rows=[],
            failures=failures,
        )

    for fragment_record in fetch_result.records:
        fragment_row, fragment_failures = process_fragment(
            protein,
            fragment_record=fragment_record,
            metadata_path=resolved_metadata_path or metadata_path.resolve(),
            client=client,
            config=config,
        )
        fragment_rows.append(fragment_row)
        failures.extend(fragment_failures)

    training_row = build_training_row(
        protein,
        metadata_path=resolved_metadata_path,
        fragment_rows=fragment_rows,
        fetch_result=fetch_result,
        failures=failures,
    )
    return EntryProcessingResult(
        order=protein.order,
        training_row=training_row,
        fragment_rows=fragment_rows,
        failures=failures,
    )


def fragment_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("taxonomy_id", pa.string()),
            pa.field("model_entity_id", pa.string()),
            pa.field("uniprot_accession", pa.string()),
            pa.field("sequence_start", pa.int64()),
            pa.field("sequence_end", pa.int64()),
            pa.field("fragment_length", pa.int64()),
            pa.field("latest_version", pa.int64()),
            pa.field("model_created_date", pa.string()),
            pa.field("global_metric_value", pa.float64()),
            pa.field("fraction_plddt_very_low", pa.float64()),
            pa.field("fraction_plddt_low", pa.float64()),
            pa.field("fraction_plddt_confident", pa.float64()),
            pa.field("fraction_plddt_very_high", pa.float64()),
            pa.field("pdb_path", pa.string()),
            pa.field("pae_path", pa.string()),
            pa.field("metadata_path", pa.string()),
            pa.field("pae_mean", pa.float64()),
            pa.field("pae_median", pa.float64()),
            pa.field("pae_p90", pa.float64()),
            pa.field("pae_max", pa.float64()),
            pa.field("all_versions", pa.list_(pa.int64())),
            pa.field("fragment_status", pa.string()),
            pa.field("download_error", pa.string()),
        ]
    )


def training_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("taxonomy_id", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("sequence_length", pa.int64()),
            pa.field("go_terms_bpo", pa.list_(pa.string())),
            pa.field("go_terms_cco", pa.list_(pa.string())),
            pa.field("go_terms_mfo", pa.list_(pa.string())),
            pa.field("af_status", pa.string()),
            pa.field("af_found", pa.bool_()),
            pa.field("af_fragment_count", pa.int64()),
            pa.field("af_model_versions", pa.list_(pa.int64())),
            pa.field("af_model_entity_ids", pa.list_(pa.string())),
            pa.field("af_pdb_paths", pa.list_(pa.string())),
            pa.field("af_pae_paths", pa.list_(pa.string())),
            pa.field("af_metadata_path", pa.string()),
            pa.field("af_mean_plddt", pa.float64()),
            pa.field("af_fraction_plddt_very_low", pa.float64()),
            pa.field("af_fraction_plddt_low", pa.float64()),
            pa.field("af_fraction_plddt_confident", pa.float64()),
            pa.field("af_fraction_plddt_very_high", pa.float64()),
            pa.field("af_pae_mean", pa.float64()),
            pa.field("af_pae_median", pa.float64()),
            pa.field("af_pae_p90", pa.float64()),
            pa.field("af_pae_max", pa.float64()),
            pa.field("download_error", pa.string()),
        ]
    )


def normalise_rows_for_schema(rows: list[dict[str, Any]], schema: pa.Schema) -> list[dict[str, Any]]:
    field_names = [field.name for field in schema]
    normalised = []
    for row in rows:
        normalised.append({name: row.get(name) for name in field_names})
    return normalised


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    ensure_parent(path)
    table = pa.Table.from_pylist(normalise_rows_for_schema(rows, schema), schema=schema)
    pq.write_table(table, path)


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "entry_id",
        "taxonomy_id",
        "model_entity_id",
        "stage",
        "status",
        "http_status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def create_progress_bar(total: int):
    if not HAVE_TQDM:
        return None
    return tqdm(
        total=total,
        desc="Processing CAFA5 entries",
        unit="protein",
        dynamic_ncols=True,
        file=sys.stdout,
    )


def run_pipeline(
    args: argparse.Namespace,
    client: AlphaFoldClientProtocol | None = None,
) -> dict[str, Any]:
    taxonomy_filter = parse_taxonomy_filter(args.taxonomy_ids)
    output_dir = args.output_dir.resolve()
    config = PipelineConfig(
        output_dir=output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        request_delay=args.request_delay,
        workers=args.workers,
    )

    proteins = build_protein_records(
        taxonomy_path=args.train_taxonomy,
        sequences_path=args.train_sequences,
        terms_path=args.train_terms,
        taxonomy_filter=taxonomy_filter,
        limit=args.limit,
    )
    if not proteins:
        raise ValueError("No CAFA5 entries matched the provided filters.")

    LOG.info("Selected %d CAFA5 entries for processing.", len(proteins))
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    client = client or AlphaFoldClient(request_delay=args.request_delay)

    results: list[EntryProcessingResult] = []
    progress = create_progress_bar(len(proteins))
    status_counts = {
        "ok": 0,
        "partial_error": 0,
        "not_found": 0,
        "other": 0,
    }
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(process_entry, protein, client, config) for protein in proteins]
        completed = 0
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1

                status = result.training_row["af_status"]
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts["other"] += 1

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        ok=status_counts["ok"],
                        partial=status_counts["partial_error"],
                        missing=status_counts["not_found"],
                    )
                elif completed == len(futures) or completed % 25 == 0:
                    LOG.info(
                        "Progress %d/%d | ok=%d partial=%d missing=%d other=%d",
                        completed,
                        len(futures),
                        status_counts["ok"],
                        status_counts["partial_error"],
                        status_counts["not_found"],
                        status_counts["other"],
                    )
        finally:
            if progress is not None:
                progress.close()

    results.sort(key=lambda result: result.order)
    fragment_rows = [
        fragment_row
        for result in results
        for fragment_row in sorted(result.fragment_rows, key=lambda row: row["model_entity_id"] or "")
    ]
    training_rows = [result.training_row for result in results]
    failure_rows = [failure for result in results for failure in result.failures]

    manifests_dir = output_dir / "manifests"
    fragment_manifest = manifests_dir / "alphafold_fragments.parquet"
    training_manifest = manifests_dir / "training_index.parquet"
    failure_manifest = manifests_dir / "download_failures.csv"

    write_parquet(fragment_manifest, fragment_rows, fragment_schema())
    write_parquet(training_manifest, training_rows, training_schema())
    write_failures_csv(failure_manifest, failure_rows)

    summary = {
        "entries": len(training_rows),
        "found": sum(1 for row in training_rows if row["af_found"]),
        "ok": sum(1 for row in training_rows if row["af_status"] == "ok"),
        "partial_error": sum(1 for row in training_rows if row["af_status"] == "partial_error"),
        "not_found": sum(1 for row in training_rows if row["af_status"] == "not_found"),
        "failures": len(failure_rows),
        "fragment_rows": len(fragment_rows),
        "output_dir": str(output_dir),
    }
    LOG.info("Finished. Summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
