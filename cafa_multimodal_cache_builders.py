#!/usr/bin/env python3
"""
Build optional multimodal caches that overlay onto the protein graph dataset.

Two cache types are supported:

  - ESM2 residue embeddings keyed by entry_id + cafa_residue_index
  - DSSP/SASA residue features keyed by model_entity_id + residue_index

The builders are designed for a remote machine with the required heavyweight
dependencies and binaries installed. The local graph dataset can load these
caches later without any API changes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import alphafold_feature_extractor as feature_extractor
import cafa5_alphafold_pipeline as pipeline
import cafa_graph_dataset as graphs

try:  # pragma: no cover - optional in the default py313 env
    import torch
except ImportError:  # pragma: no cover
    torch = None


DEFAULT_ESM2_MODEL = "facebook/esm2_t30_150M_UR50D"
DEFAULT_MAX_RESIDUES_PER_CHUNK = 1000
DEFAULT_CHUNK_OVERLAP = 128

SS3_ORDER = ("helix", "sheet", "coil")
HELIX_CODES = {"H", "G", "I"}
SHEET_CODES = {"E", "B"}

# Tien et al. / Wilke maximum ASA values, commonly used for RSA normalization.
MAX_ASA_BY_AA = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
    "X": 1.0,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def parse_entry_filter(values: list[str] | None) -> set[str] | None:
    return graphs.parse_entry_filter(values)


def parse_esm2_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-entry ESM2 residue embedding caches for the graph dataset."
    )
    parser.add_argument("--training-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_ESM2_MODEL,
        help=f"Hugging Face model name. Default: {DEFAULT_ESM2_MODEL}",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection. Default prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--entry-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-residues-per-chunk",
        type=positive_int,
        default=DEFAULT_MAX_RESIDUES_PER_CHUNK,
    )
    parser.add_argument(
        "--chunk-overlap",
        type=non_negative_int,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    return parser.parse_args(argv)


def parse_structure_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-fragment DSSP/SASA caches for the graph dataset."
    )
    parser.add_argument("--fragment-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--entry-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mkdssp-exe", default="mkdssp")
    parser.add_argument("--freesasa-exe", default="freesasa")
    return parser.parse_args(argv)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required for multimodal cache builders. "
            "Run them in the Python 3.11/remote environment."
        )
    return torch


def resolve_device(device_name: str):
    torch_module = require_torch()
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def iter_selected_training_rows(
    training_rows: list[dict[str, Any]],
    entry_filter: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in training_rows:
        entry_id = str(row.get("entry_id") or "").strip()
        if not entry_id:
            continue
        if entry_filter and entry_id not in entry_filter:
            continue
        if row.get("af_status") != "ok":
            continue
        if not row.get("sequence"):
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def iter_selected_fragment_rows(
    fragment_rows: list[dict[str, Any]],
    entry_filter: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in fragment_rows:
        entry_id = str(row.get("entry_id") or "").strip()
        if not entry_id:
            continue
        if entry_filter and entry_id not in entry_filter:
            continue
        if row.get("fragment_status") != "ok":
            continue
        if not row.get("pdb_path"):
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def chunk_sequence_windows(
    sequence_length: int,
    max_residues_per_chunk: int = DEFAULT_MAX_RESIDUES_PER_CHUNK,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    if sequence_length <= 0:
        return []
    if max_residues_per_chunk <= 0:
        raise ValueError("max_residues_per_chunk must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= max_residues_per_chunk:
        raise ValueError("chunk_overlap must be smaller than max_residues_per_chunk")

    windows: list[tuple[int, int]] = []
    start = 0
    while start < sequence_length:
        end = min(start + max_residues_per_chunk, sequence_length)
        windows.append((start, end))
        if end >= sequence_length:
            break
        start = end - chunk_overlap
    return windows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    pipeline.ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_transformers():
    try:  # pragma: no cover - optional dependency in the local env
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for the ESM2 cache builder. "
            "Install it in the remote multimodal environment."
        ) from exc
    return AutoTokenizer, AutoModel


def load_mdtraj():
    try:  # pragma: no cover - optional dependency
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mdtraj is required for the local structure-feature fallback when "
            "mkdssp/freesasa are unavailable."
        ) from exc
    return md


def _encode_sequence_chunks(
    sequence: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_residues_per_chunk: int,
    chunk_overlap: int,
) -> tuple[Any, Any]:
    torch_module = require_torch()
    windows = chunk_sequence_windows(
        len(sequence),
        max_residues_per_chunk=max_residues_per_chunk,
        chunk_overlap=chunk_overlap,
    )
    hidden_dim = int(getattr(model.config, "hidden_size", graphs.ESM2_DIM))
    residue_sum = torch_module.zeros((len(sequence), hidden_dim), dtype=torch_module.float32)
    residue_count = torch_module.zeros((len(sequence), 1), dtype=torch_module.float32)

    with torch_module.inference_mode():
        for start, end in windows:
            chunk_sequence = sequence[start:end]
            batch = tokenizer(chunk_sequence, return_tensors="pt", add_special_tokens=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            chunk_embedding = outputs.last_hidden_state[0, 1:-1, :].detach().to(
                device="cpu",
                dtype=torch_module.float32,
            )
            if chunk_embedding.shape[0] != end - start:
                raise ValueError(
                    f"Tokenized chunk length mismatch: expected {end - start}, got {chunk_embedding.shape[0]}"
                )
            residue_sum[start:end] += chunk_embedding
            residue_count[start:end] += 1.0

    residue_embedding = residue_sum / residue_count.clamp_min(1.0)
    protein_embedding = residue_embedding.mean(dim=0)
    return residue_embedding, protein_embedding


def build_esm2_cache(args: argparse.Namespace) -> dict[str, Any]:
    torch_module = require_torch()
    AutoTokenizer, AutoModel = load_transformers()
    entry_filter = parse_entry_filter(args.entry_ids)
    training_rows = read_parquet_rows(args.training_index)
    selected_rows = iter_selected_training_rows(training_rows, entry_filter=entry_filter, limit=args.limit)

    output_dir = args.output_dir
    pipeline.ensure_parent(output_dir / "placeholder")
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    if int(getattr(model.config, "hidden_size", 0)) != graphs.ESM2_DIM:
        raise ValueError(
            f"Model hidden size {getattr(model.config, 'hidden_size', None)} "
            f"does not match expected ESM2 width {graphs.ESM2_DIM}"
        )

    built = 0
    skipped_existing = 0
    failures: list[dict[str, str]] = []
    for row in selected_rows:
        entry_id = str(row["entry_id"])
        cache_path = output_dir / f"{entry_id}.pt"
        if args.resume and cache_path.exists():
            skipped_existing += 1
            continue

        try:
            residue_embedding, protein_embedding = _encode_sequence_chunks(
                sequence=str(row["sequence"]),
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_residues_per_chunk=int(args.max_residues_per_chunk),
                chunk_overlap=int(args.chunk_overlap),
            )
            payload = {
                "entry_id": entry_id,
                "model_name": args.model_name,
                "cafa_residue_index": torch_module.arange(
                    1,
                    residue_embedding.shape[0] + 1,
                    dtype=torch_module.long,
                ),
                "residue_embedding": residue_embedding,
                "protein_embedding": protein_embedding,
            }
            torch_module.save(payload, cache_path)
            built += 1
        except Exception as exc:
            failures.append({"entry_id": entry_id, "error": str(exc)})

    summary = {
        "entries_selected": len(selected_rows),
        "entries_built": built,
        "skipped_existing": skipped_existing,
        "failures": failures,
        "model_name": args.model_name,
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
    }
    write_summary(output_dir / "_builder_summary.json", summary)
    return summary


def _float_or_default(text: str, default: float = 0.0) -> float:
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError:
        return default


def _int_or_default(text: str, default: int = 0) -> int:
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return int(stripped)
    except ValueError:
        return default


def normalize_chain_id(value: str | None) -> str:
    stripped = (value or "").strip()
    return stripped or "A"


def parse_dssp_text(text: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    lines = text.splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith("  #  RESIDUE AA STRUCTURE"))
    except StopIteration:
        return records

    for line in lines[header_index + 1 :]:
        if len(line) < 115:
            continue
        aa_code = line[13].strip()
        if not aa_code or aa_code == "!":
            continue

        residue_number = _int_or_default(line[5:10], default=0)
        chain_id = normalize_chain_id(line[11:12])
        insertion_code = line[10].strip()
        secondary_code = line[16].strip()
        accessibility = _float_or_default(line[34:38])
        phi = _float_or_default(line[103:109])
        psi = _float_or_default(line[109:115])
        records[(chain_id, residue_number, insertion_code)] = {
            "aa": aa_code.upper(),
            "secondary_code": secondary_code,
            "accessibility": accessibility,
            "phi": phi,
            "psi": psi,
        }
    return records


def ss3_one_hot(secondary_code: str) -> list[float]:
    code = (secondary_code or "").strip().upper()
    if code in HELIX_CODES:
        return [1.0, 0.0, 0.0]
    if code in SHEET_CODES:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def build_dssp_feature_row(aa: str, secondary_code: str, accessibility: float, phi: float, psi: float) -> list[float]:
    max_asa = MAX_ASA_BY_AA.get((aa or "X").upper(), MAX_ASA_BY_AA["X"])
    phi_rad = math.radians(phi)
    psi_rad = math.radians(psi)
    return [
        *ss3_one_hot(secondary_code),
        math.sin(phi_rad),
        math.cos(phi_rad),
        math.sin(psi_rad),
        math.cos(psi_rad),
        float(accessibility),
        float(accessibility) / max_asa if max_asa > 0 else 0.0,
        0.0,
    ]


def parse_freesasa_residue_number(value: str) -> tuple[int, str]:
    stripped = (value or "").strip()
    digits = []
    suffix = []
    for character in stripped:
        if character.isdigit() or (character == "-" and not digits):
            digits.append(character)
        else:
            suffix.append(character)
    residue_number = int("".join(digits)) if digits else 0
    insertion_code = "".join(suffix).strip()
    return residue_number, insertion_code


def parse_freesasa_json(payload: dict[str, Any]) -> dict[tuple[str, int, str], float]:
    residues: dict[tuple[str, int, str], float] = {}
    results = payload.get("results")
    if not results:
        results = [{"structures": payload.get("structures", [])}]

    for result in results:
        for structure in result.get("structures", []):
            chains = structure.get("chains", [])
            if isinstance(chains, dict):
                chain_iterable = [{"label": key, **value} for key, value in chains.items()]
            else:
                chain_iterable = chains
            for chain in chain_iterable:
                chain_id = normalize_chain_id(chain.get("label") or chain.get("chain"))
                chain_residues = chain.get("residues", [])
                if isinstance(chain_residues, dict):
                    residue_iterable = []
                    for number, residue_value in chain_residues.items():
                        residue_iterable.append({"number": number, **residue_value})
                else:
                    residue_iterable = chain_residues
                for residue in residue_iterable:
                    residue_number, insertion_code = parse_freesasa_residue_number(
                        str(residue.get("number") or "")
                    )
                    area_value = residue.get("area") or {}
                    if isinstance(area_value, dict):
                        total_area = area_value.get("total")
                    else:
                        total_area = area_value
                    area = float(total_area or 0.0)
                    residues[(chain_id, residue_number, insertion_code)] = area
    return residues


def build_structure_cache_payload_mdtraj(
    fragment_row: dict[str, Any],
    residues: list[feature_extractor.ResidueRecord],
) -> dict[str, Any]:
    torch_module = require_torch()
    md = load_mdtraj()

    pdb_path = Path(str(fragment_row["pdb_path"]))
    traj = md.load_pdb(str(pdb_path))
    topology_residues = list(traj.topology.residues)
    if len(topology_residues) != len(residues):
        raise ValueError(
            f"mdtraj residue count mismatch for {pdb_path}: "
            f"{len(topology_residues)} != {len(residues)}"
        )

    features = [[0.0] * graphs.DSSP_SASA_DIM for _ in residues]
    dssp_mask = [False] * len(residues)
    sasa_mask = [False] * len(residues)

    ss3_codes = md.compute_dssp(traj, simplified=True)[0]
    residue_sasa = md.shrake_rupley(traj, mode="residue")[0] * 100.0

    phi_angles = {}
    phi_atom_indices, phi_values = md.compute_phi(traj)
    for atom_indices, angle_value in zip(phi_atom_indices, phi_values[0], strict=False):
        residue_index = traj.topology.atom(int(atom_indices[1])).residue.index
        phi_angles[int(residue_index)] = float(angle_value)

    psi_angles = {}
    psi_atom_indices, psi_values = md.compute_psi(traj)
    for atom_indices, angle_value in zip(psi_atom_indices, psi_values[0], strict=False):
        residue_index = traj.topology.atom(int(atom_indices[1])).residue.index
        psi_angles[int(residue_index)] = float(angle_value)

    for residue_offset, residue in enumerate(residues):
        aa = feature_extractor.AA3_TO_AA1.get(residue.residue_name3, "X")
        ss3 = ss3_one_hot(str(ss3_codes[residue_offset]))
        phi = float(phi_angles.get(residue_offset, 0.0))
        psi = float(psi_angles.get(residue_offset, 0.0))
        sasa_total = float(residue_sasa[residue_offset])
        max_asa = MAX_ASA_BY_AA.get(aa, MAX_ASA_BY_AA["X"])
        features[residue_offset] = [
            *ss3,
            math.sin(phi),
            math.cos(phi),
            math.sin(psi),
            math.cos(psi),
            sasa_total,
            sasa_total / max_asa if max_asa > 0 else 0.0,
            sasa_total,
        ]
        dssp_mask[residue_offset] = True
        sasa_mask[residue_offset] = True

    return {
        "entry_id": str(fragment_row.get("entry_id") or ""),
        "model_entity_id": str(fragment_row.get("model_entity_id") or ""),
        "feature_source": "mdtraj_fallback",
        "residue_index": torch_module.tensor(
            [residue.residue_index for residue in residues],
            dtype=torch_module.long,
        ),
        "features": torch_module.tensor(features, dtype=torch_module.float32),
        "dssp_mask": torch_module.tensor(dssp_mask, dtype=torch_module.bool),
        "sasa_mask": torch_module.tensor(sasa_mask, dtype=torch_module.bool),
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def run_mkdssp(pdb_path: Path, executable: str) -> str:
    attempts: list[tuple[list[str], Path | None]] = [
        ([executable, "--output-format", "dssp", str(pdb_path)], None),
        ([executable, str(pdb_path)], None),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "out.dssp"
        attempts.append(([executable, "-i", str(pdb_path), "-o", str(output_path)], output_path))
        for command, file_output in attempts:
            try:
                completed = run_command(command)
            except (OSError, subprocess.CalledProcessError):
                continue
            if file_output is not None and file_output.exists():
                return file_output.read_text(encoding="utf-8")
            if completed.stdout.strip():
                return completed.stdout
    raise RuntimeError(f"Failed to run mkdssp on {pdb_path}")


def run_freesasa(pdb_path: Path, executable: str) -> dict[str, Any]:
    command = [executable, "--format=json", "--output-depth=residue", str(pdb_path)]
    completed = run_command(command)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode FreeSASA JSON for {pdb_path}: {exc}") from exc


def build_structure_cache_payload(
    fragment_row: dict[str, Any],
    mkdssp_exe: str | None,
    freesasa_exe: str | None,
) -> dict[str, Any] | None:
    torch_module = require_torch()
    pdb_path = Path(str(fragment_row["pdb_path"]))
    residues = feature_extractor.parse_pdb_ca_residues(pdb_path)
    residue_lookup = {
        (normalize_chain_id(residue.chain_id), residue.residue_number, residue.insertion_code): index
        for index, residue in enumerate(residues)
    }

    features = [[0.0] * graphs.DSSP_SASA_DIM for _ in residues]
    dssp_mask = [False] * len(residues)
    sasa_mask = [False] * len(residues)

    if mkdssp_exe and shutil.which(mkdssp_exe):
        try:
            dssp_records = parse_dssp_text(run_mkdssp(pdb_path, mkdssp_exe))
            for residue_key, dssp_record in dssp_records.items():
                index = residue_lookup.get(residue_key)
                if index is None:
                    continue
                features[index][:9] = build_dssp_feature_row(
                    aa=dssp_record["aa"],
                    secondary_code=dssp_record["secondary_code"],
                    accessibility=float(dssp_record["accessibility"]),
                    phi=float(dssp_record["phi"]),
                    psi=float(dssp_record["psi"]),
                )[:9]
                dssp_mask[index] = True
        except Exception:
            pass

    if freesasa_exe and shutil.which(freesasa_exe):
        try:
            area_map = parse_freesasa_json(run_freesasa(pdb_path, freesasa_exe))
            for residue_key, total_area in area_map.items():
                index = residue_lookup.get(residue_key)
                if index is None:
                    continue
                features[index][9] = float(total_area)
                sasa_mask[index] = True
        except Exception:
            pass

    if not any(dssp_mask) and not any(sasa_mask):
        try:
            return build_structure_cache_payload_mdtraj(fragment_row=fragment_row, residues=residues)
        except Exception:
            return None

    payload = {
        "entry_id": str(fragment_row.get("entry_id") or ""),
        "model_entity_id": str(fragment_row.get("model_entity_id") or ""),
        "feature_source": "+".join(
            source
            for source, present in (("mkdssp", any(dssp_mask)), ("freesasa", any(sasa_mask)))
            if present
        )
        or "unknown",
        "residue_index": torch_module.tensor(
            [residue.residue_index for residue in residues],
            dtype=torch_module.long,
        ),
        "features": torch_module.tensor(features, dtype=torch_module.float32),
        "dssp_mask": torch_module.tensor(dssp_mask, dtype=torch_module.bool),
        "sasa_mask": torch_module.tensor(sasa_mask, dtype=torch_module.bool),
    }
    return payload


def build_structure_cache(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    entry_filter = parse_entry_filter(args.entry_ids)
    fragment_rows = read_parquet_rows(args.fragment_manifest)
    selected_rows = iter_selected_fragment_rows(fragment_rows, entry_filter=entry_filter, limit=args.limit)

    output_dir = args.output_dir
    pipeline.ensure_parent(output_dir / "placeholder")

    built = 0
    skipped_existing = 0
    skipped_missing_modalities = 0
    failures: list[dict[str, str]] = []
    for fragment_row in selected_rows:
        model_entity_id = str(fragment_row["model_entity_id"])
        cache_path = output_dir / f"{model_entity_id}.pt"
        if args.resume and cache_path.exists():
            skipped_existing += 1
            continue

        try:
            payload = build_structure_cache_payload(
                fragment_row=fragment_row,
                mkdssp_exe=args.mkdssp_exe,
                freesasa_exe=args.freesasa_exe,
            )
            if payload is None:
                skipped_missing_modalities += 1
                continue
            torch.save(payload, cache_path)
            built += 1
        except Exception as exc:
            failures.append({"model_entity_id": model_entity_id, "error": str(exc)})

    summary = {
        "fragments_selected": len(selected_rows),
        "fragments_built": built,
        "skipped_existing": skipped_existing,
        "skipped_missing_modalities": skipped_missing_modalities,
        "failures": failures,
        "output_dir": str(output_dir.resolve()),
        "mkdssp_exe": args.mkdssp_exe,
        "freesasa_exe": args.freesasa_exe,
    }
    write_summary(output_dir / "_builder_summary.json", summary)
    return summary


def main_esm2(argv: list[str] | None = None) -> int:
    args = parse_esm2_args(argv)
    summary = build_esm2_cache(args)
    print(
        f"Built {summary['entries_built']} ESM2 caches into {summary['output_dir']} "
        f"(selected={summary['entries_selected']}, skipped_existing={summary['skipped_existing']})"
    )
    return 0


def main_structure(argv: list[str] | None = None) -> int:
    args = parse_structure_args(argv)
    summary = build_structure_cache(args)
    print(
        f"Built {summary['fragments_built']} structure caches into {summary['output_dir']} "
        f"(selected={summary['fragments_selected']}, skipped_existing={summary['skipped_existing']}, "
        f"missing_modalities={summary['skipped_missing_modalities']})"
    )
    return 0
