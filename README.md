# CAFA5 AlphaFold Pipeline

This repo contains the CAFA5 AlphaFold download pipeline, supporting tests, and local data/output folders for the project.

## Layout

- `cafa5_alphafold_pipeline.py`: main CAFA5 -> AlphaFold downloader and manifest builder
- `alphafold_feature_extractor.py`: convert downloaded PDB/PAE files into residue, contact-graph, and fragment-level parquet features
- `cafa_graph_dataset.py`: build protein-level graph caches and load them as PyG/DGL datasets
- `cafa_graph_dataloaders.py`: deterministic split export plus training-ready PyG/DGL dataloader builders
- `build_cafa_graph_cache.py`: CLI wrapper for graph cache materialization
- `export_graph_dataloaders.py`: CLI wrapper that writes split manifests and verifies loaders
- `benchmark_graph_dataloaders.py`: benchmark dataset/dataloader stages and report throughput plus system utilization
- `cafa_multimodal_cache_builders.py`: remote-oriented ESM2 and DSSP/SASA cache builders
- `build_esm2_cache.py`: CLI wrapper for per-entry ESM2 residue embeddings
- `build_structure_cache.py`: CLI wrapper for per-fragment DSSP/SASA caches
- `alphafold_downloader.py`: smaller standalone AlphaFold downloader demo
- `tests/`: unit tests for the pipeline
- `docs/planning/`: experiment plans and implementation checklists
- `docs/reports/`: final reports and experiment summaries
- `docs/presentations/`: presentation deck and speaking scripts
- `docs/progress_reports/`: historical checkpoint PDFs
- `notebooks/reports/`: report notebooks
- `figures/`: report and presentation figures
- `data/`: local datasets and downloaded structures
- `output/` and `outputs/`: generated manifests, caches, notebooks, and smoke/full pipeline runs

## Main command

```bash
./.venv/bin/python cafa5_alphafold_pipeline.py \
  --train-taxonomy data/kaggle_cafa5/extracted/Train/train_taxonomy.tsv \
  --train-sequences data/kaggle_cafa5/extracted/Train/train_sequences.fasta \
  --train-terms data/kaggle_cafa5/extracted/Train/train_terms.tsv \
  --output-dir outputs/cafa5_af_full \
  --workers 4 \
  --request-delay 0.5 \
  --resume
```

For a local 100-entry sample:

```bash
./.venv/bin/python cafa5_alphafold_pipeline.py \
  --train-taxonomy data/kaggle_cafa5/extracted/Train/train_taxonomy.tsv \
  --train-sequences data/kaggle_cafa5/extracted/Train/train_sequences.fasta \
  --train-terms data/kaggle_cafa5/extracted/Train/train_terms.tsv \
  --output-dir outputs/cafa5_af_100 \
  --limit 100 \
  --workers 4 \
  --request-delay 0.5 \
  --resume
```

## Progress bar

The pipeline shows a running `tqdm` progress bar with `ok`, `partial`, and `missing` counts as entries complete.

## Feature extraction

```bash
./.venv/bin/python alphafold_feature_extractor.py \
  --training-index outputs/cafa5_af_smoke/manifests/training_index.parquet \
  --fragment-manifest outputs/cafa5_af_smoke/manifests/alphafold_fragments.parquet \
  --output-dir outputs/cafa5_af_smoke/features
```

This writes:

- `fragment_features.parquet`: per-AlphaFold-fragment summary statistics
- `residue_features.parquet`: per-residue coordinates, pLDDT bins, contact degree, and PAE row summaries
- `contact_graph_edges.parquet`: residue-residue edges under the selected C-alpha distance threshold

## Graph Environment

Create a separate Python 3.11 environment for the local graph MVP:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv311
./.venv311/bin/pip install -r requirements-graph-local.txt
```

This keeps the graph stack (`torch`, `torch_geometric`, `dgl`) isolated from the existing Python 3.13 environment.

## Graph Cache Builder

```bash
./.venv311/bin/python build_cafa_graph_cache.py \
  --training-index outputs/cafa5_af_smoke/manifests/training_index.parquet \
  --fragment-features outputs/cafa5_af_smoke/features/fragment_features.parquet \
  --residue-features outputs/cafa5_af_smoke/features/residue_features.parquet \
  --edge-features outputs/cafa5_af_smoke/features/contact_graph_edges.parquet \
  --output-dir outputs/cafa5_af_smoke/graph_cache
```

This writes per-protein graph payloads under `graphs/` and metadata/vocab files under `metadata/`.

## Training Dataloaders

Export deterministic train/val/test splits and verify the PyG / DGL loaders:

```bash
./.venv311/bin/python export_graph_dataloaders.py \
  --root outputs/cafa5_af_100/graph_cache \
  --aspects BPO CCO MFO \
  --batch-size 8
```

This writes split manifests under `graph_cache/splits/<aspect>/` and an `export_summary.json` file with one-batch verification for both frameworks.

## Benchmark

Benchmark one split end-to-end for the selected frameworks:

```bash
./.venv311/bin/python benchmark_graph_dataloaders.py \
  --root outputs/cafa5_af_100/graph_cache \
  --frameworks pyg dgl \
  --aspects BPO CCO MFO \
  --split train \
  --batch-size 8 \
  --device cpu
```

On a remote NVIDIA server, switch to `--device cuda`. CPU metrics come from `psutil`; GPU metrics use `pynvml` when installed and otherwise fall back to `nvidia-smi`.

## Minimal Training Loop

Run a minimal end-to-end training loop on one aspect:

```bash
./.venv311/bin/python train_minimal_graph_model.py \
  --root outputs/cafa5_af_100/graph_cache \
  --framework pyg \
  --aspect MFO \
  --epochs 3 \
  --batch-size 8 \
  --device cpu
```

This writes checkpoints and a per-epoch JSON summary under `graph_cache/training_runs/`.

## Remote Multimodal Cache Builders

Install the remote-only Python dependencies on the training server:

```bash
python -m venv .venv_remote
./.venv_remote/bin/pip install -r requirements-remote-multimodal.txt
```

Then build ESM2 caches directly into the graph cache tree:

```bash
./.venv_remote/bin/python build_esm2_cache.py \
  --training-index outputs/cafa5_af_100/manifests/training_index.parquet \
  --output-dir outputs/cafa5_af_100/graph_cache/modality_cache/esm2 \
  --resume
```

And build DSSP/SASA caches for each AlphaFold fragment:

```bash
./.venv_remote/bin/python build_structure_cache.py \
  --fragment-manifest outputs/cafa5_af_100/manifests/alphafold_fragments.parquet \
  --output-dir outputs/cafa5_af_100/graph_cache/modality_cache/structure \
  --resume
```

`build_structure_cache.py` expects `mkdssp` and `freesasa` to be installed on the remote machine and available on `PATH`. When a cache file is missing, `CafaPyGDataset` and `CafaDGLDataset` keep the reserved feature slots zero-filled and leave the modality mask unset.

On a local machine without those binaries, `build_structure_cache.py` can still run if `mdtraj` is installed. In that case it falls back to `mdtraj`-based secondary-structure, phi/psi, and residue SASA features while keeping the same output schema.

## Tests

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Repository Hygiene

Generated graph caches, training checkpoints, run logs, notebook checkpoints, and local datasets are ignored by Git. Keep reproducible code, tests, reports, and lightweight figures in the repository; regenerate large experiment artifacts from the scripts when needed.
