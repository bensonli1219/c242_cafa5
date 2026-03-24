# CAFA5 AlphaFold Pipeline

This repo contains the CAFA5 AlphaFold download pipeline, supporting tests, and local data/output folders for the project.

## Layout

- `cafa5_alphafold_pipeline.py`: main CAFA5 -> AlphaFold downloader and manifest builder
- `alphafold_downloader.py`: smaller standalone AlphaFold downloader demo
- `tests/`: unit tests for the pipeline
- `docs/`: project documents
- `data/`: local datasets and downloaded structures
- `outputs/`: generated manifests and smoke/full pipeline runs

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

## Progress bar

The pipeline shows a running `tqdm` progress bar with `ok`, `partial`, and `missing` counts as entries complete.

## Tests

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
