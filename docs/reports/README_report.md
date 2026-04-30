# Protein Function Prediction Training Report

Generated notebook:

- `../../notebooks/reports/protein_function_prediction_training_report.ipynb`
- Generated figures are written to `../../figures/`.

Run it from the repository root:

```bash
jupyter notebook notebooks/reports/protein_function_prediction_training_report.ipynb
```

The notebook uses repo-local summary tables under `output/`. For full training
artifacts and prediction-bundle analyses, set:

```bash
export CAFA5_ARTIFACT_ROOT=/global/scratch/users/bensonli/cafa5_outputs
```

On the Savio project environment, the notebook also checks the known scratch
artifact root. If full artifacts are unavailable, it still runs and marks the
missing sections explicitly instead of fabricating results.

The updated report also summarizes the uploaded historical progress PDFs in
`../progress_reports/` and keeps those results separate from the matched
structure-cohort experiment because the cohorts, label spaces, and metrics differ.
