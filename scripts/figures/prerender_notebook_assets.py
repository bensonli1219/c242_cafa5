#!/usr/bin/env python3
"""Pre-render every code-generated figure and DataFrame in the report
notebook so the eventual PDF export has them visible without executing
the cells.

Outputs:
  figures/section1_label_space_summary.png
  figures/section2_kmer_vs_esm2_bars.png
  figures/section2_sequence_loss_trajectories.png
  figures/section2_freq_filter_comparison.png
  output/notebook_inline_tables.json   (markdown-table strings keyed by section)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from Bio import SeqIO
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
DATA_BASE = Path(os.environ.get("CAFA5_DATA_DIR", Path.home() / "cafa5"))
TRAIN_DIR = DATA_BASE / "Train"


def md_table(df: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    """Produce a GFM Markdown table from a DataFrame (no tabulate dep)."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].map(lambda v: "" if pd.isna(v) else float_format.format(v))
    df = df.astype(str)
    cols = list(df.columns)
    rows = df.values.tolist()
    # column-aware width for nice rendering
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
    sep_align = ["-" * w for w in widths]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join(s for s in sep_align) + " |"
    body = "\n".join(
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(r, widths)) + " |"
        for r in rows
    )
    return "\n".join([header, sep, body])


# ---------- Load raw data ----------
print("Loading FASTA + train_terms ...")
sequences_dict: dict[str, str] = {}
for record in SeqIO.parse(TRAIN_DIR / "train_sequences.fasta", "fasta"):
    seq = str(record.seq).strip()
    if record.id and seq:
        sequences_dict[record.id] = seq
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
sequences_dict = {p: s for p, s in sequences_dict.items() if not (set(s) - valid_aas)}

terms = pd.read_csv(TRAIN_DIR / "train_terms.tsv", sep="\t").dropna(subset=["EntryID", "term"])
labels_dict = terms.groupby("EntryID")["term"].apply(list)

protein_ids: list[str] = []
sequences: list[str] = []
all_labels: list[list[str]] = []
for pid, seq in sequences_dict.items():
    if pid in labels_dict:
        protein_ids.append(pid)
        sequences.append(seq)
        all_labels.append(labels_dict[pid])

mlb = MultiLabelBinarizer(sparse_output=True)
Y = csr_matrix(mlb.fit_transform(all_labels).astype(np.int8))
print(f"  Y shape={Y.shape}, nnz={Y.nnz:,}")

# Splits (matches teammate convention)
indices = np.arange(len(sequences))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15 / 0.85, random_state=42)
Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]


# ============================================================
# FIGURE 1 — Section 1 label space summary
# ============================================================
print("\nFigure 1: label space summary")
labels_per_protein = np.asarray(Y.sum(axis=1)).ravel()
term_frequency = np.asarray(Y.sum(axis=0)).ravel()
go_to_aspect = (
    terms[["term", "aspect"]].drop_duplicates().set_index("term")["aspect"].to_dict()
)
aspect_of = np.array([go_to_aspect.get(t, "unk") for t in mlb.classes_])
FREQ_CUTOFF = 20
raw_counts = {a: int((aspect_of == a).sum()) for a in ("BPO", "CCO", "MFO")}
kept_counts = {
    a: int(((aspect_of == a) & (term_frequency >= FREQ_CUTOFF)).sum())
    for a in ("BPO", "CCO", "MFO")
}

fig, (ax_hist, ax_freq, ax_aspect) = plt.subplots(1, 3, figsize=(13, 3.8))
ax_hist.hist(labels_per_protein, bins=60, color="#3a6ea5", edgecolor="white", linewidth=0.4)
ax_hist.axvline(
    labels_per_protein.mean(),
    color="#c92a2a", ls="--", lw=1.2,
    label=f"mean = {labels_per_protein.mean():.1f}",
)
ax_hist.set_xlabel("GO labels per protein")
ax_hist.set_ylabel("Number of proteins")
ax_hist.set_title("A. Label cardinality per protein", fontsize=10, loc="left")
ax_hist.legend(framealpha=0.9)
ax_hist.grid(True, alpha=0.25)

ax_freq.hist(
    term_frequency,
    bins=np.logspace(0, np.log10(term_frequency.max() + 1), 60),
    color="#888", edgecolor="white", linewidth=0.3,
)
ax_freq.axvline(FREQ_CUTOFF, color="#c92a2a", ls="--", lw=1.4, label=f"freq = {FREQ_CUTOFF} cutoff")
ax_freq.set_xscale("log")
ax_freq.set_yscale("log")
ax_freq.set_xlabel("GO term training frequency")
ax_freq.set_ylabel("Number of GO terms")
n_below = int((term_frequency < FREQ_CUTOFF).sum())
n_above = int((term_frequency >= FREQ_CUTOFF).sum())
ax_freq.set_title(
    f"B. GO term frequency  ({n_below:,} below cutoff, {n_above:,} above)",
    fontsize=10, loc="left",
)
ax_freq.legend(framealpha=0.9)
ax_freq.grid(True, alpha=0.25, which="both")

aspects = ["BPO", "CCO", "MFO"]
x = np.arange(len(aspects))
raw_vals = [raw_counts[a] for a in aspects]
kept_vals = [kept_counts[a] for a in aspects]
ax_aspect.bar(
    x - 0.18, raw_vals, width=0.36, color="#cccccc",
    edgecolor="#333", linewidth=0.5, label="raw vocabulary",
)
ax_aspect.bar(
    x + 0.18, kept_vals, width=0.36, color="#3a6ea5",
    edgecolor="#333", linewidth=0.5, label=f"kept (freq >= {FREQ_CUTOFF})",
)
for xi, r, k in zip(x, raw_vals, kept_vals):
    ax_aspect.text(xi - 0.18, r + 200, f"{r:,}", ha="center", fontsize=8)
    ax_aspect.text(xi + 0.18, k + 200, f"{k:,}", ha="center", fontsize=8)
ax_aspect.set_xticks(x)
ax_aspect.set_xticklabels(aspects)
ax_aspect.set_ylabel("Number of GO terms")
ax_aspect.set_title("C. Per-aspect vocabulary, raw vs filtered", fontsize=10, loc="left")
ax_aspect.legend(framealpha=0.9)
ax_aspect.grid(True, axis="y", alpha=0.25)

fig.suptitle("Figure 1. The CAFA5 label space at a glance", fontsize=11, y=1.02)
plt.tight_layout()
out = FIG_DIR / "section1_label_space_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out}")


# ============================================================
# Load JSON metric files (for the next 3 figures + summary tables)
# ============================================================
kmer_full = json.loads(Path("baselines/kmer/kmer_mlp_full_go_metrics.json").read_text())
kmer_tuned = json.loads(Path("baselines/kmer/kmer_tuning_summary.json").read_text())
esm_full = json.loads(Path("baselines/esm/esm_mlp_full_go_metrics.json").read_text())
esm_tuned = json.loads(Path("baselines/esm/esm_tuning_summary.json").read_text())
seq_per_ns = pd.read_csv("output/sequence_baseline_per_namespace.csv")


# ============================================================
# FIGURE 2 — K-mer vs ESM2 per-namespace bar chart
# ============================================================
print("\nFigure 2: K-mer vs ESM2 per-namespace")
ns_pivot = (
    seq_per_ns.pivot(index="namespace", columns="feature", values="ia_fmax")
    .reindex(["BPO", "CCO", "MFO"])
)
fig, ax = plt.subplots(figsize=(6, 4))
xx = np.arange(len(ns_pivot))
ax.bar(xx - 0.18, ns_pivot["K-mer"], width=0.36, label="K-mer (tuned)", color="#888")
ax.bar(xx + 0.18, ns_pivot["ESM2"], width=0.36, label="ESM2 (tuned)", color="#3a6ea5")
for xi, k, e in zip(xx, ns_pivot["K-mer"], ns_pivot["ESM2"]):
    ax.text(xi - 0.18, k + 0.01, f"{k:.3f}", ha="center", fontsize=9)
    ax.text(xi + 0.18, e + 0.01, f"{e:.3f}", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(ns_pivot.index)
ax.set_ylabel("IA-weighted Fmax")
ax.set_ylim(0, 0.32)
ax.set_title("Sequence baselines: K-mer vs ESM2 (Fmax by GO namespace)")
ax.legend(loc="upper left")
plt.tight_layout()
out = FIG_DIR / "section2_kmer_vs_esm2_bars.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out}")


# ============================================================
# FIGURE 3 — Sequence baseline loss trajectories
# ============================================================
print("\nFigure 3: K-mer / ESM2 loss trajectories")
fig, ax = plt.subplots(figsize=(6, 4))
epochs = list(range(1, 6))
ax.plot(epochs, kmer_full["train_losses"], "o-", color="#888", label="K-mer train")
ax.plot(epochs, kmer_full["val_losses"], "s--", color="#888", label="K-mer val")
ax.plot(epochs, esm_full["train_losses"], "o-", color="#3a6ea5", label="ESM2 train")
ax.plot(epochs, esm_full["val_losses"], "s--", color="#3a6ea5", label="ESM2 val")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE loss")
ax.set_title("Sequence baselines: training trajectories (full GO, 31,454 labels)")
ax.legend()
plt.tight_layout()
out = FIG_DIR / "section2_sequence_loss_trajectories.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out}")


# ============================================================
# FIGURE 4 — Section 2.3 freq filter before/after comparison
# ============================================================
print("\nFigure 4: freq filter before/after")
# Recompute the inputs the cell uses (keep_idx, removed_idx, bpo_idx, etc.)
Y_full = scipy.sparse.vstack([Y_train, Y_val, Y_test])
label_counts = np.asarray(Y_full.sum(axis=0)).ravel().astype(np.float32)
keep_idx = np.where(label_counts >= FREQ_CUTOFF)[0]
removed_idx = np.where(label_counts < FREQ_CUTOFF)[0]
go_terms_filtered = [mlb.classes_[i] for i in keep_idx]
bpo_idx = [i for i, g in enumerate(go_terms_filtered) if go_to_aspect.get(g) == "BPO"]
cco_idx = [i for i, g in enumerate(go_terms_filtered) if go_to_aspect.get(g) == "CCO"]
mfo_idx = [i for i, g in enumerate(go_terms_filtered) if go_to_aspect.get(g) == "MFO"]

fig, (ax_vocab, ax_metric) = plt.subplots(
    2, 1, figsize=(9, 6.5),
    gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.55},
)
total_terms = int(len(keep_idx) + len(removed_idx))
ax_vocab.barh(["Before filter"], [total_terms], color="#888", edgecolor="#333", linewidth=0.5)
ax_vocab.text(total_terms, 0, f" all {total_terms:,} GO terms", va="center", ha="left", fontsize=9)
left = 0
for name, count, color in [
    ("BPO kept", len(bpo_idx), "#3a6ea5"),
    ("CCO kept", len(cco_idx), "#2b8a3e"),
    ("MFO kept", len(mfo_idx), "#c92a2a"),
    ("removed (freq<20)", int(len(removed_idx)), "#dddddd"),
]:
    ax_vocab.barh(
        ["After filter"], [count], left=[left],
        color=color, edgecolor="#333", linewidth=0.5,
        label=f"{name}: {count:,}",
    )
    left += count
ax_vocab.set_xlim(0, total_terms)
ax_vocab.set_xlabel("Number of GO terms in label vocabulary")
ax_vocab.set_title("Vocabulary effect of the freq>=20 filter", fontsize=11, loc="left")
ax_vocab.legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol=4, fontsize=8.5, frameon=False)

groups = [
    "No filter\n(micro F1, 31,454 GO)",
    "Freq>=20 + BPO\n(IA-Fmax)",
    "Freq>=20 + CCO\n(IA-Fmax)",
    "Freq>=20 + MFO\n(IA-Fmax)",
]
kmer_vals = [
    kmer_tuned["final_best_micro"]["micro_f1"],
    float(ns_pivot.loc["BPO", "K-mer"]),
    float(ns_pivot.loc["CCO", "K-mer"]),
    float(ns_pivot.loc["MFO", "K-mer"]),
]
esm_vals = [
    esm_tuned["final_best_micro"]["micro_f1"],
    float(ns_pivot.loc["BPO", "ESM2"]),
    float(ns_pivot.loc["CCO", "ESM2"]),
    float(ns_pivot.loc["MFO", "ESM2"]),
]
xx = np.arange(len(groups))
ax_metric.bar(
    xx - 0.18, kmer_vals, width=0.36, color="#888", edgecolor="#333", linewidth=0.5,
    label="K-mer (tuned)",
)
ax_metric.bar(
    xx + 0.18, esm_vals, width=0.36, color="#3a6ea5", edgecolor="#333", linewidth=0.5,
    label="ESM2 (tuned)",
)
for xi, k, e in zip(xx, kmer_vals, esm_vals):
    ax_metric.text(xi - 0.18, k + 0.006, f"{k:.3f}", ha="center", fontsize=9)
    ax_metric.text(xi + 0.18, e + 0.006, f"{e:.3f}", ha="center", fontsize=9)
ax_metric.axvline(0.5, color="#333", ls="--", lw=0.8, alpha=0.5)
ax_metric.text(
    0.5, ax_metric.get_ylim()[1] * 0.92,
    "  filter applied  ",
    ha="left", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.25", fc="#fff8e7", ec="#999", lw=0.4),
)
ax_metric.set_xticks(xx)
ax_metric.set_xticklabels(groups, fontsize=8.5)
ax_metric.set_ylabel("Score (different metric on each side)")
ax_metric.set_ylim(0, max(max(kmer_vals), max(esm_vals)) * 1.2)
ax_metric.set_title(
    "Sequence baseline metrics: full GO (no filter) vs filtered + namespace",
    fontsize=11, loc="left",
)
ax_metric.legend(loc="upper right", framealpha=0.9)
ax_metric.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
out = FIG_DIR / "section2_freq_filter_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {out}")


# ============================================================
# DataFrame -> markdown tables
# ============================================================
print("\nDataFrames -> markdown tables")
tables: dict[str, str] = {}

# 1. summary (cell 26): K-mer / ESM2 metric comparison
summary_df = pd.DataFrame([
    {
        "model": "K-mer (full GO, 5ep)",
        "val_loss": kmer_full["best_val_loss"],
        "micro_f1": kmer_full["best_micro"]["micro_f1"],
        "thr": kmer_full["best_micro"]["threshold"],
    },
    {
        "model": "K-mer (tuned)",
        "val_loss": kmer_tuned["final_best_val_loss"],
        "micro_f1": kmer_tuned["final_best_micro"]["micro_f1"],
        "thr": kmer_tuned["final_best_micro"]["threshold"],
    },
    {
        "model": "ESM2 (full GO, 5ep)",
        "val_loss": esm_full["best_val_loss"],
        "micro_f1": esm_full["best_micro"]["micro_f1"],
        "thr": esm_full["best_micro"]["threshold"],
    },
    {
        "model": "ESM2 (tuned)",
        "val_loss": esm_tuned["final_best_val_loss"],
        "micro_f1": esm_tuned["final_best_micro"]["micro_f1"],
        "thr": esm_tuned["final_best_micro"]["threshold"],
    },
])
tables["summary"] = md_table(summary_df, float_format="{:.4f}")

# 2. splits (cell 70)
audit = pd.read_csv(
    "output/preprocessing_audit_metadata_quick2/min_frequency_scenarios.csv"
)
splits_df = (
    audit[audit["min_term_frequency"] == 20]
    .loc[:, ["aspect", "eligible_entries", "vocab_size",
             "train_count", "val_count", "test_count"]]
    .rename(columns={
        "eligible_entries": "entries", "vocab_size": "vocab",
        "train_count": "train", "val_count": "val", "test_count": "test",
    })
    .reset_index(drop=True)
)
tables["splits"] = md_table(splits_df, float_format="{:.0f}")

# 3. structure_results (cell 90) + comparison (cell 91)
graph_best = pd.read_csv(
    "output/jupyter-notebook/report_assets/graph_training_best_results.csv"
)
raw_baseline = (
    graph_best[(graph_best["model_change"] == "Raw graph baseline")
               & graph_best["aspect"].isin(["CCO", "MFO"])]
    .loc[:, ["aspect", "best_val_fmax", "best_test_fmax"]]
    .assign(model="Graph baseline (raw)")
    .rename(columns={"best_val_fmax": "val_fmax", "best_test_fmax": "test_fmax"})
)
esm2_per_ns = (
    seq_per_ns[(seq_per_ns["feature"] == "ESM2")
               & seq_per_ns["namespace"].isin(["CCO", "MFO"])]
    .loc[:, ["namespace", "ia_fmax"]]
    .rename(columns={"namespace": "aspect", "ia_fmax": "test_fmax"})
    .assign(model="ESM2 tuned (sequence)", val_fmax=pd.NA)
)
structure_results_df = (
    pd.concat([esm2_per_ns, raw_baseline], ignore_index=True)
    .loc[:, ["model", "aspect", "test_fmax"]]
    .sort_values(["aspect", "model"])
    .reset_index(drop=True)
)
tables["structure_results"] = md_table(structure_results_df, float_format="{:.4f}")

comparison_df = (
    structure_results_df.pivot(index="aspect", columns="model", values="test_fmax")
)
comparison_df["relative_gain"] = (
    comparison_df["Graph baseline (raw)"] / comparison_df["ESM2 tuned (sequence)"]
)
comparison_with_index = comparison_df.reset_index()
tables["comparison"] = md_table(comparison_with_index, float_format="{:.3f}")

# 4. tuning (cell 96)
TUNING_LABEL = {
    "Raw graph baseline": "Raw baseline",
    "Baseline rerun": "Control rerun",
    "Lower learning rate, 0.0007": "Lower lr (7e-4)",
    "Lower learning rate, 0.0005": "Lower lr (5e-4)",
    "Moderate hidden size": "Wider hidden (192)",
    "Larger hidden size": "Wider hidden (256)",
    "Weighted BCE": "Weighted BCE",
    "Broad tuned recipe": "Combined tuned bundle",
}
tuning_df = (
    graph_best[graph_best["model_change"].isin(TUNING_LABEL)
               & graph_best["aspect"].isin(["CCO", "MFO"])]
    .pivot_table(index="model_change", columns="aspect",
                 values="best_test_fmax", aggfunc="first")
    .reindex(list(TUNING_LABEL.keys()))
    .rename(columns={"CCO": "cco_test_fmax", "MFO": "mfo_test_fmax"})
    .reset_index()
)
tuning_df["run"] = tuning_df["model_change"].map(TUNING_LABEL)
raw_cco = float(tuning_df.loc[tuning_df["run"] == "Raw baseline", "cco_test_fmax"].iloc[0])
raw_mfo = float(tuning_df.loc[tuning_df["run"] == "Raw baseline", "mfo_test_fmax"].iloc[0])
tuning_df["delta_cco"] = (tuning_df["cco_test_fmax"] - raw_cco).round(4)
tuning_df["delta_mfo"] = (tuning_df["mfo_test_fmax"] - raw_mfo).round(4)
tuning_view = tuning_df[["run", "cco_test_fmax", "mfo_test_fmax", "delta_cco", "delta_mfo"]]
tables["tuning"] = md_table(tuning_view, float_format="{:.4f}")

# 5. label_aware (cell 103)
LABEL_AWARE_RENAME = {
    "Raw graph baseline": "Raw baseline",
    "Label-aware scorer": "Label-aware scorer",
}
la_df = (
    graph_best[graph_best["model_change"].isin(LABEL_AWARE_RENAME)
               & graph_best["aspect"].isin(["CCO", "MFO"])]
    .loc[:, ["model_change", "aspect", "best_val_fmax", "best_test_fmax"]]
    .rename(columns={
        "model_change": "run", "best_val_fmax": "val", "best_test_fmax": "test",
    })
    .replace({"run": LABEL_AWARE_RENAME})
    .reset_index(drop=True)
)
raw = {
    a: float(la_df[(la_df["run"] == "Raw baseline") & (la_df["aspect"] == a)]["test"].iloc[0])
    for a in ("CCO", "MFO")
}
la_df["delta_test"] = la_df.apply(
    lambda r: round(r["test"] - raw[r["aspect"]], 4), axis=1
)
tables["label_aware"] = md_table(la_df, float_format="{:.4f}")

# 6. fusion (cell 108)
fusion_df = pd.read_csv("output/fusion_n3_mfo_results.csv")
graph_only_val = float(
    fusion_df.loc[fusion_df["setting"].str.startswith("Graph only"), "val_fmax"].iloc[0]
)
graph_only_test = float(
    fusion_df.loc[fusion_df["setting"].str.startswith("Graph only"), "test_fmax"].iloc[0]
)
fusion_df["delta_val_vs_graph"] = (fusion_df["val_fmax"] - graph_only_val).round(4)
fusion_df["delta_test_vs_graph"] = (fusion_df["test_fmax"] - graph_only_test).round(4)
tables["fusion"] = md_table(fusion_df, float_format="{:.4f}")

# Save everything
TABLES_OUT = REPO / "output" / "notebook_inline_tables.json"
TABLES_OUT.write_text(json.dumps(tables, indent=2), encoding="utf-8")
print(f"  -> {TABLES_OUT}  ({len(tables)} tables)")

print("\nDone.")
