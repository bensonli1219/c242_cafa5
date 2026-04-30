#!/usr/bin/env python3
"""Render presentation-ready training result figures, in English.

One figure per data point cited in the script - no interpretive text on the
figures themselves; just clean axes, bar values, and deltas.

Outputs:
  figures/result_cco_per_ontology.png        (CCO graph experiments)
  figures/result_mfo_per_ontology.png        (MFO graph experiments)
  figures/result_matched_mfo_fmax.png        (matched MFO controlled Fmax)
  figures/result_matched_mfo_auprc.png       (matched MFO micro AUPRC)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


COLOR_RAW          = "#7f7f7f"
COLOR_WEIGHTED_BCE = "#9b59b6"
COLOR_LABEL_AWARE  = "#d62728"
COLOR_SEQUENCE     = "#1f77b4"
COLOR_STRUCTURE    = "#e67e22"
COLOR_FUSION       = "#2ca02c"


def _annotate_top(ax, bars, values, *, fmt="{:.4f}", fontsize=12, dy=0.0004):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            fmt.format(v),
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        )


def _annotate_delta(ax, bars, values, baseline_index, *, fontsize=10, dy=0.0004):
    baseline = values[baseline_index]
    for i, (bar, v) in enumerate(zip(bars, values)):
        if i == baseline_index:
            continue
        delta = (v - baseline) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy + 0.0008,
            f"({delta:+.2f} pp)",
            ha="center", va="bottom",
            fontsize=fontsize, color="#555", fontstyle="italic",
        )


def _polish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)


def make_cco(out_path: Path) -> None:
    labels = ["Raw graph", "Weighted BCE", "Label-aware scorer"]
    values = [0.5647, 0.5656, 0.5875]
    colors = [COLOR_RAW, COLOR_WEIGHTED_BCE, COLOR_LABEL_AWARE]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.4, width=0.55)
    ax.set_ylim(0.555, 0.595)
    ax.set_ylabel("Test Fmax", fontsize=12)
    ax.set_title("CCO  (Cellular Component)  -  per-ontology graph experiments",
                 fontsize=13, fontweight="bold")
    _annotate_top(ax, bars, values, dy=0.0006)
    _annotate_delta(ax, bars, values, baseline_index=0, dy=0.0010)
    _polish(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_mfo(out_path: Path) -> None:
    labels = ["Raw graph", "Weighted BCE", "Label-aware scorer"]
    values = [0.4574, 0.4605, 0.4575]
    colors = [COLOR_RAW, COLOR_WEIGHTED_BCE, COLOR_LABEL_AWARE]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.4, width=0.55)
    ax.set_ylim(0.453, 0.464)
    ax.set_ylabel("Test Fmax", fontsize=12)
    ax.set_title("MFO  (Molecular Function)  -  per-ontology graph experiments",
                 fontsize=13, fontweight="bold")
    _annotate_top(ax, bars, values, dy=0.0003)
    _annotate_delta(ax, bars, values, baseline_index=0, dy=0.0006)
    _polish(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_matched_mfo_fmax(out_path: Path) -> None:
    labels = ["Sequence-only\nESM2 MLP", "Structure GNN\n(graph)", "Late fusion\n(seq + structure)"]
    values = [0.2776, 0.4580, 0.4579]
    colors = [COLOR_SEQUENCE, COLOR_STRUCTURE, COLOR_FUSION]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.4, width=0.55)
    ax.set_ylim(0.25, 0.50)
    ax.set_ylabel("Test Fmax", fontsize=12)
    ax.set_title("MFO best training result  -  Test Fmax\n"
                 "same protein cohort, train / val / test split, MFO label vocabulary",
                 fontsize=12, fontweight="bold")
    _annotate_top(ax, bars, values, dy=0.002)
    _polish(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_matched_mfo_auprc(out_path: Path) -> None:
    labels = ["Sequence-only\nESM2 MLP", "Structure GNN\n(graph)"]
    values = [0.359, 0.367]
    colors = [COLOR_SEQUENCE, COLOR_STRUCTURE]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.4, width=0.50)
    ax.set_ylim(0.350, 0.376)
    ax.set_ylabel("Micro AUPRC", fontsize=12)
    ax.set_title("Matched MFO controlled comparison  -  Micro AUPRC",
                 fontsize=13, fontweight="bold")
    _annotate_top(ax, bars, values, fmt="{:.3f}", dy=0.0008)
    _annotate_delta(ax, bars, values, baseline_index=0, dy=0.0014)
    _polish(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_cco(out_dir / "result_cco_per_ontology.png")
    make_mfo(out_dir / "result_mfo_per_ontology.png")
    make_matched_mfo_fmax(out_dir / "result_matched_mfo_fmax.png")
    make_matched_mfo_auprc(out_dir / "result_matched_mfo_auprc.png")
