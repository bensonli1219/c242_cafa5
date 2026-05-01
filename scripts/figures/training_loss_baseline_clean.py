#!/usr/bin/env python3
"""Clean redesign of figures/training_loss_curves_baseline.png.

Reads per-epoch metrics from the canonical CSV, filters to the raw graph
baseline run only, plots a 2-panel (CCO / MFO) train-vs-val loss figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "output/jupyter-notebook/report_assets/graph_training_epoch_metrics.csv"
OUT_PATH = REPO / "figures/training_loss_curves_baseline.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    baseline = df[df["run_dir"] == "full_graph_pyg_mtf20_33234089"].copy()
    baseline = baseline[baseline["aspect"].isin(["CCO", "MFO"])]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)

    color_train = "#2b6cb0"
    color_val = "#d97706"

    for ax, aspect in zip(axes, ["CCO", "MFO"]):
        sub = baseline[baseline["aspect"] == aspect].sort_values("epoch")
        epochs = sub["epoch"].to_numpy()

        ax.plot(epochs, sub["train_loss"], "o-", color=color_train,
                lw=2, ms=7, label="train loss")
        ax.plot(epochs, sub["val_loss"], "s--", color=color_val,
                lw=2, ms=7, label="validation loss")

        best_row = sub.loc[sub["val_fmax"].idxmax()]
        ax.axvline(best_row["epoch"], color="#999", ls=":", lw=1, alpha=0.7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.set_xticks(epochs)
        ax.set_title(
            f"{aspect}  best val Fmax = {best_row['val_fmax']:.4f}  "
            f"(epoch {int(best_row['epoch'])})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.suptitle(
        "Raw graph baseline: training vs validation loss across 5 epochs",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
