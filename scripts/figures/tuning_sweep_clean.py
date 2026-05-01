#!/usr/bin/env python3
"""Clean redesign of figures/tuning_sweep_overview.png.

Reads run-level results from the canonical best-results CSV. For each
single-factor tuning variant on CCO and MFO, plots its delta vs the raw
baseline as a horizontal bar. Two panels (CCO, MFO) side by side. Variant
order is fixed (raw baseline at the top, then sorted by what changed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "output/jupyter-notebook/report_assets/graph_training_best_results.csv"
OUT_PATH = REPO / "figures/tuning_sweep_overview.png"

# Map run model_change -> short label, in display order (top to bottom).
LABEL_ORDER = [
    ("Raw graph baseline", "Raw baseline"),
    ("Baseline rerun", "Control rerun"),
    ("Lower learning rate, 0.0007", "lr = 7e-4"),
    ("Lower learning rate, 0.0005", "lr = 5e-4"),
    ("Moderate hidden size", "hidden = 192"),
    ("Larger hidden size", "hidden = 256"),
    ("Weighted BCE", "Weighted BCE"),
    ("Broad tuned recipe", "Combined tuned bundle"),
]


def baseline_value(df: pd.DataFrame, aspect: str) -> float:
    row = df[(df["model_change"] == "Raw graph baseline") & (df["aspect"] == aspect)]
    return float(row["best_test_fmax"].iloc[0])


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    keep = df["model_change"].isin([m for m, _ in LABEL_ORDER])
    df = df[keep & df["aspect"].isin(["CCO", "MFO"])].copy()

    base = {a: baseline_value(df, a) for a in ("CCO", "MFO")}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, aspect in zip(axes, ["CCO", "MFO"]):
        sub = df[df["aspect"] == aspect]
        bars_x, bars_y, labels, colors = [], [], [], []
        for i, (model_change, short) in enumerate(LABEL_ORDER):
            row = sub[sub["model_change"] == model_change]
            if row.empty:
                continue
            delta = float(row["best_test_fmax"].iloc[0]) - base[aspect]
            bars_y.append(short)
            bars_x.append(delta)
            if model_change == "Raw graph baseline":
                colors.append("#666")
            elif delta > 0:
                colors.append("#2b8a3e")
            else:
                colors.append("#c92a2a")

        y_pos = list(range(len(bars_y)))
        ax.barh(y_pos, bars_x, color=colors, edgecolor="#333", linewidth=0.5)
        ax.axvline(0, color="#000", lw=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bars_y)
        ax.invert_yaxis()  # raw baseline at top
        for y, x in zip(y_pos, bars_x):
            ax.text(
                x + (0.0008 if x >= 0 else -0.0008),
                y,
                f"{x:+.4f}" if x != 0 else "0.0000",
                va="center",
                ha="left" if x >= 0 else "right",
                fontsize=9,
            )
        ax.set_xlabel(f"Δ test Fmax  vs raw baseline ({base[aspect]:.4f})")
        ax.set_title(f"{aspect}")
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_xlim(min(bars_x) - 0.005, max(bars_x) + 0.005)

    fig.suptitle(
        "Hyperparameter sweep: how much does each single-factor change move test Fmax?",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
