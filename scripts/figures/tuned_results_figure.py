"""Composite visualization comparing baseline vs tuned graph training runs.

Reads saved Savio artifacts (`results_summary.json` and per-aspect
`summary.json`) for the baseline and tuned full-graph runs and produces a
single PNG that shows the test/validation Fmax bars, macro-F1 bars, and the
per-epoch validation Fmax learning curves for CCO and MFO.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_RUN_DIR = Path(
    "/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/"
    "full_graph_pyg_mtf20_33234089"
)
TUNED_RUN_DIR = Path(
    "/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs/"
    "full_graph_tuned_pyg_mtf20_33275343"
)
OUTPUT_PATH = Path(
    "/global/home/users/bensonli/c242_cafa5/figures/tuned_graph_model_results.png"
)
ASPECTS = ("CCO", "MFO")


def load_results(run_dir: Path) -> dict[str, dict]:
    rows = json.loads((run_dir / "results_summary.json").read_text())
    return {row["aspect"]: row for row in rows}


def load_val_fmax_curve(run_dir: Path, aspect: str) -> tuple[list[int], list[float]]:
    summary_path = run_dir / aspect.lower() / "summary.json"
    if not summary_path.exists():
        return [], []
    summary = json.loads(summary_path.read_text())
    epochs, values = [], []
    for record in summary.get("history", []) or []:
        val = (record.get("val") or {}).get("fmax")
        if val is None:
            continue
        epochs.append(int(record["epoch"]))
        values.append(float(val))
    return epochs, values


def add_value_labels(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    baseline = load_results(BASELINE_RUN_DIR)
    tuned = load_results(TUNED_RUN_DIR)

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
    ax_fmax = fig.add_subplot(gs[0, 0])
    ax_macro = fig.add_subplot(gs[0, 1])
    ax_cco = fig.add_subplot(gs[1, 0])
    ax_mfo = fig.add_subplot(gs[1, 1])

    width = 0.35
    x = np.arange(len(ASPECTS))
    baseline_color = "#4C72B0"
    tuned_color = "#DD8452"

    # --- Test Fmax bars ---
    base_fmax = [baseline[a].get("test_fmax") for a in ASPECTS]
    tune_fmax = [tuned[a].get("test_fmax") for a in ASPECTS]
    bars1 = ax_fmax.bar(x - width / 2, base_fmax, width, label="baseline", color=baseline_color)
    bars2 = ax_fmax.bar(x + width / 2, tune_fmax, width, label="tuned", color=tuned_color)
    ax_fmax.set_xticks(x)
    ax_fmax.set_xticklabels(ASPECTS)
    ax_fmax.set_ylabel("Test Fmax")
    ax_fmax.set_title("Test Fmax: baseline vs tuned")
    ax_fmax.set_ylim(0, max(filter(None, base_fmax + tune_fmax)) * 1.18)
    ax_fmax.grid(axis="y", alpha=0.25)
    ax_fmax.legend(frameon=False)
    add_value_labels(ax_fmax, bars1)
    add_value_labels(ax_fmax, bars2)

    # --- Test macro-F1 bars ---
    base_macro = [baseline[a].get("test_macro_f1") for a in ASPECTS]
    tune_macro = [tuned[a].get("test_macro_f1") for a in ASPECTS]
    bars3 = ax_macro.bar(x - width / 2, base_macro, width, label="baseline", color=baseline_color)
    bars4 = ax_macro.bar(x + width / 2, tune_macro, width, label="tuned", color=tuned_color)
    ax_macro.set_xticks(x)
    ax_macro.set_xticklabels(ASPECTS)
    ax_macro.set_ylabel("Test macro-F1")
    ax_macro.set_title("Test macro-F1: weighted BCE lifts rare labels")
    ax_macro.set_ylim(0, max(filter(None, base_macro + tune_macro)) * 1.30)
    ax_macro.grid(axis="y", alpha=0.25)
    ax_macro.legend(frameon=False)
    add_value_labels(ax_macro, bars3, fmt="{:.4f}")
    add_value_labels(ax_macro, bars4, fmt="{:.4f}")

    # --- Per-epoch validation Fmax curves ---
    for ax, aspect in ((ax_cco, "CCO"), (ax_mfo, "MFO")):
        b_epochs, b_vals = load_val_fmax_curve(BASELINE_RUN_DIR, aspect)
        t_epochs, t_vals = load_val_fmax_curve(TUNED_RUN_DIR, aspect)
        if b_epochs:
            ax.plot(b_epochs, b_vals, marker="o", linewidth=2, color=baseline_color, label="baseline val")
        if t_epochs:
            ax.plot(t_epochs, t_vals, marker="s", linewidth=2, color=tuned_color, label="tuned val")
        # Mark the tuned best (Fmax-checkpointed) epoch
        best_epoch = tuned[aspect].get("best_epoch")
        best_metric = tuned[aspect].get("best_checkpoint_metric")
        if best_epoch and best_metric is not None:
            ax.scatter(
                [best_epoch],
                [best_metric],
                s=110,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
                label="tuned best (Fmax checkpoint)",
            )
        ax.set_title(f"{aspect}: validation Fmax per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("val Fmax")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    # --- Title and footnote ---
    fig.suptitle(
        "Tuned vs baseline graph model — full CAFA5 training (mtf20)",
        fontsize=15,
        y=0.995,
    )
    footnote = (
        "Tuned changes: hidden 128→256, dropout 0.2→0.3, lr 1e-3→3e-4, weight_decay 1e-4→5e-4, "
        "weighted BCE (pos_weight^0.5, cap=20), Fmax checkpointing, plateau LR, early stopping. "
        "BPO failed in both runs and is omitted."
    )
    fig.text(0.5, 0.005, footnote, ha="center", va="bottom", fontsize=9, style="italic", color="#444")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
