"""Generate batch-level training/validation loss-vs-step figures.

The per-epoch averages in `graph_training_epoch_metrics.csv` look "flat" because:
  (a) the epoch-1 train_loss is the mean over thousands of batches that span
      many orders of magnitude (label-aware: ~199 -> ~0.03 in one epoch);
  (b) once BCE converges to the long-tail prior in epoch 1, subsequent epochs
      barely move in BCE-space — the actual optimisation signal lives in Fmax.

This script parses the `train.log` running-mean log lines, reconstructs the
per-window mean loss within each epoch, and plots loss vs cumulative batch.
That gives a real "is it training?" picture.

Outputs (in figures/):
  1. training_loss_curves_baseline.png   (CCO + MFO raw graph baseline)
  2. training_loss_curves_label_aware.png (CCO + MFO label-aware longer)
  3. val_loss_vs_val_fmax_decoupling.png  (loss min != Fmax max — kept)
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache/training_runs")
EPOCH_CSV = ROOT / "output/jupyter-notebook/report_assets/graph_training_epoch_metrics.csv"
OUT = ROOT / "figures"

PROGRESS_RE = re.compile(
    r"\[progress\] epoch=(?P<epoch>\d+) (?P<aspect>[A-Z]+) (?P<phase>train|val|test) "
    r"batch=(?P<batch>\d+)/(?P<total>\d+) graphs=\d+ loss=(?P<loss>[0-9.eE+-]+)"
)


def parse_train_log(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            m = PROGRESS_RE.match(line)
            if not m:
                continue
            rows.append(dict(
                epoch=int(m["epoch"]),
                aspect=m["aspect"],
                phase=m["phase"],
                batch=int(m["batch"]),
                total=int(m["total"]),
                running_mean=float(m["loss"]),
            ))
    return pd.DataFrame(rows)


def cumulative_step(df_phase: pd.DataFrame) -> pd.DataFrame:
    df_phase = df_phase.sort_values(["epoch", "batch"]).reset_index(drop=True)
    total_per_epoch = df_phase.groupby("epoch").total.first().to_dict()
    offsets = {}
    cum = 0
    for ep in sorted(total_per_epoch):
        offsets[ep] = cum
        cum += total_per_epoch[ep]
    df_phase["cum_batch"] = df_phase.epoch.map(offsets) + df_phase.batch
    return df_phase


def epoch_boundaries(df_train: pd.DataFrame) -> list[int]:
    bounds = []
    cum = 0
    for _, total in df_train.groupby("epoch").total.first().items():
        cum += total
        bounds.append(cum)
    return bounds


def gather_run(run_dir: Path, aspect: str) -> dict:
    log = run_dir / aspect.lower() / "train.log"
    raw = parse_train_log(log)
    train = cumulative_step(raw[raw.phase == "train"].copy())
    val_running = (raw[raw.phase == "val"]
                   .sort_values(["epoch", "batch"])
                   .groupby("epoch")
                   .running_mean.last()
                   .reset_index(name="val_loss"))
    return dict(train=train, val=val_running, bounds=epoch_boundaries(train))


def plot_run(ax_loss, ax_fmax, run, val_fmax_by_epoch, *, title):
    train = run["train"]
    bounds = run["bounds"]

    # Drop the first 5 log entries of each epoch — the running mean over <50
    # batches is dominated by single-batch noise on the first few stragglers.
    parts = []
    for ep, g in train.groupby("epoch"):
        parts.append(g.iloc[5:] if len(g) > 6 else g)
    cleaned = pd.concat(parts, ignore_index=True)

    for _, g in cleaned.groupby("epoch"):
        ax_loss.plot(g.cum_batch, g.running_mean, color="#1f77b4",
                     linewidth=1.8, alpha=0.95)

    val = run["val"]
    val_x = [bounds[int(e) - 1] for e in val.epoch]
    ax_loss.plot(val_x, val.val_loss, marker="o", color="#d62728",
                 linewidth=2, markersize=7, label="Val loss (epoch end)")
    ax_loss.plot([], [], color="#1f77b4", linewidth=1.8,
                 label="Train loss (running mean within epoch)")

    for b in bounds[:-1]:
        ax_loss.axvline(b, color="#aaa", linestyle=":", linewidth=1.0)
        ax_fmax.axvline(b, color="#aaa", linestyle=":", linewidth=1.0)

    n_epochs = len(bounds)
    epoch_centers = [bounds[0] / 2] + [(bounds[i - 1] + bounds[i]) / 2
                                       for i in range(1, n_epochs)]
    ax_top = ax_loss.secondary_xaxis("top")
    ax_top.set_xticks(epoch_centers)
    ax_top.set_xticklabels([f"ep {i + 1}" for i in range(n_epochs)],
                           fontsize=8.5, color="#555")
    ax_top.tick_params(length=0)

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("BCE loss (log scale)")
    ax_loss.set_title(title)
    ax_loss.grid(alpha=0.3, which="both")
    ax_loss.legend(loc="upper right", fontsize=9)

    fmax_x = [bounds[int(e) - 1] for e in val_fmax_by_epoch.epoch]
    ax_fmax.plot(fmax_x, val_fmax_by_epoch.val_fmax, marker="s",
                 color="#2ca02c", linewidth=2, markersize=7,
                 label="Val Fmax (epoch end)")
    best_idx = val_fmax_by_epoch.val_fmax.idxmax()
    best_x = bounds[int(val_fmax_by_epoch.loc[best_idx, "epoch"]) - 1]
    best_y = val_fmax_by_epoch.loc[best_idx, "val_fmax"]
    ax_fmax.scatter([best_x], [best_y], s=140, facecolors="none",
                    edgecolors="#2ca02c", linewidths=2,
                    label=f"Best Fmax = {best_y:.4f}")
    ax_fmax.set_ylabel("Val Fmax")
    ax_fmax.set_xlabel("Cumulative training batch")
    ax_fmax.grid(alpha=0.3)
    ax_fmax.legend(loc="lower right", fontsize=9)


def _val_fmax(label: str, aspect: str, epoch_df: pd.DataFrame) -> pd.DataFrame:
    sub = epoch_df[(epoch_df.model_change == label) & (epoch_df.aspect == aspect)]
    return sub.sort_values("epoch")[["epoch", "val_fmax"]]


def figure_baseline():
    epoch_df = pd.read_csv(EPOCH_CSV)
    run_dir = RUNS / "full_graph_pyg_mtf20_33234089"
    cco = gather_run(run_dir, "CCO")
    mfo = gather_run(run_dir, "MFO")
    cco_fmax = _val_fmax("Raw graph baseline", "CCO", epoch_df)
    mfo_fmax = _val_fmax("Raw graph baseline", "MFO", epoch_df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5),
                             gridspec_kw=dict(height_ratios=[1.6, 1]))
    plot_run(axes[0, 0], axes[1, 0], cco, cco_fmax,
             title="CCO — Raw graph baseline (5 epochs, ~8.3K batches/epoch)")
    plot_run(axes[0, 1], axes[1, 1], mfo, mfo_fmax,
             title="MFO — Raw graph baseline (5 epochs, ~7.6K batches/epoch)")
    fig.suptitle(
        "Batch-level training loss + per-epoch val Fmax — "
        "BCE drops 1–2 orders of magnitude inside epoch 1, then converges",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = OUT / "training_loss_curves_baseline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_label_aware():
    epoch_df = pd.read_csv(EPOCH_CSV)
    run_dir = RUNS / "sigimp_n3_confirm_long_20260425_n3_confirm"
    cco = gather_run(run_dir, "CCO")
    mfo = gather_run(run_dir, "MFO")
    label = "Label-aware scorer, longer confirmation"
    cco_fmax = _val_fmax(label, "CCO", epoch_df)
    mfo_fmax = _val_fmax(label, "MFO", epoch_df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5),
                             gridspec_kw=dict(height_ratios=[1.6, 1]))
    plot_run(axes[0, 0], axes[1, 0], cco, cco_fmax,
             title="CCO — Label-aware head (8 epochs)")
    plot_run(axes[0, 1], axes[1, 1], mfo, mfo_fmax,
             title="MFO — Label-aware head (6 epochs)")
    fig.suptitle(
        "Batch-level training loss + per-epoch val Fmax — label-aware head "
        "starts at loss ≈ 200 (random label embeddings) and converges in epoch 1",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = OUT / "training_loss_curves_label_aware.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_loss_vs_fmax_decoupling():
    """The narrative: BCE flattens fast; further gains live in Fmax (epoch-level)."""
    df = pd.read_csv(EPOCH_CSV)
    sub = df[(df.model_change == "Label-aware scorer, longer confirmation")
             & (df.aspect == "CCO")].sort_values("epoch")

    fig, ax_loss = plt.subplots(figsize=(8.6, 4.8))
    ax_fmax = ax_loss.twinx()

    l1, = ax_loss.plot(sub.epoch, sub.val_loss, marker="o",
                       color="#d62728", linewidth=2, label="Val loss")
    l2, = ax_fmax.plot(sub.epoch, sub.val_fmax, marker="s",
                       color="#2ca02c", linewidth=2, label="Val Fmax")

    best_loss_ep = int(sub.loc[sub.val_loss.idxmin(), "epoch"])
    best_fmax_ep = int(sub.loc[sub.val_fmax.idxmax(), "epoch"])

    ax_loss.axvline(best_loss_ep, color="#d62728", linestyle=":",
                    linewidth=1.2, alpha=0.7)
    ax_fmax.axvline(best_fmax_ep, color="#2ca02c", linestyle="--",
                    linewidth=1.2, alpha=0.9)

    ax_loss.annotate(
        f"min val loss\n@ ep {best_loss_ep}",
        xy=(best_loss_ep, sub.val_loss.min()),
        xytext=(-58, 14), textcoords="offset points",
        fontsize=9, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )
    ax_fmax.annotate(
        f"max val Fmax = {sub.val_fmax.max():.4f}\n@ ep {best_fmax_ep}  ← chosen checkpoint",
        xy=(best_fmax_ep, sub.val_fmax.max()),
        xytext=(10, -36), textcoords="offset points",
        fontsize=9, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val BCE loss", color="#d62728")
    ax_fmax.set_ylabel("Val Fmax", color="#2ca02c")
    ax_loss.tick_params(axis="y", colors="#d62728")
    ax_fmax.tick_params(axis="y", colors="#2ca02c")
    ax_loss.set_xticks(sub.epoch)
    ax_loss.grid(alpha=0.3)
    ax_loss.set_title(
        "Why we checkpoint on Fmax, not loss — CCO label-aware run\n"
        "(BCE converges to long-tail prior; Fmax keeps moving)",
        fontsize=12,
    )
    ax_loss.legend(handles=[l1, l2], loc="lower right")

    fig.tight_layout()
    out = OUT / "val_loss_vs_val_fmax_decoupling.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure_baseline()
    figure_label_aware()
    figure_loss_vs_fmax_decoupling()
