#!/usr/bin/env python3
"""Render the training setup table as a figure (figures/training_setup_table.png).

Table mirrors the actual settings used in:
  - scripts/savio_train_full_graph.sh         (Baseline column)
  - scripts/savio_train_full_graph_tuned.sh   (Tuned column - CCO +2.28 pp result)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


CATEGORY_COLORS = {
    "Data":               "#cfe2f3",
    "Model":              "#d5e8d4",
    "Optimizer":          "#fde2c4",
    "Loss":               "#f4cccc",
    "Training schedule": "#fff2cc",
    "Hardware / runtime": "#e1d5e7",
}

ROWS = [
    ("Data",               "Dataset",            "CAFA5 (AlphaFold-matched cohort)",  "same"),
    ("Data",               "Aspect",             "MFO / CCO",                          "MFO / CCO"),
    ("Data",               "Min term frequency", "20",                                  "20"),
    ("Data",               "Split",              "train / val / test (fixed seed)",   "same"),

    ("Model",              "Architecture",       "2-layer GCN + global mean pool + graph-feat MLP", "same"),
    ("Model",              "Hidden dim",         "128",                                 "256"),
    ("Model",              "Dropout",            "0.2",                                 "0.3"),
    ("Model",              "Classifier head",    "flat_linear",                         "flat_linear or label_dot (label-aware)"),
    ("Model",              "Node feature dim",   "682  (32 base + DSSP/ESM2 zero-filled)", "same"),

    ("Optimizer",          "Algorithm",          "Adam",                                "Adam"),
    ("Optimizer",          "Learning rate",      "1e-3",                                "3e-4"),
    ("Optimizer",          "Weight decay",       "1e-4",                                "5e-4"),
    ("Optimizer",          "LR scheduler",       "none",                                "ReduceLROnPlateau\n(factor 0.5, patience 1, min 1e-6)"),

    ("Loss",               "Loss function",      "BCE-with-logits",                     "weighted BCE\n(pos_weight_power = 0.5, cap = 20)"),
    ("Loss",               "Logit adjustment",   "none",                                "none"),

    ("Training schedule", "Epochs",              "5",                                   "5"),
    ("Training schedule", "Batch size",          "8",                                   "8"),
    ("Training schedule", "Early stopping",      "off",                                 "patience = 2, min_delta = 5e-4"),
    ("Training schedule", "Checkpoint metric",   "val_loss",                            "val_fmax"),
    ("Training schedule", "Seed",                "2026",                                "2026"),

    ("Hardware / runtime", "Partition",          "Savio savio2_1080ti, 1 node, 4 x GTX 1080Ti", "same"),
    ("Hardware / runtime", "Parallelism",        "1 GPU per aspect (BPO / CCO / MFO independent)", "same"),
    ("Hardware / runtime", "Wall-time cap",      "72 h",                                "72 h"),
    ("Hardware / runtime", "Framework",          "PyTorch 2.3.1 + PyG",                "same"),
]

HEADERS = ["Category", "Setting", "Baseline", "Tuned (CCO label-aware run)"]


def _row_height(row) -> float:
    multi_line = any("\n" in str(cell) for cell in row[2:])
    return 0.62 if multi_line else 0.42


def main(out_path: Path) -> None:
    n_rows = len(ROWS)
    row_heights = [_row_height(r) for r in ROWS]
    body_h = sum(row_heights)
    header_h = 0.55
    title_h = 0.85
    footer_h = 0.65

    col_widths = [1.7, 2.4, 4.0, 5.6]
    total_w = sum(col_widths)
    total_h = title_h + header_h + body_h + footer_h

    fig, ax = plt.subplots(figsize=(total_w * 1.05, total_h * 0.65))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect("auto")
    ax.axis("off")

    ax.text(
        total_w / 2, total_h - 0.30,
        "Training Setup - Structure GNN classifier",
        ha="center", va="center", fontsize=15, fontweight="bold",
    )
    ax.text(
        total_w / 2, total_h - 0.60,
        "Baseline: scripts/savio_train_full_graph.sh defaults    "
        "Tuned: scripts/savio_train_full_graph_tuned.sh (used for the CCO +2.28 pp label-aware result)",
        ha="center", va="center", fontsize=9, color="#555", fontstyle="italic",
    )

    header_y_top = total_h - title_h
    header_y_bottom = header_y_top - header_h
    x = 0.0
    for h, w in zip(HEADERS, col_widths):
        rect = patches.Rectangle(
            (x, header_y_bottom), w, header_h,
            facecolor="#3c4858", edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, header_y_bottom + header_h / 2, h,
            ha="center", va="center", fontsize=11, fontweight="bold", color="white",
        )
        x += w

    last_category = None
    category_block_top = header_y_bottom
    y = header_y_bottom

    def fill_category_block(top_y, bottom_y, category):
        if category is None:
            return
        rect = patches.Rectangle(
            (0, bottom_y), col_widths[0], top_y - bottom_y,
            facecolor=CATEGORY_COLORS[category], edgecolor="white", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            col_widths[0] / 2, (top_y + bottom_y) / 2, category,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222",
        )

    for i, row in enumerate(ROWS):
        category, setting, baseline, tuned = row
        row_h = row_heights[i]

        is_new_category = category != last_category
        if is_new_category:
            if last_category is not None:
                fill_category_block(category_block_top, y, last_category)
            category_block_top = y
            last_category = category

        row_top = y
        row_bottom = y - row_h

        for c_idx, content in enumerate([setting, baseline, tuned]):
            x_left = sum(col_widths[: c_idx + 1])
            w = col_widths[c_idx + 1]

            face = "#ffffff" if i % 2 == 0 else "#f7f9fb"
            rect = patches.Rectangle(
                (x_left, row_bottom), w, row_h,
                facecolor=face, edgecolor="#dddddd", linewidth=0.8,
            )
            ax.add_patch(rect)

            color = "#222"
            weight = "normal"
            if c_idx == 0:
                weight = "bold"
            if c_idx == 2 and content not in ("same", baseline) and content != "":
                color = "#b03a2e"
                weight = "bold"

            ax.text(
                x_left + 0.10, row_bottom + row_h / 2, content,
                ha="left", va="center",
                fontsize=9, color=color, fontweight=weight,
            )

        y = row_bottom

    fill_category_block(category_block_top, y, last_category)

    footer_top = y
    footer_bottom = footer_top - footer_h
    ax.add_patch(patches.Rectangle(
        (0, footer_bottom), total_w, footer_h,
        facecolor="#fafafa", edgecolor="#cccccc", linewidth=0.8,
    ))
    ax.text(
        0.15, footer_bottom + footer_h * 0.62,
        "Red = differs from Baseline.  ESM2 / DSSP slots are reserved in the cache but zero-filled in current runs.",
        ha="left", va="center", fontsize=9, color="#444",
    )
    ax.text(
        0.15, footer_bottom + footer_h * 0.25,
        "All results in this report use these two configurations; the CCO label-aware +2.28 pp improvement comes from the Tuned column.",
        ha="left", va="center", fontsize=9, color="#444",
    )

    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("figures/training_setup_table.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
