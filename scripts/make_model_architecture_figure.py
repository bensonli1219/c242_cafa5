#!/usr/bin/env python3
"""Render a simple model architecture diagram for the structure GNN classifier.

Mirrors `MinimalPygClassifier` in train_minimal_graph_model.py:
  - Linear(682 -> 128)
  - GCNConv(128 -> 128) + ReLU + dropout
  - GCNConv(128 -> 128) + ReLU
  - global mean pool (per-protein node aggregation)
  - parallel branch: Linear(13 -> 128) + ReLU on graph_feat
  - concat -> dropout -> classifier head (flat linear or label-aware dot product)
  - sigmoid -> per-GO-term probability

Output: figures/model_architecture_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLOR_INPUT = "#cfe2f3"
COLOR_INPUT_EDGE = "#1f77b4"
COLOR_PROJ = "#fde2c4"
COLOR_PROJ_EDGE = "#d97706"
COLOR_GCN = "#d5e8d4"
COLOR_GCN_EDGE = "#2ca02c"
COLOR_POOL = "#ead1dc"
COLOR_POOL_EDGE = "#8e44ad"
COLOR_FUSE = "#fff2cc"
COLOR_FUSE_EDGE = "#bf8f00"
COLOR_HEAD = "#f4cccc"
COLOR_HEAD_EDGE = "#cc0000"
COLOR_OUT = "#e6e6e6"
COLOR_OUT_EDGE = "#444"


def add_box(ax, x, y, w, h, *, label, sub=None, face=COLOR_GCN, edge=COLOR_GCN_EDGE,
            font=11, sub_font=9, lw=1.6, radius=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=font, fontweight="bold")
    return box


def add_arrow(ax, x1, y1, x2, y2, *, color="#444", lw=2.0, label=None, label_offset=(0, 0.12)):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=lw, color=color,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            ha="center", va="center",
            fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
        )


def main(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 9.0))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")

    add_box(ax, 0.2, 6.6, 2.6, 1.05,
            label="Node features  x",
            sub="(N residues x 682)",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE)
    add_box(ax, 0.2, 5.25, 2.6, 1.05,
            label="Edge index",
            sub="(2 x E contacts)",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE)
    add_box(ax, 0.2, 3.9, 2.6, 1.05,
            label="Graph feature  graph_feat",
            sub="(13,)",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE)

    ax.text(1.5, 8.1, "Input  (one protein graph)",
            ha="center", va="center", fontsize=12, fontweight="bold", color=COLOR_INPUT_EDGE)

    add_box(ax, 3.6, 6.6, 2.5, 1.05,
            label="Linear",
            sub="682 -> 128  + ReLU",
            face=COLOR_PROJ, edge=COLOR_PROJ_EDGE)
    add_arrow(ax, 2.8, 7.12, 3.6, 7.12, label="x")

    add_box(ax, 6.7, 7.0, 2.5, 0.9,
            label="GCNConv #1",
            sub="128 -> 128 + ReLU",
            face=COLOR_GCN, edge=COLOR_GCN_EDGE)
    add_box(ax, 6.7, 5.95, 2.5, 0.9,
            label="GCNConv #2",
            sub="128 -> 128 + ReLU",
            face=COLOR_GCN, edge=COLOR_GCN_EDGE)
    add_arrow(ax, 6.1, 7.12, 6.7, 7.45, label="(N, 128)")

    ax.annotate("",
                xy=(7.95, 6.95), xytext=(7.95, 6.85),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.0))
    ax.text(8.4, 6.9, "+ dropout", fontsize=8, color="#666", va="center")

    add_arrow(ax, 2.8, 5.78, 6.7, 6.4, label="edge_index", label_offset=(0, -0.18))

    add_box(ax, 9.7, 5.95, 2.4, 1.95,
            label="global\nmean pool",
            sub="aggregate per protein\n(N, 128) -> (B, 128)",
            face=COLOR_POOL, edge=COLOR_POOL_EDGE,
            font=11, sub_font=8)
    add_arrow(ax, 9.2, 6.4, 9.7, 6.92, label="(N, 128)")

    add_box(ax, 6.7, 4.05, 2.5, 0.9,
            label="Linear",
            sub="13 -> 128 + ReLU",
            face=COLOR_PROJ, edge=COLOR_PROJ_EDGE)
    add_arrow(ax, 2.8, 4.42, 6.7, 4.50, label="graph_feat")

    add_box(ax, 9.7, 4.05, 2.4, 0.9,
            label="(B, 128)",
            sub="graph-level branch",
            face=COLOR_POOL, edge=COLOR_POOL_EDGE,
            font=10, sub_font=8)
    add_arrow(ax, 9.2, 4.50, 9.7, 4.50)

    add_box(ax, 12.7, 5.0, 2.0, 1.55,
            label="concat\n+ dropout",
            sub="(B, 256)",
            face=COLOR_FUSE, edge=COLOR_FUSE_EDGE,
            font=11, sub_font=9)
    add_arrow(ax, 12.1, 6.92, 13.05, 6.55, label="from nodes\n(B, 128)")
    add_arrow(ax, 12.1, 4.50, 13.05, 5.10, label="from graph_feat\n(B, 128)")

    head_x = 15.0
    add_box(ax, head_x, 6.0, 1.85, 1.05,
            label="Linear head",
            sub="256 -> #GO terms",
            face=COLOR_HEAD, edge=COLOR_HEAD_EDGE,
            font=10, sub_font=8)
    ax.text(head_x + 0.93, 7.18, "flat_linear head",
            ha="center", va="bottom", fontsize=9, color=COLOR_HEAD_EDGE, fontweight="bold")

    add_box(ax, head_x, 4.6, 1.85, 1.05,
            label="dot-product head",
            sub="fused . label_emb[k]",
            face=COLOR_HEAD, edge=COLOR_HEAD_EDGE,
            font=10, sub_font=8)
    ax.text(head_x + 0.93, 4.42, "label_dot head",
            ha="center", va="top", fontsize=9, color=COLOR_HEAD_EDGE, fontweight="bold")
    ax.text(head_x + 0.93, 4.05, "(label-aware scorer)",
            ha="center", va="top", fontsize=8, color="#666", fontstyle="italic")

    add_arrow(ax, 14.7, 5.95, 15.0, 6.52, label=None)
    add_arrow(ax, 14.7, 5.6, 15.0, 5.13, label=None)
    ax.text(14.5, 5.78, "either", ha="right", va="center", fontsize=8.5, color="#888", fontstyle="italic")

    add_box(ax, head_x - 0.2, 2.6, 2.25, 1.0,
            label="sigmoid",
            sub="-> per-GO-term\nprobability  (B, K)",
            face=COLOR_OUT, edge=COLOR_OUT_EDGE,
            font=10, sub_font=8)
    add_arrow(ax, head_x + 0.93, 6.0, head_x + 0.93, 3.65, color="#888")
    add_arrow(ax, head_x + 0.93, 4.6, head_x + 0.93, 3.65, color="#888")

    ax.text(head_x + 0.93, 2.05, "Output\nmulti-label GO predictions",
            ha="center", va="top", fontsize=10, fontweight="bold", color="#222")

    legend_y = 1.05
    ax.text(0.2, legend_y + 0.55, "Colour key", fontsize=10, fontweight="bold")
    legend_specs = [
        ("input tensor", COLOR_INPUT, COLOR_INPUT_EDGE),
        ("Linear projection", COLOR_PROJ, COLOR_PROJ_EDGE),
        ("Graph convolution\n(message passing)", COLOR_GCN, COLOR_GCN_EDGE),
        ("Pooling /\nper-protein vector", COLOR_POOL, COLOR_POOL_EDGE),
        ("Concat /\nfusion", COLOR_FUSE, COLOR_FUSE_EDGE),
        ("Classifier head", COLOR_HEAD, COLOR_HEAD_EDGE),
    ]
    lx = 0.2
    for label, face, edge in legend_specs:
        rect = patches.Rectangle((lx, legend_y - 0.05), 0.45, 0.3,
                                 facecolor=face, edgecolor=edge, linewidth=1.4)
        ax.add_patch(rect)
        ax.text(lx + 0.55, legend_y + 0.10, label, fontsize=8.5, va="center")
        lx += 2.55

    ax.text(8.5, 8.6,
            "Structure GNN classifier  (MinimalPygClassifier, hidden_dim = 128, dropout = 0.2)",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(8.5, 8.18,
            "shared baseline used for all structure-graph experiments in this report",
            ha="center", va="center", fontsize=10, color="#555", fontstyle="italic")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("figures/model_architecture_diagram.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
