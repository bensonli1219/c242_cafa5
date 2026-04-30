#!/usr/bin/env python3
"""Render a paper-style architecture diagram for the late-fusion classifier.

Two parallel branches that produce per-label scores, then a learned-free
weighted-sum late fusion to produce final multi-label GO predictions.

  Structure branch  (MinimalPygClassifier)
    node features (N x 682)  --[Linear]-->  (N, 128)
                              --[GCNConv x2 + ReLU + dropout]-->  (N, 128)
                              --[global mean pool]-->  (B, 128)
    graph_feat (13)           --[Linear + ReLU]-->     (B, 128)
    concat  (B, 256)          --[classifier head]-->   logits_g  (B, K)

  Sequence branch   (SequenceMlpClassifier on ESM2 embeddings)
    protein ESM2 (640)        --[Linear + ReLU + dropout]-->  (B, 256)
                              --[Linear]-->                   logits_s  (B, K)

  Late fusion
    p = alpha * sigma(logits_g) + (1 - alpha) * sigma(logits_s)

Output: figures/fusion_model_architecture.png
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLOR_INPUT = "#cfe2f3"
COLOR_INPUT_EDGE = "#1f77b4"
COLOR_PROJ = "#fde2c4"
COLOR_PROJ_EDGE = "#d97706"
COLOR_GCN = "#d5e8d4"
COLOR_GCN_EDGE = "#2ca02c"
COLOR_POOL = "#ead1dc"
COLOR_POOL_EDGE = "#8e44ad"
COLOR_HEAD = "#f4cccc"
COLOR_HEAD_EDGE = "#cc0000"
COLOR_FUSE = "#fff2cc"
COLOR_FUSE_EDGE = "#bf8f00"
COLOR_LOGIT = "#fcefcf"
COLOR_LOGIT_EDGE = "#8a6d00"
COLOR_OUT = "#e6e6e6"
COLOR_OUT_EDGE = "#444"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    sub: Optional[str] = None,
    face: str = COLOR_GCN,
    edge: str = COLOR_GCN_EDGE,
    font: float = 11,
    sub_font: float = 8.5,
    lw: float = 1.6,
    radius: float = 0.04,
) -> Tuple[float, float, float, float]:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=font, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub,
                ha="center", va="center", fontsize=sub_font, color="#333")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=font, fontweight="bold")
    return (x, y, x + w, y + h)


def add_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#444",
    lw: float = 1.8,
    label: Optional[str] = None,
    label_offset: Tuple[float, float] = (0, 0.18),
    style: str = "-|>",
    connectionstyle: Optional[str] = None,
) -> None:
    kwargs = dict(arrowstyle=style, mutation_scale=16, linewidth=lw, color=color)
    if connectionstyle is not None:
        kwargs["connectionstyle"] = connectionstyle
    arrow = FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            ha="center", va="center",
            fontsize=8.0, color="#333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )


def draw_dashed_lane(ax, x: float, y: float, w: float, h: float,
                     title: str, color: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.1, edgecolor=color, facecolor="none",
        linestyle=(0, (5, 4)),
    )
    ax.add_patch(rect)
    ax.text(x + 0.18, y + h - 0.22, title,
            ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=color)


def main(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 9.6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(9.0, 9.20,
            "Late-fusion GO classifier  (structure GNN  +  sequence MLP)",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(9.0, 8.78,
            "two parallel branches produce per-label scores;  "
            "fusion = alpha * sigma(logits_g) + (1 - alpha) * sigma(logits_s)",
            ha="center", va="center", fontsize=10.0,
            color="#555", fontstyle="italic")

    # Lane backgrounds
    draw_dashed_lane(ax, 0.10, 4.85, 13.40, 3.55,
                     "Structure branch  -  graph neural network",
                     color=COLOR_GCN_EDGE)
    draw_dashed_lane(ax, 0.10, 1.10, 13.40, 3.30,
                     "Sequence branch  -  ESM2 protein-level MLP",
                     color=COLOR_INPUT_EDGE)

    # ==================================================================
    # STRUCTURE BRANCH
    # ==================================================================
    # Inputs
    add_box(ax, 0.35, 7.20, 2.45, 0.95,
            label="node features  x", sub="(N residues, 682)",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE, font=10, sub_font=8.5)
    add_box(ax, 0.35, 6.10, 2.45, 0.95,
            label="edge_index", sub="(2, E contacts)",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE, font=10, sub_font=8.5)
    add_box(ax, 0.35, 5.00, 2.45, 0.95,
            label="graph_feat", sub="(13,) per protein",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE, font=10, sub_font=8.5)

    # node features -> Linear
    _, _, x_lin_r, _ = add_box(ax, 3.30, 7.20, 1.65, 0.95,
            label="Linear", sub="682 -> 128",
            face=COLOR_PROJ, edge=COLOR_PROJ_EDGE, font=10, sub_font=8.5)
    add_arrow(ax, 2.80, 7.68, 3.30, 7.68)

    # GCNConv x 2
    add_box(ax, 5.45, 7.20, 1.95, 0.95,
            label="GCNConv x 2", sub="128 -> 128 + ReLU + dropout",
            face=COLOR_GCN, edge=COLOR_GCN_EDGE, font=10, sub_font=8.0)
    add_arrow(ax, x_lin_r, 7.68, 5.45, 7.68, label="(N, 128)")

    # edge_index -> GCN (curved)
    add_arrow(ax, 2.80, 6.55, 5.45, 7.30,
              label="edge_index", label_offset=(-0.1, -0.20),
              connectionstyle="arc3,rad=0.10")

    # global mean pool
    add_box(ax, 7.90, 7.20, 1.85, 0.95,
            label="global  mean pool", sub="(N,128) -> (B,128)",
            face=COLOR_POOL, edge=COLOR_POOL_EDGE, font=10, sub_font=8.0)
    add_arrow(ax, 7.40, 7.68, 7.90, 7.68)

    # graph_feat -> Linear -> graph-branch vector
    add_box(ax, 5.45, 5.00, 1.95, 0.95,
            label="Linear + ReLU", sub="13 -> 128",
            face=COLOR_PROJ, edge=COLOR_PROJ_EDGE, font=10, sub_font=8.5)
    add_arrow(ax, 2.80, 5.46, 5.45, 5.46)
    add_box(ax, 7.90, 5.00, 1.85, 0.95,
            label="graph branch", sub="(B, 128)",
            face=COLOR_POOL, edge=COLOR_POOL_EDGE, font=10, sub_font=8.5)
    add_arrow(ax, 7.40, 5.46, 7.90, 5.46)

    # Concat + dropout (single combined block)
    add_box(ax, 9.95, 5.95, 1.45, 1.30,
            label="concat\n+ dropout", sub="(B, 256)",
            face=COLOR_FUSE, edge=COLOR_FUSE_EDGE, font=10, sub_font=8.5)
    add_arrow(ax, 9.75, 7.68, 10.20, 7.25)
    add_arrow(ax, 9.75, 5.46, 10.20, 5.95)

    # Classifier head
    add_box(ax, 11.65, 5.95, 1.85, 1.30,
            label="classifier head",
            sub="flat_linear / label_dot\n256 -> K",
            face=COLOR_HEAD, edge=COLOR_HEAD_EDGE, font=10, sub_font=8.0)
    add_arrow(ax, 11.40, 6.60, 11.65, 6.60)

    # logits_g pill (small explicit box that catches the structure-branch output)
    add_box(ax, 13.90, 6.20, 1.55, 0.85,
            label="logits_g", sub="(B, K)",
            face=COLOR_LOGIT, edge=COLOR_LOGIT_EDGE, font=11, sub_font=8.0)
    add_arrow(ax, 13.50, 6.60, 13.90, 6.60)

    # ==================================================================
    # SEQUENCE BRANCH
    # ==================================================================
    add_box(ax, 0.35, 2.55, 2.45, 0.95,
            label="protein sequence", sub="amino acid string",
            face=COLOR_INPUT, edge=COLOR_INPUT_EDGE, font=10, sub_font=8.5)

    add_box(ax, 3.30, 2.55, 2.55, 0.95,
            label="ESM2 (t30, 150M)", sub="frozen encoder -> 640-d",
            face=COLOR_GCN, edge=COLOR_GCN_EDGE, font=10, sub_font=8.0)
    add_arrow(ax, 2.80, 3.02, 3.30, 3.02)

    add_box(ax, 6.30, 2.55, 2.05, 0.95,
            label="Linear + ReLU", sub="640 -> 256 + dropout",
            face=COLOR_PROJ, edge=COLOR_PROJ_EDGE, font=10, sub_font=8.0)
    add_arrow(ax, 5.85, 3.02, 6.30, 3.02, label="(B, 640)")

    add_box(ax, 8.85, 2.55, 1.95, 0.95,
            label="Linear", sub="256 -> K",
            face=COLOR_HEAD, edge=COLOR_HEAD_EDGE, font=10, sub_font=8.5)
    add_arrow(ax, 8.35, 3.02, 8.85, 3.02, label="(B, 256)")

    # logits_s pill
    add_box(ax, 11.65, 2.55, 1.55, 0.95,
            label="logits_s", sub="(B, K)",
            face=COLOR_LOGIT, edge=COLOR_LOGIT_EDGE, font=11, sub_font=8.0)
    add_arrow(ax, 10.80, 3.02, 11.65, 3.02)

    # ==================================================================
    # LATE FUSION + OUTPUT
    # ==================================================================
    add_box(ax, 15.65, 4.40, 2.20, 1.40,
            label="late fusion",
            sub="alpha * sigma(logits_g)\n+ (1-alpha) * sigma(logits_s)",
            face=COLOR_FUSE, edge=COLOR_FUSE_EDGE, font=11, sub_font=8.0)

    # logits_g -> fusion (curved)
    add_arrow(ax, 15.45, 6.20, 16.40, 5.80,
              connectionstyle="arc3,rad=0.05", color="#666")
    # logits_s -> fusion (curved)
    add_arrow(ax, 13.20, 3.02, 16.40, 4.40,
              connectionstyle="arc3,rad=-0.10", color="#666")

    # Output
    add_box(ax, 15.65, 2.20, 2.20, 1.30,
            label="multi-label\nGO predictions",
            sub="probabilities  (B, K)",
            face=COLOR_OUT, edge=COLOR_OUT_EDGE, font=11, sub_font=8.5)
    add_arrow(ax, 16.75, 4.40, 16.75, 3.50)

    ax.text(16.75, 1.78, "alpha selected on val grid {0.0, 0.1, ..., 1.0}",
            ha="center", va="center", fontsize=8.5, color="#555",
            fontstyle="italic")

    # ==================================================================
    # Legend
    # ==================================================================
    legend_y = 0.30
    ax.text(0.10, legend_y + 0.40, "Colour key",
            fontsize=10, fontweight="bold")
    legend_specs = [
        ("input tensor",                 COLOR_INPUT,  COLOR_INPUT_EDGE),
        ("Linear projection",            COLOR_PROJ,   COLOR_PROJ_EDGE),
        ("Encoder (GCN / ESM2)",         COLOR_GCN,    COLOR_GCN_EDGE),
        ("Pooling / per-protein vec",    COLOR_POOL,   COLOR_POOL_EDGE),
        ("Concat / fusion",              COLOR_FUSE,   COLOR_FUSE_EDGE),
        ("Classifier head",              COLOR_HEAD,   COLOR_HEAD_EDGE),
        ("Logits (B, K)",                COLOR_LOGIT,  COLOR_LOGIT_EDGE),
        ("Output",                       COLOR_OUT,    COLOR_OUT_EDGE),
    ]
    lx = 0.10
    for label_text, face, edge in legend_specs:
        rect = patches.Rectangle(
            (lx, legend_y - 0.04), 0.40, 0.30,
            facecolor=face, edgecolor=edge, linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(lx + 0.50, legend_y + 0.10, label_text,
                fontsize=8.5, va="center")
        lx += 2.20

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("figures/fusion_model_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
