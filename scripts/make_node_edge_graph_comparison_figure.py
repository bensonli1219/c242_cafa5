#!/usr/bin/env python3
"""Render an English comparison table for node / edge / graph-level data
in the structure-graph dataset.

Output: figures/node_edge_graph_data_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


COL_NODE  = "#cfe2f3"
COL_NODE_E = "#1f77b4"
COL_EDGE  = "#d5e8d4"
COL_EDGE_E = "#2ca02c"
COL_GRAPH = "#fde2c4"
COL_GRAPH_E = "#d97706"

ROW_HEADER_BG = "#f3f3f3"
ROW_ALT_BG    = "#fafafa"
LABEL_BG      = "#ececec"


ROWS = [
    (
        "Attached to",
        "each residue\n(one amino acid)",
        "each residue-residue contact\n(one edge)",
        "the whole protein\n(not residues, not contacts)",
    ),
    (
        "Tensor shape\n(per protein)",
        "x : (N, 682)",
        "edge_index : (2, E)\nedge_attr : (E, 6)",
        "graph_feat : (13,)",
    ),
    (
        "Count per protein",
        "N\n(= sequence length)",
        "E\n(roughly ~10 x N for our graphs)",
        "1\n(single fixed-size vector)",
    ),
    (
        "Example features",
        "amino-acid one-hot,\npLDDT + bucket flags,\nPAE row stats,\ncontact degree,\nposition fraction (i / L)",
        "C-alpha distance,\nsequence separation |i - j|,\nPAE on this pair,\nstrict-3D-contact indicator,\nbackbone-vs-non-seq flag",
        "protein length, fragment count,\nproportion of residues in each\npLDDT bucket, mean PAE,\nedge density,\nradius of gyration, ...",
    ),
]

# Row index that should be rendered taller than the others (1.0 = same as base).
ROW_HEIGHT_WEIGHTS = [1.0, 1.0, 1.0, 2.0]

COLUMN_LABELS = [
    ("Node data",  "per-residue",  COL_NODE,  COL_NODE_E),
    ("Edge data",  "per-contact",  COL_EDGE,  COL_EDGE_E),
    ("Graph data", "per-protein",  COL_GRAPH, COL_GRAPH_E),
]


def main(out_path: Path) -> None:
    n_rows = len(ROWS)
    weights = ROW_HEIGHT_WEIGHTS if len(ROW_HEIGHT_WEIGHTS) == n_rows else [1.0] * n_rows
    weight_total = sum(weights)
    base_row_h = 1.05
    fig_w = 16.0
    fig_h = 2.2 + base_row_h * weight_total
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("auto")
    ax.axis("off")

    # Title
    ax.text(fig_w / 2, fig_h - 0.45,
            "Node, Edge, and Graph data in the protein structure-graph dataset",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(fig_w / 2, fig_h - 0.95,
            "what each tensor describes, its shape, and the kind of features it stores",
            ha="center", va="center", fontsize=10, color="#555", fontstyle="italic")

    # Layout
    x_label_left = 0.30
    x_label_right = 3.20
    label_w = x_label_right - x_label_left
    col_w = (fig_w - x_label_right - 0.30) / 3.0

    header_y = fig_h - 2.20
    header_h = 0.85
    body_top = header_y
    body_bottom = 0.30
    body_h = body_top - body_bottom
    row_heights = [body_h * (w / weight_total) for w in weights]

    # Header label cell (top-left, blank corner with "feature aspect")
    corner = Rectangle(
        (x_label_left, header_y), label_w, header_h,
        facecolor=ROW_HEADER_BG, edgecolor="#bbbbbb", linewidth=1.0,
    )
    ax.add_patch(corner)
    ax.text(x_label_left + label_w / 2, header_y + header_h / 2,
            "feature aspect",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#444", fontstyle="italic")

    # Column headers
    for col_idx, (title, sub, face, edge) in enumerate(COLUMN_LABELS):
        cx = x_label_right + col_idx * col_w
        rect = Rectangle(
            (cx, header_y), col_w, header_h,
            facecolor=face, edgecolor=edge, linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(cx + col_w / 2, header_y + header_h * 0.62,
                title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=edge)
        ax.text(cx + col_w / 2, header_y + header_h * 0.27,
                sub, ha="center", va="center",
                fontsize=9.5, color="#444", fontstyle="italic")

    # Body rows
    cum = 0.0
    for r_idx, row in enumerate(ROWS):
        cum += row_heights[r_idx]
        y = body_top - cum
        row_h = row_heights[r_idx]
        is_big = weights[r_idx] > 1.0
        cell_font = 11.5 if is_big else 9.8

        # Row label cell
        label_text = row[0]
        label_face = LABEL_BG
        label_rect = Rectangle(
            (x_label_left, y), label_w, row_h,
            facecolor=label_face, edgecolor="#bbbbbb", linewidth=1.0,
        )
        ax.add_patch(label_rect)
        ax.text(x_label_left + label_w / 2, y + row_h / 2,
                label_text,
                ha="center", va="center",
                fontsize=11.5 if is_big else 10.5,
                fontweight="bold", color="#222")

        # Three data cells
        for col_idx in range(3):
            cx = x_label_right + col_idx * col_w
            face = ROW_ALT_BG if r_idx % 2 == 0 else "white"
            rect = Rectangle(
                (cx, y), col_w, row_h,
                facecolor=face,
                edgecolor="#cccccc", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(cx + col_w / 2, y + row_h / 2,
                    row[col_idx + 1],
                    ha="center", va="center",
                    fontsize=cell_font, color="#222", linespacing=1.40)

        # Coloured side strip on left of each data column to keep colour anchor
        for col_idx, (_, _, face, edge) in enumerate(COLUMN_LABELS):
            cx = x_label_right + col_idx * col_w
            strip = Rectangle(
                (cx, y), 0.12, row_h,
                facecolor=face, edgecolor=edge, linewidth=0.8,
            )
            ax.add_patch(strip)

    # Outer border
    border = Rectangle(
        (x_label_left, body_bottom),
        fig_w - x_label_left - 0.30,
        body_top - body_bottom + header_h,
        fill=False, edgecolor="#888", linewidth=1.4,
    )
    ax.add_patch(border)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    out = Path("figures/node_edge_graph_data_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
