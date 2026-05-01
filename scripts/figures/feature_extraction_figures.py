#!/usr/bin/env python3
"""Generate human-readable Feature Extraction figures for the presentation.

Loads the cached protein graph for a single example (Q8IXT2 by default) from the
graph cache and renders three figures:

  1. feature_extraction_node_features.png
       - one-residue feature card (annotated)
       - full N x summary-column node feature heatmap

  2. feature_extraction_contact_map.png
       - residue x residue contact matrix with sequential vs. long-range edges

  3. feature_extraction_edges_and_graph.png
       - distance / sequence separation / PAE histograms over edges
       - graph-level 13-dim scalar card

Run from the repo root with the `cafa5` conda env (torch available):

    python scripts/make_feature_extraction_figures.py \
        --graph output/graph_cache_normalized_smoke/graphs/Q8IXT2.pt \
        --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import torch

AA_ORDER = list("ARNDCQEGHILKMFPSTWYVX")

NODE_FEATURE_COLUMNS = [
    ("AA one-hot", slice(0, 21)),
    ("pLDDT", slice(21, 22)),
    ("pLDDT bins", slice(22, 26)),
    ("PAE row stats", slice(26, 29)),
    ("Contact degree", slice(29, 31)),
    ("Position frac.", slice(31, 32)),
]

GRAPH_FEAT_LABELS = [
    "n residues",
    "n fragments",
    "mean pLDDT",
    "median pLDDT",
    "frac pLDDT<50",
    "frac 50-70",
    "frac 70-90",
    "frac >=90",
    "mean degree",
    "edge density",
    "mean PAE",
    "p90 PAE",
    "radius of gyration",
]

EDGE_FEAT_LABELS = [
    "distance_ca (A)",
    "seq separation",
    "pae_mean_pair",
    "is_seq_neighbor",
    "is_short_range",
    "is_strict_contact",
]


def load_graph(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def fig1_node_features(graph: dict, out_path: Path) -> None:
    x = graph["x"].numpy()
    n_residues = x.shape[0]
    base = x[:, :32]

    aa_index = np.argmax(base[:, :21], axis=1)
    plddt = base[:, 21]
    pae_mean = base[:, 26]
    pae_min = base[:, 27]
    pae_p90 = base[:, 28]
    contact_deg = base[:, 29]
    strict_deg = base[:, 30]
    position_frac = base[:, 31]

    fig = plt.figure(figsize=(15.5, 11.0))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.05, 0.18, 2.6],
        hspace=0.55,
    )

    ax_card = fig.add_subplot(gs[0, 0])

    example_idx = int(np.argmax(plddt))
    example_row = base[example_idx]
    aa_letter = AA_ORDER[int(np.argmax(example_row[:21]))]

    ax_card.set_xlim(0, 32)
    ax_card.set_ylim(-0.85, 2.0)
    ax_card.set_yticks([])
    ax_card.set_xticks([])
    ax_card.spines[:].set_visible(False)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    cell_norm = np.zeros(32)
    cell_norm[:21] = example_row[:21]
    cell_norm[21] = example_row[21] / 100.0
    cell_norm[22:26] = example_row[22:26]
    pae_block = example_row[26:29]
    cell_norm[26:29] = pae_block / 32.0
    deg_block = example_row[29:31]
    cell_norm[29:31] = deg_block / 25.0
    cell_norm[31] = example_row[31]

    text_values = []
    for i in range(32):
        if i < 21:
            text_values.append("1" if example_row[i] > 0.5 else "")
        elif i == 21:
            text_values.append(f"{example_row[21]:.1f}")
        elif 22 <= i <= 25:
            text_values.append("1" if example_row[i] > 0.5 else "0")
        elif 26 <= i <= 28:
            text_values.append(f"{example_row[i]:.1f}")
        elif 29 <= i <= 30:
            text_values.append(f"{int(example_row[i])}")
        else:
            text_values.append(f"{example_row[i]:.2f}")

    for i in range(32):
        rect = patches.Rectangle(
            (i, 0), 1, 1,
            facecolor=cmap(norm(cell_norm[i])),
            edgecolor="white", linewidth=0.6,
        )
        ax_card.add_patch(rect)
        if text_values[i]:
            ax_card.text(
                i + 0.5, 0.5, text_values[i],
                ha="center", va="center",
                fontsize=7,
                color="black" if cell_norm[i] < 0.55 else "white",
            )

    section_specs = [
        (0, 21, f"AA one-hot\n(amino-acid identity = '{aa_letter}')", "#1f77b4", 1.85),
        (21, 22, "pLDDT", "#2ca02c", 1.85),
        (22, 26, "pLDDT bins\n(very_low / low / confident / very_high)", "#2ca02c", 1.30),
        (26, 29, "PAE row stats\n(mean / min / p90)", "#ff7f0e", 1.85),
        (29, 31, "contact degree\n(all / strict)", "#9467bd", 1.30),
        (31, 32, "position\nfrac", "#8c564b", 1.85),
    ]
    for start, end, label, color, ytext in section_specs:
        ax_card.add_patch(patches.Rectangle(
            (start, 0), end - start, 1,
            fill=False, edgecolor=color, linewidth=2.0,
        ))
        ax_card.plot([(start + end) / 2.0, (start + end) / 2.0], [1.02, ytext - 0.05],
                     color=color, linewidth=0.8)
        ax_card.text(
            (start + end) / 2.0, ytext, label,
            ha="center", va="bottom",
            fontsize=8.5, color=color, fontweight="bold",
        )

    for i in range(21):
        ax_card.text(
            i + 0.5, -0.25, AA_ORDER[i],
            ha="center", va="top", fontsize=7, color="#666",
        )
    for i in range(21, 32):
        ax_card.text(
            i + 0.5, -0.25, str(i),
            ha="center", va="top", fontsize=7, color="#666",
        )

    ax_card.text(
        16, -0.55,
        f"One residue -> 32-dim base feature vector  "
        f"(example: residue idx {example_idx}, AA = '{aa_letter}', "
        f"pLDDT = {example_row[21]:.1f}, contact_degree = {int(example_row[29])})",
        ha="center", va="top", fontsize=10, fontweight="bold",
    )

    ax_aa = fig.add_subplot(gs[1, 0])
    ax_aa.imshow(
        aa_index.reshape(1, -1),
        aspect="auto",
        cmap="tab20",
        interpolation="nearest",
    )
    ax_aa.set_yticks([0])
    ax_aa.set_yticklabels(["AA"], fontsize=9)
    ax_aa.set_xlabel("Residue index (0 .. N-1)", fontsize=9)
    ax_aa.set_title(
        f"AA per residue (21-dim one-hot collapsed to AA index, colored by AA identity) - "
        f"protein has {n_residues} residues",
        fontsize=10,
    )
    ax_aa.tick_params(axis="x", labelsize=8)

    ax_heat = fig.add_subplot(gs[2, 0])

    rows = [
        ("pLDDT (0-100)\nAlphaFold confidence", plddt, "viridis", 0, 100),
        ("PAE row mean (A)\nlower = more confident", pae_mean, "magma_r", None, None),
        ("PAE row min (A)", pae_min, "magma_r", None, None),
        ("PAE row p90 (A)", pae_p90, "magma_r", None, None),
        ("contact degree\n(# 3D neighbors)", contact_deg, "Purples", 0, None),
        ("strict contact deg\n(close neighbors)", strict_deg, "Purples", 0, None),
        ("position fraction\n(i / sequence length)", position_frac, "Greys", 0, 1),
    ]

    for r_idx, (label, vec, cmap_name, vmin, vmax) in enumerate(rows):
        kwargs = {}
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        ax_heat.imshow(
            vec.reshape(1, -1),
            aspect="auto",
            cmap=cmap_name,
            interpolation="nearest",
            extent=[0, n_residues, r_idx + 1, r_idx],
            **kwargs,
        )
    ax_heat.set_xlim(0, n_residues)
    ax_heat.set_ylim(len(rows), 0)
    ax_heat.set_yticks([i + 0.5 for i in range(len(rows))])
    ax_heat.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax_heat.set_xlabel("Residue index (0 .. N-1)")
    ax_heat.set_title(
        f"Continuous node features along the residue axis  "
        f"(each row is one feature column; each column is one residue)",
        fontsize=11,
    )

    fig.suptitle(
        f"Stage 2: Feature Extraction - Node side  (entry {graph['entry_id']})",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_contact_map(graph: dict, out_path: Path) -> None:
    n = graph["x"].shape[0]
    edge_index = graph["edge_index"].numpy()
    edge_attr = graph["edge_attr"].numpy()

    seq_neighbor = edge_attr[:, 3] > 0.5
    strict_contact = edge_attr[:, 5] > 0.5

    contact_map = np.zeros((n, n), dtype=np.int8)
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        if strict_contact[k]:
            contact_map[i, j] = max(contact_map[i, j], 2)
        elif seq_neighbor[k]:
            contact_map[i, j] = max(contact_map[i, j], 1)
        else:
            contact_map[i, j] = max(contact_map[i, j], 1)

    cmap = mcolors.ListedColormap(["#f7f7f7", "#bdbdbd", "#d62728"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, (ax_map, ax_legend) = plt.subplots(
        1, 2, figsize=(11, 8),
        gridspec_kw={"width_ratios": [4.5, 1.0], "wspace": 0.1},
    )

    ax_map.imshow(contact_map, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
    ax_map.set_xlabel("Residue j")
    ax_map.set_ylabel("Residue i")
    ax_map.set_title(
        f"Contact map ({n} x {n})\n"
        f"{int(strict_contact.sum() // 2)} strict contacts, "
        f"{int(seq_neighbor.sum() // 2)} sequential, "
        f"{int(edge_index.shape[1] // 2)} undirected edges total",
        fontsize=11,
    )

    ax_legend.axis("off")
    legend_handles = [
        patches.Patch(facecolor="#bdbdbd", edgecolor="#888", label="sequential / weak contact\n(usually backbone neighbors)"),
        patches.Patch(facecolor="#d62728", edgecolor="#888", label="strict 3D contact\n(< distance threshold)"),
    ]
    ax_legend.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
        title="Edge type",
        title_fontsize=11,
    )
    ax_legend.text(
        0.0, 0.45,
        "Reading the map\n"
        "- diagonal: residue i ~ i+1\n"
        "  (sequential backbone)\n"
        "- off-diagonal red dots:\n"
        "  long-range 3D contacts\n"
        "  (the structural signal\n"
        "   that sequence alone\n"
        "   cannot see)",
        transform=ax_legend.transAxes,
        fontsize=9,
        va="top",
    )

    fig.suptitle(
        f"Stage 2: Feature Extraction - Edge side / contact map  (entry {graph['entry_id']})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_edges_and_graph(graph: dict, out_path: Path) -> None:
    edge_attr = graph["edge_attr"].numpy()
    distance = edge_attr[:, 0]
    seq_sep = edge_attr[:, 1]
    pae = edge_attr[:, 2]
    is_seq_neighbor = edge_attr[:, 3] > 0.5
    is_strict = edge_attr[:, 5] > 0.5

    graph_feat = graph["graph_feat"].numpy()

    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.32)

    ax_dist = fig.add_subplot(gs[0, 0])
    ax_dist.hist(distance[~is_seq_neighbor], bins=40, color="#d62728", alpha=0.75, label="non-seq edges")
    ax_dist.hist(distance[is_seq_neighbor], bins=40, color="#bdbdbd", alpha=0.75, label="sequential edges")
    ax_dist.set_xlabel("distance_ca  (Angstroms)")
    ax_dist.set_ylabel("# edges")
    ax_dist.set_title("Edge feature 0: C-alpha distance\n(threshold ~10 A built into the graph)")
    ax_dist.legend(fontsize=8)

    ax_sep = fig.add_subplot(gs[0, 1])
    ax_sep.hist(seq_sep[is_seq_neighbor], bins=40, color="#bdbdbd", alpha=0.85, label="seq neighbors (|i-j|=1)")
    ax_sep.hist(seq_sep[~is_seq_neighbor], bins=40, color="#1f77b4", alpha=0.85, label="long-range")
    ax_sep.set_xlabel("|i - j| in sequence")
    ax_sep.set_ylabel("# edges (log)")
    ax_sep.set_title("Edge feature 1: sequence separation\n(>1 = real structural contact, not backbone neighbor)")
    ax_sep.set_yscale("log")
    ax_sep.legend(fontsize=8)

    ax_pae = fig.add_subplot(gs[0, 2])
    ax_pae.hist(pae[is_strict], bins=40, color="#2ca02c", alpha=0.75, label="strict contact")
    ax_pae.hist(pae[~is_strict], bins=40, color="#ff7f0e", alpha=0.65, label="other edges")
    ax_pae.set_xlabel("pae_mean_pair  (lower = AlphaFold more confident)")
    ax_pae.set_ylabel("# edges")
    ax_pae.set_title("Edge feature 2: PAE on the residue pair\n(strict contacts cluster at low PAE)")
    ax_pae.legend(fontsize=8)

    ax_card = fig.add_subplot(gs[1, :])
    ax_card.set_xlim(-0.05, len(graph_feat) + 0.05)
    ax_card.set_ylim(-1.55, 1.55)
    ax_card.set_yticks([])
    ax_card.spines[:].set_visible(False)
    ax_card.set_xticks([])

    abs_max = max(abs(graph_feat).max(), 1.0)
    cmap = plt.cm.Purples
    for i, (v, label) in enumerate(zip(graph_feat, GRAPH_FEAT_LABELS)):
        norm_v = min(abs(v) / abs_max, 1.0)
        rect = patches.Rectangle(
            (i, 0), 1, 1,
            facecolor=cmap(0.20 + 0.55 * norm_v),
            edgecolor="white", linewidth=0.6,
        )
        ax_card.add_patch(rect)
        if abs(v) >= 100:
            value_text = f"{v:.0f}"
        elif abs(v) >= 10:
            value_text = f"{v:.1f}"
        else:
            value_text = f"{v:.3f}"
        ax_card.text(
            i + 0.5, 0.5, value_text,
            ha="center", va="center", fontsize=9, color="black",
        )
        ytext = -0.18 if i % 2 == 0 else -0.75
        ax_card.plot([i + 0.5, i + 0.5], [-0.02, ytext + 0.05],
                     color="#888", linewidth=0.6)
        ax_card.text(
            i + 0.5, ytext, label,
            ha="center", va="top", fontsize=8, color="#333",
        )

    ax_card.set_title(
        f"Graph-level feature vector graph_feat: shape (13,) - one summary vector per whole protein",
        fontsize=11, pad=10,
    )

    fig.suptitle(
        f"Stage 2: Feature Extraction - Edge feature distributions and graph-level summary  (entry {graph['entry_id']})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path,
                        default=Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache/graphs/Q8IXT2.pt"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    graph = load_graph(args.graph)

    fig1_node_features(graph, args.out / "feature_extraction_node_features.png")
    fig2_contact_map(graph, args.out / "feature_extraction_contact_map.png")
    fig3_edges_and_graph(graph, args.out / "feature_extraction_edges_and_graph.png")

    print("Wrote:")
    for name in [
        "feature_extraction_node_features.png",
        "feature_extraction_contact_map.png",
        "feature_extraction_edges_and_graph.png",
    ]:
        print(f"  {args.out / name}")


if __name__ == "__main__":
    main()
