"""Generate low-frequency GO-term distribution figures.

All three figures explicitly mark the `min_term_frequency = 20` cutoff that
is used by the training pipeline (see scripts/savio_train_full_graph.sh).

Inputs:
  /global/scratch/users/bensonli/cafa5_outputs/graph_cache/metadata/term_counts.json

Outputs (in figures/):
  1. go_term_frequency_long_tail.png
       Rank-vs-count log-log curves per ontology, with the freq=20 line marked.
  2. go_term_frequency_histogram_with_cutoff.png
       Log-binned histograms per ontology, with the dropped region shaded.
  3. min_freq_cutoff_sweep.png
       Sweep min_term_frequency across many candidate cutoffs and show how
       vocabulary size and annotation coverage trade off; the chosen 20 is
       highlighted on each curve.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COUNTS = Path("/global/scratch/users/bensonli/cafa5_outputs/graph_cache/metadata/term_counts.json")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

CUTOFF = 20

ASPECT_COLOR = {
    "BPO": "#1f77b4",
    "CCO": "#2ca02c",
    "MFO": "#d62728",
}
ASPECT_LABEL = {
    "BPO": "Biological Process",
    "CCO": "Cellular Component",
    "MFO": "Molecular Function",
}


def load_counts() -> dict[str, np.ndarray]:
    raw = json.load(COUNTS.open())
    return {asp: np.array(sorted(d.values(), reverse=True), dtype=int)
            for asp, d in raw.items()}


def figure_long_tail(counts: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, asp in zip(axes, ("BPO", "CCO", "MFO")):
        c = counts[asp]
        rank = np.arange(1, len(c) + 1)
        ax.loglog(rank, c, color=ASPECT_COLOR[asp], linewidth=2)

        kept = int((c >= CUTOFF).sum())
        dropped = len(c) - kept
        cutoff_rank = kept  # rank of the last-kept term
        ax.axhline(CUTOFF, color="#d62728", linestyle="--", linewidth=1.4,
                   alpha=0.85, label=f"min_term_frequency = {CUTOFF}")
        ax.axvline(cutoff_rank, color="#d62728", linestyle=":",
                   linewidth=1.0, alpha=0.7)

        ax.fill_between(rank, c, where=(c < CUTOFF), color="#d62728",
                        alpha=0.18, label=f"dropped: {dropped:,} terms")
        ax.scatter([cutoff_rank], [CUTOFF], s=70, color="#d62728",
                   zorder=5, edgecolor="white", linewidth=1.2)
        ax.annotate(
            f"kept {kept:,} terms\n→ usable label space",
            xy=(cutoff_rank, CUTOFF),
            xytext=(-130, 70), textcoords="offset points",
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9),
        )

        ax.set_title(f"{asp} — {ASPECT_LABEL[asp]}\n"
                     f"{len(c):,} unique terms total")
        ax.set_xlabel("GO term rank (most → least frequent)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8.5)

    axes[0].set_ylabel("Annotation count (log)")
    fig.suptitle(
        f"GO term frequency long-tail per ontology — most terms fall below the "
        f"min_term_frequency = {CUTOFF} cutoff",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = OUT / "go_term_frequency_long_tail.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_histogram(counts: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = np.logspace(0, np.log10(max(c.max() for c in counts.values())) + 0.05, 40)

    for ax, asp in zip(axes, ("BPO", "CCO", "MFO")):
        c = counts[asp]
        ax.hist(c, bins=bins, color=ASPECT_COLOR[asp], alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")

        kept = int((c >= CUTOFF).sum())
        dropped = len(c) - kept
        ax.axvline(CUTOFF, color="#d62728", linestyle="--", linewidth=1.6,
                   alpha=0.9)
        ax.axvspan(1, CUTOFF, color="#d62728", alpha=0.10)

        ax.text(0.04, 0.95,
                f"DROPPED (count < {CUTOFF})\n  {dropped:,} terms "
                f"({dropped / len(c):.1%})",
                transform=ax.transAxes, fontsize=9, color="#a32",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#d62728", alpha=0.85))
        ax.text(0.96, 0.95,
                f"KEPT (count ≥ {CUTOFF})\n  {kept:,} terms "
                f"({kept / len(c):.1%})",
                transform=ax.transAxes, fontsize=9, color="#173",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=ASPECT_COLOR[asp], alpha=0.85))

        ax.set_title(f"{asp} — {ASPECT_LABEL[asp]}")
        ax.set_xlabel("Annotation count per GO term (log)")
        ax.set_ylabel("Number of GO terms (log)")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Distribution of GO-term annotation counts — long tail of rare terms "
        f"removed at cutoff = {CUTOFF}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = OUT / "go_term_frequency_histogram_with_cutoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_cutoff_sweep(counts: dict[str, np.ndarray]):
    cutoffs = np.unique(np.concatenate([
        np.arange(1, 21),
        np.array([25, 30, 50, 75, 100, 150, 200, 300, 500, 1000]),
    ]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    for asp in ("BPO", "CCO", "MFO"):
        c = counts[asp]
        total_terms = len(c)
        total_anno = c.sum()
        vocab = [(c >= t).sum() for t in cutoffs]
        anno_kept_pct = [c[c >= t].sum() / total_anno * 100 for t in cutoffs]

        axes[0].plot(cutoffs, vocab, marker=".", color=ASPECT_COLOR[asp],
                     linewidth=2, label=f"{asp} (raw {total_terms:,} terms)")
        axes[1].plot(cutoffs, anno_kept_pct, marker=".",
                     color=ASPECT_COLOR[asp], linewidth=2, label=asp)

        v_at_cut = (c >= CUTOFF).sum()
        a_at_cut = c[c >= CUTOFF].sum() / total_anno * 100
        axes[0].scatter([CUTOFF], [v_at_cut], s=110, color=ASPECT_COLOR[asp],
                        edgecolor="black", linewidth=1.2, zorder=5)
        axes[0].annotate(f"  {asp}: {v_at_cut:,}", (CUTOFF, v_at_cut),
                         fontsize=8.5, color=ASPECT_COLOR[asp],
                         va="center", ha="left")
        axes[1].scatter([CUTOFF], [a_at_cut], s=110, color=ASPECT_COLOR[asp],
                        edgecolor="black", linewidth=1.2, zorder=5)
        axes[1].annotate(f"  {asp}: {a_at_cut:.1f}%", (CUTOFF, a_at_cut),
                         fontsize=8.5, color=ASPECT_COLOR[asp],
                         va="center", ha="left")

    for ax, ylabel, title in [
        (axes[0], "Vocabulary size (# GO terms kept)",
         "Vocabulary size vs min_term_frequency"),
        (axes[1], "Annotation coverage (%)",
         "Annotation coverage vs min_term_frequency"),
    ]:
        ax.axvline(CUTOFF, color="#444", linestyle="--", linewidth=1.4,
                   alpha=0.8, label=f"chosen cutoff = {CUTOFF}")
        ax.set_xscale("log")
        ax.set_xlabel("min_term_frequency cutoff (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper right" if "Vocabulary" in title else "lower left",
                  fontsize=9)

    axes[0].set_yscale("log")

    fig.suptitle(
        f"Effect of min_term_frequency on label space and annotation coverage "
        f"(chosen cutoff = {CUTOFF}: drops > 80% of rare terms but keeps > 95% "
        f"of annotations)",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    out = OUT / "min_freq_cutoff_sweep.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_pooled_distribution(counts: dict[str, np.ndarray]):
    """Minimal pooled GO-term frequency distribution with the freq=20 line."""
    pooled = np.concatenate(list(counts.values()))
    pooled.sort()
    pooled = pooled[::-1]
    rank = np.arange(1, len(pooled) + 1)
    kept = int((pooled >= CUTOFF).sum())
    dropped = len(pooled) - kept

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogy(rank, pooled, color="#1f3b66", linewidth=2)
    ax.axhline(CUTOFF, color="#d62728", linestyle="--", linewidth=1.4)

    ax.set_xlim(0, len(pooled) * 1.02)

    ax.text(0.98, 0.95,
            f"Kept: {kept:,} GO terms",
            transform=ax.transAxes,
            color="#1f3b66", fontsize=13, fontweight="bold",
            ha="right", va="top")
    ax.text(0.98, 0.87,
            f"Dropped: {dropped:,} GO terms",
            transform=ax.transAxes,
            color="#d62728", fontsize=13, fontweight="bold",
            ha="right", va="top")
    ax.text(0.98, 0.79,
            f"min_term_frequency = {CUTOFF}",
            transform=ax.transAxes,
            color="#d62728", fontsize=11,
            ha="right", va="top")

    ax.set_xlabel("GO term rank")
    ax.set_ylabel("Annotation count")
    ax.set_title("GO term frequency distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT / "go_term_frequency_distribution_pooled.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_pooled_kept_vs_dropped_bar(counts: dict[str, np.ndarray]):
    """Simple two-bar chart: how many GO terms we dropped vs kept."""
    pooled = np.concatenate(list(counts.values()))
    total = len(pooled)
    kept = int((pooled >= CUTOFF).sum())
    dropped = total - kept

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"Dropped\n(count < {CUTOFF})", f"Kept\n(count ≥ {CUTOFF})"],
        [dropped, kept],
        color=["#d62728", "#1f77b4"],
        width=0.55,
    )
    for bar, value in zip(bars, [dropped, kept]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + total * 0.012,
                f"{value:,}\n({value / total:.1%})",
                ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Number of GO terms")
    ax.set_title(f"GO terms dropped vs kept at min_term_frequency = {CUTOFF}\n"
                 f"(total = {total:,})")
    ax.set_ylim(0, total * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_yticks([])

    fig.tight_layout()
    out = OUT / "go_terms_kept_vs_dropped.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    counts = load_counts()
    figure_pooled_distribution(counts)
