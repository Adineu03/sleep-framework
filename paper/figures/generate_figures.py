"""Generate all figures for the SLEEP paper.

Produces:
  - figure_substrate_comparison.pdf  (Section 6.1)
  - figure_pareto_frontier.pdf       (Section 6.5)
  - figure_multi_cycle.pdf            (Section 6.6)
  - figure_architecture.pdf           (placeholder schematic)

Run from `paper/figures/`:
    python generate_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Use a clean, publication-friendly style.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Figure: W_fast Substrate Comparison
# ---------------------------------------------------------------------------

def fig_substrate_comparison():
    conditions = [
        "LoRA W_fast\n(α=1e-4)",
        "LoRA W_fast\n(α=1e-3)",
        "KV memory\n(no gating)",
        "KV memory\n(k=16, gated)",
    ]
    tagged = [0.23, 0.24, 0.27, 0.28]
    untagged = [0.24, 0.24, 0.16, 0.12]
    bcp = [0.99, 1.17, 96.88, 1.08]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.arange(len(conditions))
    w = 0.36
    ax1.bar(x - w/2, tagged, w, label="Tagged", color="#1f77b4")
    ax1.bar(x + w/2, untagged, w, label="Untagged", color="#aec7e8")
    ax1.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance (25%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9)
    ax1.set_ylabel("Multiple-choice accuracy")
    ax1.set_title("Recognition signal by substrate", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(0, 0.35)

    # BCP — log scale because of the no-gating outlier
    ax2.bar(x, bcp, color=["#ff7f0e" if b > 1.05 else "#2ca02c" for b in bcp])
    ax2.set_yscale("log")
    ax2.axhline(1.05, color="grey", linestyle="--", linewidth=0.8, label="BCP threshold (1.05)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=9)
    ax2.set_ylabel("BCP (log scale)")
    ax2.set_title("Base capability preservation", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.suptitle("KV memory injection produces a recognition signal that LoRA does not",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_substrate_comparison.pdf"))
    plt.savefig(os.path.join(OUT, "figure_substrate_comparison.png"), dpi=200)
    plt.close()
    print("Wrote figure_substrate_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure: Pareto Frontier
# ---------------------------------------------------------------------------

def fig_pareto_frontier():
    # Single-cycle data from Table 2 (Section 6.5)
    sleep_settings = ["SLEEP\ndefault", "Mild (A)", "Moderate (B)", "Aggressive (C)"]
    sleep_dra = [0.012, 0.050, 0.067, 0.103]
    sleep_bcp = [1.29, 1.67, 2.33, 2.73]
    naive_dra = 0.275
    naive_bcp = 2.94

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    # SLEEP curve. Labels go up-and-right (the curve ascends, so that
    # quadrant is open) to avoid colliding with the line or each other.
    ax.plot(sleep_bcp, sleep_dra, "o-", color="#1f77b4", linewidth=2,
            markersize=9, label="SLEEP (sweep)", zorder=3)
    for i, lbl in enumerate(sleep_settings):
        ax.annotate(lbl.replace("\n", " "), xy=(sleep_bcp[i], sleep_dra[i]),
                    xytext=(7, 7), textcoords="offset points", fontsize=9,
                    zorder=5)
    # Naive LoRA point — label placed below-left so it clears the marker
    # and does not run into the plot frame at the top-right.
    ax.plot(naive_bcp, naive_dra, "s", color="#d62728", markersize=11,
            label="Naive LoRA", zorder=4)
    ax.annotate("Naive LoRA", xy=(naive_bcp, naive_dra),
                xytext=(-58, -6), textcoords="offset points", fontsize=9,
                zorder=5)

    # Preservation threshold band
    ax.axvspan(1.0, 1.05, color="green", alpha=0.10,
               label="Preservation OK (BCP < 1.05)")
    ax.axhline(0.05, color="grey", linestyle=":", linewidth=0.8,
               label="Useful recall threshold (0.05)")

    ax.set_xlabel("BCP (lower = better preservation)")
    ax.set_ylabel("DRA (higher = better recall)")
    ax.set_title("Single-cycle stability–plasticity Pareto frontier",
                 fontsize=12, pad=12)
    ax.set_xlim(0.95, 3.30)
    ax.set_ylim(-0.02, 0.34)
    ax.grid(True, alpha=0.3, linestyle=":")
    # Upper-left quadrant (low BCP, high DRA) holds no data points, so the
    # legend sits there without overlapping the curve or annotations.
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_pareto_frontier.pdf"))
    plt.savefig(os.path.join(OUT, "figure_pareto_frontier.png"), dpi=200)
    plt.close()
    print("Wrote figure_pareto_frontier.pdf")


# ---------------------------------------------------------------------------
# Figure: Multi-Cycle Continual Learning
# ---------------------------------------------------------------------------

def fig_multi_cycle():
    cycles = [1, 2, 3]
    sleep_dra_cum = [0.010, 0.052, 0.058]
    sleep_bcp = [1.10, 2.26, 2.31]
    naive_dra_cum = [0.129, 0.080, 0.112]
    naive_bcp = [3.84, 3.42, 4.74]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # DRA cumulative
    ax1.plot(cycles, sleep_dra_cum, "o-", linewidth=2.5, markersize=10,
             color="#1f77b4", label="SLEEP-A")
    ax1.plot(cycles, naive_dra_cum, "s-", linewidth=2.5, markersize=10,
             color="#d62728", label="Naive LoRA")
    ax1.set_xticks(cycles)
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("DRA on cumulative facts")
    ax1.set_title("Recall (cumulative)", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_ylim(0, 0.18)

    # BCP
    ax2.plot(cycles, sleep_bcp, "o-", linewidth=2.5, markersize=10,
             color="#1f77b4", label="SLEEP-A")
    ax2.plot(cycles, naive_bcp, "s-", linewidth=2.5, markersize=10,
             color="#d62728", label="Naive LoRA")
    ax2.axhspan(1.0, 1.05, color="green", alpha=0.10, label="Preservation target")
    ax2.set_xticks(cycles)
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("BCP (lower = better preservation)")
    ax2.set_title("Preservation across cycles", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_ylim(0.5, 5.5)

    plt.suptitle("Multi-cycle: SLEEP preserves BCP ~2x better than naive LoRA at every cycle",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_multi_cycle.pdf"))
    plt.savefig(os.path.join(OUT, "figure_multi_cycle.png"), dpi=200)
    plt.close()
    print("Wrote figure_multi_cycle.pdf")


# ---------------------------------------------------------------------------
# Architecture schematic (placeholder — TikZ would be cleaner; this is a stub)
# ---------------------------------------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    # Boxes
    boxes = [
        # (x, y, w, h, text, color)
        (0.05, 0.65, 0.18, 0.20, "Input tokens\n+ surprise", "#cee5f4"),
        (0.30, 0.65, 0.18, 0.20, "Tagging Layer\n(Q1.1-1.6)", "#1f77b4"),
        (0.55, 0.78, 0.18, 0.13, "PRP Allocator\n(Q2.1-2.5)", "#9467bd"),
        (0.55, 0.55, 0.18, 0.18, "KV Memory Bank\n(W_fast)", "#ff7f0e"),
        (0.80, 0.55, 0.18, 0.36,
         "Sleep Engine\n(Q4.1-4.6)\n\nGenerate -> QC ->\nTrain W_cons ->\nValidate -> Cleanup",
         "#2ca02c"),
        (0.55, 0.15, 0.43, 0.20, "Frozen base model W_slow + W_cons (LoRA)",
         "#bbbbbb"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor="black", linewidth=1.0, alpha=0.85))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=9, color="black")

    # Arrows
    arrows = [
        # (x1, y1, x2, y2)
        (0.23, 0.75, 0.30, 0.75),     # input -> tagging
        (0.48, 0.78, 0.55, 0.84),     # tagging -> prp
        (0.48, 0.72, 0.55, 0.65),     # tagging -> kv
        (0.73, 0.84, 0.80, 0.80),     # prp -> sleep
        (0.73, 0.65, 0.80, 0.70),     # kv -> sleep
        (0.89, 0.55, 0.80, 0.35),     # sleep -> base (cons update)
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    ax.text(0.5, 0.98, "SLEEP architecture: wake (left/middle) and sleep (right)",
            ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(OUT, "figure_architecture.pdf"))
    plt.savefig(os.path.join(OUT, "figure_architecture.png"), dpi=200)
    plt.close()
    print("Wrote figure_architecture.pdf")


def main():
    fig_substrate_comparison()
    fig_pareto_frontier()
    fig_multi_cycle()
    fig_architecture()
    print("All figures written.")


if __name__ == "__main__":
    main()
