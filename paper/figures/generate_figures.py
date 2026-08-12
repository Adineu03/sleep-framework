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
    # Ten-cycle run (2026-08-05, A6000): 10 cycles x 20 facts, seed 0.
    # Supersedes the earlier 3-cycle data — the advantage inverts by cycle 10.
    cycles = list(range(1, 11))
    sleep_dra_cum = [0.000, 0.0917, 0.0833, 0.0375, 0.040,
                     0.0444, 0.0452, 0.0458, 0.0444, 0.050]
    sleep_bcp = [1.179, 1.980, 2.042, 4.403, 4.570,
                 4.453, 4.592, 4.465, 4.520, 4.535]
    naive_dra_cum = [0.300, 0.1583, 0.1167, 0.1042, 0.1267,
                     0.1139, 0.0810, 0.0771, 0.0593, 0.0850]
    naive_bcp = [3.006, 3.318, 3.740, 3.100, 3.054,
                 2.559, 2.018, 2.290, 2.406, 3.363]

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
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.set_ylim(0, 0.33)

    # BCP
    ax2.plot(cycles, sleep_bcp, "o-", linewidth=2.5, markersize=10,
             color="#1f77b4", label="SLEEP-A")
    ax2.plot(cycles, naive_bcp, "s-", linewidth=2.5, markersize=10,
             color="#d62728", label="Naive LoRA")
    ax2.axhspan(1.0, 1.05, color="green", alpha=0.10, label="Preservation target")
    # Mark the crossover: SLEEP's advantage ends between cycles 3 and 4.
    ax2.axvline(3.5, color="grey", linestyle="--", linewidth=1.0)
    ax2.annotate("advantage ends", xy=(3.5, 5.15), xytext=(3.65, 5.15),
                 fontsize=8.5, color="dimgrey", va="center")
    ax2.set_xticks(cycles)
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("BCP (lower = better preservation)")
    ax2.set_title("Preservation across cycles", fontsize=11)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.set_ylim(0.5, 5.5)

    plt.suptitle("Multi-cycle: SLEEP's preservation advantage is transient, not architectural",
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


def fig_warmup_extension():
    """Warm-up extension result: recognition rises, recall does not follow.

    Data from the 2026-08-05 A6000 runs (200 facts, 300 warm-up steps).
    Gate-only is the mean of seeds 0-1; +LoRA is the mean of seeds 0-1.
    """
    conditions = ["No warm-up", "Gate only", "+ LoRA"]
    mc = [0.220, 0.238, 0.255]          # recognition (multiple choice)
    dra = [0.0017, 0.0033, 0.0033]      # free-form recall
    bcp = [1.073, 1.008, 2.736]         # preservation cost

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    x = np.arange(len(conditions))
    w = 0.36

    # Left: recognition vs recall on the same axis makes the gap visible.
    ax1.bar(x - w/2, mc, w, label="Recognition (MC)", color="#1f77b4")
    ax1.bar(x + w/2, dra, w, label="Recall (DRA)", color="#d62728")
    ax1.axhline(0.25, color="grey", linestyle="--", linewidth=0.8,
                label="MC chance (0.25)")
    ax1.axhline(0.10, color="green", linestyle=":", linewidth=1.0,
                label="Pre-registered DRA target")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylabel("Accuracy / score")
    ax1.set_title("The gap does not close", fontsize=11)
    ax1.set_ylim(0, 0.33)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Right: what the warm-up cost in base capability.
    ax2.bar(x, bcp, color=["#2ca02c" if b < 1.5 else "#ff7f0e" for b in bcp])
    ax2.axhline(1.5, color="grey", linestyle="--", linewidth=0.8,
                label="Acceptable BCP (<1.5)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel("BCP (lower = better preservation)")
    ax2.set_title("Preservation cost of the warm-up", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax2.set_ylim(0, 3.2)

    plt.suptitle("Retrieval-aware warm-up: recognition improves, recall stays at floor",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_warmup_extension.pdf"))
    plt.savefig(os.path.join(OUT, "figure_warmup_extension.png"), dpi=200)
    plt.close()
    print("Wrote figure_warmup_extension.pdf")


# ---------------------------------------------------------------------------
# Figure: The arc (Figure 1) — before/after the localisation repair
# ---------------------------------------------------------------------------

MODELS = ["Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B", "Qwen2.5-1.5B"]
C_ACCENT = "#3D52C9"
C_GOOD = "#2C7D57"
C_WARN = "#A96D12"
C_GREY = "#8A93A6"


def fig_arc():
    """Left: the diagnosis-era dissociation (in-context vs consolidated).
    Right: single-cycle trained-subset DRA after the localisation repair
    (5 seeds, mean +/- SD), with the old floor marked."""
    incontext = [0.927, 0.987, 0.760, 0.700]
    old_weights = [0.007, 0.006, 0.006, 0.007]
    repaired = [0.750, 0.512, 0.189, 0.105]
    repaired_sd = [0.033, 0.048, 0.031, 0.020]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = np.arange(len(MODELS))
    w = 0.38

    ax1.bar(x - w / 2, incontext, w, color=C_GREY, label="fact in context (ceiling)")
    ax1.bar(x + w / 2, old_weights, w, color=C_WARN,
            label="from consolidated weights")
    for i, v in enumerate(old_weights):
        ax1.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8,
                 color=C_WARN)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("-", "-\n", 1) for m in MODELS], fontsize=8.5)
    ax1.set_ylabel("Free-form recall (DRA)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Before: recognition without recall\n(original placement, 5 seeds/model)",
                  fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    ax2.bar(x, repaired, 0.55, yerr=repaired_sd, capsize=3, color=C_ACCENT)
    ax2.axhline(0.006, color=C_WARN, lw=1.2, ls="--")
    ax2.text(len(MODELS) - 0.45, 0.017, "old floor (0.006)", fontsize=8.5,
             color=C_WARN, ha="right")
    for i, (v, s) in enumerate(zip(repaired, repaired_sd)):
        ax2.text(i, v + s + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("-", "-\n", 1) for m in MODELS], fontsize=8.5)
    ax2.set_ylabel("Trained-subset DRA")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("After: the localisation repair\n(mid-MLP + paraphrases + distillation, 5 seeds/model)",
                  fontsize=10.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_arc.pdf"))
    plt.savefig(os.path.join(OUT, "figure_arc.png"), dpi=200)
    plt.close()
    print("Wrote figure_arc.pdf")


# ---------------------------------------------------------------------------
# Figure: Localisation ladder
# ---------------------------------------------------------------------------

def fig_localisation_ladder():
    """Five-arm isolation on Qwen2.5-7B: recall and damage per arm."""
    arms = [
        "attention-top,\n1 wording\n(original)",
        "mid-MLP,\n1 wording",
        "mid-MLP\n+ paraphrases",
        "mid-MLP + para.\n+ distillation",
        "attention-top\n+ distillation\n(control)",
    ]
    dra = [0.01, 0.03, 0.22, 0.36, 0.06]
    bcp = [3.6, 1.5, 1.4, 0.83, 2.9]
    colors = [C_WARN, C_GREY, C_ACCENT, C_GOOD, C_WARN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = np.arange(len(arms))

    ax1.bar(x, dra, 0.6, color=colors)
    for i, v in enumerate(dra):
        ax1.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, fontsize=7.8)
    ax1.set_ylabel("Free-form recall (DRA)")
    ax1.set_title("Recall by arm (50 facts, matched exposure)", fontsize=10.5)
    ax1.set_ylim(0, 0.42)

    ax2.bar(x, bcp, 0.6, color=colors)
    ax2.axhline(1.0, color=C_GREY, lw=1.0, ls=":")
    for i, v in enumerate(bcp):
        ax2.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arms, fontsize=7.8)
    ax2.set_ylabel("BCP (lower = better)")
    ax2.set_title("Damage by arm", fontsize=10.5)
    ax2.set_ylim(0, 4.1)

    plt.suptitle("The localisation ladder: substrate, diversity, and objective each isolated (Qwen2.5-7B)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_localisation_ladder.pdf"))
    plt.savefig(os.path.join(OUT, "figure_localisation_ladder.png"), dpi=200)
    plt.close()
    print("Wrote figure_localisation_ladder.pdf")


# ---------------------------------------------------------------------------
# Figure: Ten-cycle trajectories, three arms (from result JSONs)
# ---------------------------------------------------------------------------

def _load_traj(path):
    import json
    with open(path) as f:
        d = json.load(f)
    ev = d["eval_results"]
    return ([e["cycle"] for e in ev],
            [e["dra_cumulative"] for e in ev],
            [e["bcp"] for e in ev])


def fig_matrix_multi_cycle():
    """Qwen2.5-7B ten-cycle trajectories for the three arms, both seeds,
    loaded from the released result JSONs."""
    res = os.path.abspath(os.path.join(OUT, "../../experiments/results"))
    arms = {
        "SLEEP v2-moderate": (
            [os.path.join(res, "pod_run_2026-08-11_phase_c", f"c_v2moderate_s{s}.json") for s in (0, 1)],
            C_GOOD),
        "EWC-only (matched)": (
            [os.path.join(res, "pod_run_2026-08-12_phase_d", f"d4_qwen7b_ewc_s{s}.json") for s in (0, 1)],
            C_ACCENT),
        "naive LoRA": (
            [os.path.join(res, "pod_run_2026-08-11_phase_c", f"c_naive_s{s}.json") for s in (0, 1)],
            C_WARN),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for name, (paths, color) in arms.items():
        for i, p in enumerate(paths):
            cycles, dra, bcp = _load_traj(p)
            ax1.plot(cycles, dra, color=color, lw=1.8, alpha=0.9 if i == 0 else 0.55,
                     label=name if i == 0 else None,
                     ls="-" if i == 0 else "--")
            ax2.plot(cycles, bcp, color=color, lw=1.8, alpha=0.9 if i == 0 else 0.55,
                     ls="-" if i == 0 else "--")

    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Cumulative DRA (all facts seen so far)")
    ax1.set_title("Recall accumulates only under the repaired pipeline", fontsize=10.5)
    ax1.legend(fontsize=8.5, framealpha=0.9)
    ax1.set_xticks(range(1, 11))

    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("BCP (lower = better)")
    ax2.set_title("Damage drifts smoothly, no step-cliffs", fontsize=10.5)
    ax2.set_xticks(range(1, 11))

    plt.suptitle("Ten cycles, three arms, Qwen2.5-7B (solid = seed 0, dashed = seed 1; from released run files)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_matrix_multi_cycle.pdf"))
    plt.savefig(os.path.join(OUT, "figure_matrix_multi_cycle.png"), dpi=200)
    plt.close()
    print("Wrote figure_matrix_multi_cycle.pdf")


# ---------------------------------------------------------------------------
# Figure: Multi-cycle endpoints across the matrix + the 1.5B catastrophe
# ---------------------------------------------------------------------------

def fig_matrix_endpoints():
    """Left: cycle-10 cumulative DRA, v2 vs naive, per model (seed pairs).
    Right: cycle-10 BCP on log scale showing the small-model blow-up."""
    models = ["Qwen2.5-7B", "Mistral-7B", "Qwen2.5-1.5B", "Llama-3.1-8B"]
    v2_dra = [(0.173, 0.175), (0.173, 0.168), (0.097, 0.102), (0.102, 0.062)]
    nv_dra = [(0.053, 0.060), (0.118, 0.160), (0.038, 0.050), (0.082, 0.042)]
    v2_bcp = [(1.87, 2.24), (1.18, 1.18), (2.35, 2.49), (4.33, 9.20)]
    nv_bcp = [(1.66, 1.80), (1.71, 1.57), (24.5, 11.4), (2.24, 1.67)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = np.arange(len(models))
    w = 0.38

    ax1.bar(x - w / 2, [np.mean(v) for v in v2_dra], w, color=C_GOOD,
            label="SLEEP v2-moderate")
    ax1.bar(x + w / 2, [np.mean(v) for v in nv_dra], w, color=C_WARN,
            label="naive LoRA")
    for i, (v2, nv) in enumerate(zip(v2_dra, nv_dra)):
        ax1.scatter([i - w / 2] * 2, v2, color="black", s=9, zorder=3)
        ax1.scatter([i + w / 2] * 2, nv, color="black", s=9, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("-", "-\n", 1) for m in models], fontsize=8.5)
    ax1.set_ylabel("Cumulative DRA at cycle 10")
    ax1.set_title("Recall: v2 leads on all four models (dots = seeds)", fontsize=10.5)
    ax1.legend(fontsize=8.5, framealpha=0.9)

    ax2.bar(x - w / 2, [np.mean(v) for v in v2_bcp], w, color=C_GOOD)
    ax2.bar(x + w / 2, [np.mean(v) for v in nv_bcp], w, color=C_WARN)
    for i, (v2, nv) in enumerate(zip(v2_bcp, nv_bcp)):
        ax2.scatter([i - w / 2] * 2, v2, color="black", s=9, zorder=3)
        ax2.scatter([i + w / 2] * 2, nv, color="black", s=9, zorder=3)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color=C_GREY, lw=1.0, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("-", "-\n", 1) for m in models], fontsize=8.5)
    ax2.set_ylabel("BCP at cycle 10 (log scale)")
    ax2.set_title("Damage: naive detonates the 1.5B model;\nv2's flagged exception is Llama",
                  fontsize=10.5)
    ax2.annotate("24.5 / 11.4", xy=(2 + w / 2, 17), fontsize=8.5, ha="center",
                 color=C_WARN, fontweight="bold")

    plt.suptitle("Ten-cycle endpoints across the matrix (2 seeds per arm)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_matrix_endpoints.pdf"))
    plt.savefig(os.path.join(OUT, "figure_matrix_endpoints.png"), dpi=200)
    plt.close()
    print("Wrote figure_matrix_endpoints.pdf")


def main():
    fig_substrate_comparison()
    fig_pareto_frontier()
    fig_multi_cycle()
    fig_warmup_extension()
    fig_architecture()
    fig_arc()
    fig_localisation_ladder()
    fig_matrix_multi_cycle()
    fig_matrix_endpoints()
    print("All figures written.")


if __name__ == "__main__":
    main()
