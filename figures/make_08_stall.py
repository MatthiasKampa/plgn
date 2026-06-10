#!/usr/bin/env python
"""
Figure 08 (stall) — Deep chip-native LGN training stalls.

A single honest summary bar chart: the one configuration that actually
*works* (a single LUT4 gate, clamped positive phase, exact gradient) reaches
64/64, while every deep / sampling-based training method we tried plateaus in
a tight band just above chance (~32/64).

Pure plotting: every number below is a baked-in measured literal. No training,
no heavy compute. Agg backend.

Run:
    /Users/kamp/Documents/energy/.venv/bin/python figures/make_08_stall.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1420", "axes.facecolor": "#0f1420", "savefig.facecolor": "#0f1420",
        "text.color": "#e6ebf5", "axes.labelcolor": "#e6ebf5", "axes.titlecolor": "#e6ebf5",
        "axes.edgecolor": "#475569", "xtick.color": "#cbd5e1", "ytick.color": "#cbd5e1", "grid.color": "#22304d",
    }
)

# ---------------------------------------------------------------- palette / style
AMBER = "#ffb703"   # the configuration that WORKS (single LUT4)
CYAN = "#4cc9f0"    # stalled deep / sampling methods
MUTED = "#94a3b8"   # reference lines (chance / solved)
TEXT = "#e6ebf5"    # light text
EDGE = "#475569"    # axis edges
GRID = "#22304d"    # grid
PANEL = "#1a2233"   # dark fill for annotation bboxes

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "08_training_stall.png")

# ---------------------------------------------------------------- data (best acc /64)
# (label, best accuracy out of 64, bar color, "works" flag)
BARS = [
    ("single LUT4*",            64, AMBER, True),
    ("whole-network CD",        44, CYAN,  False),
    ("persistent CD\n(tuned)",  38, CYAN,  False),
    ("MF$^{+}$ + PCD$^{-}$",    40, CYAN,  False),
    ("Glorot+curric\n+MF+PCD",  39, CYAN,  False),
]

CHANCE = 32
SOLVED = 64


def main():
    labels = [b[0] for b in BARS]
    values = np.array([b[1] for b in BARS], dtype=float)
    colors = [b[2] for b in BARS]

    x = np.arange(len(BARS))

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # --- "near-chance" band behind everything (where bars 2-5 cluster) -------
    band_lo, band_hi = CHANCE, 45
    ax.axhspan(band_lo, band_hi, color=CYAN, alpha=0.08, zorder=0)

    # --- reference lines (behind the bars) ----------------------------------
    ax.axhline(CHANCE, color=MUTED, ls="--", lw=1.5, zorder=1)
    ax.axhline(SOLVED, color=MUTED, ls=":", lw=1.5, zorder=1)

    # --- the bars -----------------------------------------------------------
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="#e6ebf5",
        lw=1.0,
        zorder=3,
    )

    # value label on top of each bar
    for b, c, (_, _, _, works) in zip(bars, colors, BARS):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.9,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=c,
            zorder=5,
        )

    # --- reference-line labels (right edge, dark bbox) ----------------------
    ax.text(
        len(BARS) - 0.5, SOLVED, "solved (64/64)",
        va="center", ha="right", fontsize=9, color=MUTED, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.9),
        zorder=6,
    )
    ax.text(
        len(BARS) - 0.5, CHANCE - 0.5, "chance (32/64)",
        va="top", ha="right", fontsize=9, color=MUTED, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.9),
        zorder=6,
    )

    # --- "near-chance" band callout (brace over bars 2-5) -------------------
    band_label_x = (x[1] + x[4]) / 2
    ax.annotate(
        "near-chance band",
        xy=(band_label_x, band_hi),
        xytext=(band_label_x, 56),
        ha="center", va="center", fontsize=10, color=CYAN, fontweight="bold",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=CYAN, lw=1.1, alpha=0.95),
        arrowprops=dict(
            arrowstyle="-[, widthB=8.4, lengthB=0.6",
            color=CYAN, lw=1.6, shrinkA=4, shrinkB=2,
        ),
    )

    # --- asterisk note near the single-LUT4 bar -----------------------------
    ax.annotate(
        "*clamped positive phase,\nno latent — exact gradient",
        xy=(x[0], 50),
        xytext=(x[0] + 0.18, 30),
        ha="left", va="center", fontsize=8.8, color=AMBER, fontweight="bold",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=AMBER, lw=1.1, alpha=0.95),
        arrowprops=dict(
            arrowstyle="-|>", color=AMBER, lw=1.6,
            connectionstyle="arc3,rad=-0.25", shrinkA=2, shrinkB=4,
        ),
    )

    # --- footnote-style diagnosis (dark bbox, light text) -------------------
    ax.text(
        0.015, 0.025,
        "Diagnosis: interior gate outputs are 2 hops from supervision "
        "$\\rightarrow$ contrastive gradient $\\approx 0$ on them  "
        "($\\langle o_A\\rangle_{+} \\approx \\langle o_A\\rangle_{-}$).  "
        "The single gate has no latent to infer.",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8.6, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=EDGE, lw=1.0, alpha=0.96),
        zorder=8,
    )

    # --- axes cosmetics -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=TEXT)
    ax.set_ylabel("best accuracy  (/64)", fontsize=11, color=TEXT)
    ax.set_ylim(0, 66)
    ax.set_yticks(range(0, 65, 8))
    ax.set_xlim(-0.6, len(BARS) - 0.4)

    ax.yaxis.grid(True, color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EDGE)
    ax.tick_params(colors="#cbd5e1")

    ax.set_title(
        "Deep chip-native LGN training stalls "
        "— every sampling method plateaus near chance",
        fontsize=13.5, fontweight="bold", color=TEXT, pad=12,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
