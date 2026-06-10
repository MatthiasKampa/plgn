#!/usr/bin/env python
"""
Figure 4 — How many hidden nodes a LUT4 needs (404 Boolean functions).

Grouped bar chart of the minimal hidden-unit count across 404 functions,
comparing RBM vs RBM+skip (a general Boltzmann machine with direct
input->output edges). Parity is the worst case.

Reproducible, pure-plotting (no training / heavy compute).

Run:
    /Users/kamp/Documents/energy/.venv/bin/python figures/make_04_capacity.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- palette / style
WARM = "#ffb703"      # RBM
COOL = "#4cc9f0"
HILITE = "#ef476f"
TEAL = "#2ec4b6"      # RBM + skip
TEXT = "#e6ebf5"      # text (light)
EDGE = "#475569"      # edges / lines (dark-mode)
GRID = "#22304d"      # dark grid
SPECIAL = "#fb8500"   # callout accent

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": "#0f1420", "axes.facecolor": "#0f1420", "savefig.facecolor": "#0f1420",
        "text.color": "#e6ebf5", "axes.labelcolor": "#e6ebf5", "axes.titlecolor": "#e6ebf5",
        "axes.edgecolor": "#475569", "xtick.color": "#cbd5e1", "ytick.color": "#cbd5e1", "grid.color": "#22304d",
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "04_lut4_capacity.png")

# ---------------------------------------------------------------- data (404 each)
H_VALUES = [0, 1, 2, 3, 4]
RBM = [0, 9, 247, 147, 1]            # sum = 404
RBM_SKIP = [9, 303, 92, 0, 0]        # sum = 404

assert sum(RBM) == 404, sum(RBM)
assert sum(RBM_SKIP) == 404, sum(RBM_SKIP)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(H_VALUES))
    width = 0.38

    bars_rbm = ax.bar(
        x - width / 2,
        RBM,
        width,
        label="RBM",
        color=WARM,
        edgecolor="#e6ebf5",
        lw=1.0,
        zorder=3,
    )
    bars_skip = ax.bar(
        x + width / 2,
        RBM_SKIP,
        width,
        label="RBM + skip",
        color=TEAL,
        edgecolor="#e6ebf5",
        lw=1.0,
        zorder=3,
    )

    # value labels on top of each (non-zero) bar
    for bars in (bars_rbm, bars_skip):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 5,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=TEXT,
                    fontweight="bold",
                )

    # ---- axes / grid ----------------------------------------------------------
    ax.set_xticks(x)
    # X tick labels carry the H->nodes mapping (nodes = 5 + H)
    ax.set_xticklabels(
        [f"H={h}\n({5 + h} nodes)" for h in H_VALUES],
        fontsize=10,
        color=TEXT,
    )
    ax.set_xlabel("minimal hidden  H   (nodes = 5 + H)", fontsize=11, color=TEXT)
    ax.set_ylabel("number of functions", fontsize=11, color=TEXT)
    ax.set_ylim(0, 360)

    ax.yaxis.grid(True, color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EDGE)
    ax.tick_params(colors="#cbd5e1")

    # ---- note: linearly-separable functions -----------------------------------
    # Placed in open space on the lower-left, arrow aimed at the H=0 skip bar.
    ax.annotate(
        "linearly-separable:\n1 hidden (RBM) · 0 (skip)",
        xy=(x[0] + width / 2, RBM_SKIP[0] + 2),     # the H=0 skip bar (9 funcs)
        xytext=(x[0] + 0.05, 120),
        ha="left",
        va="center",
        fontsize=8.8,
        color=TEAL,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>",
            color=TEAL,
            lw=1.5,
            connectionstyle="arc3,rad=0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#1a2233",
            edgecolor=TEAL,
            lw=1.1,
        ),
    )

    # ---- callout: parity = the ceiling (near H=4) -----------------------------
    ax.annotate(
        "parity = the ceiling:\nRBM needs 4 hidden (9 nodes);\n"
        "skip edges cut it to 2 (7 nodes)",
        xy=(x[4] - width / 2, RBM[4] + 2),       # the lone H=4 RBM bar
        xytext=(x[2] + 0.15, 285),
        ha="left",
        va="center",
        fontsize=9,
        color=SPECIAL,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>",
            color=SPECIAL,
            lw=1.8,
            connectionstyle="arc3,rad=-0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a2233",
            edgecolor=SPECIAL,
            lw=1.2,
        ),
    )

    # ---- legend + footer total ------------------------------------------------
    legend_obj = ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        facecolor="#1a2233",
        edgecolor="#475569",
    )
    for t in legend_obj.get_texts():
        t.set_color("#e6ebf5")
    ax.text(
        0.99,
        0.74,
        "404 functions each",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TEXT,
        style="italic",
    )

    # ---- title ----------------------------------------------------------------
    ax.set_title(
        "How many hidden nodes a LUT4 needs (404 functions)\n"
        "— parity is the worst case",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
