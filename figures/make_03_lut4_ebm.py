#!/usr/bin/env python
"""
Figure 3 — A LUT4 as an energy-based model (RBM / EBM schematic).

Bipartite layout:
  LEFT  "visible (5)": 4 clamped inputs x0..x3 (lock icon) + 1 sampled output y.
  RIGHT "hidden (H)":  H=4 hidden nodes.
  Edges: every visible <-> every hidden (RBM bipartite coupling).

Reproducible, pure-plotting (no training / heavy compute).

Run:
    /Users/kamp/Documents/energy/.venv/bin/python figures/make_03_lut4_ebm.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ---------------------------------------------------------------- palette / style
WARM = "#ffb703"      # logic / +1 (inputs)
COOL = "#4cc9f0"      # prob / -1 (output)
HILITE = "#ef476f"    # highlight / positive
TEAL = "#2ec4b6"      # hidden nodes
TEXT = "#e6ebf5"      # text (light)
EDGE = "#475569"      # edges / lines (dark-mode)
GRID = "#22304d"      # dark edges / grid
SPECIAL = "#fb8500"   # special accent

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
OUT = os.path.join(HERE, "03_lut4_ebm.png")

# ---------------------------------------------------------------- geometry
# Coordinate system roughly 0..10 (x) by 0..10 (y).
VIS_X = 3.0          # x of the visible column
HID_X = 7.0          # x of the hidden column
NODE_R = 0.34        # node radius
LOCK_R = NODE_R      # lock badge sits just left of clamped nodes

# Visible nodes: 4 inputs (top) + 1 output (bottom). Spread vertically.
visible = [
    ("x0", 8.6, WARM, True),
    ("x1", 7.3, WARM, True),
    ("x2", 6.0, WARM, True),
    ("x3", 4.7, WARM, True),
    ("y", 2.6, COOL, False),   # output: sampled, not clamped
]

# Hidden nodes (H = 4), centred against the visible column.
H = 4
hid_top, hid_bot = 7.9, 3.3
hidden = []
for i in range(H):
    yy = hid_top - i * (hid_top - hid_bot) / (H - 1)
    hidden.append((f"h{i}", yy))


def draw_lock(ax, cx, cy, color=TEXT, scale=1.0):
    """Tiny padlock glyph to mark a clamped node."""
    bw, bh = 0.26 * scale, 0.20 * scale          # lock body
    body = plt.Rectangle(
        (cx - bw / 2, cy - bh / 2),
        bw,
        bh,
        facecolor=color,
        edgecolor=color,
        zorder=6,
    )
    ax.add_patch(body)
    # shackle (half-circle arc above the body)
    arc = matplotlib.patches.Arc(
        (cx, cy + bh / 2),
        0.18 * scale,
        0.22 * scale,
        angle=0,
        theta1=0,
        theta2=180,
        lw=1.8 * scale,
        edgecolor=color,
        zorder=6,
    )
    ax.add_patch(arc)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---- bipartite edges (every visible <-> every hidden), light --------------
    for _, vy, _, _ in visible:
        for _, hy in hidden:
            ax.plot(
                [VIS_X, HID_X],
                [vy, hy],
                color=EDGE,
                lw=1.0,
                zorder=1,
                solid_capstyle="round",
            )

    # ---- visible nodes --------------------------------------------------------
    for name, vy, color, clamped in visible:
        circ = Circle(
            (VIS_X, vy),
            NODE_R,
            facecolor=color,
            edgecolor="#e6ebf5",
            lw=1.6,
            zorder=5,
        )
        ax.add_patch(circ)
        ax.text(
            VIS_X,
            vy,
            name,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#0f1420",
            fontweight="bold",
            zorder=7,
        )
        if clamped:
            # lock badge + small "clamped" tag to the left
            draw_lock(ax, VIS_X - NODE_R - 0.42, vy, color=SPECIAL)
            ax.text(
                VIS_X - NODE_R - 0.72,
                vy,
                "clamped",
                ha="right",
                va="center",
                fontsize=8.5,
                color=SPECIAL,
                style="italic",
            )

    # ---- output "sampled ->" tag ---------------------------------------------
    ax.annotate(
        "sampled →",
        xy=(VIS_X + NODE_R, 2.6),
        xytext=(VIS_X + NODE_R + 0.55, 2.6),
        ha="left",
        va="center",
        fontsize=9.5,
        color=HILITE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=HILITE, lw=1.8),
    )

    # ---- hidden nodes ---------------------------------------------------------
    for name, hy in hidden:
        circ = Circle(
            (HID_X, hy),
            NODE_R,
            facecolor=TEAL,
            edgecolor="#e6ebf5",
            lw=1.6,
            zorder=5,
        )
        ax.add_patch(circ)
        ax.text(
            HID_X,
            hy,
            name,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#0f1420",
            fontweight="bold",
            zorder=7,
        )

    # ---- column headers -------------------------------------------------------
    ax.text(
        VIS_X,
        9.5,
        "visible (5)",
        ha="center",
        va="center",
        fontsize=12,
        color=TEXT,
        fontweight="bold",
    )
    ax.text(
        HID_X,
        9.5,
        "hidden (H = 4)",
        ha="center",
        va="center",
        fontsize=12,
        color=TEXT,
        fontweight="bold",
    )

    # subtle bracket-ish sub-labels for the visible split
    # (pushed well left of the "clamped" tags so nothing overlaps)
    SUBLABEL_X = 0.55
    ax.text(
        SUBLABEL_X,
        (8.6 + 4.7) / 2,
        "4 inputs",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color=WARM,
        fontweight="bold",
    )
    ax.text(
        SUBLABEL_X,
        2.6,
        "1 output",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color=COOL,
        fontweight="bold",
    )

    # ---- "self-bias (+1)" annotation (one node points to the per-node concept)-
    ax.annotate(
        "every node:\nself-bias (+1)",
        xy=(HID_X + NODE_R, hidden[0][1]),
        xytext=(HID_X + 0.95, hidden[0][1] + 0.55),
        ha="left",
        va="center",
        fontsize=8.8,
        color=TEXT,
        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.3),
    )

    # ---- flow note: clamp -> block-Gibbs -> read ------------------------------
    ax.text(
        5.0,
        0.95,
        "clamp the 4 inputs  →  block-Gibbs  →  read the output",
        ha="center",
        va="center",
        fontsize=10,
        color=TEXT,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#1a2233",
            edgecolor="#475569",
            lw=1.2,
        ),
    )

    # ---- title + caption ------------------------------------------------------
    ax.set_title(
        "A LUT4 as an energy-based model",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )
    fig.text(
        0.5,
        -0.015,
        "A gate = a tiny EBM. Inputs clamped, output (and hidden) sampled. "
        "Worst case (parity) needs 4 hidden → 9 nodes.",
        ha="center",
        va="top",
        fontsize=9.2,
        color=TEXT,
        style="italic",
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
