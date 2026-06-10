"""
Figure 8 — Deep-LGN chip-native training accuracy vs step for two chip-native
methods. Honest, readable line plot (no compute, no training — the data points
are fixed measurements baked in below).

  * "whole-network CD"          (warm) — peaks (~94%) then regresses (unstable).
  * "persistent CD + centering" (cool) — stalls near chance.

Reference lines: chance (32/64) and solved (64/64).

Run:  /Users/kamp/Documents/energy/.venv/bin/python figures/make_08_curves.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'figure.facecolor': '#0f1420', 'axes.facecolor': '#0f1420', 'savefig.facecolor': '#0f1420',
    'text.color': '#e6ebf5', 'axes.labelcolor': '#e6ebf5', 'axes.titlecolor': '#e6ebf5',
    'axes.edgecolor': '#475569', 'xtick.color': '#cbd5e1', 'ytick.color': '#cbd5e1', 'grid.color': '#22304d',
})

# ---- shared style -----------------------------------------------------------
WARM   = "#ffb703"   # whole-network CD
COOL   = "#4cc9f0"   # persistent CD + centering
TEAL   = "#2ec4b6"   # accent
HILITE = "#ef476f"   # highlight / positive
TEXT   = "#e6ebf5"   # text (light on dark)
EDGE   = "#475569"   # edges / lines
GRID   = "#22304d"   # grid
SPECIAL = "#fb8500"  # special
MUTED  = "#94a3b8"   # muted light for reference lines
PANEL  = "#1a2233"   # dark fill for annotation bboxes

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
})

# --- measured points (out of 64) ---
CD = [(1, 29), (20, 32), (40, 33), (60, 38), (80, 36), (100, 53),
      (120, 60), (140, 55), (160, 52), (180, 56), (200, 54), (220, 50)]
PCD = [(1, 31), (15, 33), (30, 34), (45, 32), (60, 32), (75, 31), (90, 31),
       (105, 31), (120, 37), (135, 34), (150, 33), (165, 30), (180, 33),
       (195, 32), (210, 30), (225, 25)]

CHANCE = 32
SOLVED = 64


def main():
    cd = np.array(CD, dtype=float)
    pcd = np.array(PCD, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    # reference bands/lines first (behind the data)
    ax.axhline(SOLVED, color=MUTED, ls=":", lw=1.4, alpha=0.8, zorder=1)
    ax.axhline(CHANCE, color=MUTED, ls="--", lw=1.4, zorder=1)
    # shade the "below chance" region very lightly to anchor the eye
    ax.axhspan(20, CHANCE, color=MUTED, alpha=0.12, zorder=0)

    # data lines
    ax.plot(cd[:, 0], cd[:, 1], color=WARM, lw=2.4, marker="o", ms=5.5,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4,
            label="whole-network CD")
    ax.plot(pcd[:, 0], pcd[:, 1], color=COOL, lw=2.4, marker="s", ms=5.0,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            label="persistent CD + centering")

    # --- reference line labels (placed at the right edge) ---
    ax.text(228, SOLVED, "solved (64/64)", va="center", ha="right",
            fontsize=9, color=TEXT, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.85),
            zorder=5)
    ax.text(228, CHANCE - 0.4, "chance (32/64)", va="top", ha="right",
            fontsize=9, color=MUTED, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.85),
            zorder=5)

    # --- annotate the CD peak at (120, 60) and the subsequent regression ---
    ax.scatter([120], [60], s=130, facecolor="none", edgecolor=HILITE,
               linewidths=2.0, zorder=6)
    ax.annotate(
        "$\\approx$94% — credit assignment DID\n"
        "propagate through the latent layer",
        xy=(120, 60), xytext=(70, 70),
        fontsize=9, color=SPECIAL, fontweight="bold", ha="left", va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=EDGE, lw=0.8, alpha=0.92),
        arrowprops=dict(arrowstyle="-|>", color=SPECIAL, lw=1.6,
                        shrinkA=0, shrinkB=6),
    )

    # arrow showing the regression after the peak (120 -> ~220)
    reg = FancyArrowPatch((128, 59), (218, 51),
                          connectionstyle="arc3,rad=-0.28",
                          arrowstyle="-|>", mutation_scale=16,
                          color=HILITE, lw=1.8, zorder=6)
    ax.add_patch(reg)
    ax.text(178, 57.5, "then regresses\n(unstable)", fontsize=8.8, color=HILITE,
            ha="center", va="center", fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=EDGE, lw=0.8, alpha=0.92))

    # --- annotate the PCD stall ---
    ax.annotate(
        "this tuning stalled near chance",
        xy=(180, 33), xytext=(178, 24.0),
        fontsize=9, color=COOL, fontweight="bold", ha="center", va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=EDGE, lw=0.8, alpha=0.92),
        arrowprops=dict(arrowstyle="-|>", color=COOL, lw=1.5,
                        connectionstyle="arc3,rad=-0.2", shrinkA=2, shrinkB=4),
    )

    # axes cosmetics
    ax.set_xlim(0, 232)
    ax.set_ylim(20, 75)
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("accuracy  (out of 64)", fontsize=11)
    ax.set_xticks(range(0, 226, 25))
    ax.set_yticks(range(20, 71, 10))
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(EDGE)

    ax.set_title(
        "Deep chip-native training is still open:\n"
        "CD peaks then regresses; PCD stalls",
        fontsize=14, fontweight="bold", pad=10)

    ax.legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=0.95,
              facecolor=PANEL, edgecolor=EDGE, labelcolor=TEXT, ncol=1)

    out = "figures/08_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
