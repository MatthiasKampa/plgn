"""
Figure 2 — the 2-colour block-Gibbs update cycle on the Z-1 chip.

Pure-plotting illustration (no compute, no training). Three panels show an 8x8
checkerboard:
  (a) state        — both colours shown (warm/cool checker).
  (b) Phase A      — update the warm sublattice given the cool one (warm cells
                     highlighted/glowing + a sampling marker; cool cells greyed).
  (c) Phase B      — update the cool sublattice given the warm one.
A curved arrow from (c) back to (b) is labelled "repeat".

Run:  /Users/kamp/Documents/energy/.venv/bin/python figures/make_02_block_gibbs.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# ---- shared style -----------------------------------------------------------
WARM   = "#ffb703"
COOL   = "#4cc9f0"
TEAL   = "#2ec4b6"
HILITE = "#ef476f"
TEXT   = "#e6ebf5"   # text (light)
EDGE   = "#475569"   # edges / lines (dark-mode)
GRID   = "#22304d"   # dark grid
CENTER = "#fb8500"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": "#0f1420", "axes.facecolor": "#0f1420", "savefig.facecolor": "#0f1420",
    "text.color": "#e6ebf5", "axes.labelcolor": "#e6ebf5", "axes.titlecolor": "#e6ebf5",
    "axes.edgecolor": "#475569", "xtick.color": "#cbd5e1", "ytick.color": "#cbd5e1", "grid.color": "#22304d",
})

N = 8                       # 8x8 checkerboard
WARM_LIGHT = "#3a2f1a"      # inactive / dimmed warm tone
COOL_LIGHT = "#15324a"      # inactive / dimmed cool tone
GREY       = "#2a3344"      # conditioned (held-fixed) cells in a phase


def draw_board(ax, mode):
    """mode in {'state','warm','cool'}.
    'state'      -> full warm/cool checker.
    'warm'/'cool'-> highlight that colour's sublattice, grey the other.
    Returns lists of (x,y) cells for the warm and cool classes."""
    warm_cells, cool_cells = [], []
    for x in range(N):
        for y in range(N):
            is_warm = (x + y) % 2 == 0
            (warm_cells if is_warm else cool_cells).append((x, y))

            if mode == "state":
                fc = WARM if is_warm else COOL
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc,
                                       edgecolor="#0f1420", lw=0.8, zorder=1))
            else:
                active = (mode == "warm" and is_warm) or (mode == "cool" and not is_warm)
                if active:
                    fc = WARM if is_warm else COOL
                    # glow: a slightly larger soft halo behind the active cell
                    ax.add_patch(Rectangle((x - 0.06, y - 0.06), 1.12, 1.12,
                                           facecolor=HILITE, alpha=0.22, lw=0,
                                           zorder=1))
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fc,
                                           edgecolor=HILITE, lw=1.6, zorder=3))
                else:
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=GREY,
                                           edgecolor="#0f1420", lw=0.8, zorder=1))

    ax.set_xlim(-0.3, N + 0.3)
    ax.set_ylim(-0.3, N + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return warm_cells, cool_cells


def mark_sampling(ax, cells):
    """Put a small 'sampling' marker (a tilde for stochastic update) on active cells."""
    for (x, y) in cells:
        ax.text(x + 0.5, y + 0.5, "~", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#0f1420", zorder=4)
    # legend-style note inside the panel
    ax.add_patch(Circle((N - 0.55, -0.05), 0.0, fill=False))  # no-op anchor


def main():
    fig, axes = plt.subplots(1, 3, figsize=(8, 5))
    axA, axB, axC = axes

    # (a) state
    draw_board(axA, "state")
    axA.set_title("(a) state", fontsize=11, fontweight="bold", pad=8)

    # (b) Phase A: update warm given cool
    warm_cells, _ = draw_board(axB, "warm")
    mark_sampling(axB, warm_cells)
    axB.set_title("(b) Phase A: update " + r"$\bullet$" +
                  " sublattice\ngiven " + r"$\circ$",
                  fontsize=11, fontweight="bold", pad=8)

    # (c) Phase B: update cool given warm
    _, cool_cells = draw_board(axC, "cool")
    mark_sampling(axC, cool_cells)
    axC.set_title("(c) Phase B: update " + r"$\circ$" +
                  "\ngiven " + r"$\bullet$",
                  fontsize=11, fontweight="bold", pad=8)

    # legend mapping bullet/circle to colours (figure-level, top)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WARM,
               markeredgecolor="#e6ebf5", markersize=10, label=r"$\bullet$  warm sublattice  $(x+y)$ even"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COOL,
               markeredgecolor="#e6ebf5", markersize=10, label=r"$\circ$  cool sublattice  $(x+y)$ odd"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY,
               markeredgecolor="#e6ebf5", markersize=10, label="held fixed (conditioned)"),
    ]
    legend_obj = fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.96),
                            ncol=3, fontsize=8.5, frameon=False, handletextpad=0.4,
                            columnspacing=1.4)
    for t in legend_obj.get_texts():
        t.set_color("#e6ebf5")

    # curved "repeat" arrow from (c) back to (b), drawn in figure coordinates
    fig.canvas.draw()  # ensure positions are realised before transforming
    arrow = FancyArrowPatch(
        (0.82, 0.21), (0.53, 0.21),
        connectionstyle="arc3,rad=-0.40",
        arrowstyle="-|>", mutation_scale=18,
        lw=2.0, color=TEAL, transform=fig.transFigure, zorder=10,
    )
    fig.patches.append(arrow)
    # label placed just above the arrow's lowest point so it never overlaps the line
    fig.text(0.675, 0.155, "repeat", ha="center", va="center",
             fontsize=10, fontweight="bold", color=TEAL,
             bbox=dict(boxstyle="round,pad=0.15", fc="#1a2233", ec="#475569"))

    # caption under the row
    fig.text(0.5, 0.005,
             "Each colour is conditionally independent given the other  $\\rightarrow$  "
             "a whole sublattice\nupdates in parallel (the chip's native cycle).",
             ha="center", va="bottom", fontsize=9.5, color=TEXT)

    fig.suptitle("2-colour block-Gibbs sampling", fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.18, top=0.82, bottom=0.18, left=0.03, right=0.97)

    out = "figures/02_block_gibbs_2cycle.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
