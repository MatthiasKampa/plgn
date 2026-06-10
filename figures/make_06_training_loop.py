"""
Figure 6 — Chip-native training: a contrastive gradient from sampling.

Pure-plotting illustration (no compute, no training). A loop diagram of
chip-native EBM training -- there is NO off-chip backprop. Two big rounded
boxes and a cycle of arrows between them:

  * "Z-1 TSU (samples)"  -- the thermodynamic sampling chip:
        FREE phase:    clamp inputs           -> block-Gibbs -> <s_i s_j>_-
        CLAMPED phase: clamp inputs + target  -> block-Gibbs -> <s_i s_j>_+
  * "FPGA (manages)"     -- the orchestrator:
        dW = <s_i s_j>_+ - <s_i s_j>_-
        update couplings
        persist negative chain

  Cycle:  FPGA sets clamps -> chip samples both phases -> chip returns the two
  correlation matrices -> FPGA computes dW and writes the new weights back ->
  repeat.

  Callout banner: the gradient IS the sampled correlations -- the exact EBM
  maximum-likelihood gradient -- so no autodiff is needed anywhere.

Run:  /Users/kamp/Documents/energy/.venv/bin/python figures/make_06_training_loop.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'figure.facecolor': '#0f1420', 'axes.facecolor': '#0f1420', 'savefig.facecolor': '#0f1420',
    'text.color': '#e6ebf5', 'axes.labelcolor': '#e6ebf5', 'axes.titlecolor': '#e6ebf5',
    'axes.edgecolor': '#475569', 'xtick.color': '#cbd5e1', 'ytick.color': '#cbd5e1', 'grid.color': '#22304d',
})

# ---- shared style -----------------------------------------------------------
WARM   = "#ffb703"   # logic / +1 / chip
COOL   = "#4cc9f0"   # prob / -1
TEAL   = "#2ec4b6"   # accent / FPGA
HILITE = "#ef476f"   # highlight / positive
TEXT   = "#e6ebf5"   # text (light on dark)
EDGE   = "#475569"   # edges / lines
GRID   = "#22304d"   # grid
CENTER = "#fb8500"   # special
PANEL  = "#1a2233"   # dark fill for sub-step panels

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
})

# semi-transparent tints for the big boxes (warm chip / teal fpga) over dark
WARM_FILL = WARM
TEAL_FILL = TEAL


def big_box(ax, x, y, w, h, title, subtitle, edge, fill):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor=to_rgba(fill, 0.16), edgecolor=edge, linewidth=2.4, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=edge, zorder=4)
    ax.text(x + w / 2, y + h - 0.64, subtitle, ha="center", va="center",
            fontsize=8.6, style="italic", color=edge, zorder=4)


def sub_step(ax, x, y, w, h, head, body, edge):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=PANEL, edgecolor=edge, linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x + 0.18, y + h - 0.27, head, ha="left", va="center",
            fontsize=9.6, fontweight="bold", color=edge, zorder=4)
    ax.text(x + 0.18, y + h - 0.62, body, ha="left", va="center",
            fontsize=9.0, color=TEXT, zorder=4)


def step_line(ax, x, y, body, edge):
    """One bulleted FPGA step (head-less)."""
    ax.text(x, y, body, ha="left", va="center", fontsize=9.4, color=TEXT, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # =====================================================================
    # Big boxes:  chip on the LEFT, fpga on the RIGHT
    # =====================================================================
    chip_x, chip_y, chip_w, chip_h = 0.30, 0.55, 4.05, 4.30
    fpga_x, fpga_y, fpga_w, fpga_h = 5.65, 0.85, 3.95, 3.70

    big_box(ax, chip_x, chip_y, chip_w, chip_h,
            "Z-1 TSU", "(samples)", WARM, WARM_FILL)
    big_box(ax, fpga_x, fpga_y, fpga_w, fpga_h,
            "FPGA", "(manages)", TEAL, TEAL_FILL)

    # ---- chip sub-steps: FREE phase (top) and CLAMPED phase (bottom) ----
    sub_step(ax, chip_x + 0.25, 2.78, chip_w - 0.50, 1.05,
             "FREE phase",
             "clamp inputs $\\rightarrow$ block-Gibbs", WARM)
    ax.text(chip_x + 0.43, 2.97,
            "$\\rightarrow\\ \\langle s_i s_j\\rangle_{-}$",
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=COOL, zorder=4)

    sub_step(ax, chip_x + 0.25, 1.05, chip_w - 0.50, 1.05,
             "CLAMPED phase",
             "clamp inputs $+$ target", WARM)
    ax.text(chip_x + 0.43, 1.24,
            "$\\rightarrow\\ \\langle s_i s_j\\rangle_{+}$",
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=HILITE, zorder=4)

    # ---- fpga sub-steps -------------------------------------------------
    step_line(ax, fpga_x + 0.30, 3.42,
              "$\\Delta W = \\langle s_i s_j\\rangle_{+} - \\langle s_i s_j\\rangle_{-}$", TEAL)
    # small color cues on the +/- terms
    step_line(ax, fpga_x + 0.30, 2.78, "$\\bullet$  update couplings  $W \\leftarrow W + \\eta\\,\\Delta W$", TEAL)
    step_line(ax, fpga_x + 0.30, 2.20, "$\\bullet$  persist negative chain", TEAL)
    # thin divider under the dW formula
    ax.plot([fpga_x + 0.30, fpga_x + fpga_w - 0.30], [3.10, 3.10],
            color=TEAL, lw=1.0, alpha=0.5, zorder=3)

    # =====================================================================
    # Cycle arrows between the two boxes
    # =====================================================================
    # TOP arrow: FPGA -> chip  (sets clamps / writes weights back)
    a_top = FancyArrowPatch((fpga_x + 0.10, fpga_y + fpga_h - 0.35),
                            (chip_x + chip_w - 0.10, chip_y + chip_h - 0.35),
                            connectionstyle="arc3,rad=-0.28",
                            arrowstyle="-|>", mutation_scale=20,
                            lw=2.2, color=TEAL, zorder=5)
    ax.add_patch(a_top)
    ax.text(4.95, 5.18,
            "sets clamps  +  writes new $W$ back",
            ha="center", va="center", fontsize=9.0, color=TEAL, fontweight="bold",
            zorder=6)

    # BOTTOM arrow: chip -> FPGA  (returns the two correlation matrices)
    a_bot = FancyArrowPatch((chip_x + chip_w - 0.10, chip_y + 0.35),
                            (fpga_x + 0.10, fpga_y + 0.35),
                            connectionstyle="arc3,rad=-0.28",
                            arrowstyle="-|>", mutation_scale=20,
                            lw=2.2, color=WARM, zorder=5)
    ax.add_patch(a_bot)
    ax.text(4.95, 0.30,
            "returns correlations  $\\langle s_i s_j\\rangle_{\\pm}$",
            ha="center", va="center", fontsize=9.0, color=WARM, fontweight="bold",
            zorder=6)

    # tiny "repeat" loop glyph near the bottom arrow
    ax.text(4.95, 0.70, "$\\circlearrowleft$ repeat", ha="center", va="center",
            fontsize=10.5, color=TEXT, zorder=6)

    # =====================================================================
    # Callout banner
    # =====================================================================
    banner = FancyBboxPatch((0.30, -0.78), 9.30, 0.78,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=HILITE, edgecolor="white", linewidth=1.4,
                            zorder=7)
    ax.add_patch(banner)
    ax.text(4.95, -0.39,
            "No off-chip backprop  —  the gradient IS sampled correlations\n"
            "(the exact EBM maximum-likelihood gradient)",
            ha="center", va="center", fontsize=10.3, fontweight="bold",
            color="white", zorder=8)

    # =====================================================================
    ax.set_xlim(0.0, 9.9)
    ax.set_ylim(-0.95, 5.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Chip-native training: a contrastive gradient from sampling",
                 fontsize=14, fontweight="bold", pad=10)

    out = "figures/06_chip_native_training.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
