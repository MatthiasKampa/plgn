"""
Figure 7 — One thermodynamic fabric hosting BOTH probabilistic compute and
learned logic, the difference being temperature (the "sharpness" knob beta).

Conceptual / pure-plotting illustration (no compute, no training). A single
chip rectangle is partitioned into regions:
  * LOGIC regions        — cold beta (sharp LUTs, deterministic gates), warm-shaded.
  * PROBABILISTIC regions — warm beta (sampling, inference, priors), cool-shaded.
A few wires connect regions (mixed circuits crossing the temperature gradient).
A horizontal "temperature beta / sharpness" colorbar runs from cold/logic
(left, warm colour) to warm/probabilistic (right, cool colour) to convey
"same substrate, one knob".

Run:  /Users/kamp/Documents/energy/.venv/bin/python figures/make_07_fabric.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.cm as cm

plt.rcParams.update({
    'figure.facecolor': '#0f1420', 'axes.facecolor': '#0f1420', 'savefig.facecolor': '#0f1420',
    'text.color': '#e6ebf5', 'axes.labelcolor': '#e6ebf5', 'axes.titlecolor': '#e6ebf5',
    'axes.edgecolor': '#475569', 'xtick.color': '#cbd5e1', 'ytick.color': '#cbd5e1', 'grid.color': '#22304d',
})

# ---- shared style -----------------------------------------------------------
WARM   = "#ffb703"   # logic / cold beta / +1
COOL   = "#4cc9f0"   # probabilistic / warm beta / -1
TEAL   = "#2ec4b6"   # accent
HILITE = "#ef476f"   # highlight / positive
TEXT   = "#e6ebf5"   # text (light on dark)
EDGE   = "#475569"   # edges / lines
GRID   = "#22304d"   # grid
SPECIAL = "#fb8500"  # special
PANEL  = "#1a2233"   # dark fill for bbox panels

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
})

# A cold(warm-colour) -> warm(cool-colour) colormap for the "beta / sharpness" axis.
# Midpoint is a muted slate so the gradient reads warm -> neutral -> cool on dark.
FABRIC_CMAP = LinearSegmentedColormap.from_list(
    "fabric", [(0.0, WARM), (0.5, "#64748b"), (1.0, COOL)]
)


# Region layout on a 0..10 (x) by 0..10 (y) chip rectangle.
# kind: "logic" (cold beta, warm shade) or "prob" (warm beta, cool shade).
REGIONS = [
    {"xy": (0.55, 5.45), "w": 3.7, "h": 3.7, "kind": "logic",
     "title": "LOGIC", "sub": "cold $\\beta$\nsharp LUTs\ndeterministic gates"},
    {"xy": (0.55, 0.95), "w": 3.7, "h": 3.7, "kind": "logic",
     "title": "LOGIC", "sub": "cold $\\beta$\ncombinational\ndecode / control"},
    {"xy": (5.75, 5.45), "w": 3.7, "h": 3.7, "kind": "prob",
     "title": "PROBABILISTIC", "sub": "warm $\\beta$\nsampling\ninference"},
    {"xy": (5.75, 0.95), "w": 3.7, "h": 3.7, "kind": "prob",
     "title": "PROBABILISTIC", "sub": "warm $\\beta$\nlearned priors\nstochastic features"},
]

# A wire = a list of waypoints (x,y) in chip coords. These cross between regions
# (mixed circuits: logic feeding samplers, posteriors feeding gates, etc.).
WIRES = [
    [(4.05, 7.3), (4.9, 7.3), (4.9, 6.9), (5.95, 6.9)],   # top logic -> top prob
    [(4.05, 6.2), (5.0, 6.2), (5.0, 2.7), (5.95, 2.7)],   # top logic -> bottom prob
    [(4.05, 2.5), (4.9, 2.5), (4.9, 2.2), (5.95, 2.2)],   # bottom logic -> bottom prob
    [(5.95, 6.0), (5.0, 6.0), (5.0, 3.0), (4.05, 3.0)],   # top prob -> bottom logic (feedback)
    [(2.4, 5.35), (2.4, 4.85)],                            # logic <-> logic (intra)
    [(7.6, 5.35), (7.6, 4.85)],                            # prob  <-> prob (intra)
]


def draw_region(ax, r):
    cold = (r["kind"] == "logic")
    # semi-transparent brightened fill + saturated border, rounded "tile" look
    edge = WARM if cold else COOL
    fill = to_rgba(edge, 0.16)
    x, y = r["xy"]
    box = FancyBboxPatch(
        (x, y), r["w"], r["h"],
        boxstyle="round,pad=0.0,rounding_size=0.28",
        linewidth=2.2, edgecolor=edge, facecolor=fill, zorder=2,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)

    cx, cy = x + r["w"] / 2.0, y + r["h"] / 2.0

    # faint pbit lattice inside the tile to read as "same grid everywhere"
    nx, ny = 7, 7
    gx = np.linspace(x + 0.35, x + r["w"] - 0.35, nx)
    gy = np.linspace(y + 0.35, y + r["h"] - 0.35, ny)
    GX, GY = np.meshgrid(gx, gy)
    ax.scatter(GX.ravel(), GY.ravel(), s=7, color=edge, alpha=0.30,
               zorder=3, linewidths=0)

    ax.text(cx, y + r["h"] - 0.46, r["title"], ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=edge, zorder=5)
    ax.text(cx, cy - 0.35, r["sub"], ha="center", va="center",
            fontsize=8.6, color=TEXT, zorder=5, linespacing=1.35)


def draw_wire(ax, pts):
    pts = np.array(pts, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], color="#94a3b8", lw=1.4, alpha=0.9,
            solid_capstyle="round", zorder=4)
    # small contact pads at the endpoints
    for (px, py) in (pts[0], pts[-1]):
        ax.add_patch(Circle((px, py), 0.085, facecolor=HILITE,
                            edgecolor="#e6ebf5", lw=0.8, zorder=6))


def main():
    fig = plt.figure(figsize=(8, 5.4))
    # main chip axis on top, slim colorbar-style "beta knob" axis underneath
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.58])   # [left, bottom, w, h]
    cax = fig.add_axes([0.16, 0.135, 0.68, 0.045])

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # outer chip / fabric boundary with a soft horizontal temperature gradient
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(0.2, 9.8, 0.6, 9.55), aspect="auto",
              cmap=FABRIC_CMAP, alpha=0.16, zorder=0)
    chip = FancyBboxPatch(
        (0.2, 0.6), 9.6, 8.95,
        boxstyle="round,pad=0.0,rounding_size=0.35",
        linewidth=2.6, edgecolor="#64748b", facecolor="none", zorder=1,
    )
    ax.add_patch(chip)
    ax.text(5.0, 9.33, "one thermodynamic fabric  (single pbit grid, one EBM)",
            ha="center", va="center", fontsize=9.3, style="italic",
            color=TEXT, zorder=5)

    for r in REGIONS:
        draw_region(ax, r)
    for w in WIRES:
        draw_wire(ax, w)

    # label the inter-region wires once
    ax.text(5.0, 4.05, "mixed circuits\n(wires cross regions)", ha="center",
            va="center", fontsize=8.2, color=TEXT, zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL,
                      edgecolor=EDGE, alpha=0.92))

    ax.set_title("One fabric: probabilistic compute + learned logic",
                 fontsize=14, fontweight="bold", pad=12)

    # ---- the "one knob" temperature / sharpness colorbar ----
    sm = cm.ScalarMappable(cmap=FABRIC_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(EDGE)
    cb.outline.set_linewidth(1.0)
    cax.set_xticks([0.0, 1.0])
    cax.set_xticklabels(["cold $\\beta$ (high)\nsharp $\\to$ LOGIC",
                         "warm $\\beta$ (low)\nsoft $\\to$ PROBABILISTIC"],
                        fontsize=8.6)
    cax.tick_params(length=0)
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.set_title("temperature  $\\beta$  /  sharpness  —  same substrate, one knob",
                  fontsize=9.6, fontweight="bold", pad=6)

    # caption under the colorbar
    fig.text(0.5, 0.028,
             "Logic and probabilistic compute are the same EBM at different "
             "temperatures — one grid, one contrastive trainer, FPGA-orchestrated.",
             ha="center", va="center", fontsize=9.0, color=TEXT, style="italic")

    out = "figures/07_combined_fabric.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
