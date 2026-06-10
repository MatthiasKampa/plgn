"""
Figure 5 — A logic gate network = one bipartite deep EBM.

Pure-plotting illustration (no compute, no training). A 2-layer Logic Gate
Network is drawn as ONE bipartite deep energy-based model, laid out bottom->top:

  * 6 clamped input units  x0..x5 (locked to the data).
  * Gate A: hidden block hidA + a *latent* output unit oA, with oA depending on
    x0,x1,x2   (edges x0,x1,x2 <-> hidA, and oA <-> hidA).
  * Gate B: hidden block hidB + latent output oB, depending on x3,x4,x5.
  * Gate C: hidden block hidC + output unit oC, with oC depending on oA,oB
    (edges oA,oB <-> hidC, oC <-> hidC).

The interior outputs oA, oB are LATENT (drawn dashed, "no target" badge): they
get no supervision. Only oC is the trained output / target.

The whole graph is bipartite -- one part is the *units* {x*, o*}, the other is
the *hidden* blocks -- so it still runs as the chip's 2-colour block-Gibbs cycle.

Run:  /Users/kamp/Documents/energy/.venv/bin/python figures/make_05_lgn.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({
    'figure.facecolor': '#0f1420', 'axes.facecolor': '#0f1420', 'savefig.facecolor': '#0f1420',
    'text.color': '#e6ebf5', 'axes.labelcolor': '#e6ebf5', 'axes.titlecolor': '#e6ebf5',
    'axes.edgecolor': '#475569', 'xtick.color': '#cbd5e1', 'ytick.color': '#cbd5e1', 'grid.color': '#22304d',
})

# ---- shared style -----------------------------------------------------------
WARM   = "#ffb703"   # logic / +1 / units
COOL   = "#4cc9f0"   # prob / -1 / hidden
TEAL   = "#2ec4b6"   # accent
HILITE = "#ef476f"   # highlight / positive / output target
TEXT   = "#e6ebf5"   # text (light on dark)
EDGE   = "#475569"   # edges / lines
GRID   = "#22304d"   # grid / gate boxes
CENTER = "#fb8500"   # special / latent
PANEL  = "#1a2233"   # dark fill for nodes / bbox panels

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
})

# node radii
R_UNIT = 0.34   # input / output units (the bipartite "unit" part)
R_OUT  = 0.40   # output units (slightly bigger)


def unit_node(ax, x, y, label, color, dashed=False, r=R_UNIT, z=4):
    """A circular *unit* node. dashed => latent (no target)."""
    ls = (0, (4, 3)) if dashed else "solid"
    c = Circle((x, y), r, facecolor=PANEL, edgecolor=color,
               linewidth=2.2 if dashed else 1.8, linestyle=ls, zorder=z)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=color, zorder=z + 1)
    return (x, y, r)


def hidden_block(ax, x, y, label, w=1.55, h=0.74, z=3):
    """A rounded rectangle standing for a *block* of hidden pbits."""
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.16",
                         facecolor=COOL, edgecolor=COOL, linewidth=1.8, alpha=0.22, zorder=z)
    ax.add_patch(box)
    ax.text(x, y + 0.10, label, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=TEXT, zorder=z + 1)
    ax.text(x, y - 0.17, "hidden pbits", ha="center", va="center",
            fontsize=7.6, color=COOL, zorder=z + 1)
    return (x, y, w, h)


def edge(ax, x0, y0, x1, y1, z=2):
    ax.plot([x0, x1], [y0, y1], color=EDGE, lw=1.15, alpha=0.75, zorder=z,
            solid_capstyle="round")


def gate_box(ax, x0, y0, x1, y1, label, accent=GRID):
    """A dashed group box around one gate, with a corner label."""
    box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                         boxstyle="round,pad=0.02,rounding_size=0.22",
                         facecolor="none", edgecolor=accent, linewidth=1.6,
                         linestyle=(0, (6, 4)), zorder=1)
    ax.add_patch(box)
    ax.text(x0 + 0.16, y1 - 0.22, label, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=accent, zorder=6)


def latent_badge(ax, x, y, text="latent — no target"):
    ax.annotate(text, xy=(x, y), xytext=(x, y),
                ha="center", va="center", fontsize=8.0, color="white",
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=CENTER,
                          edgecolor="white", linewidth=1.0))


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # ----- coordinates -------------------------------------------------------
    # bottom row: 6 inputs.  Two triples feeding gates A and B.
    y_in = 0.55
    xin = [0.7, 1.7, 2.7, 5.3, 6.3, 7.3]
    inp_lbl = [f"x{i}" for i in range(6)]

    # hidden blocks for gates A,B (layer 1)
    y_h1 = 2.35
    hidA = (1.7, y_h1)
    hidB = (6.3, y_h1)

    # latent outputs oA, oB (top of layer 1)
    y_o1 = 3.65
    oA = (1.7, y_o1)
    oB = (6.3, y_o1)

    # gate C (layer 2): hidden block + output oC
    y_hC = 4.65
    hidC = (4.0, y_hC)
    y_oC = 5.95
    oC = (4.0, y_oC)

    # ----- gate group boxes (draw first, behind everything) ------------------
    gate_box(ax, 0.18, 0.05, 3.22, 4.20, "gate A", accent=WARM)
    gate_box(ax, 4.78, 0.05, 7.82, 4.20, "gate B", accent=WARM)
    gate_box(ax, 2.55, 4.18, 5.45, 6.45, "gate C", accent=HILITE)

    # ----- edges (under nodes) ----------------------------------------------
    # gate A: x0,x1,x2 <-> hidA ; oA <-> hidA
    for i in (0, 1, 2):
        edge(ax, xin[i], y_in, *hidA)
    edge(ax, *oA, *hidA)
    # gate B: x3,x4,x5 <-> hidB ; oB <-> hidB
    for i in (3, 4, 5):
        edge(ax, xin[i], y_in, *hidB)
    edge(ax, *oB, *hidB)
    # gate C: oA,oB <-> hidC ; oC <-> hidC
    edge(ax, *oA, *hidC)
    edge(ax, *oB, *hidC)
    edge(ax, *oC, *hidC)

    # ----- nodes -------------------------------------------------------------
    # inputs (clamped)
    for x, lbl in zip(xin, inp_lbl):
        unit_node(ax, x, y_in, lbl, WARM)
    # hidden blocks
    hidden_block(ax, *hidA, "hidA")
    hidden_block(ax, *hidB, "hidB")
    hidden_block(ax, *hidC, "hidC")
    # latent outputs (dashed)
    unit_node(ax, *oA, "oA", CENTER, dashed=True)
    unit_node(ax, *oB, "oB", CENTER, dashed=True)
    # trained output
    unit_node(ax, *oC, "oC", HILITE, r=R_OUT)

    # ----- annotations -------------------------------------------------------
    # "clamped" bracket under the inputs
    ax.annotate("", xy=(xin[0] - 0.45, y_in - 0.62), xytext=(xin[-1] + 0.45, y_in - 0.62),
                arrowprops=dict(arrowstyle="-", color=WARM, lw=1.6))
    ax.text(4.0, y_in - 0.92,
            "[ clamped ]  6 input units locked to the data",
            ha="center", va="center", fontsize=9.5, color=WARM, fontweight="bold")

    # latent badges below oA, oB (in the gap above the hidden blocks, outer flank
    # -- clear of the vertical o<->hid edges and of the node glyphs)
    badge_yA = (oA[1] + hidA[1]) / 2 + 0.05   # midway between oA and hidA
    latent_badge(ax, 0.62, badge_yA)
    latent_badge(ax, 7.38, badge_yA)
    # short connectors from each badge up to its latent node's lower rim
    ax.annotate("", xy=(oA[0] - 0.20, oA[1] - R_UNIT + 0.02), xytext=(0.95, badge_yA + 0.22),
                arrowprops=dict(arrowstyle="-", color=CENTER, lw=1.2))
    ax.annotate("", xy=(oB[0] + 0.20, oB[1] - R_UNIT + 0.02), xytext=(7.05, badge_yA + 0.22),
                arrowprops=dict(arrowstyle="-", color=CENTER, lw=1.2))

    # output / target badge to the LEFT of oC, dropped slightly below its centre
    # so it clears the "gate C" corner label; keeps the top-right free for the
    # bipartite note.
    ax.annotate("output / target",
                xy=(oC[0] - R_OUT * 0.72, oC[1] - R_OUT * 0.72),
                xytext=(oC[0] - 1.10, oC[1] - 0.34),
                ha="right", va="center", fontsize=8.6, color="white", fontweight="bold",
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=HILITE,
                          edgecolor="white", linewidth=1.0),
                arrowprops=dict(arrowstyle="-", color=HILITE, lw=1.2))

    # bipartite note (top-right corner, clear of gate C and the output badge)
    ax.text(0.985, 0.985,
            "bipartite:  units $\\{x_*,\\,o_*\\}$  vs  hidden blocks\n"
            "$\\Rightarrow$ still a 2-colour block-Gibbs cycle",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL,
                      edgecolor=EDGE, linewidth=1.0, alpha=0.95))

    # ----- legend ------------------------------------------------------------
    handles = [
        Line2D([0], [0], marker="o", color=PANEL, markerfacecolor=PANEL,
               markeredgecolor=WARM, markeredgewidth=1.8, markersize=12,
               label="clamped input unit"),
        Line2D([0], [0], marker="s", color=PANEL, markerfacecolor=COOL,
               markeredgecolor=COOL, markeredgewidth=1.8, markersize=12,
               alpha=0.55, label="hidden pbit block"),
        Line2D([0], [0], marker="o", color=PANEL, markerfacecolor=PANEL,
               markeredgecolor=CENTER, markeredgewidth=2.2, markersize=12,
               linestyle="none", label="latent output (no target)"),
        Line2D([0], [0], marker="o", color=PANEL, markerfacecolor=PANEL,
               markeredgecolor=HILITE, markeredgewidth=1.8, markersize=13,
               label="trained output / target"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.205),
              fontsize=8.6, frameon=False, ncol=4, handletextpad=0.4,
              columnspacing=1.3)

    # ----- caption -----------------------------------------------------------
    fig.text(0.5, -0.075,
             "Task  $f = \\mathrm{MAJ3}(x_0,x_1,x_2)\\ \\oplus\\ \\mathrm{MAJ3}(x_3,x_4,x_5)$: "
             "needs composition (a single LUT4 can't);\n"
             "interior outputs $o_A, o_B$ get no targets.",
             ha="center", va="top", fontsize=9.5, color=TEXT)

    ax.set_xlim(-0.55, 9.15)
    ax.set_ylim(-0.55, 6.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A logic gate network = one bipartite deep EBM",
                 fontsize=14, fontweight="bold", pad=12)

    out = "figures/05_lgn_deep_ebm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
