"""Presentation Figure B -- MODEL-STRUCTURE FIGURE (not a data plot).

Shows how the design doc's four-quadrant attachment-style assignment
(placed at the CORNERS of the (K_out, K_self) square, i.e. defined by
the two EDGES K_out and K_self) relates to the model's own generative
axes -- the fixed point s* = K_self / (K_self + K_out) and the total
coupling K_self + K_out -- which sit along the square's two DIAGONALS,
45 degrees off from the edges.

Everything drawn here is derived from equations/results already
established analytically or by direct simulation, not fit to new data:
  - s* = K_self/(K_self+K_out): the noiseless fixed point of
    update_internal_state (see analyze_internal_state_quadrants.py's
    docstring derivation).
  - The four corner labels/signatures (secure/dismissive/preoccupied/
    fearful-ish dynamics) are the measured results from
    analyze_internal_state_quadrants.py's four-corner test-signal
    simulation (HIGH=0.8, LOW=0.1, HIGH-HIGH rescaled to 0.5/0.5):
      HIGH k_out, LOW k_self  -> s*=0.111 (drawn into partner's defection)
      LOW k_out, HIGH k_self  -> s*=0.889 (holds own position)
      HIGH k_out, HIGH k_self -> s*=0.5, reached instantly, low noise
                                  sensitivity (std=0.041 across realizations)
      LOW k_out, LOW k_self   -> s*=0.5 too, but slow (decay 0.8/round)
                                  and noisier (std=0.084)

This is a labeled geometric diagram, not a rendering of simulation
output -- caption and on-figure text are written in the descriptive
("this is how the model maps") register, not a claim about what
evolution actually produced (that's Figure A / the density-heatmap
result, which found no interior 4-quadrant structure emerged).

Output:
    scripts/output/figures/figure_B_generative_axes_vs_quadrants.png

Usage:
    python3 scripts/plot_figure_B_generative_axes_vs_quadrants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("scripts/output/figures")

EDGE_COLOR = "#2a78d6"      # K_out/K_self edges = the square's own axes
DIAG_COLOR = "#eb6834"      # generative diagonals
CORNER_COLOR = "#1baf7a"
MUTED_INK = "#8f8f88"
BG_TINT = "#f5f3ee"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # the square itself: K_out (x) and K_self (y) as the model's literal parameters
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=BG_TINT, edgecolor="none", zorder=0))
    for x in [0, 1]:
        ax.axvline(x, color=EDGE_COLOR, linewidth=2.2, zorder=2)
    for y in [0, 1]:
        ax.axhline(y, color=EDGE_COLOR, linewidth=2.2, zorder=2)

    # edge labels: the square's own two axes = K_out / K_self, standing in for the
    # design doc's "attachment two-axis" framework (each edge is one axis' 0->1 range)
    ax.text(0.5, -0.09, "K_out  (edge axis 1)", ha="center", va="top", fontsize=11, color=EDGE_COLOR, fontweight="bold")
    ax.text(-0.09, 0.5, "K_self  (edge axis 2)", ha="center", va="center", fontsize=11, color=EDGE_COLOR,
            fontweight="bold", rotation=90)

    # generative diagonals
    ax.annotate("", xy=(1, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=DIAG_COLOR, linewidth=2.6, mutation_scale=22), zorder=3)
    ax.plot([0, 1], [1, 0], color=DIAG_COLOR, linewidth=2.6, zorder=3)

    ax.text(1.03, 1.03, "sum axis:\nK_out+K_self\n(total anchor)", fontsize=9.5, color=DIAG_COLOR, ha="left", va="bottom")
    ax.text(-0.03, 1.05, "ratio axis: s*=K_self/(K_self+K_out)\n(self vs. other balance)",
            fontsize=9.5, color=DIAG_COLOR, ha="left", va="bottom")

    # the 45-degree mismatch callout
    ax.text(0.5, 0.5, "45°", fontsize=13, color=MUTED_INK, ha="center", va="center",
            fontweight="bold", bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor=MUTED_INK))

    # four corners: measured dynamical signature from analyze_internal_state_quadrants.py
    corner_labels = [
        (0.02, 0.02, "LOW/LOW\ns*=0.5, slow (decay 0.8/rd)\nnoisy: std=0.084\n\"fearful\"-like", "left", "bottom"),
        (0.98, 0.02, "HIGH k_out, LOW k_self\ns*=0.111\ndrawn into partner's defection\n\"preoccupied\"-like", "right", "bottom"),
        (0.02, 0.98, "LOW k_out, HIGH k_self\ns*=0.889\nholds own position\n\"dismissive\"-like", "left", "top"),
        (0.98, 0.98, "HIGH/HIGH (rescaled 0.5/0.5)\ns*=0.5, instant, robust\nstd=0.041\n\"secure\"-like", "right", "top"),
    ]
    for x, y, label, ha, va in corner_labels:
        ax.scatter([x], [y], s=90, color=CORNER_COLOR, edgecolor="white", linewidth=1.2, zorder=5)
        offset_x = -0.04 if ha == "right" else 0.04
        offset_y = -0.04 if va == "top" else 0.04
        ax.text(x + offset_x, y + offset_y, label, fontsize=8, ha=ha, va=va, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=CORNER_COLOR, alpha=0.92))

    ax.set_xlim(-0.42, 1.55)
    ax.set_ylim(-0.32, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("The design doc's quadrant corners vs. the model's own generative axes",
                 fontsize=14, fontweight="bold", pad=14)
    fig.text(0.5, 0.955,
              "Corner labels are the measured fixed-point/noise-sensitivity signature (test-signal simulation); "
              "the two diagonals are the update rule's own axes.\nOn this model, they sit 45° apart from the "
              "K_out/K_self edges the quadrant assignment is drawn on.",
              fontsize=9, color=MUTED_INK, ha="center")

    fig.text(0.01, 0.01,
              "Model-structure diagram: derived from the update-rule fixed point and the four-corner test-signal "
              "simulation (analyze_internal_state_quadrants.py). Describes how the model maps corners to dynamics -- "
              "not a claim about what evolution produced (see Figure showing no interior 4-peak structure emerged).",
              fontsize=7.5, color=MUTED_INK, wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig_path = OUT_DIR / "figure_B_generative_axes_vs_quadrants.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
