"""Presentation Figure C -- SCHEMATIC / HYPOTHETICAL INTERPRETATION.

This is NOT a data figure and NOT a model-structure figure like A/B --
it is an interpretive diagram proposing how this project's confirmed
result (Figure A: K_self is selected on) relates to the broader,
untested claim that four attachment styles emerge from personality
evolution. Everything above the dashed divider in this figure is
speculative framing, explicitly labeled as such, not a result.

Two-layer proposal:
  Foundation layer (what this model actually captured, per Figure A):
    selection on K_self = acquisition of self-anchoring capacity --
    a single, one-dimensional capacity, not four types.
  Derived layer (NOT produced by this model -- see the density-heatmap
    result, which found a single broad basin, not four interior peaks):
    the four classical attachment styles, hypothesized to require
    additional structure this model doesn't have (e.g. social
    buffering / relationship-specific conditioning) to differentiate
    out of the single foundation capacity.

Output:
    scripts/output/figures/figure_C_source_and_derived_layers.png

Usage:
    python3 scripts/plot_figure_C_source_and_derived_layers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("scripts/output/figures")

FOUNDATION_COLOR = "#2a78d6"
DERIVED_COLOR = "#8f8f88"
MUTED_INK = "#8f8f88"
WARN_COLOR = "#eb6834"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 8.5))

    # overall dashed frame -- visually marks the whole figure as schematic/speculative
    frame = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01",
                            transform=ax.transAxes, facecolor="none", edgecolor=MUTED_INK,
                            linewidth=1.8, linestyle=(0, (6, 4)), zorder=10)
    ax.add_patch(frame)

    # "SCHEMATIC" badge
    ax.text(0.98, 0.98, "SCHEMATIC — hypothetical interpretation, not a result",
             transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
             color="white", zorder=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=WARN_COLOR, edgecolor="none"))

    # --- derived layer (top): four attachment styles, dashed/uncertain styling ---
    derived_box = FancyBboxPatch((0.5, 6.3), 9, 2.0, boxstyle="round,pad=0.1",
                                  facecolor="#f2f2ef", edgecolor=DERIVED_COLOR, linewidth=1.6,
                                  linestyle=(0, (5, 3)))
    ax.add_patch(derived_box)
    ax.text(5, 8.55, "Derived layer: four attachment styles (hypothesized)",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold", color="#444")

    style_names = ["Secure", "Preoccupied\n(anxious)", "Dismissive\n(avoidant)", "Fearful\n(disorganized)"]
    for i, name in enumerate(style_names):
        cx = 1.6 + i * 2.15
        box = FancyBboxPatch((cx - 0.85, 6.6), 1.7, 1.4, boxstyle="round,pad=0.08",
                              facecolor="white", edgecolor=DERIVED_COLOR, linewidth=1.2,
                              linestyle=(0, (3, 2)))
        ax.add_patch(box)
        ax.text(cx, 7.3, name, ha="center", va="center", fontsize=9.5, color="#444")

    ax.text(5, 6.42, "requires additional structure not in this model (e.g. social buffering / relationship-specific conditioning)",
            ha="center", va="top", fontsize=8.5, color=MUTED_INK, style="italic")

    ax.text(9.35, 7.3, "NOT\nobserved\nhere", ha="center", va="center", fontsize=9, fontweight="bold",
            color=WARN_COLOR,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=WARN_COLOR, linewidth=1.4))

    # arrow up from foundation to derived
    arrow = FancyArrowPatch((5, 4.9), (5, 6.25), arrowstyle="-|>", mutation_scale=24,
                             color=MUTED_INK, linewidth=2.2, linestyle="--")
    ax.add_patch(arrow)
    ax.text(5.25, 5.55, "differentiation\n(hypothesized, untested)", ha="left", va="center",
            fontsize=8.5, color=MUTED_INK, style="italic")

    # --- foundation layer (bottom): K_self selection, solid/confirmed styling ---
    foundation_box = FancyBboxPatch((0.5, 1.0), 9, 3.7, boxstyle="round,pad=0.1",
                                     facecolor="#eaf1fb", edgecolor=FOUNDATION_COLOR, linewidth=2.2)
    ax.add_patch(foundation_box)
    ax.text(5, 4.45, "Foundation layer: self-anchoring capacity",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=FOUNDATION_COLOR)

    ax.text(5, 3.3, "Selection on K_self\n(hold to one's own read of the interaction,\nindependent of the partner's last move)",
            ha="center", va="center", fontsize=11, color="#222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=FOUNDATION_COLOR, linewidth=1.6))

    ax.text(5, 1.35, "CONFIRMED — Figure A: K_self median ρ(K,payoff)=+0.28 (partial +0.26), n=50 seeds, p<0.0001",
            ha="center", va="center", fontsize=9, fontweight="bold", color=FOUNDATION_COLOR)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    ax.set_title("A single foundation capacity, not four emergent types\n(this simulation's result placed inside a broader, unverified hypothesis)",
                 fontsize=13.5, fontweight="bold", pad=10)

    fig.text(0.5, 0.015,
              "Schematic / hypothetical interpretation. Only the foundation layer (K_self selection) is a result of this "
              "project (Figure A). The derived four-style layer is a proposed hypothesis this model was NOT designed or "
              "shown to produce — the density-heatmap analysis instead found a single broad basin, not four interior peaks.",
              fontsize=8, color=MUTED_INK, ha="center", wrap=True)

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig_path = OUT_DIR / "figure_C_source_and_derived_layers.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
