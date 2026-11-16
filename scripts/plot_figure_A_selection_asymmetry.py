"""Presentation Figure A -- CONFIRMED DATA FIGURE.

The selection asymmetry: K_self carries a real, positive,
depth-independent selection signature; K_out does not. This is the
most load-bearing empirical result of the whole investigation, so this
figure states it plainly and the caption is declarative, not hedged.

Reads only already-saved analysis outputs -- no new simulation, no new
statistics beyond aggregating what's already on disk:
  scripts/output/selection_vs_drift_correlation/correlation_by_snapshot.csv
    (zero-order Spearman rho(K, payoff), one row per seed x generation
    x channel, gen>=2000)
  scripts/output/selection_partial_correlation/partial_correlation_by_snapshot.csv
    (partial Spearman rho(K, payoff | memory_depth), same grain)

For each channel, collapses to one value per seed (median across that
seed's own gen>=2000 snapshots) -- n=50 independent seeds is the
sample size that matters here, not the 1500 non-independent snapshots.
Plots a violin + jittered per-seed dots for the zero-order
distribution, with a marker for the partial-correlation (depth-
controlled) median overlaid, so the figure itself shows the finding
survives controlling for the memory-depth confound.

Output:
    scripts/output/figures/figure_A_selection_asymmetry.png

Usage:
    python3 scripts/plot_figure_A_selection_asymmetry.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ZERO_ORDER_CSV = Path("scripts/output/selection_vs_drift_correlation/correlation_by_snapshot.csv")
PARTIAL_CSV = Path("scripts/output/selection_partial_correlation/partial_correlation_by_snapshot.csv")
OUT_DIR = Path("scripts/output/figures")
SEEDS = list(range(50))

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
PARTIAL_MARKER_COLOR = "#1baf7a"
MUTED_INK = "#8f8f88"


def per_seed_median(csv_path: Path, value_col: str, channel: str):
    rows = list(csv.DictReader(open(csv_path)))
    medians = []
    for seed in SEEDS:
        vals = [float(r[value_col]) for r in rows if int(r["seed"]) == seed and r["channel"] == channel]
        medians.append(np.median(vals))
    return np.array(medians)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scipy.stats import wilcoxon

    zero_order = {}
    partial = {}
    for channel in ["K_out", "K_self"]:
        zero_order[channel] = per_seed_median(ZERO_ORDER_CSV, "spearman_rho", channel)
        partial[channel] = per_seed_median(PARTIAL_CSV, "partial_spearman_given_depth", channel)

    print("=== Figure A data: per-seed median Spearman rho(K, payoff), gen>=2000, n=50 seeds ===")
    for channel in ["K_out", "K_self"]:
        z = zero_order[channel]
        p = partial[channel]
        w_z, pval_z = wilcoxon(z)
        w_p, pval_p = wilcoxon(p)
        print(f"{channel}: zero-order median={np.median(z):+.3f} (Wilcoxon p={pval_z:.4f}, {int((z>0).sum())}/50 positive)  "
              f"partial|depth median={np.median(p):+.3f} (Wilcoxon p={pval_p:.4f}, {int((p>0).sum())}/50 positive)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    rng = np.random.default_rng(0)
    positions = [0, 1]
    colors = [KOUT_COLOR, KSELF_COLOR]
    labels = ["K_out", "K_self"]

    data = [zero_order["K_out"], zero_order["K_self"]]
    parts = ax.violinplot(data, positions=positions, widths=0.7, showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.25)
        pc.set_edgecolor(color)

    for pos, channel, color in zip(positions, labels, colors):
        vals = zero_order[channel]
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22, color=color, alpha=0.65, edgecolor="none", zorder=3)
        med = np.median(vals)
        ax.plot([pos - 0.2, pos + 0.2], [med, med], color=color, linewidth=2.5, zorder=4)

        part_med = np.median(partial[channel])
        ax.scatter([pos], [part_med], marker="D", s=110, color="white", edgecolor=PARTIAL_MARKER_COLOR,
                   linewidth=2.2, zorder=6,
                   label="partial (|memory depth), median" if pos == 0 else None)

    ax.axhline(0, color=MUTED_INK, linewidth=1.2, linestyle="--", zorder=1, label="zero (neutral)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("per-seed median Spearman ρ(K, payoff), within generation")

    kout_z_med = np.median(zero_order["K_out"])
    kself_z_med = np.median(zero_order["K_self"])
    kout_p_med = np.median(partial["K_out"])
    kself_p_med = np.median(partial["K_self"])
    subtitle = (f"K_out: median ρ={kout_z_med:+.2f} (partial {kout_p_med:+.2f}), not distinguishable from zero  |  "
                f"K_self: median ρ={kself_z_med:+.2f} (partial {kself_p_med:+.2f}), consistently positive")
    ax.set_title(subtitle, fontsize=9, color=MUTED_INK, pad=32)
    fig.suptitle("K_self is selected on; K_out drifts near-neutrally", fontsize=14, fontweight="bold", y=1.01)

    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01,
              "n=50 seeds (each point = one seed's median over its own gen≥2000 snapshots), "
              "5000-gen coupled-evolution sweep. Confirmed empirical result.",
              fontsize=7.5, color=MUTED_INK)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig_path = OUT_DIR / "figure_A_selection_asymmetry.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
