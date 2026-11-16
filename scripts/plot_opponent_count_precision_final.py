"""Plot the precision opponent-count sweep's interim results (whichever
k values have all 50 seeds done) as violins -- so the seed-to-seed
spread behind the median table is visible, not just the point estimate
-- with significance stars per k (one-sample Wilcoxon test of the
50 per-seed medians against 0).

K_out and K_self side by side (separate y-axis scales, since K_self's
effect is much larger), chapter 1's k=99 (well-mixed) included as the
rightmost reference point in both panels.

Pure re-analysis of already-saved logs -- no new simulation. Re-run
this after the sweep finishes (or at any later checkpoint) to update
the figure with newly-completed k values.

Usage:
    python3 scripts/plot_opponent_count_precision_interim.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000  # matches chapter 1 exactly

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"
SIG_COLOR = "#1baf7a"


def load_individual_log(path):
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def per_seed_medians(sweep_dir, seeds, min_gen):
    result = {"K_out": [], "K_self": []}
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        for channel, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for gen, g in by_gen.items():
                if gen < min_gen:
                    continue
                if len(set(g[key])) < 2 or len(set(g["payoff"])) < 2:
                    continue
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            result[channel].append(np.median(rhos))
    return {k: np.array(v) for k, v in result.items()}


def wilcoxon_p(values):
    values = np.asarray(values)
    values = values[values != 0]
    if len(values) == 0:
        return float("nan")
    _, p = wilcoxon(values)
    return p


def sig_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    ready_k = []
    for k in K_VALUES:
        k_dir = SCAN_ROOT / f"k{k}"
        n_ready = sum(
            1 for s in SEEDS
            if (k_dir / f"seed_{s}_timeseries.csv").exists()
            and (k_dir / f"seed_{s}_individual_log.csv").exists()
        )
        if n_ready == len(SEEDS):
            ready_k.append(k)
    print(f"plotting {len(ready_k)} complete k value(s): {ready_k} (+ chapter 1 k=99 reference)")

    data = {k: per_seed_medians(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN) for k in ready_k}
    ch1 = per_seed_medians(CH1_DIR, SEEDS, STEADY_STATE_MIN_GEN)

    all_k_labels = [str(k) for k in ready_k] + ["99\n(ch.1)"]
    positions = list(range(len(all_k_labels)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, channel, color in [(axes[0], "K_out", KOUT_COLOR), (axes[1], "K_self", KSELF_COLOR)]:
        all_vals = [data[k][channel] for k in ready_k] + [ch1[channel]]
        parts = ax.violinplot(all_vals, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(color if i < len(ready_k) else MUTED_INK)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color if i < len(ready_k) else MUTED_INK)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        rng = np.random.default_rng(0)
        for pos, vals in zip(positions, all_vals):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10,
                       color=MUTED_INK, alpha=0.4, edgecolor="none", zorder=3)

        y_top = max(np.percentile(v, 97) for v in all_vals)
        y_bottom = min(np.percentile(v, 3) for v in all_vals)
        y_span = y_top - y_bottom
        for pos, vals in zip(positions, all_vals):
            p = wilcoxon_p(vals)
            stars = sig_stars(p)
            label = f"{stars}\np={p:.3f}" if stars != "n.s." else f"n.s.\np={p:.3f}"
            ax.text(pos, y_top + 0.05 * y_span, label, ha="center", va="bottom", fontsize=8,
                    color=SIG_COLOR if stars not in ("", "n.s.") else MUTED_INK)

        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(all_k_labels)
        ax.set_xlabel("k (opponent count)")
        ax.set_ylabel(f"Spearman ρ({channel}, payoff)\n(per-seed median, n=50)")
        ax.set_title(f"{channel} vs opponent count k\n(gen≥{STEADY_STATE_MIN_GEN}, one-sample Wilcoxon vs 0)")
        ax.set_ylim(y_bottom - 0.05 * y_span, y_top + 0.22 * y_span)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Precision opponent-count sweep -- FINAL (n=50 seeds × 5000 gen per k)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "opponent_count_precision_sweep_final.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
