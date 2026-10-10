"""Local-placement production: 2x2 attachment-quadrant breakdown.

Classifies each agent at each generation into one of 4 quadrants by a
fixed threshold of 0.5 (the midpoint of K_out/K_self's clipped [0,1]
range, not a per-sample median):
    K_out < 0.5, K_self < 0.5  -> "secure"       (low avoidance, low anxiety)
    K_out < 0.5, K_self >= 0.5 -> "preoccupied"   (low avoidance, high anxiety)
    K_out >= 0.5, K_self < 0.5 -> "dismissing"    (high avoidance, low anxiety)
    K_out >= 0.5, K_self >= 0.5-> "fearful"       (high avoidance, high anxiety)

These labels follow Bartholomew & Horowitz's four-category attachment
model under the user's explicit analogy (K_out <-> avoidance of
intimacy, K_self <-> fear of abandonment). This is a metaphorical
overlay the user chose to apply; the simulation itself has no such
construct.

For each (condition, k, seed), per generation (gen>=2000):
  - relative fitness per quadrant = mean(payoff / generation-mean-payoff)
    over agents in that quadrant (this is the fitness-proportional
    weight that actually drives reproduction in this model -- there is
    no separate "survival" step, so this is reported as "relative
    fitness", not "survival probability").
  - population share per quadrant = fraction of the population in that
    quadrant.
Then averaged over gen>=2000 within each seed, giving one value per
(condition, k, seed, quadrant). n=50 seeds.

Per-agent cooperation rate is NOT available in the production
individual logs (only the population-level cooperation_rate per
generation is recorded in timeseries.csv) -- so a cooperation-rate
breakdown by quadrant cannot be computed from existing data without
new instrumentation and a re-run. This script reports fitness and
population share only, and prints a completeness note for this gap.

Usage:
    python3 scripts/analyze_local_placement_quadrants.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
SHUFFLE_ROOT = Path("scripts/output/local_placement_sweep_shuffle_replay")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000
THRESHOLD = 0.5

QUADRANTS = ["secure", "preoccupied", "dismissing", "fearful"]
QUADRANT_TITLES = {
    "secure": "secure\n(K_out<0.5, K_self<0.5)",
    "preoccupied": "preoccupied\n(K_out<0.5, K_self>=0.5)",
    "dismissing": "dismissing\n(K_out>=0.5, K_self<0.5)",
    "fearful": "fearful\n(K_out>=0.5, K_self>=0.5)",
}

LOCAL_COLOR = "#2a78d6"
SHUFFLE_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def quadrant_of(k_out, k_self):
    if k_out < THRESHOLD and k_self < THRESHOLD:
        return "secure"
    if k_out < THRESHOLD and k_self >= THRESHOLD:
        return "preoccupied"
    if k_out >= THRESHOLD and k_self < THRESHOLD:
        return "dismissing"
    return "fearful"


def seed_quadrant_summary(sweep_dir, seed, min_gen):
    rows_by_gen = defaultdict(list)
    with open(Path(sweep_dir) / f"seed_{seed}_individual_log.csv") as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            if gen < min_gen:
                continue
            rows_by_gen[gen].append(
                (float(row["k_out"]), float(row["k_self"]), float(row["payoff"]))
            )

    fitness_acc = {q: [] for q in QUADRANTS}
    share_acc = {q: [] for q in QUADRANTS}
    for gen, rows in rows_by_gen.items():
        payoffs = [p for _, _, p in rows]
        mean_payoff = np.mean(payoffs)
        if mean_payoff == 0:
            continue
        counts = {q: 0 for q in QUADRANTS}
        sums = {q: 0.0 for q in QUADRANTS}
        for k_out, k_self, payoff in rows:
            q = quadrant_of(k_out, k_self)
            counts[q] += 1
            sums[q] += payoff / mean_payoff
        n = len(rows)
        for q in QUADRANTS:
            share_acc[q].append(counts[q] / n)
            if counts[q] > 0:
                fitness_acc[q].append(sums[q] / counts[q])

    fitness = {q: (np.mean(fitness_acc[q]) if fitness_acc[q] else np.nan) for q in QUADRANTS}
    share = {q: np.mean(share_acc[q]) for q in QUADRANTS}
    n_gens_present = {q: len(fitness_acc[q]) for q in QUADRANTS}
    return fitness, share, n_gens_present


def main():
    print("computing per-seed quadrant summaries (reads every individual_log.csv)...", flush=True)
    data = {}  # (cond, k) -> {"fitness": {q: array[50]}, "share": {q: array[50]}}
    completeness = defaultdict(lambda: defaultdict(int))  # (cond, k, q) -> n_seeds_with_zero_presence_gens
    for cond_label, root in [("local", LOCAL_ROOT), ("shuffle", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            fitness_lists = {q: [] for q in QUADRANTS}
            share_lists = {q: [] for q in QUADRANTS}
            for s in SEEDS:
                fitness, share, n_gens_present = seed_quadrant_summary(root / f"k{k}", s, STEADY_STATE_MIN_GEN)
                for q in QUADRANTS:
                    fitness_lists[q].append(fitness[q])
                    share_lists[q].append(share[q])
                    if n_gens_present[q] == 0:
                        completeness[(cond_label, k, q)] += 1
            data[(cond_label, k)] = dict(
                fitness={q: np.array(v) for q, v in fitness_lists.items()},
                share={q: np.array(v) for q, v in share_lists.items()},
            )
    print("done.\n", flush=True)

    print("=" * 100)
    print("Completeness: seeds (out of 50) where a quadrant had ZERO members in every gen>=2000 generation")
    print("(if >0, that quadrant's relative-fitness violin for this (cond,k) is based on fewer than 50 seeds)")
    print("=" * 100)
    any_missing = False
    for cond_label in ["local", "shuffle"]:
        for k in K_VALUES:
            for q in QUADRANTS:
                n_missing = completeness.get((cond_label, k, q), 0)
                if n_missing > 0:
                    any_missing = True
                    print(f"  {cond_label:>8}  k={k:>2}  {q:>12}: {n_missing}/50 seeds had no members in this quadrant")
    if not any_missing:
        print("  (none -- every quadrant had at least one member in at least one gen>=2000 generation, in every seed)")

    print("\n" + "=" * 100)
    print("Population share by quadrant (mean fraction of population, gen>=2000, seed mean, n=50)")
    print("=" * 100)
    print(f"\n{'cond':>8}  {'k':>4}  " + "  ".join(f"{q:>12}" for q in QUADRANTS))
    for cond_label in ["local", "shuffle"]:
        for k in K_VALUES:
            shares = data[(cond_label, k)]["share"]
            cells = "  ".join(f"{np.nanmean(shares[q]):>12.4f}" for q in QUADRANTS)
            print(f"{cond_label:>8}  {k:>4}  {cells}")

    print("\n" + "=" * 100)
    print("Relative fitness by quadrant (payoff / generation-mean-payoff, seed mean, n=50)")
    print("=" * 100)
    print(f"\n{'cond':>8}  {'k':>4}  " + "  ".join(f"{q:>12}" for q in QUADRANTS))
    for cond_label in ["local", "shuffle"]:
        for k in K_VALUES:
            fitness = data[(cond_label, k)]["fitness"]
            cells = "  ".join(f"{np.nanmean(fitness[q]):>12.4f}" for q in QUADRANTS)
            print(f"{cond_label:>8}  {k:>4}  {cells}")

    print("\nNote: per-agent cooperation rate is not recorded in the production individual logs "
          "(only the population-level cooperation_rate per generation exists). A cooperation-rate "
          "breakdown by quadrant is not computed here.")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(K_VALUES), 2, figsize=(16, 20))
    rng = np.random.default_rng(0)

    for row_i, k in enumerate(K_VALUES):
        for col_i, (metric_key, metric_title) in enumerate(
            [("fitness", "relative fitness (payoff / gen-mean-payoff)"),
             ("share", "population share")]
        ):
            ax = axes[row_i, col_i]
            positions = []
            all_vals = []
            colors = []
            labels = []
            pos = 0
            for q in QUADRANTS:
                for cond_label, color in [("local", LOCAL_COLOR), ("shuffle", SHUFFLE_COLOR)]:
                    vals = data[(cond_label, k)][metric_key][q]
                    vals = vals[~np.isnan(vals)]
                    all_vals.append(vals)
                    positions.append(pos)
                    colors.append(color)
                    labels.append(f"{q}\n{cond_label}")
                    pos += 1
                pos += 0.5

            plot_vals = [v if len(v) >= 2 else np.array([np.nan, np.nan]) for v in all_vals]
            parts = ax.violinplot(plot_vals, positions=positions, widths=0.8, showmedians=True, showextrema=False)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.55)
                pc.set_edgecolor(c)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)

            for p, vals, c in zip(positions, all_vals, colors):
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.1, 0.1, size=len(vals))
                ax.scatter(np.full(len(vals), p) + jitter, vals, s=6, color=MUTED_INK, alpha=0.35, zorder=3)

            if metric_key == "fitness":
                ax.axhline(1.0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"k={k}: {metric_title}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Attachment-quadrant breakdown (K_out/K_self high-low, threshold=0.5): "
        "local vs shuffle(replay), k=4,8,16,32 (gen>=2000, per-seed mean, n<=50)",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "local_placement_quadrants.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
