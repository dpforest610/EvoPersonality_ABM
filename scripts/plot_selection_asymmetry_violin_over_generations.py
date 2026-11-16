"""Static companion to Animation 1: the line chart shows the median
trend but hides how much the 50 seeds actually agree/disagree with
each other at any given point in time. This adds that back as violins.

For each of the 50 seeds and every logged generation (every 100 gen,
0..4900), computes Spearman rho(individual K, individual payoff)
within that generation's 100 agents -- same computation as
animate_selection_asymmetry_over_time.py. Then groups the 50 snapshot
generations into --bin-size (default 5, i.e. ~500 generations) chunks;
within each chunk, takes each seed's own median across that chunk's
generations, giving 50 values per chunk per channel -- the seeds stay
the unit of replication, matching every other seed-level analysis in
this project.

Plots the median line (thin, for trend context) with a violin at each
chunk's center generation, K_out and K_self side by side, so both the
trend AND the seed-to-seed spread at each point in time are visible in
one static figure.

Pure re-analysis of the already-saved individual logs -- no new
simulation.

Output:
    scripts/output/figures/selection_asymmetry_violin_over_generations.png

Usage:
    python3 scripts/plot_selection_asymmetry_violin_over_generations.py
    python3 scripts/plot_selection_asymmetry_violin_over_generations.py --bin-size 10
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
OUT_DIR = Path("scripts/output/figures")
SEEDS = list(range(50))

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin-size", type=int, default=5, help="snapshot points per violin (5 snapshots ~= 500 gen)")
    return p.parse_args()


def load_all_generations(seed: int):
    path = SWEEP_DIR / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scipy.stats import spearmanr

    print("Loading all 50 seeds' individual logs and computing per-(seed,generation) correlations...")
    per_seed_by_gen = {seed: load_all_generations(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())

    # rho_by_seed_gen[channel][seed] = [rho at gen0, rho at gen100, ...] (aligned with `generations`)
    rho_by_seed_gen = {"K_out": defaultdict(list), "K_self": defaultdict(list)}
    median_rho = {"K_out": [], "K_self": []}
    for gen in generations:
        for channel_name, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos_this_gen = []
            for seed in SEEDS:
                g = per_seed_by_gen[seed][gen]
                rho, _ = spearmanr(g[key], g["payoff"])
                rho_by_seed_gen[channel_name][seed].append(rho)
                rhos_this_gen.append(rho)
            median_rho[channel_name].append(np.median(rhos_this_gen))

    # bin into chunks of --bin-size snapshot points; per seed, median within the chunk
    n = len(generations)
    bin_size = args.bin_size
    bin_starts = list(range(0, n, bin_size))
    bin_center_gen = []
    bin_values = {"K_out": [], "K_self": []}  # list of arrays, one per bin, each length 50 (one per seed)

    for start in bin_starts:
        end = min(start + bin_size, n)
        gens_in_bin = generations[start:end]
        bin_center_gen.append(int(np.mean(gens_in_bin)))
        for channel_name in ["K_out", "K_self"]:
            per_seed_bin_median = [
                np.median(rho_by_seed_gen[channel_name][seed][start:end]) for seed in SEEDS
            ]
            bin_values[channel_name].append(np.array(per_seed_bin_median))

    print(f"{len(generations)} snapshots binned into {len(bin_starts)} groups of ~{bin_size} "
          f"(~{bin_size*100} generations each)")
    for i, gen_center in enumerate(bin_center_gen):
        ko = bin_values["K_out"][i]
        ks = bin_values["K_self"][i]
        print(f"  gen~{gen_center}: K_out median={np.median(ko):+.3f} IQR=[{np.percentile(ko,25):+.3f},{np.percentile(ko,75):+.3f}]   "
              f"K_self median={np.median(ks):+.3f} IQR=[{np.percentile(ks,25):+.3f},{np.percentile(ks,75):+.3f}]")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # thin median line for trend context (raw per-generation, not binned)
    ax.plot(generations, median_rho["K_out"], color=KOUT_COLOR, linewidth=1.0, alpha=0.45, zorder=1)
    ax.plot(generations, median_rho["K_self"], color=KSELF_COLOR, linewidth=1.0, alpha=0.45, zorder=1)

    violin_width = bin_size * 100 * 0.35
    offset = bin_size * 100 * 0.12
    for i, gen_center in enumerate(bin_center_gen):
        for channel_name, color, sign in [("K_out", KOUT_COLOR, -1), ("K_self", KSELF_COLOR, +1)]:
            vals = bin_values[channel_name][i]
            parts = ax.violinplot([vals], positions=[gen_center + sign * offset], widths=violin_width,
                                   showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
                pc.set_edgecolor(color)
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(2)

    ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlim(generations[0] - bin_size * 60, generations[-1] + bin_size * 60)
    ax.set_xlabel("generation")
    ax.set_ylabel("Spearman ρ(individual K, individual payoff)")
    ax.set_title(f"Selection asymmetry over time, with seed-to-seed spread\n"
                 f"(each violin: n=50 seeds' own median over ~{bin_size*100} generations; line: raw per-generation median)")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=KOUT_COLOR, alpha=0.55, label="K_out"),
        Patch(facecolor=KSELF_COLOR, alpha=0.55, label="K_self"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = OUT_DIR / "selection_asymmetry_violin_over_generations.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
