"""Does memory depth evolve over generations, or stay fixed?

Plots mean_depth (already recorded in each seed's timeseries.csv) across
all 50 seeds of the coupling sweep -- pure re-analysis of already-saved
data, no new simulation.

Usage:
    python3 scripts/plot_memory_depth_evolution_50seed.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
SEEDS = list(range(50))
OUT_PATH = Path("scripts/output/50seed_confirmation/memory_depth_evolution_50seed.png")

CLOUD_COLOR = "#8f8f88"
MEAN_COLOR = "#2a78d6"
MUTED_INK = "#8f8f88"


def load_depth_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, depths = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            depths.append(float(row["mean_depth"]))
    return np.array(gens), np.array(depths)


def main():
    all_depths = []
    gens = None
    for seed in SEEDS:
        g, d = load_depth_series(seed)
        gens = g
        all_depths.append(d)
    all_depths = np.vstack(all_depths)
    mean_across_seeds = all_depths.mean(axis=0)
    std_across_seeds = all_depths.std(axis=0)

    print(f"generation 0: mean depth across seeds = {all_depths[:,0].mean():.2f}")
    print(f"generation {gens[-1]}: mean depth across seeds = {all_depths[:,-1].mean():.2f} "
          f"(range {all_depths[:,-1].min():.2f}-{all_depths[:,-1].max():.2f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for d in all_depths:
        ax.plot(gens, d, color=CLOUD_COLOR, linewidth=0.5, alpha=0.3)
    ax.plot(gens, mean_across_seeds, color=MEAN_COLOR, linewidth=2.5, label="mean across 50 seeds")
    ax.fill_between(
        gens, mean_across_seeds - std_across_seeds, mean_across_seeds + std_across_seeds,
        color=MEAN_COLOR, alpha=0.15, label="±1 s.d. across seeds",
    )
    ax.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle="--", label="initial depth (m=1)")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean memory depth m")
    ax.set_title("Memory depth over generations, all 50 seeds\n(grows via duplication, shrinks via split -- not fixed)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
