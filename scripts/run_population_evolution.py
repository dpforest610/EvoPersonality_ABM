"""Run the Lindgren population loop and report how memory depth moves.

First step from a single duel to population-level evolution (see
HBES2026_ABM骨子_二成分K.md section 4): ~50 agents play a round-robin
iterated Prisoner's Dilemma, payoff sets fitness-proportional
reproduction, offspring get neutral memory-depth duplication and point
mutation. No spatial structure, no odor gradient, no K_out/K_self yet.

Usage:
    python3 scripts/run_population_evolution.py
    python3 scripts/run_population_evolution.py --generations 80 --seed 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evopersonality_abm.population import run_evolution


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-size", type=int, default=50)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--rounds-per-duel", type=int, default=150)
    p.add_argument("--initial-memory-depth", type=int, default=1)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=6)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=str, default="scripts/output/memory_depth_evolution.png"
    )
    p.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    return p.parse_args()


def print_table(stats):
    header = f"{'gen':>4} {'mean_m':>7} {'std_m':>6} {'min':>4} {'max':>4} {'mean_payoff':>12}"
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s.generation:>4} {s.mean_depth:>7.2f} {s.std_depth:>6.2f} "
            f"{s.min_depth:>4} {s.max_depth:>4} {s.mean_payoff:>12.3f}"
        )


def plot(stats, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    generations = [s.generation for s in stats]
    means = np.array([s.mean_depth for s in stats])
    stds = np.array([s.std_depth for s in stats])
    mins = [s.min_depth for s in stats]
    maxs = [s.max_depth for s in stats]

    max_depth_seen = max(s.max_depth for s in stats)
    heat = np.zeros((len(stats), max_depth_seen + 1))
    for gi, s in enumerate(stats):
        for d in s.memory_depths:
            heat[gi, d] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(generations, means, color="tab:blue", label="mean memory depth m")
    ax1.fill_between(
        generations,
        means - stds,
        means + stds,
        color="tab:blue",
        alpha=0.2,
        label="±1 s.d.",
    )
    ax1.plot(generations, mins, color="gray", linewidth=0.8, linestyle="--", label="min/max")
    ax1.plot(generations, maxs, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("memory depth m")
    ax1.set_title("Mean memory depth over generations")
    ax1.legend(fontsize=8)

    im = ax2.imshow(
        heat.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[generations[0] - 0.5, generations[-1] + 0.5, -0.5, max_depth_seen + 0.5],
    )
    ax2.set_xlabel("generation")
    ax2.set_ylabel("memory depth m")
    ax2.set_title("Memory depth distribution over generations")
    fig.colorbar(im, ax=ax2, label="# agents")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


def main():
    args = parse_args()
    stats = run_evolution(
        population_size=args.population_size,
        generations=args.generations,
        rounds_per_duel=args.rounds_per_duel,
        initial_memory_depth=args.initial_memory_depth,
        point_mutation_rate=args.point_mutation_rate,
        duplication_rate=args.duplication_rate,
        split_rate=args.split_rate,
        max_memory_depth=args.max_memory_depth,
        min_memory_depth=args.min_memory_depth,
        seed=args.seed,
    )
    print_table(stats)

    if not args.no_plot:
        plot(stats, Path(args.out))


if __name__ == "__main__":
    main()
