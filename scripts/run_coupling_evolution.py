"""Evolve the two-channel coupling genes (K_out, K_self) alongside the
Lindgren strategy -- HBES2026_ABM骨子_二成分K.md §3-5.

Starts a population uniformly over the (K_out, K_self) plane and asks
where the evolutionary attractor(s) land: one peak, several, or none.
No spatial structure yet -- this sits on the same round-robin population
loop used for the pure-Lindgren experiments.

Usage:
    python3 scripts/run_coupling_evolution.py
    python3 scripts/run_coupling_evolution.py --generations 1000 --seed 1
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evopersonality_abm.coupling import run_coupled_evolution

# HBES dataviz palette: sequential blue ramp (density) and categorical
# slots 1/2 (blue/orange, for the two named K_out/K_self series).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_ORANGE = "#eb6834"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--initial-memory-depth", type=int, default=1)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=8)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--k-mutation-std", type=float, default=0.05)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="scripts/output/coupling_evolution.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="if set, write timeseries.csv (per-generation summary) and "
        "snapshot_gen<N>.csv (per-agent K_out/K_self/depth) for the first "
        "and last generation into this directory",
    )
    return p.parse_args()


def write_csvs(stats, csv_dir: Path) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = csv_dir / "timeseries.csv"
    with open(timeseries_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self", "mean_depth", "mean_payoff"]
        )
        for s in stats:
            w.writerow(
                [
                    s.generation,
                    f"{s.mean_k_out:.6f}",
                    f"{statistics.pstdev(s.k_out):.6f}",
                    f"{s.mean_k_self:.6f}",
                    f"{statistics.pstdev(s.k_self):.6f}",
                    f"{statistics.fmean(s.memory_depths):.4f}",
                    f"{s.mean_payoff:.6f}",
                ]
            )
    print(f"saved timeseries to {timeseries_path}")

    for snap in (stats[0], stats[-1]):
        snap_path = csv_dir / f"snapshot_gen{snap.generation}.csv"
        with open(snap_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent_index", "k_out", "k_self", "memory_depth"])
            for i, (k_out, k_self, depth) in enumerate(
                zip(snap.k_out, snap.k_self, snap.memory_depths)
            ):
                w.writerow([i, f"{k_out:.6f}", f"{k_self:.6f}", depth])
        print(f"saved snapshot to {snap_path}")


def print_table(stats):
    header = (
        f"{'gen':>4} {'mean_Kout':>10} {'mean_Kself':>11} "
        f"{'mean_m':>7} {'mean_payoff':>12}"
    )
    print(header)
    print("-" * len(header))
    step = max(1, len(stats) // 40)
    for s in stats[:-1:step]:
        print(
            f"{s.generation:>4} {s.mean_k_out:>10.3f} {s.mean_k_self:>11.3f} "
            f"{sum(s.memory_depths) / len(s.memory_depths):>7.2f} {s.mean_payoff:>12.3f}"
        )
    last = stats[-1]
    print(
        f"{last.generation:>4} {last.mean_k_out:>10.3f} {last.mean_k_self:>11.3f} "
        f"{sum(last.memory_depths) / len(last.memory_depths):>7.2f} {last.mean_payoff:>12.3f}"
    )


def plot(stats, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)

    fig, ((ax_scatter, ax_heat), (ax_k_traj, ax_payoff)) = plt.subplots(2, 2, figsize=(12, 11))

    first, last = stats[0], stats[-1]
    ax_scatter.scatter(
        first.k_out, first.k_self, s=22, color=MUTED_INK, alpha=0.5,
        label=f"gen {first.generation} (initial, uniform)",
    )
    ax_scatter.scatter(
        last.k_out, last.k_self, s=22, color=CATEGORICAL_BLUE, alpha=0.7,
        label=f"gen {last.generation} (final)",
    )
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_xlabel("K_out (automatic channel)")
    ax_scatter.set_ylabel("K_self (deliberate channel)")
    ax_scatter.set_title("Population in the (K_out, K_self) plane: start vs. end")
    ax_scatter.legend(fontsize=8, loc="lower right")
    ax_scatter.spines[["top", "right"]].set_visible(False)

    h = ax_heat.hist2d(last.k_out, last.k_self, bins=20, range=[[0, 1], [0, 1]], cmap=cmap)
    fig.colorbar(h[3], ax=ax_heat, label="# agents")
    ax_heat.set_xlabel("K_out")
    ax_heat.set_ylabel("K_self")
    ax_heat.set_title(f"Density at generation {last.generation}")

    gens = [s.generation for s in stats]
    ax_k_traj.plot(gens, [s.mean_k_out for s in stats], color=CATEGORICAL_BLUE, linewidth=2, label="mean K_out")
    ax_k_traj.plot(gens, [s.mean_k_self for s in stats], color=CATEGORICAL_ORANGE, linewidth=2, label="mean K_self")
    ax_k_traj.set_ylim(-0.02, 1.02)
    ax_k_traj.set_xlabel("generation")
    ax_k_traj.set_ylabel("mean value")
    ax_k_traj.set_title("Mean K_out / K_self over generations")
    ax_k_traj.legend(fontsize=8)
    ax_k_traj.spines[["top", "right"]].set_visible(False)

    ax_payoff.plot(gens, [s.mean_payoff for s in stats], color=CATEGORICAL_BLUE, linewidth=2)
    ax_payoff.axhline(1.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    ax_payoff.axhline(3.0, color=MUTED_INK, linewidth=0.8, linestyle="--")
    ax_payoff.set_xlabel("generation")
    ax_payoff.set_ylabel("mean payoff")
    ax_payoff.set_title("Mean payoff over generations (dashed: 1=defection, 3=cooperation)")
    ax_payoff.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved plot to {out_path}")


def main():
    args = parse_args()
    stats = run_coupled_evolution(
        population_size=args.population_size,
        generations=args.generations,
        rounds_per_duel=args.rounds_per_duel,
        initial_memory_depth=args.initial_memory_depth,
        point_mutation_rate=args.point_mutation_rate,
        duplication_rate=args.duplication_rate,
        split_rate=args.split_rate,
        max_memory_depth=args.max_memory_depth,
        min_memory_depth=args.min_memory_depth,
        k_mutation_std=args.k_mutation_std,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print_table(stats)

    if args.csv_dir is not None:
        write_csvs(stats, Path(args.csv_dir))

    if not args.no_plot:
        plot(stats, Path(args.out))


if __name__ == "__main__":
    main()
