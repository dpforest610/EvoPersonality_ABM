"""Run the population evolution loop across multiple seeds in parallel,
and compare which payoff regime (defection near 1.0, cooperation near
3.0) each run ends up in and whether/when a mid-run transition happens.

Usage:
    python3 scripts/run_seed_sweep.py --seeds 10 --workers 4

Writes one CSV of per-generation stats per seed, a summary.csv, and (by
default) a comparison plot with all seeds' trajectories overlaid.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evopersonality_abm.population import run_evolution

DEFECT_PAYOFF = 1.0
COOP_PAYOFF = 3.0
COOP_THRESHOLD = (DEFECT_PAYOFF + COOP_PAYOFF) / 2  # 2.0

Row = Tuple[int, float, float, int, int, float]  # gen, mean_m, std_m, min_m, max_m, mean_payoff


def run_one_seed(
    seed: int,
    population_size: int,
    generations: int,
    rounds_per_duel: int,
    point_mutation_rate: float,
    duplication_rate: float,
    split_rate: float,
    max_memory_depth: int,
    min_memory_depth: int,
) -> Tuple[int, List[Row]]:
    stats = run_evolution(
        population_size=population_size,
        generations=generations,
        rounds_per_duel=rounds_per_duel,
        point_mutation_rate=point_mutation_rate,
        duplication_rate=duplication_rate,
        split_rate=split_rate,
        max_memory_depth=max_memory_depth,
        min_memory_depth=min_memory_depth,
        seed=seed,
    )
    rows = [
        (s.generation, s.mean_depth, s.std_depth, s.min_depth, s.max_depth, s.mean_payoff)
        for s in stats
    ]
    return seed, rows


def detect_transition(rows: List[Row]) -> Optional[int]:
    """Earliest generation after which mean_payoff never again drops
    below COOP_THRESHOLD -- a permanent defection -> cooperation regime
    change. None if the run never settles into cooperation for good."""
    payoffs = [r[5] for r in rows]
    for i in range(len(payoffs)):
        if all(p >= COOP_THRESHOLD for p in payoffs[i:]):
            return rows[i][0]
    return None


def write_csv(path: Path, rows: List[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_depth", "std_depth", "min_depth", "max_depth", "mean_payoff"])
        w.writerows(rows)


def plot_sweep(all_rows: List[Tuple[int, List[Row]]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)

    for seed, rows in all_rows:
        gens = [r[0] for r in rows]
        means = [r[1] for r in rows]
        payoffs = [r[5] for r in rows]
        final_regime = "cooperation" if payoffs[-1] >= COOP_THRESHOLD else "defection"
        color = "tab:blue" if final_regime == "cooperation" else "tab:red"
        ax1.plot(gens, means, color=color, alpha=0.6, linewidth=1)
        ax2.plot(gens, payoffs, color=color, alpha=0.6, linewidth=1)

        transition_gen = detect_transition(rows)
        if transition_gen is not None:
            ax2.axvline(transition_gen, color=color, alpha=0.25, linewidth=0.8, linestyle=":")

    ax1.set_xlabel("generation")
    ax1.set_ylabel("mean memory depth m")
    ax1.set_title("Mean memory depth per seed")

    ax2.axhline(COOP_THRESHOLD, color="gray", linewidth=0.8, linestyle="--", label="classification threshold")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("mean payoff")
    ax2.set_title("Mean payoff per seed (blue=cooperation, red=defection)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved comparison plot to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10, help="number of seeds, seed-offset..seed-offset+seeds-1")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=8)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="scripts/output/seed_sweep")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    all_rows: List[Tuple[int, List[Row]]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one_seed,
                seed,
                args.population_size,
                args.generations,
                args.rounds_per_duel,
                args.point_mutation_rate,
                args.duplication_rate,
                args.split_rate,
                args.max_memory_depth,
                args.min_memory_depth,
            ): seed
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed, rows = fut.result()
            write_csv(out_dir / f"seed_{seed}.csv", rows)
            all_rows.append((seed, rows))

            final_payoff = rows[-1][5]
            final_regime = "cooperation" if final_payoff >= COOP_THRESHOLD else "defection"
            transition_gen = detect_transition(rows)
            print(
                f"seed={seed} done: final_payoff={final_payoff:.3f} "
                f"regime={final_regime} transition_gen={transition_gen}",
                flush=True,
            )

    all_rows.sort(key=lambda t: t[0])
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "final_mean_payoff", "final_regime", "transition_generation"])
        for seed, rows in all_rows:
            final_payoff = rows[-1][5]
            final_regime = "cooperation" if final_payoff >= COOP_THRESHOLD else "defection"
            transition_gen = detect_transition(rows)
            w.writerow([seed, f"{final_payoff:.4f}", final_regime, transition_gen if transition_gen is not None else ""])

    print("\n=== summary ===")
    print(f"{'seed':>4} {'final_payoff':>12} {'regime':>11} {'transition_gen':>14}")
    n_coop = 0
    for seed, rows in all_rows:
        final_payoff = rows[-1][5]
        final_regime = "cooperation" if final_payoff >= COOP_THRESHOLD else "defection"
        n_coop += final_regime == "cooperation"
        transition_gen = detect_transition(rows)
        print(f"{seed:>4} {final_payoff:>12.3f} {final_regime:>11} {transition_gen if transition_gen is not None else '-':>14}")
    print(f"\n{n_coop}/{len(all_rows)} seeds ended in cooperation")

    if not args.no_plot:
        plot_sweep(all_rows, out_dir / "seed_sweep_comparison.png")


if __name__ == "__main__":
    main()
