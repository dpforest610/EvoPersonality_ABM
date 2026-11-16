"""第二章: coarse scan across opponent count k, to check whether K_out's
spatial selection signal (experiment A, k=8: median rho=+0.044, rank-
biserial r=0.802 -- "small but consistent," not "significant but
essentially zero", per the effect-size check) scales monotonically with
k, or is a k=8-specific artifact.

Uses `run_coupled_evolution_opponent_count` (1D ring, not the 2D grid):
every generation, agents are placed on a freshly-shuffled ring and each
duels its k nearest ring-neighbors for one continuous rounds_per_duel-
round duel (no movement, no duel-chopping, reshuffled every generation
-- the three variables already ruled out or shown not to help). k=99
(population_size-1, the complete graph) is NOT re-simulated here --
chapter 1's own well-mixed data (median rho~-0.03) is the reference
point for that end of the sweep.

COARSE pass, explicitly not the full-precision scale (per chat): 10
seeds x 1500 generations per k (half of experiment A's 3000, a fifth of
its seed count), evaluated gen>=750. Purpose is the SHAPE of K_out vs k
(monotonic or not), not a precise estimate at each point -- if the
answer is "yes, monotonic," specific k values can get the full 50x3000
treatment afterward.

k values: 4, 8, 16, 32, 50. Not tuned after results.

Usage:
    python3 scripts/run_opponent_count_scan.py
"""

import csv
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_opponent_count

OUT_ROOT = Path("scripts/output/opponent_count_scan")
K_VALUES = [4, 8, 16, 32, 50]
SEEDS = list(range(10))
GENERATIONS = 1500
LOG_INTERVAL = 25

PARAMS = dict(
    population_size=100,
    rounds_per_duel=50,
    point_mutation_rate=0.05,
    duplication_rate=0.02,
    split_rate=0.02,
    max_memory_depth=8,
    min_memory_depth=0,
    k_mutation_std=0.05,
    noise_std=0.05,
)

AggRow = Tuple[int, float, float, float, float, float, float]
IndividualRow = Tuple[int, int, float, float, int, float]


def run_one_seed(k: int, seed: int) -> Tuple[int, List[AggRow], List[IndividualRow], float]:
    t0 = time.time()
    stats = run_coupled_evolution_opponent_count(seed=seed, generations=GENERATIONS, k=k, **PARAMS)
    elapsed = time.time() - t0

    agg_rows: List[AggRow] = []
    individual_rows: List[IndividualRow] = []
    for s in stats:
        agg_rows.append(
            (
                s.generation,
                s.mean_k_out,
                statistics.pstdev(s.k_out),
                s.mean_k_self,
                statistics.pstdev(s.k_self),
                statistics.fmean(s.memory_depths),
                s.mean_payoff,
            )
        )
        if s.generation % LOG_INTERVAL == 0:
            for i in range(len(s.k_out)):
                individual_rows.append(
                    (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i])
                )

    return seed, agg_rows, individual_rows, elapsed


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
                    "mean_depth", "mean_payoff"])
        for gen, mko, sko, mks, sks, md, mp in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                        f"{md:.4f}", f"{mp:.6f}"])


def write_individual_csv(path: Path, rows: List[IndividualRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff"])
        for gen, idx, k_out, k_self, depth, payoff in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}"])


def main():
    print(f"opponent-count coarse scan: k={K_VALUES}, seeds={SEEDS}, generations={GENERATIONS}")
    t_start = time.time()

    for k in K_VALUES:
        out_dir = OUT_ROOT / f"k{k}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pending_seeds = [
            seed for seed in SEEDS
            if not (out_dir / f"seed_{seed}_timeseries.csv").exists()
            or not (out_dir / f"seed_{seed}_individual_log.csv").exists()
        ]
        if not pending_seeds:
            print(f"k={k}: all seeds already done, skipping")
            continue
        print(f"\nk={k}: running {len(pending_seeds)} seed(s)")

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_one_seed, k, seed): seed for seed in pending_seeds}
            for future in as_completed(futures):
                seed, agg_rows, individual_rows, elapsed = future.result()
                write_agg_csv(out_dir / f"seed_{seed}_timeseries.csv", agg_rows)
                write_individual_csv(out_dir / f"seed_{seed}_individual_log.csv", individual_rows)
                per_gen_ms = elapsed / GENERATIONS * 1000
                print(f"  k={k} seed={seed} done: {elapsed:.1f}s total, {per_gen_ms:.1f} ms/generation")

    print(f"\nall k values done in {time.time() - t_start:.1f}s wall-clock")


if __name__ == "__main__":
    main()
