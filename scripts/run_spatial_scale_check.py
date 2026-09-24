"""第二章 scale-finding run -- condition (3) alone, k=8 (Moore), larger
scale than the first smoke.

Purpose (per the chat discussion after the first condition-3 smoke):
the first smoke (3 seeds x 2000 generations, k=4/von Neumann) was too
small a "measurement window" to even recover chapter 1's ALREADY
CONFIRMED K_self selection signal at full strength -- so it could not
be trusted to detect a same-or-weaker K_out signal either. Before
adding condition (2) or judging K_out, find a scale at which K_self
comes through clearly as a calibration check on the measurement
pipeline itself.

Two changes from the first smoke, both requested directly (not chosen
to chase an effect):
  - neighbor_offsets=MOORE_OFFSETS (k=8, not k=4/von Neumann): the
    hypothesis needs a large-enough neighborhood for partner choice to
    have room to matter, and k=8 matches last year's (undocumented,
    non-existent-as-code) reference point.
  - larger scale: seeds=range(10) (was 3), generations=3000 (was 2000).

All other parameters are unchanged from the first smoke's fixed set
(grid 12x12 ~30.6% vacant, n_phases=5, tau=1.0, move_radius=1,
population_size=100, and all non-spatial mutation/game parameters
identical to chapter 1's defaults). Not tuned after seeing results.

Evaluation region for the companion analysis script: generation >= 1000
(this run is only 3000 generations total, well short of chapter 1's
5000, so "steady state" here means the back two-thirds of the run, not
chapter 1's gen>=2000 convention).

Usage:
    python3 scripts/run_spatial_scale_check.py
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

from evopersonality_abm.spatial import MOORE_OFFSETS, run_coupled_evolution_spatial

OUT_DIR = Path("scripts/output/spatial_scale_check_k8_condition3")

SEEDS = list(range(10))
GENERATIONS = 3000
LOG_INTERVAL = 50

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
    grid_width=12,
    grid_height=12,
    n_phases=5,
    tau=1.0,
    move_radius=1,
    mobile=True,  # condition (3)
    neighbor_offsets=MOORE_OFFSETS,  # k=8
)

AggRow = Tuple[int, float, float, float, float, float, float, float]
IndividualRow = Tuple[int, int, float, float, int, float, int, int]


def run_one_seed(seed: int) -> Tuple[int, List[AggRow], List[IndividualRow], float]:
    t0 = time.time()
    stats = run_coupled_evolution_spatial(seed=seed, generations=GENERATIONS, **PARAMS)
    elapsed = time.time() - t0

    agg_rows: List[AggRow] = []
    individual_rows: List[IndividualRow] = []
    for s in stats:
        n_isolated = sum(1 for r in s.rounds_played if r == 0)
        agg_rows.append(
            (
                s.generation,
                s.mean_k_out,
                statistics.pstdev(s.k_out),
                s.mean_k_self,
                statistics.pstdev(s.k_self),
                statistics.fmean(s.memory_depths),
                s.mean_payoff,
                n_isolated / len(s.payoffs),
            )
        )
        if s.generation % LOG_INTERVAL == 0:
            for i in range(len(s.k_out)):
                x, y = s.positions[i]
                individual_rows.append(
                    (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i], x, y)
                )

    return seed, agg_rows, individual_rows, elapsed


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
                    "mean_depth", "mean_payoff", "frac_isolated"])
        for gen, mko, sko, mks, sks, md, mp, fi in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                        f"{md:.4f}", f"{mp:.6f}", f"{fi:.4f}"])


def write_individual_csv(path: Path, rows: List[IndividualRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff", "x", "y"])
        for gen, idx, k_out, k_self, depth, payoff, x, y in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}", x, y])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"condition (3) scale check: seeds={SEEDS}, generations={GENERATIONS}, params={PARAMS}")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_one_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed, agg_rows, individual_rows, elapsed = future.result()
            write_agg_csv(OUT_DIR / f"seed_{seed}_timeseries.csv", agg_rows)
            write_individual_csv(OUT_DIR / f"seed_{seed}_individual_log.csv", individual_rows)
            per_gen_ms = elapsed / GENERATIONS * 1000
            print(f"seed={seed} done: {elapsed:.1f}s total, {per_gen_ms:.1f} ms/generation")

    print(f"\nall {len(SEEDS)} seeds done in {time.time() - t_start:.1f}s wall-clock")


if __name__ == "__main__":
    main()
