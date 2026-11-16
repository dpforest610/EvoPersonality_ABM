"""第二章 smoke run -- condition (3) alone (spatial + individual-based
softmax movement), per the design memo (chat, not committed to this
repo as a doc). Three checks only:

  1. does the phase/duel/movement loop complete for a realistic
     population size and generation count without crashing or
     corrupting the agent<->cell bookkeeping?
  2. does corr(individual K_out, individual payoff) show ANY
     directional hint of moving off zero -- direction only, not
     significance (that needs the full multi-generation pooling, as
     chapter 1's Figure A established)?
  3. what is the real ms/generation, to firm up the full-scale
     (3 conditions x 50 seeds x 5000 gen) time estimate?

Parameter set is fixed up front and NOT tuned after seeing results
(explicit instruction: no reaching for an effect by turning knobs).
This is the one straightforward default chosen in the design
discussion:

  population_size=100        -- same scale as chapter 1, so any hint
                                 seen here is informative for the
                                 eventual full run at the same scale.
  grid_width=grid_height=12  -- 144 cells for 100 agents, ~30.6% vacant
                                 (a plain, unforced vacancy ratio -- not
                                 picked to make movement easier or
                                 harder).
  neighbor topology: von Neumann (k=4) -- spatial.py's default and the
                                 simpler of the two standard choices
                                 (vs. Moore/k=8).
  n_phases=5                 -- divides rounds_per_duel=50 into 5
                                 phases of 10 rounds each, a round
                                 number, not tuned.
  move_radius=1               -- candidates are only the moving agent's
                                 immediate grid neighbors (the minimal
                                 choice; a longer-range variant is
                                 explicitly deferred in spatial.py).
  tau=1.0                    -- softmax temperature in payoff units;
                                 PD per-round payoffs range over
                                 [0,5] (mutual C=3, mutual D=1), so
                                 tau=1.0 is "one payoff-unit" of
                                 softness -- a plain, unforced choice,
                                 not fit to produce a stronger effect.
  All non-spatial parameters (rounds_per_duel, mutation rates, noise,
  memory-depth bounds) are identical to chapter 1's defaults
  (run_coupled_evolution / run_selection_vs_drift_sweep.py), so this
  smoke is comparable to the existing well-mixed data.

Smoke scale: seeds=[0,1,2], generations=2000 (well short of chapter
1's 5000, but ~20x chapter 1's first snapshot at gen=100, where its
K_self signal was already directionally visible -- enough runway for
a hint to show up if there is one). Measured at ~33ms/generation
(see design memo), this is a few minutes total, not hours -- chosen
for feasibility BEFORE running, not adjusted after.

Two files per seed (same shape as chapter 1's, x/y position columns
added to the individual log):

  seed_<n>_timeseries.csv      -- generation, mean_k_out, std_k_out,
                                   mean_k_self, std_k_self, mean_depth,
                                   mean_payoff, frac_isolated (fraction
                                   of agents with 0 rounds played that
                                   generation -- a diagnostic, not part
                                   of chapter 1's format).
  seed_<n>_individual_log.csv  -- generation, agent_index, k_out,
                                   k_self, memory_depth, payoff, x, y --
                                   logged every --log-interval
                                   generations (default 50).

Usage:
    python3 scripts/run_spatial_smoke.py
"""

import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_spatial

OUT_DIR = Path("scripts/output/spatial_smoke_condition3")

SEEDS = [0, 1, 2]
GENERATIONS = 2000
LOG_INTERVAL = 50

# the one straightforward parameter set -- see module docstring; not tuned after results.
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
)

AggRow = Tuple[int, float, float, float, float, float, float, float]
IndividualRow = Tuple[int, int, float, float, int, float, int, int]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    return p.parse_args()


def run_one_seed(seed: int, log_interval: int) -> Tuple[int, List[AggRow], List[IndividualRow], float]:
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
        if s.generation % log_interval == 0:
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
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"condition (3) smoke: seeds={SEEDS}, generations={GENERATIONS}, params={PARAMS}")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=min(4, len(SEEDS))) as executor:
        futures = {
            executor.submit(run_one_seed, seed, args.log_interval): seed for seed in SEEDS
        }
        for future in as_completed(futures):
            seed, agg_rows, individual_rows, elapsed = future.result()
            write_agg_csv(OUT_DIR / f"seed_{seed}_timeseries.csv", agg_rows)
            write_individual_csv(OUT_DIR / f"seed_{seed}_individual_log.csv", individual_rows)
            per_gen_ms = elapsed / GENERATIONS * 1000
            print(f"seed={seed} done: {elapsed:.1f}s total, {per_gen_ms:.1f} ms/generation")

    print(f"\nall {len(SEEDS)} seeds done in {time.time() - t_start:.1f}s wall-clock")


if __name__ == "__main__":
    main()
