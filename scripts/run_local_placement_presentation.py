"""Presentation run for the local-placement experiment.

Full per-generation, per-agent logging (ring position = array index,
K_out, K_self, payoff, immediate-parent index) for k=8, both placement
conditions, seeds 0-1, 2000 generations -- NOT used for the main
statistical analysis (that's run_local_placement_sweep.py's
every-100-gen individual_log.csv). This exists purely to drive the
conference animation and space-time figures, which need every single
generation, not a sampled subset.

Same PARAMS and seed numbering as the main sweep, so these two runs'
seed 0/1 populations start identically.

Output: scripts/output/local_placement_presentation/{placement}_seed{seed}.npz
  arrays (shape (n_gen, population_size) unless noted):
    k_out, k_self, payoff
    cooperation_rate, k_out_assortativity, k_self_assortativity,
      same_parent_fraction (shape (n_gen,) -- population-level scalars)
    parent_of (int, -1 for "no parent yet" at generation 0)

Usage:
    python3 scripts/run_local_placement_presentation.py
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_opponent_count

OUT_DIR = Path("scripts/output/local_placement_presentation")
K = 8
SEEDS = [0, 1]
GENERATIONS = 2000

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


def run_one(placement: str, seed: int) -> dict:
    stats = run_coupled_evolution_opponent_count(
        seed=seed, k=K, generations=GENERATIONS, placement=placement,
        record_parent_history=True, **PARAMS,
    )
    n = PARAMS["population_size"]
    n_gen = len(stats)
    k_out = np.empty((n_gen, n), dtype=np.float32)
    k_self = np.empty((n_gen, n), dtype=np.float32)
    payoff = np.empty((n_gen, n), dtype=np.float32)
    parent_of = np.empty((n_gen, n), dtype=np.int32)
    cooperation_rate = np.empty(n_gen, dtype=np.float32)
    k_out_assortativity = np.empty(n_gen, dtype=np.float32)
    k_self_assortativity = np.empty(n_gen, dtype=np.float32)
    same_parent_fraction = np.empty(n_gen, dtype=np.float32)

    for t, s in enumerate(stats):
        k_out[t] = s.k_out
        k_self[t] = s.k_self
        payoff[t] = s.payoffs
        parent_of[t] = [-1 if p is None else p for p in s.parent_of]
        cooperation_rate[t] = s.cooperation_rate
        k_out_assortativity[t] = s.k_out_assortativity
        k_self_assortativity[t] = s.k_self_assortativity
        same_parent_fraction[t] = s.same_parent_fraction

    return dict(
        k_out=k_out, k_self=k_self, payoff=payoff, parent_of=parent_of,
        cooperation_rate=cooperation_rate,
        k_out_assortativity=k_out_assortativity,
        k_self_assortativity=k_self_assortativity,
        same_parent_fraction=same_parent_fraction,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for placement in ("shuffle", "local"):
        for seed in SEEDS:
            out_path = OUT_DIR / f"{placement}_seed{seed}.npz"
            if out_path.exists():
                print(f"skip (already exists): {out_path}", flush=True)
                continue
            t0 = time.time()
            print(f"running placement={placement} seed={seed} ...", flush=True)
            data = run_one(placement, seed)
            np.savez_compressed(out_path, **data)
            print(f"saved {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
