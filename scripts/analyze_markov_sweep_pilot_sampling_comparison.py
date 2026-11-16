"""Compare virtual-participant sampling schemes (A) within-seed vs (B)
cross-lineage on the pilot's probe data -- requested 2026-10-17,
exploratory. Only n=10 is available in the pilot (10 seeds total, so
scheme (B) can draw at most 10 distinct seeds).

(A) 集団内抽出: resample_within_seed_group_with_identity -- n individuals
    from ONE seed's population, weighted by genotype frequency (the
    existing scheme used throughout section 3.3's main analysis).
(B) 系統間抽出: resample_cross_seed_group -- n DISTINCT seeds, one
    individual per seed (frequency-weighted within that seed).

For each (noise condition, opponent, age), both schemes are applied
with group_size=10 (all 10 pilot seeds), n_boot bootstrap draws each,
reporting: widest-path recovery/descent + undefined rates, largest
j>=.40 SCC, correlation to each human group (Table 7 j* and edge-Jaccard
to human groups are design-doc-standard but segregated from the mean
reported here into the full CSV; see plot script for the figures) and
the two group-diversity metrics (n_distinct_genotypes, mean pairwise
3-gram-set Jaccard distance).

Usage:
    python3 scripts/analyze_markov_sweep_pilot_sampling_comparison.py --n-boot 200 --workers 4
"""

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_apply import (
    OPPONENT_INFO,
    bootstrap_aggregate_with_sampler,
    load_census,
    load_genotype_identities,
    load_human_weights,
    load_series_by_genotype_all_opponents,
    resample_cross_seed_group,
    resample_within_seed_group_with_identity,
)
from evopersonality_abm.markov_recovery import load_vocab

CONDITIONS = [
    ("noise_rate=0.01", "scripts/output/markov_sweep_pilot_noise01", "noise0p01"),
    ("noise_rate=0.0", "scripts/output/markov_sweep_pilot_noise00", "noise0"),
]
GROUP_SIZE = 10  # all 10 pilot seeds -- the most scheme (B) can draw without replacement
OPPONENTS = list(OPPONENT_INFO)


def load_all_seed_pools(out_dir: Path, moves_suffix: str, age: int, seeds: range):
    """{seed: (frequencies, series_by_genotype, identities)} for every
    seed at one probe age, for scheme (B)'s cross-seed draws; also reused
    directly for scheme (A) (each seed's own pool)."""
    pools = {}
    series_by_opponent_by_seed = {}
    for seed in seeds:
        census_path = out_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
        moves_path = out_dir / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        if not census_path.exists() or not moves_path.exists():
            continue
        frequencies = load_census(census_path)
        identities = load_genotype_identities(census_path)
        series_by_opponent = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=20)
        pools[seed] = (frequencies, identities, series_by_opponent)
    return pools


def process_one_age(
    condition_name: str, out_dir: str, moves_suffix: str, age: int, n_seeds: int,
    n_boot: int, human_weights_path: str,
) -> List[dict]:
    out_dir_p = Path(out_dir)
    human_weights = load_human_weights(Path(human_weights_path))
    rng = random.Random(age)
    rows: List[dict] = []

    pools = load_all_seed_pools(out_dir_p, moves_suffix, age, range(n_seeds))
    if len(pools) < GROUP_SIZE:
        return rows

    for opponent in OPPONENTS:
        label, vocab_name = OPPONENT_INFO[opponent]
        vocab = load_vocab(vocab_name)
        human_weights_by_group = {g: human_weights.get((label, g), {}) for g in ("Secure", "Mix", "Fearful")}

        # scheme (A): group_size=10 individuals from ONE seed's population.
        # Loop over all seeds that have data, average results across them
        # (each seed contributes its own (A)-sampled distribution).
        a_results = []
        for seed, (frequencies, identities, series_by_opponent) in pools.items():
            series_by_genotype = series_by_opponent.get(opponent, {})
            if not series_by_genotype:
                continue

            def sample_fn_a(rng, frequencies=frequencies, series_by_genotype=series_by_genotype, identities=identities):
                return resample_within_seed_group_with_identity(frequencies, series_by_genotype, identities, GROUP_SIZE, rng)

            a_results.append(bootstrap_aggregate_with_sampler(
                sample_fn_a, opponent, n_boot, rng, human_weights_by_group, vocab, with_diversity=True,
            ))

        if a_results:
            # different seeds' results can have different sets of mean_X
            # keys present (a seed where a metric was undefined for every
            # bootstrap draw simply has no mean_X key for it, same as
            # bootstrap_aggregate's own per-seed behavior) -- average over
            # whichever subset of seeds actually defines each key.
            all_keys = {k for r in a_results for k in r}
            avg_a = {}
            for k in all_keys:
                vals = [r[k] for r in a_results if k in r]
                avg_a[k] = sum(vals) / len(vals) if vals else float("nan")
            rows.append({"condition": condition_name, "scheme": "A_within_seed", "age": age, "opponent": opponent, **avg_a})

        # scheme (B): group_size=10 DISTINCT seeds, one individual each.
        seed_pools_b = {
            s: (pools[s][0], pools[s][2].get(opponent, {}), pools[s][1])
            for s in pools if pools[s][2].get(opponent, {})
        }
        if len(seed_pools_b) >= GROUP_SIZE:
            def sample_fn_b(rng, seed_pools_b=seed_pools_b):
                return resample_cross_seed_group(seed_pools_b, GROUP_SIZE, rng)

            result_b = bootstrap_aggregate_with_sampler(
                sample_fn_b, opponent, n_boot, rng, human_weights_by_group, vocab, with_diversity=True,
            )
            rows.append({"condition": condition_name, "scheme": "B_cross_seed", "age": age, "opponent": opponent, **result_b})

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--max-age", type=int, default=2000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--human-weights", type=str, default="docs/human_network_weights.csv")
    p.add_argument("--out", type=str, default="scripts/output/markov_sweep_pilot_sampling_comparison.csv")
    args = p.parse_args()

    ages = list(range(0, args.max_age + 1, args.probe_interval))
    t0 = time.time()
    all_rows: List[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for condition_name, out_dir, moves_suffix in CONDITIONS:
            for age in ages:
                fut = ex.submit(
                    process_one_age, condition_name, out_dir, moves_suffix, age,
                    args.n_seeds, args.n_boot, args.human_weights,
                )
                futures[fut] = (condition_name, age)
        for fut in as_completed(futures):
            condition_name, age = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            print(f"done: {condition_name} age={age} ({len(rows)} rows) elapsed={time.time()-t0:.0f}s", flush=True)

    if not all_rows:
        print("no rows produced", flush=True)
        return

    fieldnames = list(all_rows[0].keys())
    for row in all_rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {out_path} in {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
