"""Apply the full Markov-network metric suite (markov_apply.py) to the
pilot's probe data -- docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md sections
3.3/7.2, exploratory/pilot-scoped per the user's 2026-10-16 request.

For each (noise condition, seed, probe age, opponent, group_size),
resamples --n-boot virtual-participant groups (weighted by genotype
frequency) and aggregates: widest-path recovery/descent, rate of
底底底/和和和 presence, rate of widest-path undefined, Table 7 j* for
the opponent's named circuits, largest j>=.40 strongly-connected
component, and similarity (correlation + edge-Jaccard@.30) to each
human group's own network.

Usage:
    python3 scripts/analyze_markov_sweep_pilot_network.py --n-boot 200 --workers 4

Writes scripts/output/markov_sweep_pilot_network.csv (committed,
long-format: one row per seed x age x opponent x group_size).
"""

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_apply import (
    OPPONENT_INFO,
    bootstrap_aggregate,
    load_census,
    load_human_weights,
    load_series_by_genotype_all_opponents,
)

CONDITIONS = [
    ("noise_rate=0.01", "scripts/output/markov_sweep_pilot_noise01", "noise0p01"),
    ("noise_rate=0.0", "scripts/output/markov_sweep_pilot_noise00", "noise0"),
]
GROUP_SIZES = (56, 46, 69)
OPPONENTS = list(OPPONENT_INFO)  # TFT, AllC, AllD, Random, Gradual


def process_one_seed(
    condition_name: str, out_dir: str, moves_suffix: str, seed: int,
    ages: List[int], n_boot: int, human_weights_path: str,
) -> List[dict]:
    out_dir_p = Path(out_dir)
    human_weights = load_human_weights(Path(human_weights_path))
    rng = random.Random(seed)
    rows: List[dict] = []

    for age in ages:
        census_path = out_dir_p / f"seed_{seed}_probe_age_{age}_census.csv.gz"
        moves_path = out_dir_p / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        if not census_path.exists() or not moves_path.exists():
            continue
        frequencies = load_census(census_path)
        series_by_opponent = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=20)

        for opponent in OPPONENTS:
            label, _vocab = OPPONENT_INFO[opponent]
            human_weights_by_group = {
                g: human_weights.get((label, g), {}) for g in ("Secure", "Mix", "Fearful")
            }
            series_by_genotype = series_by_opponent.get(opponent, {})
            for group_size in GROUP_SIZES:
                result = bootstrap_aggregate(
                    frequencies, series_by_genotype, opponent, group_size, n_boot, rng,
                    human_weights_by_group,
                )
                row = {
                    "condition": condition_name, "seed": seed, "age": age,
                    "opponent": opponent, "group_size": group_size,
                }
                row.update(result)
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--max-age", type=int, default=2000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--human-weights", type=str, default="docs/human_network_weights.csv")
    p.add_argument("--out", type=str, default="scripts/output/markov_sweep_pilot_network.csv")
    args = p.parse_args()

    ages = list(range(0, args.max_age + 1, args.probe_interval))
    t0 = time.time()
    all_rows: List[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for condition_name, out_dir, moves_suffix in CONDITIONS:
            for seed in range(args.seeds):
                fut = ex.submit(
                    process_one_seed, condition_name, out_dir, moves_suffix, seed,
                    ages, args.n_boot, args.human_weights,
                )
                futures[fut] = (condition_name, seed)
        for fut in as_completed(futures):
            condition_name, seed = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            print(f"done: {condition_name} seed={seed} ({len(rows)} rows) elapsed={time.time()-t0:.0f}s", flush=True)

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
