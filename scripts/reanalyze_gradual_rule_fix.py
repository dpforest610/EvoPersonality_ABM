"""Recompute ONLY Gradual/GradualAppeasement's descriptive rows (section 6
of the preregistration) from the re-probed data written by
scripts/reprobe_gradual_rule_fix.py, and splice them into a copy of the
existing confirmatory descriptive CSV -- leaving the other 5 opponents'
rows (and the confirmatory H1-H3 results, which don't use Gradual at all)
byte-for-byte untouched. Also writes a before/after comparison summary,
per docs/DEVIATION_gradual_rule.md's "report the pre-fix values alongside".

Reads:
  - --data-dir: original production dir (for the unchanged genotype
    censuses -- same as scripts/reprobe_gradual_rule_fix.py read).
  - --reprobe-dir: output of scripts/reprobe_gradual_rule_fix.py (the
    corrected Gradual/GradualAppeasement moves+summary files).
  - --original-descriptive-csv: the already-committed
    markov_confirmatory_descriptive_<role>.csv from the pre-fix run.

Writes, to --results-dir (does not touch --original-descriptive-csv):
  - markov_confirmatory_descriptive_<role>.csv (merged: all opponents,
    with Gradual/GradualAppeasement rows replaced by the corrected ones)
  - markov_confirmatory_gradual_rule_fix_comparison_<role>.csv
    (window-averaged before/after values per opponent/scheme/group_size/metric)

The merged CSV is then a drop-in --descriptive-csv for the existing
figure scripts (plot_markov_confirmatory_trends.py etc.), which can be
re-run on Mac or elsewhere exactly as in runbook section 8-6.

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 9.
"""

import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import reprobe_gradual_rule_fix as reprobe_mod  # noqa: E402
import run_markov_confirmatory_analysis as rmca  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_apply import (  # noqa: E402
    load_census,
    load_genotype_identities,
    load_series_by_genotype_all_opponents,
)

REPROBE_OPPONENTS = reprobe_mod.REPROBE_OPPONENTS  # ["Gradual", "GradualAppeasement"]

# metrics compared before/after (bootstrap_aggregate_with_sampler's output
# keys that are meaningful for opponents with no human reference data --
# correlation/Jaccard keys are skipped since Gradual/GradualAppeasement
# have no human_label and those columns are always empty either way)
COMPARISON_METRICS = [
    "mean_recovery", "mean_descent", "mean_n_distinct_genotypes",
    "mean_mean_pairwise_jaccard_distance", "mean_rate_bottom3_present",
    "mean_rate_top3_present", "mean_largest_scc_size",
]


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def load_seed_pools_for_age_reprobe(
    data_dir: Path, reprobe_dir: Path, probe_noise_rate: float, age: int, seeds: List[int],
) -> Dict[int, dict]:
    pools: Dict[int, dict] = {}
    for seed in seeds:
        census_path = data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
        moves_path = reprobe_mod.reprobe_moves_path(reprobe_dir, seed, age, probe_noise_rate)
        if not census_path.exists() or not moves_path.exists():
            continue
        frequencies = load_census(census_path)
        identities = load_genotype_identities(census_path)
        series_by_opponent = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=20)
        per_opponent = {}
        for opponent in REPROBE_OPPONENTS:
            series_by_genotype = series_by_opponent.get(opponent, {})
            if series_by_genotype:
                per_opponent[opponent] = (frequencies, series_by_genotype, identities)
        pools[seed] = per_opponent
    return pools


def process_one_age_reprobe(
    data_dir_str: str, reprobe_dir_str: str, probe_noise_rate: float, age: int, seeds: List[int],
    human_weights_path: str, n_boot_descriptive: int, n_boot_scheme_a: int, master_seed: str,
    group_for_human: Dict[str, int], cycle_thresholds: Optional[List[float]],
) -> dict:
    pools = load_seed_pools_for_age_reprobe(
        Path(data_dir_str), Path(reprobe_dir_str), probe_noise_rate, age, seeds
    )
    return rmca.process_one_age(
        data_dir_str, "", age, seeds, human_weights_path,
        n_boot_descriptive,  # n_boot_point unused (neither opponent is in H_OPPONENTS)
        n_boot_descriptive, n_boot_scheme_a,
        0,  # n_bootstrap_seed=0: no seed-level bootstrap for non-confirmatory opponents regardless
        master_seed, group_for_human, cycle_thresholds,
        opponents=REPROBE_OPPONENTS, pools=pools,
    )


def run_all_ages_reprobe(
    data_dir: Path, reprobe_dir: Path, probe_noise_rate: float, ages: List[int], seeds: List[int],
    human_weights_path: str, n_boot_descriptive: int, n_boot_scheme_a: int, master_seed: str,
    workers: int, cache_dir: Path, group_for_human: Dict[str, int],
    cycle_thresholds: Optional[List[float]],
) -> List[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age_reprobe, str(data_dir), str(reprobe_dir), probe_noise_rate, age, seeds,
                human_weights_path, n_boot_descriptive, n_boot_scheme_a, master_seed,
                group_for_human, cycle_thresholds,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    descriptive_rows: List[dict] = []
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        descriptive_rows.extend(result["descriptive_rows"])
    return descriptive_rows


def load_csv_rows(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_merged_csv(path: Path, kept_rows: List[dict], new_rows: List[dict]) -> None:
    all_rows = kept_rows + new_rows
    fieldnames = list({k for row in all_rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


def get_float(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean_ignoring_nan(values: List[float]) -> float:
    clean = [v for v in values if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def window_average_by_key(rows: List[dict], window: Tuple[int, int]) -> Dict[Tuple[str, str, str, str], float]:
    """{(opponent, scheme, group_size, metric): window-averaged value}."""
    by_key: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        age = int(row["age"])
        if not (window[0] <= age <= window[1]):
            continue
        for metric in COMPARISON_METRICS:
            if metric in row:
                by_key[(row["opponent"], row["scheme"], str(row["group_size"]), metric)].append(get_float(row, metric))
    return {k: mean_ignoring_nan(v) for k, v in by_key.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="original production dir (for the unchanged genotype censuses)")
    p.add_argument("--reprobe-dir", type=str, required=True, help="output of scripts/reprobe_gradual_rule_fix.py")
    p.add_argument("--probe-noise-rate", type=float, required=True)
    p.add_argument("--role", choices=["primary", "sensitivity"], required=True)
    p.add_argument("--original-descriptive-csv", type=str, required=True, help="the already-committed pre-fix markov_confirmatory_descriptive_<role>.csv")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--n-boot-descriptive", type=int, default=200)
    p.add_argument("--n-boot-scheme-a", type=int, default=30)
    p.add_argument("--master-seed", type=str, default="markov_confirmatory_v1", help="same master seed as the original confirmatory run, for RNG-stream continuity")
    p.add_argument("--human-weights", type=str, default="docs/human_network_weights.csv")
    p.add_argument("--group-sizes", type=str, default="Secure:56,Mix:46,Fearful:69")
    p.add_argument(
        "--cycle-thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
    )
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--results-dir", type=str, required=True)
    args = p.parse_args()

    cycle_thresholds = [float(x) for x in args.cycle_thresholds.split(",") if x.strip()] or None
    group_for_human: Dict[str, int] = {}
    for part in args.group_sizes.split(","):
        name, size = part.split(":")
        group_for_human[name] = int(size)

    data_dir = Path(args.data_dir)
    reprobe_dir = Path(args.reprobe_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / f"_age_cache_{args.role}_gradual_reprobe"

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.window_start, args.window_end + 1, args.probe_interval))
    window = (args.window_start, args.window_end)

    t0 = time.time()
    new_gradual_rows = run_all_ages_reprobe(
        data_dir, reprobe_dir, args.probe_noise_rate, ages, seeds, args.human_weights,
        args.n_boot_descriptive, args.n_boot_scheme_a, args.master_seed, args.workers, cache_dir,
        group_for_human, cycle_thresholds,
    )
    print(f"all ages done in {time.time() - t0:.0f}s, {len(new_gradual_rows)} new Gradual/GradualAppeasement rows", flush=True)

    original_rows = load_csv_rows(Path(args.original_descriptive_csv))
    kept_rows = [r for r in original_rows if r["opponent"] not in REPROBE_OPPONENTS]
    old_gradual_rows = [r for r in original_rows if r["opponent"] in REPROBE_OPPONENTS]

    merged_path = results_dir / f"markov_confirmatory_descriptive_{args.role}.csv"
    write_merged_csv(merged_path, kept_rows, new_gradual_rows)
    print(f"wrote merged descriptive CSV ({len(kept_rows)} unchanged + {len(new_gradual_rows)} corrected rows) to {merged_path}", flush=True)

    before = window_average_by_key(old_gradual_rows, window)
    after = window_average_by_key(new_gradual_rows, window)
    comparison_path = results_dir / f"markov_confirmatory_gradual_rule_fix_comparison_{args.role}.csv"
    with open(comparison_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "scheme", "group_size", "metric", "value_before_fix", "value_after_fix", "delta"])
        for key in sorted(set(before) | set(after)):
            opponent, scheme, group_size, metric = key
            b, a = before.get(key, float("nan")), after.get(key, float("nan"))
            delta = a - b if (a == a and b == b) else float("nan")
            w.writerow([opponent, scheme, group_size, metric, f"{b:.4f}" if b == b else "", f"{a:.4f}" if a == a else "", f"{delta:.4f}" if delta == delta else ""])
    print(f"wrote before/after comparison to {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
