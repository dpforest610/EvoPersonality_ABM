"""Fixed median-split K_self/K_out thresholds for the K-model's 4 types
(white/blue/green/pink -- docs/K_MODEL_MARKOV_DESIGN.md sections 3 and
8-1 point 3). Pooled median across ALL probed seeds and ages >=
--min-generation (gen 1000, matching the pilot's own probe window), from
this script's own run (scripts/run_coupling_markov_pilot.py)'s census
files -- NOT the spatial N=1000 model's thresholds
(0.7770470/0.5025085), which come from an unrelated population (design
doc section 3 explicitly rules this out: candidate (A), the non-spatial
model this pilot uses, needs its own thresholds recomputed from its own
data).

What is fixed here is the PROCEDURE (compute once from the whole
dataset, gen>=1000, and use that one fixed pair of numbers everywhere
else), not any particular numeric value -- the pilot computes its own
pilot-scale values; the eventual production run recomputes from its own
(larger) data, per Annen's decision 3.

Also writes a per-generation median CSV (design doc section 3's "案2",
requested as a sensitivity analysis alongside the pooled fixed
threshold, not a replacement for it) -- at each probed age, the pooled
median across seeds at JUST that age.

Usage:
    python3 scripts/compute_k_model_type_thresholds.py \\
        --data-dir scripts/output/coupling_markov_pilot --seeds 4 --seed-offset 100 \\
        --min-generation 1000 --out-dir scripts/output/coupling_markov_pilot/analysis
"""

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path
from typing import List, Tuple


def census_path(data_dir: Path, seed: int, age: int) -> Path:
    return data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"


def discover_ages(data_dir: Path, seeds: List[int]) -> List[int]:
    """Ages actually present on disk (any seed), sorted -- rather than
    assuming a fixed --probe-interval here too, which could drift out of
    sync with whatever run_coupling_markov_pilot.py was actually run with."""
    ages = set()
    for p in data_dir.glob("seed_*_probe_age_*_census.csv.gz"):
        parts = p.name.split("_")
        age = int(parts[parts.index("age") + 1])
        ages.add(age)
    return sorted(ages)


def load_k_values(path: Path) -> List[Tuple[float, float]]:
    """[(k_out, k_self), ...] repeated `frequency` times each -- same
    convention as compute_scent_n1000_type_thresholds.py's unweighted
    pooled list (frequency is almost always 1 under this project's
    genotype_key, design doc section 8-3, so this rarely differs from
    one row per individual)."""
    out = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            freq = int(row["frequency"])
            k_out = float(row["k_out"])
            k_self = float(row["k_self"])
            out.extend([(k_out, k_self)] * freq)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--min-generation", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default=None, help="defaults to <data-dir>/analysis")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_offset + i for i in range(args.seeds)]
    all_ages = discover_ages(data_dir, seeds)
    pooled_ages = [a for a in all_ages if a >= args.min_generation]

    pooled_k_out: List[float] = []
    pooled_k_self: List[float] = []
    n_per_seed_age = {}
    for age in pooled_ages:
        for seed in seeds:
            cpath = census_path(data_dir, seed, age)
            if not cpath.exists():
                continue
            values = load_k_values(cpath)
            n_per_seed_age[(seed, age)] = len(values)
            for k_out, k_self in values:
                pooled_k_out.append(k_out)
                pooled_k_self.append(k_self)

    threshold_kself = statistics.median(pooled_k_self)
    threshold_kout = statistics.median(pooled_k_out)
    n_total = len(pooled_k_self)

    print(f"pooled n = {n_total} (seeds={seeds}, ages>={args.min_generation}: {pooled_ages})")
    print(f"median K_self (threshold_kself) = {threshold_kself:.7f}")
    print(f"median K_out  (threshold_kout)  = {threshold_kout:.7f}")

    out_json = out_dir / "type_thresholds.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "threshold_kself": threshold_kself,
                "threshold_kout": threshold_kout,
                "n_total": n_total,
                "seeds": seeds,
                "ages_pooled": pooled_ages,
                "min_generation": args.min_generation,
                "note": (
                    "Pooled median across all probed seeds and ages >= min_generation, "
                    "from this run's own coupling_markov_pilot census files -- NOT the "
                    "spatial N=1000 model's thresholds (0.7770470/0.5025085), which come "
                    "from an unrelated population (docs/K_MODEL_MARKOV_DESIGN.md section 3)."
                ),
            },
            f, indent=2,
        )
        f.write("\n")
    print(f"wrote {out_json}")

    # per-generation median sensitivity variant (design doc section 3 "案2")
    per_gen_rows = []
    for age in all_ages:
        age_k_out: List[float] = []
        age_k_self: List[float] = []
        for seed in seeds:
            cpath = census_path(data_dir, seed, age)
            if not cpath.exists():
                continue
            for k_out, k_self in load_k_values(cpath):
                age_k_out.append(k_out)
                age_k_self.append(k_self)
        if not age_k_self:
            continue
        per_gen_rows.append({
            "age": age, "n": len(age_k_self),
            "median_kself": round(statistics.median(age_k_self), 7),
            "median_kout": round(statistics.median(age_k_out), 7),
        })

    out_csv = out_dir / "type_thresholds_by_generation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["age", "n", "median_kself", "median_kout"])
        w.writeheader()
        w.writerows(per_gen_rows)
    print(f"wrote {out_csv} ({len(per_gen_rows)} rows)")


if __name__ == "__main__":
    main()
