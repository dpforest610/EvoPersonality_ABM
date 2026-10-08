"""Local-placement production: operational-check table for the local
condition only (k=4,8,16,32, gen>=2000, seed median/IQR, n=50 seeds).
same_parent_fraction, k_out_assortativity, k_self_assortativity --
directly from timeseries.csv's per-generation columns (no recomputation
needed, already saved by the simulation).

Usage:
    python3 scripts/analyze_local_placement_operational_check.py
"""

import csv
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000

COLUMNS = ["same_parent_fraction", "k_out_assortativity", "k_self_assortativity"]


def seed_means(sweep_dir, seed):
    rows = list(csv.DictReader(open(Path(sweep_dir) / f"seed_{seed}_timeseries.csv")))
    sums = {c: 0.0 for c in COLUMNS}
    counts = {c: 0 for c in COLUMNS}
    for row in rows:
        if int(row["generation"]) < STEADY_STATE_MIN_GEN:
            continue
        for c in COLUMNS:
            v = row[c]
            if v == "":
                continue
            sums[c] += float(v)
            counts[c] += 1
    return {c: (sums[c] / counts[c] if counts[c] else float("nan")) for c in COLUMNS}


def main():
    print(f"=== local placement: operational check (gen>={STEADY_STATE_MIN_GEN}, n=50 seeds) ===\n")
    print(f"{'k':>4}  {'same_parent_fraction':>24}  {'k_out_assortativity':>23}  {'k_self_assortativity':>24}")
    for k in K_VALUES:
        per_seed = {c: [] for c in COLUMNS}
        for s in SEEDS:
            means = seed_means(LOCAL_ROOT / f"k{k}", s)
            for c in COLUMNS:
                per_seed[c].append(means[c])
        cells = []
        for c in COLUMNS:
            arr = np.array(per_seed[c])
            med = np.nanmedian(arr)
            iqr = (np.nanpercentile(arr, 25), np.nanpercentile(arr, 75))
            cells.append(f"{med:.4f} [{iqr[0]:.4f},{iqr[1]:.4f}]")
        print(f"{k:>4}  {cells[0]:>24}  {cells[1]:>23}  {cells[2]:>24}")


if __name__ == "__main__":
    main()
