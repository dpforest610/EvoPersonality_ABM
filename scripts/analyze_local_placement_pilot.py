"""Pilot operational-check report for the local-placement experiment:
k=8, 10 seeds, 1000 generations, local vs shuffle (both run at the same
pilot scale with the same instrumentation, so directly comparable).

Reports, per seed and pooled: same_parent_fraction, k_out_assortativity,
k_self_assortativity, cooperation_rate (gen>=100, skipping the initial
transient). No significance testing -- this is the pilot manipulation
check, not the production analysis.

Usage:
    python3 scripts/analyze_local_placement_pilot.py
"""

import csv
from pathlib import Path

LOCAL_DIR = Path("scripts/output/local_placement_sweep_pilot/k8")
SHUFFLE_DIR = Path("scripts/output/local_placement_sweep_pilot_shuffle/k8")
SEEDS = list(range(10))
BURN_IN_GEN = 100

COLUMNS = ["same_parent_fraction", "k_out_assortativity", "k_self_assortativity", "cooperation_rate"]


def load_timeseries(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def seed_means(sweep_dir, seed):
    rows = load_timeseries(sweep_dir / f"seed_{seed}_timeseries.csv")
    sums = {c: 0.0 for c in COLUMNS}
    counts = {c: 0 for c in COLUMNS}
    for row in rows:
        if int(row["generation"]) < BURN_IN_GEN:
            continue
        for c in COLUMNS:
            v = row[c]
            if v == "":
                continue
            sums[c] += float(v)
            counts[c] += 1
    return {c: (sums[c] / counts[c] if counts[c] else float("nan")) for c in COLUMNS}


def main():
    print(f"=== Pilot operational check: k=8, n={len(SEEDS)} seeds, gen>={BURN_IN_GEN} (of 1000) ===\n")
    print(f"{'seed':>4}  " + "  ".join(f"{c + '(local)':>24}" for c in COLUMNS))
    local_means = {c: [] for c in COLUMNS}
    shuffle_means = {c: [] for c in COLUMNS}
    for seed in SEEDS:
        local = seed_means(LOCAL_DIR, seed)
        shuffle = seed_means(SHUFFLE_DIR, seed)
        for c in COLUMNS:
            local_means[c].append(local[c])
            shuffle_means[c].append(shuffle[c])

    print(f"{'seed':>4}  " + "  ".join(f"{c:>22}(local / shuffle)" for c in COLUMNS))
    for i, seed in enumerate(SEEDS):
        cells = "  ".join(
            f"{local_means[c][i]:>10.4f} / {shuffle_means[c][i]:<9.4f}" for c in COLUMNS
        )
        print(f"{seed:>4}  {cells}")

    print(f"\n{'mean over 10 seeds':>19}  " + "  ".join(f"{c:>22}" for c in COLUMNS))
    local_pooled = {c: sum(local_means[c]) / len(SEEDS) for c in COLUMNS}
    shuffle_pooled = {c: sum(shuffle_means[c]) / len(SEEDS) for c in COLUMNS}
    print("local  : " + "  ".join(f"{local_pooled[c]:>22.4f}" for c in COLUMNS))
    print("shuffle: " + "  ".join(f"{shuffle_pooled[c]:>22.4f}" for c in COLUMNS))


if __name__ == "__main__":
    main()
