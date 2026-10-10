"""Recalibrated scent-model pilot report (M0 unchanged, M1 at
p_move=0.549684, M2 at tau=0.033333): operational check, cooperation
rate, and scent-following index, 5 seeds x 1000 generations each. No
significance tests (per PREREGISTRATION_scent_model.md section 0).

Usage:
    python3 scripts/analyze_scent_pilot_v2.py
"""

import csv
import platform
import re
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
SEEDS = list(range(5))
BURN_IN_MIN_GEN = 100

COLUMNS = [
    "k_out_assortativity", "k_self_assortativity", "same_parent_fraction",
    "move_rate", "scent_index_mean", "isolated_fraction",
    "mean_opponent_count", "max_opponent_count", "cooperation_rate",
]


def machine_tag() -> str:
    raw = platform.node() or "unknown-machine"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "unknown-machine"


def seed_means(root, seed):
    rows = list(csv.DictReader(open(root / f"seed_{seed}_timeseries.csv")))
    out = {}
    for c in COLUMNS:
        if c not in rows[0]:
            out[c] = float("nan")
            continue
        vals = [float(r[c]) for r in rows if int(r["generation"]) >= BURN_IN_MIN_GEN and r[c] != ""]
        out[c] = np.mean(vals) if vals else float("nan")
    return out


def main():
    tag = machine_tag()
    conditions = [
        ("M0 (none)", "none_pilot"),
        ("M1 (random, p_move=0.549684)", "random_pilot_v2"),
        ("M2 (scent, tau=0.033333)", "scent_pilot_v2"),
    ]

    print(f"=== recalibrated scent-model pilot report (5 seeds x 1000 gens, gen>={BURN_IN_MIN_GEN}, "
          f"machine={tag}) ===\n")

    for label, dirname in conditions:
        root = EXPERIMENT_ROOT / dirname / tag
        per_seed = {c: [] for c in COLUMNS}
        for s in SEEDS:
            means = seed_means(root, s)
            for c in COLUMNS:
                per_seed[c].append(means[c])
        print(f"--- {label} ---")
        for c in COLUMNS:
            print(f"  {c:>24}: {np.nanmean(per_seed[c]):.6f}")
        print()

    print("--- max_opponent_count (overall max across seeds) ---")
    for label, dirname in conditions:
        root = EXPERIMENT_ROOT / dirname / tag
        maxes = []
        for s in SEEDS:
            rows = list(csv.DictReader(open(root / f"seed_{s}_timeseries.csv")))
            vals = [int(r["max_opponent_count"]) for r in rows if int(r["generation"]) >= BURN_IN_MIN_GEN]
            maxes.append(max(vals) if vals else 0)
        print(f"{label}: per-seed max = {maxes}, overall max = {max(maxes)}")


if __name__ == "__main__":
    main()
