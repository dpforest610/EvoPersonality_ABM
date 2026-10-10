"""Scent-model pilot report: operational check, cooperation rate, and
per-generation timing, M0 vs M1 vs M2 (5 seeds x 1000 generations each).
No significance tests (per PREREGISTRATION_scent_model.md section 0:
pilot data is reported but never used for the production analysis A/B).

Usage:
    python3 scripts/analyze_scent_pilot.py
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
    "move_rate", "isolated_fraction", "mean_opponent_count", "max_opponent_count",
    "cooperation_rate",
]


def machine_tag() -> str:
    raw = platform.node() or "unknown-machine"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "unknown-machine"


def seed_means(root, seed):
    rows = list(csv.DictReader(open(root / f"seed_{seed}_timeseries.csv")))
    out = {}
    for c in COLUMNS:
        vals = [float(r[c]) for r in rows if int(r["generation"]) >= BURN_IN_MIN_GEN and r[c] != ""]
        out[c] = np.mean(vals) if vals else float("nan")
    return out


def main():
    tag = machine_tag()
    conditions = [("M0 (none)", "none_pilot"), ("M1 (random)", "random_pilot"), ("M2 (scent)", "scent_pilot")]

    print(f"=== scent-model pilot report (5 seeds x 1000 gens, gen>={BURN_IN_MIN_GEN}, machine={tag}) ===\n")

    header = f"{'condition':>14}  " + "  ".join(f"{c:>22}" for c in COLUMNS)
    print(header)
    for label, dirname in conditions:
        root = EXPERIMENT_ROOT / dirname / tag
        per_seed = {c: [] for c in COLUMNS}
        for s in SEEDS:
            means = seed_means(root, s)
            for c in COLUMNS:
                per_seed[c].append(means[c])
        cells = "  ".join(f"{np.nanmean(per_seed[c]):>22.6f}" for c in COLUMNS)
        print(f"{label:>14}  {cells}")

    print("\n--- max_opponent_count (max over seeds, not mean) ---")
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
