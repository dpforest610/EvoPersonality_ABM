"""第二章 smoke analysis -- condition (3) alone.

Same statistic as chapter 1 (analyze_selection_vs_drift_correlation.py):
within-generation Spearman corr(individual K, individual payoff), per
seed, then the per-seed values are the unit of replication (median
across seeds, not pooled individuals -- same convention throughout this
project). Direction only is the question here; significance needs the
full multi-generation pooling chapter 1 used, which this smoke is too
short and too few-seeded to attempt honestly.

Usage:
    python3 scripts/analyze_spatial_smoke.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SWEEP_DIR = Path("scripts/output/spatial_smoke_condition3")
SEEDS = [0, 1, 2]


def load_individual_log(seed: int):
    path = SWEEP_DIR / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def load_timeseries(seed: int):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    rows = []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    per_seed_by_gen = {seed: load_individual_log(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())
    print(f"generations logged: {generations[0]}..{generations[-1]} ({len(generations)} snapshots)")

    print("\n--- diagnostics: frac_isolated (agents with 0 rounds played that generation) ---")
    for seed in SEEDS:
        ts = load_timeseries(seed)
        frac = [float(r["frac_isolated"]) for r in ts]
        print(f"seed={seed}: mean={np.mean(frac):.4f}  max={np.max(frac):.4f}")

    rho_by_gen = {"K_out": [], "K_self": []}
    for gen in generations:
        for channel, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for seed in SEEDS:
                g = per_seed_by_gen[seed][gen]
                if len(set(g[key])) < 2 or len(set(g["payoff"])) < 2:
                    continue
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            rho_by_gen[channel].append(np.median(rhos) if rhos else float("nan"))

    print("\n--- per-generation median-across-3-seeds Spearman rho(individual K, individual payoff) ---")
    print(f"{'gen':>6}  {'K_out':>8}  {'K_self':>8}")
    for i, gen in enumerate(generations):
        print(f"{gen:>6}  {rho_by_gen['K_out'][i]:>+8.3f}  {rho_by_gen['K_self'][i]:>+8.3f}")

    def summarize(channel):
        vals = np.array(rho_by_gen[channel])
        first_half = vals[: len(vals) // 2]
        second_half = vals[len(vals) // 2 :]
        return {
            "overall_median": np.median(vals),
            "first_half_median": np.median(first_half),
            "second_half_median": np.median(second_half),
            "n_positive": int(np.sum(vals > 0)),
            "n_total": len(vals),
        }

    print("\n--- summary (direction only -- no significance test at this scale) ---")
    for channel in ["K_out", "K_self"]:
        s = summarize(channel)
        print(f"{channel}: overall median={s['overall_median']:+.3f}  "
              f"first-half median={s['first_half_median']:+.3f}  "
              f"second-half median={s['second_half_median']:+.3f}  "
              f"positive in {s['n_positive']}/{s['n_total']} snapshots")


if __name__ == "__main__":
    main()
