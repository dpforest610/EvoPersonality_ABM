"""第二章 scale-check analysis -- condition (3) alone, k=8, 10 seeds x
3000 generations.

Calibration question first: does K_self (chapter 1's confirmed, robust
signal) come through clearly at this scale? If yes, this measurement
pipeline/scale can be trusted to judge K_out. If K_self itself is still
too weak/noisy here, the scale (or pipeline) needs fixing before K_out
can be judged at all.

Same statistic as chapter 1: within-generation Spearman corr(individual
K, individual payoff), per seed (seeds are the unit of replication),
pooled over generation >= 1000 (this run's "steady state" region --
shorter than chapter 1's gen>=2000 since this run is only 3000
generations total).

Usage:
    python3 scripts/analyze_spatial_scale_check.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SWEEP_DIR = Path("scripts/output/spatial_scale_check_k8_condition3")
SEEDS = list(range(10))
STEADY_STATE_MIN_GEN = 1000


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
    with open(path) as f:
        return list(csv.DictReader(f))


def per_seed_median_over_steady_state(per_seed_by_gen, seed, key, min_gen):
    rhos = []
    for gen, g in per_seed_by_gen[seed].items():
        if gen < min_gen:
            continue
        if len(set(g[key])) < 2 or len(set(g["payoff"])) < 2:
            continue
        rho, _ = spearmanr(g[key], g["payoff"])
        rhos.append(rho)
    return np.median(rhos) if rhos else float("nan")


def main():
    per_seed_by_gen = {seed: load_individual_log(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())
    print(f"generations logged: {generations[0]}..{generations[-1]} ({len(generations)} snapshots)")
    steady_gens = [g for g in generations if g >= STEADY_STATE_MIN_GEN]
    print(f"steady-state region: gen>={STEADY_STATE_MIN_GEN} ({len(steady_gens)} snapshots)")

    print("\n--- diagnostics: frac_isolated (agents with 0 rounds played that generation) ---")
    for seed in SEEDS:
        ts = load_timeseries(seed)
        frac = [float(r["frac_isolated"]) for r in ts]
        print(f"seed={seed}: mean={np.mean(frac):.4f}  max={np.max(frac):.4f}")

    print("\n--- per-seed median rho over steady state (gen>=1000), the unit of replication (n=10) ---")
    per_seed_median = {"K_out": [], "K_self": []}
    print(f"{'seed':>4}  {'K_out':>8}  {'K_self':>8}")
    for seed in SEEDS:
        ko = per_seed_median_over_steady_state(per_seed_by_gen, seed, "k_out", STEADY_STATE_MIN_GEN)
        ks = per_seed_median_over_steady_state(per_seed_by_gen, seed, "k_self", STEADY_STATE_MIN_GEN)
        per_seed_median["K_out"].append(ko)
        per_seed_median["K_self"].append(ks)
        print(f"{seed:>4}  {ko:>+8.3f}  {ks:>+8.3f}")

    print("\n--- summary across 10 seeds (n=10, the correct unit) ---")
    for channel in ["K_out", "K_self"]:
        vals = np.array(per_seed_median[channel])
        stat, pval = wilcoxon(vals) if not np.allclose(vals, 0) else (float("nan"), 1.0)
        print(f"{channel}: median={np.median(vals):+.3f}  "
              f"IQR=[{np.percentile(vals,25):+.3f},{np.percentile(vals,75):+.3f}]  "
              f"positive in {int(np.sum(vals>0))}/10 seeds  "
              f"Wilcoxon (vs 0) stat={stat:.1f} p={pval:.4f}")

    print("\n--- per-generation median-across-10-seeds trend (direction over time) ---")
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
    print(f"{'gen':>6}  {'K_out':>8}  {'K_self':>8}")
    for i, gen in enumerate(generations):
        print(f"{gen:>6}  {rho_by_gen['K_out'][i]:>+8.3f}  {rho_by_gen['K_self'][i]:>+8.3f}")

    print("\n--- chapter 1 reference (well-mixed, gen>=2000, n=50 seeds) ---")
    print("K_out: median rho=-0.03 (not significant, Wilcoxon p=0.7522)")
    print("K_self: median rho=+0.28 (Wilcoxon p<0.0001, 43-44/50 seeds positive)")


if __name__ == "__main__":
    main()
