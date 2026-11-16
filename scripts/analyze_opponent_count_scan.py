"""Analyze the coarse opponent-count scan (run_opponent_count_scan.py):
does K_out's selection signal move monotonically with opponent count k,
or is the k=8 effect (experiment A) an isolated artifact?

Same statistic throughout this project: within-generation Spearman
corr(individual K, individual payoff), per seed (seeds are the unit),
median over the steady-state region (gen>=750 here, half of experiment
A's gen>=1000 since this scan is only 1500 generations total). k=99
uses chapter 1's existing well-mixed data (gen>=2000, n=50 seeds)
instead of re-simulating -- ring k=population_size-1 is mathematically
the complete graph, i.e. exactly chapter 1's pairing structure.

Usage:
    python3 scripts/analyze_opponent_count_scan.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_scan")
K_VALUES = [4, 8, 16, 32, 50]
SEEDS = list(range(10))
STEADY_STATE_MIN_GEN = 750

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
CH1_SEEDS = list(range(50))
CH1_MIN_GEN = 2000


def load_individual_log(path):
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def per_seed_medians(sweep_dir, seeds, min_gen):
    result = {"K_out": [], "K_self": []}
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        for channel, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for gen, g in by_gen.items():
                if gen < min_gen:
                    continue
                if len(set(g[key])) < 2 or len(set(g["payoff"])) < 2:
                    continue
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            result[channel].append(np.median(rhos))
    return {k: np.array(v) for k, v in result.items()}


def rank_biserial(values):
    values = np.asarray(values)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


def main():
    print(f"opponent-count scan analysis (gen>={STEADY_STATE_MIN_GEN}, coarse: n=10 seeds/1500 gen per k)\n")

    rows = []
    for k in K_VALUES:
        medians = per_seed_medians(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN)
        r_out, p_out = rank_biserial(medians["K_out"])
        r_self, p_self = rank_biserial(medians["K_self"])
        rows.append((k, medians["K_out"], medians["K_self"], r_out, p_out, r_self, p_self))

    print("=== Chapter 1 reference (well-mixed, k=99, n=50 seeds, gen>=2000) ===")
    ch1 = per_seed_medians(CH1_DIR, CH1_SEEDS, CH1_MIN_GEN)
    r_out_ch1, p_out_ch1 = rank_biserial(ch1["K_out"])
    r_self_ch1, p_self_ch1 = rank_biserial(ch1["K_self"])
    print(f"  k=99(full): K_out median={np.median(ch1['K_out']):+.4f}  r={r_out_ch1:.3f}  p={p_out_ch1:.4f}   "
          f"K_self median={np.median(ch1['K_self']):+.4f}  r={r_self_ch1:.3f}  p={p_self_ch1:.4f}\n")

    print(f"{'k':>4}  {'K_out median':>13}  {'K_out r':>8}  {'K_out p':>9}  |  "
          f"{'K_self median':>14}  {'K_self r':>9}  {'K_self p':>9}")
    for k, kout_vals, kself_vals, r_out, p_out, r_self, p_self in rows:
        print(f"{k:>4}  {np.median(kout_vals):>+13.4f}  {r_out:>8.3f}  {p_out:>9.4f}  |  "
              f"{np.median(kself_vals):>+14.4f}  {r_self:>9.3f}  {p_self:>9.4f}")
    print(f"{99:>4}  {np.median(ch1['K_out']):>+13.4f}  {r_out_ch1:>8.3f}  {p_out_ch1:>9.4f}  |  "
          f"{np.median(ch1['K_self']):>+14.4f}  {r_self_ch1:>9.3f}  {p_self_ch1:>9.4f}   (chapter 1, n=50, gen>=2000)")

    print("\n=== Monotonicity check (K_out median vs k) ===")
    all_k = K_VALUES + [99]
    all_kout_medians = [np.median(r[1]) for r in rows] + [np.median(ch1["K_out"])]
    print("k, K_out median:", list(zip(all_k, [f"{v:+.4f}" for v in all_kout_medians])))
    diffs = np.diff(all_kout_medians)
    print(f"consecutive differences (should be <=0 throughout if 'smaller k -> larger K_out' holds): "
          f"{[f'{d:+.4f}' for d in diffs]}")
    is_monotonic_nonincreasing = all(d <= 1e-9 for d in diffs)
    print(f"strictly monotonic non-increasing in k: {is_monotonic_nonincreasing}")

    from scipy.stats import spearmanr as sp_corr
    rho_kout_vs_k, p_kout_vs_k = sp_corr(all_k, all_kout_medians)
    print(f"\nSpearman correlation of (k, K_out median) across the {len(all_k)} points: "
          f"rho={rho_kout_vs_k:+.3f} p={p_kout_vs_k:.4f} "
          f"(negative = K_out shrinks as k grows, i.e. monotonic in the expected direction)")


if __name__ == "__main__":
    main()
