"""Analyze the precision opponent-count sweep (run_opponent_count_precision_sweep.py):
full chapter-1 precision (n=50 seeds, 5000 generations, gen>=2000 steady
state) at k=4,6,8,12,16,24,32,40,50, plus chapter 1's own well-mixed
data as the k=99 reference point (ring k=population_size-1 is
mathematically the complete graph).

Only analyzes k values whose 50 seeds are ALL present -- prints
whichever subset is ready, so this can be run as an interim check
before the full sweep finishes.

Usage:
    python3 scripts/analyze_opponent_count_precision_sweep.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000  # matches chapter 1 exactly (this sweep is 5000 gen)

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
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
    ready_k = []
    for k in K_VALUES:
        k_dir = SCAN_ROOT / f"k{k}"
        n_ready = sum(
            1 for s in SEEDS
            if (k_dir / f"seed_{s}_timeseries.csv").exists()
            and (k_dir / f"seed_{s}_individual_log.csv").exists()
        )
        if n_ready == len(SEEDS):
            ready_k.append(k)
        print(f"k={k}: {n_ready}/{len(SEEDS)} seeds ready" + ("" if n_ready == len(SEEDS) else "  (SKIPPED, incomplete)"))

    print(f"\nanalyzing {len(ready_k)} complete k value(s): {ready_k}\n")
    print(f"(gen>={STEADY_STATE_MIN_GEN}, n=50 seeds per k, matching chapter 1 exactly)\n")

    print("=== Chapter 1 reference (well-mixed, k=99, n=50 seeds, gen>=2000) ===")
    ch1 = per_seed_medians(CH1_DIR, SEEDS, CH1_MIN_GEN)
    r_out_ch1, p_out_ch1 = rank_biserial(ch1["K_out"])
    r_self_ch1, p_self_ch1 = rank_biserial(ch1["K_self"])
    print(f"  k=99: K_out median={np.median(ch1['K_out']):+.4f}  r={r_out_ch1:.3f}  p={p_out_ch1:.4f}   "
          f"K_self median={np.median(ch1['K_self']):+.4f}  r={r_self_ch1:.3f}  p={p_self_ch1:.4f}\n")

    rows = []
    for k in ready_k:
        medians = per_seed_medians(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN)
        r_out, p_out = rank_biserial(medians["K_out"])
        r_self, p_self = rank_biserial(medians["K_self"])
        rows.append((k, medians["K_out"], medians["K_self"], r_out, p_out, r_self, p_self))

    print(f"{'k':>4}  {'K_out median':>13}  {'K_out IQR':>22}  {'K_out r':>8}  {'K_out p':>9}  |  "
          f"{'K_self median':>14}  {'K_self r':>9}  {'K_self p':>9}")
    for k, kout_vals, kself_vals, r_out, p_out, r_self, p_self in rows:
        iqr = f"[{np.percentile(kout_vals,25):+.3f},{np.percentile(kout_vals,75):+.3f}]"
        print(f"{k:>4}  {np.median(kout_vals):>+13.4f}  {iqr:>22}  {r_out:>8.3f}  {p_out:>9.4f}  |  "
              f"{np.median(kself_vals):>+14.4f}  {r_self:>9.3f}  {p_self:>9.4f}")
    print(f"{99:>4}  {np.median(ch1['K_out']):>+13.4f}  {'':>22}  {r_out_ch1:>8.3f}  {p_out_ch1:>9.4f}  |  "
          f"{np.median(ch1['K_self']):>+14.4f}  {r_self_ch1:>9.3f}  {p_self_ch1:>9.4f}   (chapter 1)")

    if len(ready_k) >= 2:
        print("\n=== Shape check (K_out median vs k, complete points only) ===")
        all_k = ready_k + [99]
        all_kout_medians = [np.median(r[1]) for r in rows] + [np.median(ch1["K_out"])]
        for k, v in zip(all_k, all_kout_medians):
            print(f"  k={k}: K_out median={v:+.4f}")
        peak_idx = int(np.argmax(all_kout_medians[:-1])) if len(all_kout_medians) > 1 else 0
        print(f"\n  peak among ready points (excluding k=99 reference): k={all_k[peak_idx]} "
              f"(K_out median={all_kout_medians[peak_idx]:+.4f})")


if __name__ == "__main__":
    main()
