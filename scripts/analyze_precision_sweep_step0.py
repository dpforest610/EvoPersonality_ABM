"""第二章 精密kスイープ ステップ0: 既存9点データのみでの事前解析(新規シミュレーション不要)。

1. 9点のWilcoxon(vs 0)p値にHolm補正(K_out・K_self両方)。
2. k=40 vs k=50 のK_out対比較(Mann-Whitney U、rank-biserial効果量)。
3. シードごとのK_out相関 ÷ K_self相関 をkに対して要約
   (|K_self相関|<0.05のシードは除外、除外数を併記、中央値・IQR)。

Usage:
    python3 scripts/analyze_precision_sweep_step0.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000
RATIO_EXCLUDE_THRESHOLD = 0.05


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


def rank_biserial_one_sample(values):
    values = np.asarray(values)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the ORIGINAL order of p_values (standard: monotone non-decreasing
    enforced via running max, capped at 1.0)."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        multiplier = n - rank
        val = min(1.0, p_values[idx] * multiplier)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def main():
    data = {k: per_seed_medians(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN) for k in K_VALUES}

    print("=== 1. Holm補正付きWilcoxon検定表(n=50、gen>=2000) ===\n")
    for channel in ["K_out", "K_self"]:
        raw_p = []
        r_vals = []
        medians = []
        for k in K_VALUES:
            r, p = rank_biserial_one_sample(data[k][channel])
            raw_p.append(p)
            r_vals.append(r)
            medians.append(np.median(data[k][channel]))
        holm_p = holm_correction(raw_p)
        print(f"--- {channel} ---")
        print(f"{'k':>4}  {'median':>9}  {'r':>7}  {'raw p':>9}  {'Holm p':>9}  {'sig(Holm)':>10}")
        for k, med, r, p, hp in zip(K_VALUES, medians, r_vals, raw_p, holm_p):
            sig = "***" if hp < 0.001 else "**" if hp < 0.01 else "*" if hp < 0.05 else "n.s."
            print(f"{k:>4}  {med:>+9.4f}  {r:>7.3f}  {p:>9.4f}  {hp:>9.4f}  {sig:>10}")
        print()

    print("=== 2. k=40 vs k=50 のK_out対比較(Mann-Whitney U、対応なし、n=50ずつ) ===")
    a = data[40]["K_out"]
    b = data[50]["K_out"]
    u_stat, p = mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    print(f"k=40 median={np.median(a):+.4f}  k=50 median={np.median(b):+.4f}")
    print(f"Mann-Whitney U={u_stat:.1f}  p={p:.4f}  rank-biserial r={r_rb:+.3f}\n")

    print(f"=== 3. per-seed K_out/K_self比(|K_self median rho|<{RATIO_EXCLUDE_THRESHOLD}のシードを除外) ===")
    print(f"{'k':>4}  {'n_excluded':>10}  {'n_used':>7}  {'ratio median':>13}  {'ratio IQR':>24}")
    ratio_by_k = {}
    for k in K_VALUES:
        kout = data[k]["K_out"]
        kself = data[k]["K_self"]
        mask = np.abs(kself) >= RATIO_EXCLUDE_THRESHOLD
        n_excluded = int(np.sum(~mask))
        ratios = kout[mask] / kself[mask]
        ratio_by_k[k] = ratios
        iqr = f"[{np.percentile(ratios,25):+.3f},{np.percentile(ratios,75):+.3f}]"
        print(f"{k:>4}  {n_excluded:>10}  {int(np.sum(mask)):>7}  {np.median(ratios):>+13.4f}  {iqr:>24}")

    return data, ratio_by_k


if __name__ == "__main__":
    main()
