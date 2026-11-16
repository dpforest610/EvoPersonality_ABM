"""第二章: is K_out's spatial-condition effect (+0.044, experiment A)
practically meaningful, or "significant but essentially zero" next to
K_self's selection signal? Pure re-analysis of already-saved logs (chapter
1's well-mixed sweep and experiment A's k=8/n_phases=1/static sweep) --
no new simulation.

Reports, for each channel in each condition:
  - median and IQR of the per-seed median rho(individual K, individual payoff)
    over the steady-state region (seeds are the unit, n=50 throughout)
  - the matched-pairs rank-biserial correlation r for the one-sample
    Wilcoxon test against 0 -- a standard, scale-free non-parametric
    effect size (r=1: every seed's sign agrees and dominates; r=0: no
    net signal) -- so K_out and K_self can be compared on the same
    footing regardless of their raw rho units.

Usage:
    python3 scripts/analyze_kout_effect_size.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon


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


def rank_biserial_effect_size(values):
    """Matched-pairs rank-biserial correlation for a one-sample Wilcoxon
    test of `values` against 0: r = 1 - 2*W/(n*(n+1)), where W is the
    smaller of the positive/negative rank sums (scipy's default
    `statistic`). r in [0,1]; 0 = no net signal, 1 = every value agrees
    in sign and the signed-rank test is maximally lopsided."""
    values = np.asarray(values)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p, n


def report(name, values):
    r, p, n = rank_biserial_effect_size(values)
    print(f"  {name}: median={np.median(values):+.4f}  IQR=[{np.percentile(values,25):+.4f},"
          f"{np.percentile(values,75):+.4f}]  n={n}  positive={int(np.sum(values>0))}/{n}  "
          f"Wilcoxon p={p:.4f}  rank-biserial r={r:.3f}")


def main():
    seeds = list(range(50))

    print("=== Chapter 1: well-mixed (n=100, all pairs, gen>=2000) ===")
    ch1 = per_seed_medians("scripts/output/selection_vs_drift_sweep_with_depth", seeds, min_gen=2000)
    report("K_out ", ch1["K_out"])
    report("K_self", ch1["K_self"])

    print("\n=== Experiment A: spatial k=8/Moore, n_phases=1, static, reshuffled every gen (gen>=1000) ===")
    expA = per_seed_medians("scripts/output/spatial_scale_check_k8_n1_static", seeds, min_gen=1000)
    report("K_out ", expA["K_out"])
    report("K_self", expA["K_self"])

    print("\n=== Ratio: how big is K_out's effect relative to K_self's, on the SAME rank-biserial scale? ===")
    r_kout_ch1, _, _ = rank_biserial_effect_size(ch1["K_out"])
    r_kself_ch1, _, _ = rank_biserial_effect_size(ch1["K_self"])
    r_kout_expA, _, _ = rank_biserial_effect_size(expA["K_out"])
    r_kself_expA, _, _ = rank_biserial_effect_size(expA["K_self"])
    print(f"  chapter 1:   r(K_out)={r_kout_ch1:.3f}  r(K_self)={r_kself_ch1:.3f}  "
          f"ratio K_out/K_self={r_kout_ch1/r_kself_ch1:.3f}" if r_kself_ch1 else "n/a")
    print(f"  experiment A: r(K_out)={r_kout_expA:.3f}  r(K_self)={r_kself_expA:.3f}  "
          f"ratio K_out/K_self={r_kout_expA/r_kself_expA:.3f}")
    print(f"\n  raw-rho ratio (experiment A): median(K_out)/median(K_self) = "
          f"{np.median(expA['K_out'])/np.median(expA['K_self']):.3f}")
    print(f"  raw-rho ratio (K_out expA vs K_self chapter1, cross-experiment reference): "
          f"{np.median(expA['K_out'])/np.median(ch1['K_self']):.3f}")


if __name__ == "__main__":
    main()
