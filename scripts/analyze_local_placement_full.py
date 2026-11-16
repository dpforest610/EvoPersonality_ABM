"""Local-placement production: FULL ANALYSIS (sections B, C, D of
docs/PREREGISTRATION_local_placement.md -- the primary endpoint,
section A, is reported separately by
analyze_local_placement_primary_endpoint.py and is not repeated here).

Run only after:
  1. analyze_local_placement_primary_endpoint.py has been run/reported.
  2. verify_shuffle_replay_matches_existing.py reports OK (if it does
     not, stop and fix the replay before trusting anything here).

B. Context analysis (Heisler & Damuth 1987): per seed/generation
   (gen>=2000), regress relative fitness on own K_self/K_out AND the
   generation-standardized mean K_self/K_out of this generation's
   ACTUAL duel partners (opponent_mean_k_self/opponent_mean_k_out,
   already in individual_log.csv). Seed-median beta_self(K_self),
   beta_self(K_out), beta_ctx(K_self), beta_ctx(K_out); local vs
   shuffle(replay) per k, Mann-Whitney + Holm (4 k's), per coefficient.

C. Environment table: condition x k, seed median (IQR) of
   cooperation_rate, mean_k_out, mean_k_self (gen>=2000, from
   timeseries.csv).

D. Variance/bimodality (descriptive only, no tests): per seed, median
   (gen>=2000, individual-log cadence) of within-population variance
   and Sarle's bimodality coefficient for K_out and K_self, local vs
   shuffle per k.

Usage:
    python3 scripts/analyze_local_placement_full.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, mannwhitneyu, skew

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
SHUFFLE_ROOT = Path("scripts/output/local_placement_sweep_shuffle_replay")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000


def load_individual_log(path):
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": [],
                                   "opp_k_self": [], "opp_k_out": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
            g["opp_k_self"].append(float(row["opponent_mean_k_self"]))
            g["opp_k_out"].append(float(row["opponent_mean_k_out"]))
    return by_gen


def load_timeseries(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
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


def mannwhitney_effect(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask_x, mask_y = ~np.isnan(x), ~np.isnan(y)
    x, y = x[mask_x], y[mask_y]
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * stat) / (len(x) * len(y))
    return stat, p, r


def sig_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------- B

def context_lande_arnold(k_self, k_out, opp_k_self, opp_k_out, payoff):
    """Linear-only regression (no quadratic/interaction terms): relative
    fitness ~ 1 + z1_self + z2_self + z1_ctx + z2_ctx, all 4 predictors
    standardized within this generation. Returns (beta_self_kself,
    beta_self_kout, beta_ctx_kself, beta_ctx_kout)."""
    def standardize(v):
        v = np.asarray(v)
        sd = np.std(v)
        return (v - np.mean(v)) / sd if sd > 0 else np.zeros_like(v)

    z1_self = standardize(k_self)
    z2_self = standardize(k_out)
    z1_ctx = standardize(opp_k_self)
    z2_ctx = standardize(opp_k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1_self), z1_self, z2_self, z1_ctx, z2_ctx])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[1], coef[2], coef[3], coef[4]


def seed_median_context_betas(sweep_dir, seed, min_gen):
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    names = ["beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout"]
    per_gen = {n: [] for n in names}
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if (len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2
                or len(set(g["opp_k_self"])) < 2 or len(set(g["opp_k_out"])) < 2):
            continue
        betas = context_lande_arnold(g["k_self"], g["k_out"], g["opp_k_self"], g["opp_k_out"], g["payoff"])
        for n, b in zip(names, betas):
            per_gen[n].append(b)
    return {n: (np.median(v) if v else np.nan) for n, v in per_gen.items()}


def section_B():
    print("=" * 100)
    print("B. Context analysis (Heisler & Damuth 1987): local vs shuffle(replay), per k, Holm (4 k's)")
    print("=" * 100)

    labels = {
        "beta_self_kself": "beta_self(K_self)",
        "beta_self_kout": "beta_self(K_out)",
        "beta_ctx_kself": "beta_ctx(K_self)",
        "beta_ctx_kout": "beta_ctx(K_out)",
    }

    for name, label in labels.items():
        print(f"\n--- {label} ---")
        raw_p, rows = [], []
        for k in K_VALUES:
            local_vals = [seed_median_context_betas(LOCAL_ROOT / f"k{k}", s, STEADY_STATE_MIN_GEN)[name] for s in SEEDS]
            shuffle_vals = [seed_median_context_betas(SHUFFLE_ROOT / f"k{k}", s, STEADY_STATE_MIN_GEN)[name] for s in SEEDS]
            stat, p, r = mannwhitney_effect(local_vals, shuffle_vals)
            raw_p.append(p)
            rows.append((k, np.nanmedian(local_vals), np.nanmedian(shuffle_vals), stat, p, r))
        holm_p = holm_correction(np.array(raw_p))
        print(f"{'k':>4}  {'local med':>11}  {'shuffle med':>12}  {'U':>9}  {'raw p':>9}  {'Holm p':>9}  {'r':>7}  {'sig':>5}")
        for (k, loc, shuf, stat, p, r), hp in zip(rows, holm_p):
            print(f"{k:>4}  {loc:>+11.5f}  {shuf:>+12.5f}  {stat:>9.1f}  {p:>9.4f}  {hp:>9.4f}  {r:>+7.3f}  {sig_stars(hp):>5}")


# ---------------------------------------------------------------- C

def seed_mean_env(sweep_dir, seed, min_gen):
    rows = load_timeseries(Path(sweep_dir) / f"seed_{seed}_timeseries.csv")
    sums = {"cooperation_rate": 0.0, "mean_k_out": 0.0, "mean_k_self": 0.0}
    counts = {k: 0 for k in sums}
    for row in rows:
        if int(row["generation"]) < min_gen:
            continue
        for col in sums:
            v = row[col]
            if v == "":
                continue
            sums[col] += float(v)
            counts[col] += 1
    return {c: (sums[c] / counts[c] if counts[c] else np.nan) for c in sums}


def section_C():
    print("\n" + "=" * 100)
    print("C. Environment table: condition x k, seed median (IQR), gen>=2000")
    print("=" * 100 + "\n")
    print(f"{'condition':>10}  {'k':>4}  {'cooperation_rate':>20}  {'mean K_out':>20}  {'mean K_self':>20}")
    for label, root in [("local", LOCAL_ROOT), ("shuffle", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            vals = {"cooperation_rate": [], "mean_k_out": [], "mean_k_self": []}
            for s in SEEDS:
                env = seed_mean_env(root / f"k{k}", s, STEADY_STATE_MIN_GEN)
                for c in vals:
                    vals[c].append(env[c])
            cells = []
            for c in vals:
                arr = np.array(vals[c])
                med = np.nanmedian(arr)
                iqr = (np.nanpercentile(arr, 25), np.nanpercentile(arr, 75))
                cells.append(f"{med:.4f} [{iqr[0]:.4f},{iqr[1]:.4f}]")
            print(f"{label:>10}  {k:>4}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}")


# ---------------------------------------------------------------- D

def bimodality_coefficient(values):
    values = np.asarray(values)
    n = len(values)
    if n < 4 or np.std(values) == 0:
        return np.nan
    g1 = skew(values, bias=True)
    g2 = kurtosis(values, fisher=True, bias=True)
    denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 ** 2 + 1) / denom if denom != 0 else np.nan


def seed_median_variance_bimodality(sweep_dir, seed, min_gen):
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    names = ["var_kout", "var_kself", "bc_kout", "bc_kself"]
    per_gen = {n: [] for n in names}
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        per_gen["var_kout"].append(np.var(g["k_out"]))
        per_gen["var_kself"].append(np.var(g["k_self"]))
        per_gen["bc_kout"].append(bimodality_coefficient(g["k_out"]))
        per_gen["bc_kself"].append(bimodality_coefficient(g["k_self"]))
    return {n: (np.nanmedian(v) if v else np.nan) for n, v in per_gen.items()}


def section_D():
    print("\n" + "=" * 100)
    print("D. Variance / bimodality (descriptive, no tests): local vs shuffle(replay) per k, gen>=2000")
    print("=" * 100 + "\n")
    print(f"{'cond':>8}  {'k':>4}  {'var(K_out) med':>15}  {'var(K_self) med':>16}  "
          f"{'BC(K_out) med':>14}  {'BC(K_self) med':>15}")
    for label, root in [("local", LOCAL_ROOT), ("shuffle", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            per_seed = [seed_median_variance_bimodality(root / f"k{k}", s, STEADY_STATE_MIN_GEN) for s in SEEDS]
            med = {n: np.nanmedian([p[n] for p in per_seed]) for n in ["var_kout", "var_kself", "bc_kout", "bc_kself"]}
            print(f"{label:>8}  {k:>4}  {med['var_kout']:>15.5f}  {med['var_kself']:>16.5f}  "
                  f"{med['bc_kout']:>14.4f}  {med['bc_kself']:>15.4f}")


def check_complete(root, k, seeds):
    k_dir = root / f"k{k}"
    return sum(
        1 for s in seeds
        if (k_dir / f"seed_{s}_timeseries.csv").exists() and (k_dir / f"seed_{s}_individual_log.csv").exists()
    )


def main():
    print("Checking completeness (local + shuffle replay, k=4,8,16,32, 50 seeds each)...\n")
    missing = []
    for label, root in [("local", LOCAL_ROOT), ("shuffle(replay)", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            n = check_complete(root, k, SEEDS)
            print(f"{label} k={k}: {n}/50")
            if n < 50:
                missing.append((label, k))
    if missing:
        print(f"\nNot yet complete: {missing}. Stopping.")
        return
    print()

    section_B()
    section_C()
    section_D()

    print("\n解釈はせず、データが示す形の記述にとどめて停止します。")


if __name__ == "__main__":
    main()
