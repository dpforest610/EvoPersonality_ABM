"""EXPLORATORY, OUTSIDE ANY PRE-REGISTRATION. Uses only already-collected
N=100 production data (M1, M1b, M2) -- no new simulation runs.

Fits an ANCOVA-style OLS model to the per-seed data pooled across M1
(p_move=0.549684), M1b (p_move=0.5605), and M2 (scent, tau=0.033333):

    k_out_assortativity ~ 1 + is_scent + move_rate

(gen>=2000, per-seed mean, n=50 each -> 150 rows total). is_scent is a
binary indicator (1 for M2, 0 for M1/M1b, pooled together as the
"random movement" class). The coefficient on is_scent is the estimated
difference between M2 and random movement AT THE SAME move_rate (i.e.
holding move_rate constant), with its 95% CI -- this is what the
preregistered B(a) comparison and the M2-vs-M1b exploratory comparison
could not isolate on their own (both compared conditions at whatever
move rate each happened to produce, not at a matched rate via a
continuous covariate).

Usage:
    python3 scripts/analyze_scent_assortativity_ancova.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

import csv

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
SCENT_ROOT = EXPERIMENT_ROOT / "scent" / "vm"
RANDOM_ROOT = EXPERIMENT_ROOT / "random" / "vm"
RANDOM_B_ROOT = EXPERIMENT_ROOT / "random_b" / "vm"

SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000


def seed_mean_timeseries_col(root, seed, min_gen, col):
    rows = list(csv.DictReader(open(root / f"seed_{seed}_timeseries.csv")))
    vals = [float(r[col]) for r in rows if int(r["generation"]) >= min_gen and r[col] != ""]
    return np.mean(vals) if vals else np.nan


def ols_with_ci(X, y, alpha=0.05):
    n, k = X.shape
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - k
    sigma2 = np.sum(resid ** 2) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    tcrit = t_dist.ppf(1 - alpha / 2, dof)
    ci_lo = coef - tcrit * se
    ci_hi = coef + tcrit * se
    tstat = coef / se
    pval = 2 * (1 - t_dist.cdf(np.abs(tstat), dof))
    return coef, se, ci_lo, ci_hi, tstat, pval, dof


def main():
    print("EXPLORATORY, OUTSIDE ANY PRE-REGISTRATION. Uses only already-collected N=100 data")
    print("(M1, M1b, M2) -- no new simulation runs.\n")

    rows = []  # (assortativity, is_scent, move_rate)
    for root, is_scent in [(RANDOM_ROOT, 0), (RANDOM_B_ROOT, 0), (SCENT_ROOT, 1)]:
        for s in SEEDS:
            a = seed_mean_timeseries_col(root, s, STEADY_STATE_MIN_GEN, "k_out_assortativity")
            m = seed_mean_timeseries_col(root, s, STEADY_STATE_MIN_GEN, "move_rate")
            rows.append((a, is_scent, m))

    rows = [r for r in rows if not any(np.isnan(x) for x in r)]
    y = np.array([r[0] for r in rows])
    is_scent = np.array([r[1] for r in rows], dtype=float)
    move_rate = np.array([r[2] for r in rows], dtype=float)
    n = len(rows)
    print(f"n = {n} (pooled per-seed rows: M1 n=50, M1b n=50, M2 n=50)\n")

    X = np.column_stack([np.ones(n), is_scent, move_rate])
    coef, se, ci_lo, ci_hi, tstat, pval, dof = ols_with_ci(X, y)

    names = ["intercept", "is_scent (M2 indicator)", "move_rate"]
    print("=" * 100)
    print("OLS: k_out_assortativity ~ 1 + is_scent + move_rate")
    print("=" * 100)
    print(f"\n{'term':>28}  {'coef':>12}  {'se':>12}  {'95% CI':>26}  {'t':>8}  {'p':>10}")
    for name, c, s, lo, hi, tt, pp in zip(names, coef, se, ci_lo, ci_hi, tstat, pval):
        print(f"{name:>28}  {c:>+12.6f}  {s:>12.6f}  [{lo:+.6f},{hi:+.6f}]  {tt:>8.3f}  {pp:>10.6f}")

    print(f"\ndof = {dof}")
    print(f"\nInterpretation-free summary: at the same move_rate, M2 (scent) is predicted to have "
          f"k_out_assortativity {coef[1]:+.6f} (95% CI [{ci_lo[1]:+.6f}, {ci_hi[1]:+.6f}]) relative to "
          f"random movement (M1/M1b pooled).")

    print("\n探索的・事前登録外の分析です。解釈はせず、データが示す形の記述にとどめて停止します。")


if __name__ == "__main__":
    main()
