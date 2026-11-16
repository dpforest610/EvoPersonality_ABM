"""Pre-registration input for the scent-movement model (Chapter 3):
fits a line to the ring-model local-placement experiment's 4 points
(k_out_assortativity vs beta_ctx(K_out), local condition, gen>=2000,
seed median, n=50 seeds, k=4,8,16,32) and reports the OLS line and its
95% prediction interval as a function of x -- computed BEFORE any
scent-model pilot or production data exists, per
docs/PREREGISTRATION_scent_model.md section 3(b).

x = k_out_assortativity (Pearson correlation of K_out between actually-
    duelled pairs, seed-median per k).
y = beta_ctx(K_out) (Heisler & Damuth context-model coefficient on the
    opponent-mean K_out, seed-median per k).
Both read from the already-completed, already-verified local-placement
production data (scripts/output/local_placement_sweep/).

Usage:
    python3 scripts/precompute_ring_assortativity_vs_beta_ctx_prediction.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000


def seed_median_assortativity(k, seed):
    rows = list(csv.DictReader(open(LOCAL_ROOT / f"k{k}" / f"seed_{seed}_timeseries.csv")))
    vals = [
        float(r["k_out_assortativity"]) for r in rows
        if int(r["generation"]) >= STEADY_STATE_MIN_GEN and r["k_out_assortativity"] != ""
    ]
    return np.median(vals) if vals else np.nan


def load_individual_log(path):
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": [], "opp_k_self": [], "opp_k_out": []})
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


def context_beta_ctx_kout(k_self, k_out, opp_k_self, opp_k_out, payoff):
    def standardize(v):
        v = np.asarray(v)
        sd = np.std(v)
        return (v - np.mean(v)) / sd if sd > 0 else np.zeros_like(v)
    z1_self, z2_self = standardize(k_self), standardize(k_out)
    z1_ctx, z2_ctx = standardize(opp_k_self), standardize(opp_k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1_self), z1_self, z2_self, z1_ctx, z2_ctx])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[4]  # beta_ctx(K_out)


def seed_median_beta_ctx_kout(k, seed):
    by_gen = load_individual_log(LOCAL_ROOT / f"k{k}" / f"seed_{seed}_individual_log.csv")
    vals = []
    for gen, g in by_gen.items():
        if gen < STEADY_STATE_MIN_GEN:
            continue
        if (len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2
                or len(set(g["opp_k_self"])) < 2 or len(set(g["opp_k_out"])) < 2):
            continue
        vals.append(context_beta_ctx_kout(g["k_self"], g["k_out"], g["opp_k_self"], g["opp_k_out"], g["payoff"]))
    return np.median(vals) if vals else np.nan


def main():
    print("computing per-k seed-median assortativity and beta_ctx(K_out) (local condition)...", flush=True)
    xs, ys = [], []
    for k in K_VALUES:
        assort_per_seed = [seed_median_assortativity(k, s) for s in SEEDS]
        beta_per_seed = [seed_median_beta_ctx_kout(k, s) for s in SEEDS]
        x = np.nanmedian(assort_per_seed)
        y = np.nanmedian(beta_per_seed)
        xs.append(x)
        ys.append(y)
        print(f"  k={k:>2}: k_out_assortativity (seed median) = {x:.6f}, beta_ctx(K_out) (seed median) = {y:.6f}")

    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    y_hat = intercept + slope * xs
    resid = ys - y_hat
    rss = np.sum(resid ** 2)
    dof = n - 2
    s = np.sqrt(rss / dof)
    x_bar = np.mean(xs)
    sxx = np.sum((xs - x_bar) ** 2)
    t_crit = stats.t.ppf(0.975, dof)

    print("\n" + "=" * 90)
    print("OLS fit: beta_ctx(K_out) = intercept + slope * k_out_assortativity")
    print("=" * 90)
    print(f"n = {n} (points, one per k)")
    print(f"intercept = {intercept:.8f}")
    print(f"slope     = {slope:.8f}")
    print(f"r         = {r_value:.6f}  (r^2 = {r_value**2:.6f})")
    print(f"p (slope != 0, n={n}, dof={dof}) = {p_value:.6f}")
    print(f"residual standard error s = {s:.8f}  (dof={dof})")
    print(f"x_bar (mean assortativity) = {x_bar:.6f}")
    print(f"Sxx = {sxx:.8f}")
    print(f"t_crit (two-sided 95%, dof={dof}) = {t_crit:.6f}")

    def predict(x0):
        yhat0 = intercept + slope * x0
        se_pred = s * np.sqrt(1 + 1 / n + (x0 - x_bar) ** 2 / sxx)
        half_width = t_crit * se_pred
        return yhat0, yhat0 - half_width, yhat0 + half_width

    print("\n" + "=" * 90)
    print("95% prediction interval for a NEW single observed k_out_assortativity x0:")
    print("  y_hat(x0) = intercept + slope * x0")
    print("  PI(x0) = y_hat(x0) +/- t_crit * s * sqrt(1 + 1/n + (x0 - x_bar)^2 / Sxx)")
    print("=" * 90)
    print(f"\n{'x0 (assortativity)':>20}  {'y_hat':>12}  {'95% PI low':>12}  {'95% PI high':>12}")
    for x0 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        yhat0, lo, hi = predict(x0)
        print(f"{x0:>20.2f}  {yhat0:>12.6f}  {lo:>12.6f}  {hi:>12.6f}")

    print(f"\nAlso the 4 original (x, y) points used for the fit:")
    for k, x, y in zip(K_VALUES, xs, ys):
        print(f"  k={k:>2}: x={x:.6f}, y={y:.6f}")


if __name__ == "__main__":
    main()
