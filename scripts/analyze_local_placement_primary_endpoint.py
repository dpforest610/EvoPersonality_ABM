"""Local-placement production: PRIMARY ENDPOINT ONLY (see
docs/PREREGISTRATION_local_placement.md section A -- this is the one
locked comparison, run and reported before anything else).

Own-trait-only Lande-Arnold beta2 (K_out's linear/directional selection
gradient, Phillips & Arnold 1989 convention, gen>=2000, per-seed median,
n=50 seeds), local (new production run) vs shuffle (EXISTING precision
k-sweep data -- individual-level K_self/K_out/payoff already present,
no re-run needed for this comparison), per k in {4,8,16,32}.
Mann-Whitney U + rank-biserial effect size per k, Holm-corrected across
the 4 k's.

Requires scripts/output/local_placement_sweep/k{4,8,16,32}/ to be fully
complete (50/50 seeds each); checks this and reports what's missing
rather than silently analyzing a partial set.

Usage:
    python3 scripts/analyze_local_placement_primary_endpoint.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
SHUFFLE_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000


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


def lande_arnold_beta2(k_self, k_out, payoff):
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[2]  # beta2


def seed_median_beta2(sweep_dir, seed, min_gen):
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    vals = []
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2:
            continue
        vals.append(lande_arnold_beta2(g["k_self"], g["k_out"], g["payoff"]))
    return np.median(vals) if vals else np.nan


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


def check_complete(root, k, seeds):
    k_dir = root / f"k{k}"
    n_ready = sum(
        1 for s in seeds
        if (k_dir / f"seed_{s}_timeseries.csv").exists()
        and (k_dir / f"seed_{s}_individual_log.csv").exists()
    )
    return n_ready


def main():
    print("Checking local-placement production completeness (k=4,8,16,32, 50 seeds each)...\n")
    missing = []
    for k in K_VALUES:
        n_local = check_complete(LOCAL_ROOT, k, SEEDS)
        n_shuffle = check_complete(SHUFFLE_ROOT, k, SEEDS)
        print(f"k={k}: local {n_local}/50, shuffle(existing) {n_shuffle}/50")
        if n_local < 50:
            missing.append(k)
    if missing:
        print(f"\nNot yet complete for k={missing}. Stopping -- re-run once local production finishes.")
        return

    print(f"\n=== PRIMARY ENDPOINT: own-trait-only beta2 (K_out), local vs shuffle(existing) ===")
    print(f"(gen>={STEADY_STATE_MIN_GEN}, per-seed median, n=50 seeds, Mann-Whitney U, Holm across 4 k's)\n")

    raw_p = []
    rows = []
    for k in K_VALUES:
        local_vals = np.array([seed_median_beta2(LOCAL_ROOT / f"k{k}", s, STEADY_STATE_MIN_GEN) for s in SEEDS])
        shuffle_vals = np.array([seed_median_beta2(SHUFFLE_ROOT / f"k{k}", s, STEADY_STATE_MIN_GEN) for s in SEEDS])
        stat, p = mannwhitneyu(local_vals, shuffle_vals, alternative="two-sided")
        r = 1 - (2 * stat) / (len(local_vals) * len(shuffle_vals))
        raw_p.append(p)
        rows.append((k, np.nanmedian(local_vals), np.nanmedian(shuffle_vals), stat, p, r))

    holm_p = holm_correction(np.array(raw_p))

    print(f"{'k':>4}  {'local median':>13}  {'shuffle median':>15}  {'U':>10}  {'raw p':>9}  {'Holm p':>9}  {'r':>7}")
    for (k, loc_med, shuf_med, stat, p, r), hp in zip(rows, holm_p):
        print(f"{k:>4}  {loc_med:>+13.5f}  {shuf_med:>+15.5f}  {stat:>10.1f}  {p:>9.4f}  {hp:>9.4f}  {r:>+7.3f}")

    print("\n解釈はせず、ここで停止します(主要評価のみ)。")


if __name__ == "__main__":
    main()
