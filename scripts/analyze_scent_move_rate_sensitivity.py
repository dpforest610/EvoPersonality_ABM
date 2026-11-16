"""EXPLORATORY, OUTSIDE THE PRE-REGISTRATION (docs/PREREGISTRATION_scent_model.md).

Checks whether B(a)'s finding (K_out assortativity: M2 < M1,
Mann-Whitney p<0.000001) is attributable to the ~3 percentage-point gap
in observed move rate (M2=0.5605 vs M1=0.5424 in production) rather
than to scent per se, by re-running M1 with p_move set to M2's own
observed production move rate (0.5605) -- "M1b" -- and comparing M2 vs
M1b instead of M2 vs M1.

Runs:
    python3 scripts/run_scent_spatial_sweep.py --movement random \
        --p-move 0.5605 --out-suffix _b --workers 4 --push-checkpoints

Usage (after M1b's production run completes, 50/50 seeds):
    python3 scripts/analyze_scent_move_rate_sensitivity.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
SCENT_ROOT = EXPERIMENT_ROOT / "scent" / "vm"
RANDOM_ROOT = EXPERIMENT_ROOT / "random" / "vm"       # M1, p_move=0.549684
RANDOM_B_ROOT = EXPERIMENT_ROOT / "random_b" / "vm"   # M1b, p_move=0.5605

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


def context_lande_arnold(k_self, k_out, opp_k_self, opp_k_out, payoff):
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


def seed_median_beta_ctx_kout(root, seed, min_gen):
    by_gen = load_individual_log(root / f"seed_{seed}_individual_log.csv")
    vals = []
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if (len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2
                or len(set(g["opp_k_self"])) < 2 or len(set(g["opp_k_out"])) < 2):
            continue
        vals.append(context_lande_arnold(g["k_self"], g["k_out"], g["opp_k_self"], g["opp_k_out"], g["payoff"]))
    return np.median(vals) if vals else np.nan


def seed_mean_timeseries_col(root, seed, min_gen, col):
    rows = list(csv.DictReader(open(root / f"seed_{seed}_timeseries.csv")))
    vals = [float(r[col]) for r in rows if int(r["generation"]) >= min_gen and r[col] != ""]
    return np.mean(vals) if vals else np.nan


def mannwhitney_effect(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * stat) / (len(x) * len(y))
    return stat, p, r


def check_complete(root, label):
    n = sum(1 for s in SEEDS if (root / f"seed_{s}_timeseries.csv").exists()
            and (root / f"seed_{s}_individual_log.csv").exists())
    print(f"{label}: {n}/50")
    return n == 50


def main():
    print("EXPLORATORY, OUTSIDE THE PRE-REGISTRATION (docs/PREREGISTRATION_scent_model.md).")
    print("Checks whether B(a)'s M2 vs M1 assortativity gap survives once M1's move rate is")
    print("matched to M2's own observed production move rate.\n")

    print("Checking completeness...")
    ok_m1b = check_complete(RANDOM_B_ROOT, "M1b (p_move=0.5605)")
    if not ok_m1b:
        print("\nM1b not yet complete. Stopping.")
        return
    print()

    m2_assort = [seed_mean_timeseries_col(SCENT_ROOT, s, STEADY_STATE_MIN_GEN, "k_out_assortativity") for s in SEEDS]
    m1_assort = [seed_mean_timeseries_col(RANDOM_ROOT, s, STEADY_STATE_MIN_GEN, "k_out_assortativity") for s in SEEDS]
    m1b_assort = [seed_mean_timeseries_col(RANDOM_B_ROOT, s, STEADY_STATE_MIN_GEN, "k_out_assortativity") for s in SEEDS]

    m2_move = [seed_mean_timeseries_col(SCENT_ROOT, s, STEADY_STATE_MIN_GEN, "move_rate") for s in SEEDS]
    m1_move = [seed_mean_timeseries_col(RANDOM_ROOT, s, STEADY_STATE_MIN_GEN, "move_rate") for s in SEEDS]
    m1b_move = [seed_mean_timeseries_col(RANDOM_B_ROOT, s, STEADY_STATE_MIN_GEN, "move_rate") for s in SEEDS]

    m2_beta = [seed_median_beta_ctx_kout(SCENT_ROOT, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
    m1b_beta = [seed_median_beta_ctx_kout(RANDOM_B_ROOT, s, STEADY_STATE_MIN_GEN) for s in SEEDS]

    print("=" * 100)
    print("Move rate check (gen>=2000, per-seed mean, n=50)")
    print("=" * 100)
    print(f"M1  (p_move=0.549684): median move_rate = {np.nanmedian(m1_move):.6f}")
    print(f"M1b (p_move=0.5605):   median move_rate = {np.nanmedian(m1b_move):.6f}")
    print(f"M2  (scent):           median move_rate = {np.nanmedian(m2_move):.6f}")

    print("\n" + "=" * 100)
    print("K_out assortativity: M2 vs M1b, Mann-Whitney U (two-sided) -- same test as B(a), M1b in place of M1")
    print("=" * 100)
    stat, p, r = mannwhitney_effect(m2_assort, m1b_assort)
    print(f"M2  median k_out_assortativity = {np.nanmedian(m2_assort):.6f}")
    print(f"M1b median k_out_assortativity = {np.nanmedian(m1b_assort):.6f}")
    print(f"(reference) M1  median k_out_assortativity = {np.nanmedian(m1_assort):.6f}")
    print(f"U = {stat:.1f}, p = {p:.6f}, r = {r:+.4f}")

    print("\n" + "=" * 100)
    print("beta_ctx(K_out): M2 vs M1b (descriptive, seed median)")
    print("=" * 100)
    print(f"M2  median beta_ctx(K_out) = {np.nanmedian(m2_beta):+.6f}")
    print(f"M1b median beta_ctx(K_out) = {np.nanmedian(m1b_beta):+.6f}")

    # --- scatter plot: move_rate vs k_out_assortativity, per seed, M1/M1b/M2 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    groups = [
        ("M1 (p_move=0.549684)", m1_move, m1_assort, "#eb6834"),
        ("M1b (p_move=0.5605)", m1b_move, m1b_assort, "#8f8f88"),
        ("M2 (scent, tau=0.033333)", m2_move, m2_assort, "#2a78d6"),
    ]
    for label, x, y, color in groups:
        ax.scatter(x, y, label=label, color=color, alpha=0.7, s=30, edgecolors="none")
    ax.set_xlabel("move_rate (per-seed mean, gen>=2000)")
    ax.set_ylabel("k_out_assortativity (per-seed mean, gen>=2000)")
    ax.set_title("EXPLORATORY (outside pre-registration): move rate vs K_out assortativity, per seed")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = Path("scripts/output/figures/scent_move_rate_vs_assortativity_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_path}")

    print("\n探索的・事前登録外の分析です。解釈はせず、データが示す形の記述にとどめて停止します。")


if __name__ == "__main__":
    main()
