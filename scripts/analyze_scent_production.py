"""Scent-model production: FULL ANALYSIS per
docs/PREREGISTRATION_scent_model.md (primary endpoint A, secondary B(a)
(b)(c), descriptive C), plus the user-requested operational-check table
comparing the pilot's K_out-assortativity drop (M0 ~0.37 -> M1/M2
~0.08) against the production-scale result.

M0=none, M1=random (p_move=0.549684), M2=scent (tau=0.033333), each 50
seeds x 5000 generations, gen>=2000 steady-state convention (matching
the ring-model chapters), per-seed median/mean, n=50.

Usage:
    python3 scripts/analyze_scent_production.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, mannwhitneyu, skew

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
NONE_ROOT = EXPERIMENT_ROOT / "none" / "vm"
RANDOM_ROOT = EXPERIMENT_ROOT / "random" / "vm"
SCENT_ROOT = EXPERIMENT_ROOT / "scent" / "vm"
RING_LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
RING_K_VALUES = [4, 8, 16, 32]

SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(0)

CONDITIONS = [("M0", NONE_ROOT), ("M1", RANDOM_ROOT), ("M2", SCENT_ROOT)]

# pre-registered OLS fit (ring-model local-placement 4 points), computed
# before any scent-model pilot/production data existed -- see
# docs/PREREGISTRATION_scent_model.md section B(b) and
# scripts/precompute_ring_assortativity_vs_beta_ctx_prediction.py
PRED_INTERCEPT = -0.00016799
PRED_SLOPE = -0.00719578
PRED_S = 0.00041078
PRED_XBAR = 0.496515
PRED_SXX = 0.11212395
PRED_TCRIT = 4.302653
PRED_N = 4


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


def mannwhitney_effect(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
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


# ---------------------------------------------------------------- context model (A, B(a)(b), C)

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


# ---------------------------------------------------------------- own-trait Lande-Arnold / z* (C)

def lande_arnold_coeffs(k_self, k_out, payoff):
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, beta1, beta2, g11_raw, g22_raw, g12 = coef
    return beta1, beta2, 2 * g11_raw, 2 * g22_raw, g12


def seed_summary(sweep_dir, seed, min_gen):
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    beta1s, beta2s, g11s, g22s, g12s = [], [], [], [], []
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2:
            continue
        b1, b2, g11, g22, g12 = lande_arnold_coeffs(g["k_self"], g["k_out"], g["payoff"])
        beta1s.append(b1)
        beta2s.append(b2)
        g11s.append(g11)
        g22s.append(g22)
        g12s.append(g12)
    if not beta1s:
        return dict(median_beta1=np.nan, median_beta2=np.nan, median_g11=np.nan, median_g22=np.nan,
                    median_g12=np.nan, beta_mean=None, gamma_mean=None)
    return dict(
        median_beta1=np.median(beta1s), median_beta2=np.median(beta2s),
        median_g11=np.median(g11s), median_g22=np.median(g22s), median_g12=np.median(g12s),
        beta_mean=np.array([np.mean(beta1s), np.mean(beta2s)]),
        gamma_mean=np.array([[np.mean(g11s), np.mean(g12s)], [np.mean(g12s), np.mean(g22s)]]),
    )


def fitness_peak_for_group(summaries):
    valid_betas, valid_gammas = [], []
    n_excluded = 0
    for s in summaries:
        if s["beta_mean"] is None:
            continue
        eigvals = np.linalg.eigvalsh(s["gamma_mean"])
        if not np.all(eigvals < 0):
            n_excluded += 1
            continue
        valid_betas.append(s["beta_mean"])
        valid_gammas.append(s["gamma_mean"])
    return dict(
        n_excluded=n_excluded, n_total=len(summaries),
        valid_betas=np.array(valid_betas) if valid_betas else np.empty((0, 2)),
        valid_gammas=np.array(valid_gammas) if valid_gammas else np.empty((0, 2, 2)),
    )


def bootstrap_grand_zstar(valid_betas, valid_gammas, n_bootstrap=N_BOOTSTRAP):
    n = len(valid_betas)
    if n == 0:
        return np.array([]), np.array([]), 0
    boot_z1, boot_z2 = [], []
    n_skipped = 0
    for _ in range(n_bootstrap):
        idx = RNG.integers(0, n, size=n)
        mean_beta = valid_betas[idx].mean(axis=0)
        mean_gamma = valid_gammas[idx].mean(axis=0)
        eigvals = np.linalg.eigvalsh(mean_gamma)
        if not np.all(eigvals < 0):
            n_skipped += 1
            continue
        z = -np.linalg.solve(mean_gamma, mean_beta)
        boot_z1.append(z[0])
        boot_z2.append(z[1])
    return np.array(boot_z1), np.array(boot_z2), n_skipped


# ---------------------------------------------------------------- environment / operational check (C)

def seed_mean_env(sweep_dir, seed, min_gen):
    rows = load_timeseries(Path(sweep_dir) / f"seed_{seed}_timeseries.csv")
    cols = ["cooperation_rate", "mean_k_out", "mean_k_self"]
    sums = {c: 0.0 for c in cols}
    counts = {c: 0 for c in cols}
    for row in rows:
        if int(row["generation"]) < min_gen:
            continue
        for c in cols:
            v = row[c]
            if v == "":
                continue
            sums[c] += float(v)
            counts[c] += 1
    return {c: (sums[c] / counts[c] if counts[c] else np.nan) for c in cols}


OP_COLS = ["k_out_assortativity", "k_self_assortativity", "same_parent_fraction",
           "move_rate", "isolated_fraction", "mean_opponent_count", "max_opponent_count",
           "scent_index_mean"]


def seed_mean_operational(sweep_dir, seed, min_gen):
    rows = load_timeseries(Path(sweep_dir) / f"seed_{seed}_timeseries.csv")
    sums = {c: 0.0 for c in OP_COLS}
    counts = {c: 0 for c in OP_COLS}
    for row in rows:
        if int(row["generation"]) < min_gen:
            continue
        for c in OP_COLS:
            if c not in row:
                continue
            v = row[c]
            if v == "":
                continue
            sums[c] += float(v)
            counts[c] += 1
    return {c: (sums[c] / counts[c] if counts[c] else np.nan) for c in OP_COLS}


# ---------------------------------------------------------------- variance / bimodality (C)

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


# ---------------------------------------------------------------- sections

def check_complete():
    print("Checking completeness (M0/M1/M2, 50 seeds each)...\n")
    missing = []
    for label, root in CONDITIONS:
        n = sum(1 for s in SEEDS if (root / f"seed_{s}_timeseries.csv").exists()
                and (root / f"seed_{s}_individual_log.csv").exists())
        print(f"{label}: {n}/50")
        if n < 50:
            missing.append(label)
    if missing:
        print(f"\nNot yet complete: {missing}. Stopping.")
        return False
    print()
    return True


def compute_all_context_betas():
    out = {}
    for label, root in CONDITIONS:
        out[label] = [seed_median_context_betas(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
    return out


def compute_all_operational():
    out = {}
    for label, root in CONDITIONS:
        out[label] = [seed_mean_operational(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
    return out


def section_operational_check(op_data):
    print("=" * 100)
    print("Operational check: M0 vs M1 vs M2 (gen>=2000, per-seed mean, n=50)")
    print("Checks whether the pilot's K-similarity drop (M0 ~0.37 -> M1/M2 ~0.08) reproduces at production scale")
    print("=" * 100)
    print(f"\n{'cond':>4}  {'K_out assort':>13}  {'K_self assort':>14}  {'same_parent':>12}  "
          f"{'move_rate':>10}  {'isolated':>9}  {'mean_opp':>9}  {'max_opp':>8}  {'scent_idx':>10}")
    for label, _ in CONDITIONS:
        vals = op_data[label]
        cells = {c: np.nanmean([v[c] for v in vals]) for c in OP_COLS}
        scent_idx_str = f"{cells['scent_index_mean']:.4f}" if label == "M2" else "--"
        print(f"{label:>4}  {cells['k_out_assortativity']:>13.4f}  {cells['k_self_assortativity']:>14.4f}  "
              f"{cells['same_parent_fraction']:>12.4f}  {cells['move_rate']:>10.4f}  "
              f"{cells['isolated_fraction']:>9.5f}  {cells['mean_opponent_count']:>9.3f}  "
              f"{cells['max_opponent_count']:>8.2f}  {scent_idx_str:>10}")

    print(f"\nPilot (5 seeds x 1000 gens) reference values for comparison:")
    print(f"{'cond':>4}  {'K_out assort':>13}")
    print(f"{'M0':>4}  {0.3699:>13.4f}")
    print(f"{'M1':>4}  {0.0917:>13.4f}")
    print(f"{'M2':>4}  {0.0790:>13.4f}")


def section_A(context_betas):
    print("\n" + "=" * 100)
    print("A. Primary evaluation: beta_ctx(K_out), M2 vs M1, Mann-Whitney U (two-sided, single test)")
    print("=" * 100)
    m2_vals = [v["beta_ctx_kout"] for v in context_betas["M2"]]
    m1_vals = [v["beta_ctx_kout"] for v in context_betas["M1"]]
    stat, p, r = mannwhitney_effect(m2_vals, m1_vals)
    print(f"\nM2 median beta_ctx(K_out) = {np.nanmedian(m2_vals):+.6f}")
    print(f"M1 median beta_ctx(K_out) = {np.nanmedian(m1_vals):+.6f}")
    print(f"U = {stat:.1f}, p = {p:.6f}, r = {r:+.4f}, {sig_stars(p)}")


def section_Ba(op_data):
    print("\n" + "=" * 100)
    print("B(a). Actually-dueled-pair K_out assortativity, M2 vs M1, Mann-Whitney U (two-sided)")
    print("=" * 100)
    m2_vals = [v["k_out_assortativity"] for v in op_data["M2"]]
    m1_vals = [v["k_out_assortativity"] for v in op_data["M1"]]
    stat, p, r = mannwhitney_effect(m2_vals, m1_vals)
    print(f"\nM2 median k_out_assortativity = {np.nanmedian(m2_vals):.6f}")
    print(f"M1 median k_out_assortativity = {np.nanmedian(m1_vals):.6f}")
    print(f"U = {stat:.1f}, p = {p:.6f}, r = {r:+.4f}, {sig_stars(p)}")
    direction = "higher" if np.nanmedian(m2_vals) > np.nanmedian(m1_vals) else "lower"
    print(f"\nM2's assortativity is {direction} than M1's -- hypothesis A (gathering similar individuals) "
          f"predicts higher, hypothesis B (defectors drawn to cooperators) predicts lower.")


def predict_interval(x0):
    y_hat = PRED_INTERCEPT + PRED_SLOPE * x0
    se_pred = PRED_S * np.sqrt(1 + 1 / PRED_N + (x0 - PRED_XBAR) ** 2 / PRED_SXX)
    half_width = PRED_TCRIT * se_pred
    return y_hat, y_hat - half_width, y_hat + half_width


def section_Bb(context_betas, op_data):
    print("\n" + "=" * 100)
    print("B(b). Quantitative prediction: ring-model-fitted line, checked against observed scent-model values")
    print("formula (pre-registered before any scent-model data existed): "
          f"y_hat(x) = {PRED_INTERCEPT:.8f} + {PRED_SLOPE:.8f} * x")
    print("=" * 100)
    print(f"\n{'cond':>4}  {'obs assort (x0)':>16}  {'obs beta_ctx(K_out) (y)':>24}  "
          f"{'predicted y_hat':>16}  {'95% PI':>24}  {'in PI?':>7}")
    for label, _ in CONDITIONS:
        x0 = np.nanmedian([v["k_out_assortativity"] for v in op_data[label]])
        y_obs = np.nanmedian([v["beta_ctx_kout"] for v in context_betas[label]])
        y_hat, lo, hi = predict_interval(x0)
        in_pi = "YES" if lo <= y_obs <= hi else "NO"
        print(f"{label:>4}  {x0:>16.6f}  {y_obs:>+24.6f}  {y_hat:>+16.6f}  "
              f"[{lo:+.6f},{hi:+.6f}]  {in_pi:>7}")


def section_Bc(context_betas, op_data):
    print("\n" + "=" * 100)
    print("B(c). M0 vs ring-model local placement (dimensionality); M1 vs M0 (effect of mixing) -- descriptive")
    print("=" * 100)

    print("\n--- M0 (2D, no movement) vs ring-model local placement (k=4,8,16,32) ---")
    print(f"{'cond':>14}  {'K_out assort':>13}  {'beta_ctx(K_out)':>17}")
    x0_m0 = np.nanmedian([v["k_out_assortativity"] for v in op_data["M0"]])
    y_m0 = np.nanmedian([v["beta_ctx_kout"] for v in context_betas["M0"]])
    print(f"{'M0 (2D)':>14}  {x0_m0:>13.6f}  {y_m0:>+17.6f}")
    for k in RING_K_VALUES:
        root = RING_LOCAL_ROOT / f"k{k}"
        assort_vals, beta_vals = [], []
        for s in SEEDS:
            rows = load_timeseries(root / f"seed_{s}_timeseries.csv")
            a = [float(r["k_out_assortativity"]) for r in rows
                 if int(r["generation"]) >= STEADY_STATE_MIN_GEN and r["k_out_assortativity"] != ""]
            assort_vals.append(np.median(a) if a else np.nan)
            beta_vals.append(seed_median_context_betas(root, s, STEADY_STATE_MIN_GEN)["beta_ctx_kout"])
        print(f"{'ring local k=' + str(k):>14}  {np.nanmedian(assort_vals):>13.6f}  {np.nanmedian(beta_vals):>+17.6f}")

    print("\n--- M1 (random movement) vs M0 (no movement) ---")
    print(f"{'cond':>4}  {'K_out assort':>13}  {'beta_ctx(K_out)':>17}  {'cooperation_rate':>17}")
    for label in ["M0", "M1"]:
        x0 = np.nanmedian([v["k_out_assortativity"] for v in op_data[label]])
        y = np.nanmedian([v["beta_ctx_kout"] for v in context_betas[label]])
        env = [seed_mean_env(dict(CONDITIONS)[label], s, STEADY_STATE_MIN_GEN) for s in SEEDS]
        coop = np.nanmedian([e["cooperation_rate"] for e in env])
        print(f"{label:>4}  {x0:>13.6f}  {y:>+17.6f}  {coop:>17.6f}")


def section_C(context_betas):
    print("\n" + "=" * 100)
    print("C. Descriptive-only report (no significance tests)")
    print("=" * 100)

    print("\n--- Remaining context-model coefficients (seed median, gen>=2000) ---")
    print(f"{'cond':>4}  {'beta_self(K_self)':>18}  {'beta_self(K_out)':>17}  {'beta_ctx(K_self)':>17}")
    for label, _ in CONDITIONS:
        vals = context_betas[label]
        print(f"{label:>4}  {np.nanmedian([v['beta_self_kself'] for v in vals]):>+18.6f}  "
              f"{np.nanmedian([v['beta_self_kout'] for v in vals]):>+17.6f}  "
              f"{np.nanmedian([v['beta_ctx_kself'] for v in vals]):>+17.6f}")

    print("\n--- Own-trait-only Lande-Arnold: beta1, beta2, gamma11, gamma22, gamma12 (seed median) ---")
    print(f"{'cond':>4}  {'beta1':>10}  {'beta2':>10}  {'gamma11':>10}  {'gamma22':>10}  {'gamma12':>10}")
    summaries_by_cond = {}
    for label, root in CONDITIONS:
        summaries = [seed_summary(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
        summaries_by_cond[label] = summaries
        print(f"{label:>4}  {np.nanmedian([s['median_beta1'] for s in summaries]):>+10.5f}  "
              f"{np.nanmedian([s['median_beta2'] for s in summaries]):>+10.5f}  "
              f"{np.nanmedian([s['median_g11'] for s in summaries]):>+10.5f}  "
              f"{np.nanmedian([s['median_g22'] for s in summaries]):>+10.5f}  "
              f"{np.nanmedian([s['median_g12'] for s in summaries]):>+10.5f}")

    print("\n--- Fitness peak z* = -Gamma^-1*beta (grand mean, bootstrap 95% CI) ---")
    print(f"{'cond':>4}  {'n_excl':>6}  {'n_valid':>7}  {'z*_K_self':>16}  {'95% CI':>22}  "
          f"{'z*_K_out':>16}  {'95% CI':>22}")
    for label, _ in CONDITIONS:
        r = fitness_peak_for_group(summaries_by_cond[label])
        if len(r["valid_betas"]) == 0:
            print(f"{label:>4}  {r['n_excluded']:>6}  {0:>7}  (no valid seeds)")
            continue
        grand_beta = r["valid_betas"].mean(axis=0)
        grand_gamma = r["valid_gammas"].mean(axis=0)
        grand_z = -np.linalg.solve(grand_gamma, grand_beta)
        boot_z1, boot_z2, _ = bootstrap_grand_zstar(r["valid_betas"], r["valid_gammas"])
        ci1 = (np.percentile(boot_z1, 2.5), np.percentile(boot_z1, 97.5)) if len(boot_z1) else (np.nan, np.nan)
        ci2 = (np.percentile(boot_z2, 2.5), np.percentile(boot_z2, 97.5)) if len(boot_z2) else (np.nan, np.nan)
        print(f"{label:>4}  {r['n_excluded']:>6}  {len(r['valid_betas']):>7}  "
              f"{grand_z[0]:>+16.4f}  [{ci1[0]:+.4f},{ci1[1]:+.4f}]{'':>4}  "
              f"{grand_z[1]:>+16.4f}  [{ci2[0]:+.4f},{ci2[1]:+.4f}]")

    print("\n--- Environment table (seed median [IQR], gen>=2000) ---")
    print(f"{'cond':>4}  {'cooperation_rate':>20}  {'mean K_out':>20}  {'mean K_self':>20}")
    for label, root in CONDITIONS:
        vals = {"cooperation_rate": [], "mean_k_out": [], "mean_k_self": []}
        for s in SEEDS:
            env = seed_mean_env(root, s, STEADY_STATE_MIN_GEN)
            for c in vals:
                vals[c].append(env[c])
        cells = []
        for c in vals:
            arr = np.array(vals[c])
            med = np.nanmedian(arr)
            iqr = (np.nanpercentile(arr, 25), np.nanpercentile(arr, 75))
            cells.append(f"{med:.4f} [{iqr[0]:.4f},{iqr[1]:.4f}]")
        print(f"{label:>4}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}")

    print("\n--- Variance / Sarle's bimodality coefficient (seed median, gen>=2000) ---")
    print(f"{'cond':>4}  {'var(K_out)':>12}  {'var(K_self)':>12}  {'BC(K_out)':>11}  {'BC(K_self)':>11}")
    for label, root in CONDITIONS:
        per_seed = [seed_median_variance_bimodality(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
        med = {n: np.nanmedian([p[n] for p in per_seed]) for n in ["var_kout", "var_kself", "bc_kout", "bc_kself"]}
        print(f"{label:>4}  {med['var_kout']:>12.5f}  {med['var_kself']:>12.5f}  "
              f"{med['bc_kout']:>11.4f}  {med['bc_kself']:>11.4f}")


def main():
    if not check_complete():
        return

    print("Computing per-seed context-model betas (reads every individual_log.csv -- may take a while)...", flush=True)
    context_betas = compute_all_context_betas()
    print("Computing per-seed operational-check values...", flush=True)
    op_data = compute_all_operational()
    print("done.\n", flush=True)

    section_operational_check(op_data)
    section_A(context_betas)
    section_Ba(op_data)
    section_Bb(context_betas, op_data)
    section_Bc(context_betas, op_data)
    section_C(context_betas)

    print("\n解釈はせず、データが示す形の記述にとどめて停止します。")


if __name__ == "__main__":
    main()
