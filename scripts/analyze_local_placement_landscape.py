"""Local-placement production: landscape overview (additional item 3 of
the most recent request, not part of the original pre-registration).

Own-trait-only Lande-Arnold beta1, gamma11, gamma22, gamma12
(Phillips & Arnold 1989 convention, gen>=2000, seed median, n=50) and
the fitness-peak z*=-Gamma^-1*beta (seed-averaged beta/gamma, excluding
seeds where Gamma is not negative-definite, grand-mean bootstrap 95%
CI), local vs shuffle(replay) side by side per k. Descriptive only (no
significance tests) -- framed as an overview, not a new locked
comparison alongside the primary endpoint and context analysis.

Usage:
    python3 scripts/analyze_local_placement_landscape.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
SHUFFLE_ROOT = Path("scripts/output/local_placement_sweep_shuffle_replay")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(0)


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


def lande_arnold_coeffs(k_self, k_out, payoff):
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, beta1, beta2, g11_raw, g22_raw, g12 = coef
    return beta1, beta2, 2 * g11_raw, 2 * g22_raw, g12


def rank_biserial_one_sample(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


def seed_summary(sweep_dir, seed, min_gen):
    """Per-seed median beta1/gamma11/gamma22/gamma12 (for the table) AND
    the seed-mean (beta1,beta2) + gamma matrix (for the fitness peak,
    averaged across qualifying generations BEFORE combining seeds --
    same convention as analyze_fitness_peak.py)."""
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
        return dict(median_beta1=np.nan, median_g11=np.nan, median_g22=np.nan, median_g12=np.nan,
                    beta_mean=None, gamma_mean=None)
    return dict(
        median_beta1=np.median(beta1s), median_g11=np.median(g11s),
        median_g22=np.median(g22s), median_g12=np.median(g12s),
        beta_mean=np.array([np.mean(beta1s), np.mean(beta2s)]),
        gamma_mean=np.array([[np.mean(g11s), np.mean(g12s)], [np.mean(g12s), np.mean(g22s)]]),
    )


def fitness_peak_for_group(summaries):
    z1_list, z2_list = [], []
    valid_betas, valid_gammas = [], []
    n_excluded = 0
    for s in summaries:
        if s["beta_mean"] is None:
            continue
        eigvals = np.linalg.eigvalsh(s["gamma_mean"])
        if not np.all(eigvals < 0):
            n_excluded += 1
            continue
        z_star = -np.linalg.solve(s["gamma_mean"], s["beta_mean"])
        z1_list.append(z_star[0])
        z2_list.append(z_star[1])
        valid_betas.append(s["beta_mean"])
        valid_gammas.append(s["gamma_mean"])
    return dict(
        z1=np.array(z1_list), z2=np.array(z2_list), n_excluded=n_excluded, n_total=len(summaries),
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


def section_coeff_table():
    print("=" * 100)
    print("3a. Own-trait-only Lande-Arnold: beta1, gamma11, gamma22, gamma12 (gen>=2000, seed median, n=50)")
    print("=" * 100)
    print(f"\n{'cond':>8}  {'k':>4}  {'beta1 med':>10}  {'gamma11 med':>12}  {'gamma22 med':>12}  {'gamma12 med':>12}")
    results = {}
    for label, root in [("local", LOCAL_ROOT), ("shuffle", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            summaries = [seed_summary(root / f"k{k}", s, STEADY_STATE_MIN_GEN) for s in SEEDS]
            results[(label, k)] = summaries
            b1 = np.nanmedian([s["median_beta1"] for s in summaries])
            g11 = np.nanmedian([s["median_g11"] for s in summaries])
            g22 = np.nanmedian([s["median_g22"] for s in summaries])
            g12 = np.nanmedian([s["median_g12"] for s in summaries])
            print(f"{label:>8}  {k:>4}  {b1:>+10.5f}  {g11:>+12.5f}  {g22:>+12.5f}  {g12:>+12.5f}")
    return results


def section_fitness_peak(results):
    print("\n" + "=" * 100)
    print("3b. Fitness peak z* = -Gamma^-1*beta (grand mean, bootstrap 95% CI, n=1000)")
    print("=" * 100)
    print(f"\n{'cond':>8}  {'k':>4}  {'n_excl':>6}  {'n_valid':>7}  {'z*_K_self (grand)':>18}  {'95% CI':>22}  "
          f"{'z*_K_out (grand)':>17}  {'95% CI':>22}  {'skipped':>8}")
    for label in ["local", "shuffle"]:
        for k in K_VALUES:
            summaries = results[(label, k)]
            r = fitness_peak_for_group(summaries)
            if len(r["valid_betas"]) == 0:
                print(f"{label:>8}  {k:>4}  {r['n_excluded']:>6}  {0:>7}  (no valid seeds)")
                continue
            grand_beta = r["valid_betas"].mean(axis=0)
            grand_gamma = r["valid_gammas"].mean(axis=0)
            grand_z = -np.linalg.solve(grand_gamma, grand_beta)
            boot_z1, boot_z2, n_skipped = bootstrap_grand_zstar(r["valid_betas"], r["valid_gammas"])
            ci1 = (np.percentile(boot_z1, 2.5), np.percentile(boot_z1, 97.5)) if len(boot_z1) else (np.nan, np.nan)
            ci2 = (np.percentile(boot_z2, 2.5), np.percentile(boot_z2, 97.5)) if len(boot_z2) else (np.nan, np.nan)
            print(f"{label:>8}  {k:>4}  {r['n_excluded']:>6}  {len(r['valid_betas']):>7}  "
                  f"{grand_z[0]:>+18.4f}  [{ci1[0]:+.4f},{ci1[1]:+.4f}]{'':>4}  "
                  f"{grand_z[1]:>+17.4f}  [{ci2[0]:+.4f},{ci2[1]:+.4f}]{'':>4}  {n_skipped:>8}")


def main():
    results = section_coeff_table()
    section_fitness_peak(results)
    print("\n解釈はせず、データが示す形の記述にとどめて停止します。")


if __name__ == "__main__":
    main()
