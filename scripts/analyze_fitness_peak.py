"""第二章 精密kスイープ 追加解析: 適応度頂上の位置 z* = -Γ^-1 β
(既存データのみ、新規シミュレーション不要)。

各シードで、シード内平均したβ(β1,β2)とγ行列(Phillips & Arnold規約:
γ11,γ22は回帰係数の2倍、γ12は回帰係数そのまま)から、
    z* = -Γ^-1 β
を計算する(世代内標準化のSD単位、集団平均からの変位ベクトル)。
Γが負定値でない(両固有値が負でない)シードは適応度の「頂上」として
解釈できないため除外し、除外数を報告する。

kごと(9点+ch.1)に、z*のK_self成分・K_out成分・ノルムを、シード中央値・
IQR・Wilcoxon(vs 0)で報告。全シード平均版(有効シードのβ・Γをそれぞれ
平均してからz*を計算したもの)にシードのブートストラップ(1000回)で
95%信頼区間もつける。

参考として一変数近似(-β1/γ11, -β2/γ22、γ12を無視した単純近似)も
同じ有効シード集合で並べる。

解釈はせず、表のみ出力して停止。

Usage:
    python3 scripts/analyze_fitness_peak.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
CH1_SEEDS = list(range(50))

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


def seed_mean_beta_gamma(sweep_dir, seed, min_gen):
    """Average (beta1,beta2) and the gamma matrix across qualifying
    generations WITHIN this seed (same averaging-before-eigendecomposition
    philosophy as step2b's seed_mean_gamma_matrices)."""
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    betas, gammas = [], []
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2:
            continue
        b1, b2, g11, g22, g12 = lande_arnold_coeffs(g["k_self"], g["k_out"], g["payoff"])
        betas.append([b1, b2])
        gammas.append([[g11, g12], [g12, g22]])
    if not betas:
        return None, None
    return np.mean(betas, axis=0), np.mean(gammas, axis=0)


def fitness_peak_for_k(sweep_dir, seeds, min_gen):
    """Returns dict with per-valid-seed arrays: z1 (K_self), z2 (K_out),
    norm, uni1 (-beta1/gamma11), uni2 (-beta2/gamma22); plus n_excluded
    and the valid (beta, gamma) pairs for bootstrap."""
    z1_list, z2_list, norm_list, uni1_list, uni2_list = [], [], [], [], []
    valid_betas, valid_gammas = [], []
    n_excluded = 0
    n_no_data = 0
    for seed in seeds:
        beta, gamma = seed_mean_beta_gamma(sweep_dir, seed, min_gen)
        if beta is None:
            n_no_data += 1
            continue
        eigvals = np.linalg.eigvalsh(gamma)
        if not np.all(eigvals < 0):
            n_excluded += 1
            continue
        z_star = -np.linalg.solve(gamma, beta)
        z1_list.append(z_star[0])
        z2_list.append(z_star[1])
        norm_list.append(np.linalg.norm(z_star))
        uni1_list.append(-beta[0] / gamma[0, 0])
        uni2_list.append(-beta[1] / gamma[1, 1])
        valid_betas.append(beta)
        valid_gammas.append(gamma)

    return dict(
        z1=np.array(z1_list), z2=np.array(z2_list), norm=np.array(norm_list),
        uni1=np.array(uni1_list), uni2=np.array(uni2_list),
        n_excluded=n_excluded, n_no_data=n_no_data, n_total=len(seeds),
        valid_betas=np.array(valid_betas), valid_gammas=np.array(valid_gammas),
    )


def rank_biserial_one_sample(values):
    values = np.asarray(values)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


def bootstrap_grand_zstar(valid_betas, valid_gammas, n_bootstrap=N_BOOTSTRAP):
    n = len(valid_betas)
    boot_z1, boot_z2, boot_norm = [], [], []
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
        boot_norm.append(np.linalg.norm(z))
    return np.array(boot_z1), np.array(boot_z2), np.array(boot_norm), n_skipped


def main():
    all_groups = [(str(k), SCAN_ROOT / f"k{k}", SEEDS) for k in K_VALUES]
    all_groups.append(("99(ch.1)", CH1_DIR, CH1_SEEDS))

    print("=== z* = -Γ^-1β (fitness peak, SD units) -- per-seed median/IQR/Wilcoxon ===\n")
    print(f"{'k':>8}  {'n_excl(Γ not neg.def.)':>22}  {'n_valid':>8}  |  "
          f"{'z*_K_self med':>14}  {'IQR':>18}  {'p':>8}  |  "
          f"{'z*_K_out med':>13}  {'IQR':>18}  {'p':>8}  |  "
          f"{'||z*|| med':>11}  {'p':>8}")

    results = {}
    for label, sweep_dir, seeds in all_groups:
        r = fitness_peak_for_k(sweep_dir, seeds, STEADY_STATE_MIN_GEN)
        results[label] = r
        _, p1 = rank_biserial_one_sample(r["z1"])
        _, p2 = rank_biserial_one_sample(r["z2"])
        _, pn = rank_biserial_one_sample(r["norm"])
        iqr1 = f"[{np.percentile(r['z1'],25):+.3f},{np.percentile(r['z1'],75):+.3f}]"
        iqr2 = f"[{np.percentile(r['z2'],25):+.3f},{np.percentile(r['z2'],75):+.3f}]"
        print(f"{label:>8}  {r['n_excluded']:>22}  {len(r['z1']):>8}  |  "
              f"{np.median(r['z1']):>+14.3f}  {iqr1:>18}  {p1:>8.4f}  |  "
              f"{np.median(r['z2']):>+13.3f}  {iqr2:>18}  {p2:>8.4f}  |  "
              f"{np.median(r['norm']):>+11.3f}  {pn:>8.4f}")

    print("\n=== 参考: 一変数近似 -β1/γ11, -β2/γ22(同じ有効シード集合) ===\n")
    print(f"{'k':>8}  {'uni_K_self med':>14}  {'IQR':>18}  {'uni_K_out med':>13}  {'IQR':>18}")
    for label, _, _ in all_groups:
        r = results[label]
        iqr1 = f"[{np.percentile(r['uni1'],25):+.3f},{np.percentile(r['uni1'],75):+.3f}]"
        iqr2 = f"[{np.percentile(r['uni2'],25):+.3f},{np.percentile(r['uni2'],75):+.3f}]"
        print(f"{label:>8}  {np.median(r['uni1']):>+14.3f}  {iqr1:>18}  "
              f"{np.median(r['uni2']):>+13.3f}  {iqr2:>18}")

    print(f"\n=== 全シード平均版z*のブートストラップ95%CI(n={N_BOOTSTRAP}回、有効シードのみ復元抽出) ===\n")
    print(f"{'k':>8}  {'z*_K_self (grand)':>18}  {'95% CI':>22}  {'z*_K_out (grand)':>17}  "
          f"{'95% CI':>22}  {'skipped(非負定値)':>16}")
    for label, _, _ in all_groups:
        r = results[label]
        grand_beta = r["valid_betas"].mean(axis=0)
        grand_gamma = r["valid_gammas"].mean(axis=0)
        grand_z = -np.linalg.solve(grand_gamma, grand_beta)
        boot_z1, boot_z2, boot_norm, n_skipped = bootstrap_grand_zstar(r["valid_betas"], r["valid_gammas"])
        ci1 = (np.percentile(boot_z1, 2.5), np.percentile(boot_z1, 97.5)) if len(boot_z1) else (np.nan, np.nan)
        ci2 = (np.percentile(boot_z2, 2.5), np.percentile(boot_z2, 97.5)) if len(boot_z2) else (np.nan, np.nan)
        ci1_str = f"[{ci1[0]:+.3f},{ci1[1]:+.3f}]"
        ci2_str = f"[{ci2[0]:+.3f},{ci2[1]:+.3f}]"
        print(f"{label:>8}  {grand_z[0]:>+18.3f}  {ci1_str:>22}  {grand_z[1]:>+17.3f}  "
              f"{ci2_str:>22}  {n_skipped:>16}")


if __name__ == "__main__":
    main()
