"""第二章 精密kスイープ ステップ2補足(既存データのみ、新規シミュレーション不要)。

1. 確認事項(本スクリプトのdocstringとして明記):
   - γの報告値: γ11,γ22は回帰係数を2倍したもの(Phillips & Arnold 1989の
     規約)。γ12は回帰係数そのまま(2倍しない)。
   - major-axis angle: 大きい方の固有値に対応する固有ベクトルの角度
     (K_self軸=(1,0)から反時計回り、[0,180)に正規化)。

2. 第一章データ(k=99相当, well-mixed)のLande-Arnold係数表
   (β1,β2,γ11,γ22,γ12、シード中央値・rank-biserial r・Wilcoxon p)を
   既存9点と同じ形式で報告。

3. 固有値推定ノイズ対策: 各シード内で全世代(gen>=2000)のγ行列を要素ごとに
   平均してから固有分解(世代ごとに固有分解してから中央値を取る前回の
   方法とは異なる)。シード間の中央値・IQRで報告(全9点+ch.1)。
   さらに、全シードの平均γ行列(50シードの「シード内平均行列」をさらに
   平均したもの)の固有値に、シードの単位でのブートストラップ(1000回、
   復元抽出)で95%信頼区間をつける。

4. 端効果の頑健性: K_self・K_outが両方とも[0.02,0.98]の内側にある個体
   だけを使ってLande-Arnold回帰をやり直し、全データ版と並べて報告。

解釈はせず、表のみ出力して停止。

Usage:
    python3 scripts/analyze_selection_gradients_step2b.py
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

BOUNDARY_LOW = 0.02
BOUNDARY_HIGH = 0.98
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
    """Returns (beta1, beta2, gamma11, gamma22, gamma12) -- gamma11/22
    doubled, gamma12 raw (Phillips & Arnold 1989)."""
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, beta1, beta2, g11_raw, g22_raw, g12 = coef
    return beta1, beta2, 2 * g11_raw, 2 * g22_raw, g12


def rank_biserial_one_sample(values):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


# ---------- section 2: chapter-1 Lande-Arnold table ----------

def per_seed_medians_all_stats(sweep_dir, seeds, min_gen):
    stat_names = ["beta1", "beta2", "gamma11", "gamma22", "gamma12"]
    result = {name: [] for name in stat_names}
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        per_gen = {name: [] for name in stat_names}
        for gen, g in by_gen.items():
            if gen < min_gen:
                continue
            if len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2:
                continue
            b1, b2, g11, g22, g12 = lande_arnold_coeffs(g["k_self"], g["k_out"], g["payoff"])
            per_gen["beta1"].append(b1)
            per_gen["beta2"].append(b2)
            per_gen["gamma11"].append(g11)
            per_gen["gamma22"].append(g22)
            per_gen["gamma12"].append(g12)
        for name in stat_names:
            vals = per_gen[name]
            result[name].append(np.median(vals) if vals else np.nan)
    return {name: np.array(v) for name, v in result.items()}


def section2_chapter1_table():
    print("=== 2. 第一章(well-mixed, k=99相当)Lande-Arnold係数表 ===\n")
    data = per_seed_medians_all_stats(CH1_DIR, CH1_SEEDS, STEADY_STATE_MIN_GEN)
    labels = {"beta1": "β1 (linear, K_self)", "beta2": "β2 (linear, K_out)",
              "gamma11": "γ11 (quadratic, K_self)", "gamma22": "γ22 (quadratic, K_out)",
              "gamma12": "γ12 (correlational)"}
    print(f"{'coefficient':>26}  {'median':>10}  {'r':>7}  {'Wilcoxon p':>11}")
    for name, label in labels.items():
        r, p = rank_biserial_one_sample(data[name])
        print(f"{label:>26}  {np.nanmedian(data[name]):>+10.4f}  {r:>7.3f}  {p:>11.4f}")
    print()


# ---------- section 3: seed-averaged gamma matrix + bootstrap CI ----------

def seed_mean_gamma_matrices(sweep_dir, seeds, min_gen):
    """Returns array shape (n_seeds, 2, 2): each seed's OWN gamma matrix
    averaged element-wise over qualifying generations (computed BEFORE
    eigendecomposition, unlike step2's per-generation-then-median
    approach)."""
    matrices = []
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        per_gen_matrices = []
        for gen, g in by_gen.items():
            if gen < min_gen:
                continue
            if len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2:
                continue
            _, _, g11, g22, g12 = lande_arnold_coeffs(g["k_self"], g["k_out"], g["payoff"])
            per_gen_matrices.append(np.array([[g11, g12], [g12, g22]]))
        matrices.append(np.mean(per_gen_matrices, axis=0) if per_gen_matrices else np.full((2, 2), np.nan))
    return np.array(matrices)


def eig_large_small_angle(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])) % 180
    return eigvals[0], eigvals[1], angle


def section3_seed_averaged_eigen_and_bootstrap(all_k_dirs):
    print("=== 3. シード内平均γ行列の固有分解(シード間 中央値・IQR) + ブートストラップCI ===\n")
    print(f"{'k':>8}  {'eig_large med':>13}  {'IQR':>20}  {'eig_small med':>13}  {'IQR':>20}  "
          f"{'angle med':>9}  {'IQR':>18}")

    results = {}
    for label, sweep_dir, seeds in all_k_dirs:
        matrices = seed_mean_gamma_matrices(sweep_dir, seeds, STEADY_STATE_MIN_GEN)
        eig_large, eig_small, angle = [], [], []
        for m in matrices:
            if np.any(np.isnan(m)):
                continue
            el, es, a = eig_large_small_angle(m)
            eig_large.append(el)
            eig_small.append(es)
            angle.append(a)
        eig_large, eig_small, angle = np.array(eig_large), np.array(eig_small), np.array(angle)

        # bootstrap over seeds: resample the per-seed mean matrices, average, eigendecompose
        valid_matrices = matrices[~np.isnan(matrices).any(axis=(1, 2))]
        n_seeds = len(valid_matrices)
        boot_large, boot_small = [], []
        for _ in range(N_BOOTSTRAP):
            idx = RNG.integers(0, n_seeds, size=n_seeds)
            resampled_mean = valid_matrices[idx].mean(axis=0)
            el, es, _ = eig_large_small_angle(resampled_mean)
            boot_large.append(el)
            boot_small.append(es)
        boot_large, boot_small = np.array(boot_large), np.array(boot_small)
        ci_large = (np.percentile(boot_large, 2.5), np.percentile(boot_large, 97.5))
        ci_small = (np.percentile(boot_small, 2.5), np.percentile(boot_small, 97.5))

        grand_mean_matrix = valid_matrices.mean(axis=0)
        grand_large, grand_small, grand_angle = eig_large_small_angle(grand_mean_matrix)

        results[label] = dict(eig_large=eig_large, eig_small=eig_small, angle=angle,
                               ci_large=ci_large, ci_small=ci_small,
                               grand_large=grand_large, grand_small=grand_small)

        iqr_l = f"[{np.percentile(eig_large,25):+.4f},{np.percentile(eig_large,75):+.4f}]"
        iqr_s = f"[{np.percentile(eig_small,25):+.4f},{np.percentile(eig_small,75):+.4f}]"
        iqr_a = f"[{np.percentile(angle,25):.1f},{np.percentile(angle,75):.1f}]"
        print(f"{label:>8}  {np.median(eig_large):>+13.4f}  {iqr_l:>20}  {np.median(eig_small):>+13.4f}  "
              f"{iqr_s:>20}  {np.median(angle):>9.1f}  {iqr_a:>18}")

    print(f"\n--- ブートストラップ95%CI(シード単位、n={N_BOOTSTRAP}回、全シード平均γ行列の固有値) ---")
    print(f"{'k':>8}  {'eig_large (grand)':>18}  {'95% CI':>24}  {'CIがゼロをまたぐか':>18}  |  "
          f"{'eig_small (grand)':>18}  {'95% CI':>24}  {'CIがゼロをまたぐか':>18}")
    for label, r in results.items():
        straddles_large = "またぐ" if r["ci_large"][0] < 0 < r["ci_large"][1] else "またがない"
        straddles_small = "またぐ" if r["ci_small"][0] < 0 < r["ci_small"][1] else "またがない"
        ci_l_str = f"[{r['ci_large'][0]:+.4f},{r['ci_large'][1]:+.4f}]"
        ci_s_str = f"[{r['ci_small'][0]:+.4f},{r['ci_small'][1]:+.4f}]"
        print(f"{label:>8}  {r['grand_large']:>+18.4f}  {ci_l_str:>24}  {straddles_large:>18}  |  "
              f"{r['grand_small']:>+18.4f}  {ci_s_str:>24}  {straddles_small:>18}")
    print()
    return results


# ---------- section 4: boundary-restricted Lande-Arnold ----------

def per_seed_medians_boundary_restricted(sweep_dir, seeds, min_gen):
    stat_names = ["beta1", "beta2", "gamma11", "gamma22", "gamma12"]
    result = {name: [] for name in stat_names}
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        per_gen = {name: [] for name in stat_names}
        for gen, g in by_gen.items():
            if gen < min_gen:
                continue
            k_self = np.array(g["k_self"])
            k_out = np.array(g["k_out"])
            payoff = np.array(g["payoff"])
            mask = (
                (k_self >= BOUNDARY_LOW) & (k_self <= BOUNDARY_HIGH)
                & (k_out >= BOUNDARY_LOW) & (k_out <= BOUNDARY_HIGH)
            )
            if mask.sum() < 10:
                continue
            ks, ko, pay = k_self[mask], k_out[mask], payoff[mask]
            if len(set(ks)) < 2 or len(set(ko)) < 2 or len(set(pay)) < 2:
                continue
            b1, b2, g11, g22, g12 = lande_arnold_coeffs(ks, ko, pay)
            per_gen["beta1"].append(b1)
            per_gen["beta2"].append(b2)
            per_gen["gamma11"].append(g11)
            per_gen["gamma22"].append(g22)
            per_gen["gamma12"].append(g12)
        for name in stat_names:
            vals = per_gen[name]
            result[name].append(np.median(vals) if vals else np.nan)
    return {name: np.array(v) for name, v in result.items()}


def section4_boundary_restricted(all_k_dirs):
    print(f"=== 4. 端効果の頑健性: [{BOUNDARY_LOW},{BOUNDARY_HIGH}]内側のみでLande-Arnold回帰 ===\n")
    stat_names = ["beta1", "beta2", "gamma11", "gamma22", "gamma12"]
    labels = {"beta1": "β1", "beta2": "β2", "gamma11": "γ11", "gamma22": "γ22", "gamma12": "γ12"}

    for label, sweep_dir, seeds in all_k_dirs:
        full = per_seed_medians_all_stats(sweep_dir, seeds, STEADY_STATE_MIN_GEN)
        restricted = per_seed_medians_boundary_restricted(sweep_dir, seeds, STEADY_STATE_MIN_GEN)
        print(f"--- k={label} ---")
        print(f"{'coef':>8}  {'full median':>12}  {'restricted median':>18}  {'full p':>9}  {'restricted p':>13}")
        for name in stat_names:
            r_full, p_full = rank_biserial_one_sample(full[name])
            r_rest, p_rest = rank_biserial_one_sample(restricted[name])
            print(f"{labels[name]:>8}  {np.nanmedian(full[name]):>+12.4f}  "
                  f"{np.nanmedian(restricted[name]):>+18.4f}  {p_full:>9.4f}  {p_rest:>13.4f}")
        print()


def main():
    print(__doc__.split("Usage:")[0])

    section2_chapter1_table()

    all_k_dirs = [(str(k), SCAN_ROOT / f"k{k}", SEEDS) for k in K_VALUES]
    all_k_dirs.append(("99(ch.1)", CH1_DIR, CH1_SEEDS))

    section3_seed_averaged_eigen_and_bootstrap(all_k_dirs)
    section4_boundary_restricted(all_k_dirs)


if __name__ == "__main__":
    main()
