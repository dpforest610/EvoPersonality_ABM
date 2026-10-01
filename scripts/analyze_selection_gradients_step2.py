"""第二章 精密kスイープ 追加解析ステップ2:

(a) 正準解析: 各シード・各世代のLande-Arnold γ行列
    [[γ11, γ12], [γ12, γ22]] の固有値・固有ベクトルを求め、
    シードごとに中央値をとってkごとに要約(固有値2つの分布、
    主軸の角度 -- K_self軸(1,0)から反時計回りに何度か、[0,180)に正規化)。
    γの値はanalyze_selection_gradients_by_k.pyと同じ規約: γ11,γ22は
    回帰係数を2倍したもの、γ12は回帰係数そのまま(Phillips & Arnold 1989)。

(b) 端効果チェック: kごとに、K_self・K_out個体値が<0.02または>0.98に
    張り付いている割合(全シード・全世代・全個体をプールした比率)。

(c) 第一章(well-mixed, k=99相当)データでも個体レベルK_self/K_out/利得
    の保存を確認済み。同じ(a)(b)を実施。新k=99(同一パイプライン)が
    完了したらそちらにも適用する。

解釈はせず、表と図のみ出力して停止。

Usage:
    python3 scripts/analyze_selection_gradients_step2.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
CH1_SEEDS = list(range(50))

BOUNDARY_LOW = 0.02
BOUNDARY_HIGH = 0.98


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


def lande_arnold_gamma_matrix(k_self, k_out, payoff):
    """Returns the 2x2 gamma matrix [[g11,g12],[g12,g22]] for one
    generation's population (same convention as analyze_selection_gradients_by_k.py:
    g11,g22 = 2x raw quadratic coefficient; g12 = raw cross-term coefficient)."""
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, _, _, g11_raw, g22_raw, g12 = coef
    g11, g22 = 2 * g11_raw, 2 * g22_raw
    return np.array([[g11, g12], [g12, g22]])


def canonical_analysis_per_seed(sweep_dir, seeds, min_gen):
    """Returns per-seed median (eigval1, eigval2, angle_deg) where
    eigval1 >= eigval2 and angle_deg is the eigenvector of eigval1's
    angle from the K_self axis (z1), normalized to [0,180)."""
    eig1_list, eig2_list, angle_list = [], [], []
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        eig1s, eig2s, angles = [], [], []
        for gen, g in by_gen.items():
            if gen < min_gen:
                continue
            k_self, k_out, payoff = g["k_self"], g["k_out"], g["payoff"]
            if len(set(k_self)) < 2 or len(set(k_out)) < 2 or len(set(payoff)) < 2:
                continue
            gamma = lande_arnold_gamma_matrix(k_self, k_out, payoff)
            eigvals, eigvecs = np.linalg.eigh(gamma)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            major_vec = eigvecs[:, 0]
            angle = np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180
            eig1s.append(eigvals[0])
            eig2s.append(eigvals[1])
            angles.append(angle)
        eig1_list.append(np.median(eig1s) if eig1s else np.nan)
        eig2_list.append(np.median(eig2s) if eig2s else np.nan)
        angle_list.append(np.median(angles) if angles else np.nan)
    return np.array(eig1_list), np.array(eig2_list), np.array(angle_list)


def boundary_fraction(sweep_dir, seeds, min_gen):
    """Pooled (all seeds, all gen>=min_gen, all 100 agents) fraction of
    K_self and K_out values below BOUNDARY_LOW or above BOUNDARY_HIGH."""
    n_total_self, n_boundary_self = 0, 0
    n_total_out, n_boundary_out = 0, 0
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        for gen, g in by_gen.items():
            if gen < min_gen:
                continue
            kself = np.array(g["k_self"])
            kout = np.array(g["k_out"])
            n_total_self += len(kself)
            n_boundary_self += int(np.sum((kself < BOUNDARY_LOW) | (kself > BOUNDARY_HIGH)))
            n_total_out += len(kout)
            n_boundary_out += int(np.sum((kout < BOUNDARY_LOW) | (kout > BOUNDARY_HIGH)))
    frac_self = n_boundary_self / n_total_self if n_total_self else float("nan")
    frac_out = n_boundary_out / n_total_out if n_total_out else float("nan")
    return frac_self, frac_out, n_total_self


def main():
    print("=== (a) 正準解析: γ行列の固有値・主軸角度(kごと、シード中央値のn=50分布) ===\n")
    print(f"{'k':>4}  {'eig1 median':>12}  {'eig2 median':>12}  {'angle median(deg)':>18}  {'angle IQR':>22}")

    results_a = {}
    for k in K_VALUES:
        eig1, eig2, angle = canonical_analysis_per_seed(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN)
        results_a[k] = (eig1, eig2, angle)
        iqr = f"[{np.nanpercentile(angle,25):.1f},{np.nanpercentile(angle,75):.1f}]"
        print(f"{k:>4}  {np.nanmedian(eig1):>+12.4f}  {np.nanmedian(eig2):>+12.4f}  "
              f"{np.nanmedian(angle):>18.1f}  {iqr:>22}")

    print("\n=== (b) 端効果チェック: K_self・K_out個体値が<0.02または>0.98の割合(プール) ===\n")
    print(f"{'k':>4}  {'frac K_self boundary':>20}  {'frac K_out boundary':>20}  {'n (pooled)':>12}")
    results_b = {}
    for k in K_VALUES:
        frac_self, frac_out, n = boundary_fraction(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN)
        results_b[k] = (frac_self, frac_out)
        print(f"{k:>4}  {frac_self:>20.4f}  {frac_out:>20.4f}  {n:>12}")

    print("\n=== (c) 第一章(well-mixed, k=99相当)データで同じ解析 ===\n")
    eig1_ch1, eig2_ch1, angle_ch1 = canonical_analysis_per_seed(CH1_DIR, CH1_SEEDS, STEADY_STATE_MIN_GEN)
    print(f"正準解析: eig1 median={np.nanmedian(eig1_ch1):+.4f}  eig2 median={np.nanmedian(eig2_ch1):+.4f}  "
          f"angle median={np.nanmedian(angle_ch1):.1f}deg  "
          f"IQR=[{np.nanpercentile(angle_ch1,25):.1f},{np.nanpercentile(angle_ch1,75):.1f}]")
    frac_self_ch1, frac_out_ch1, n_ch1 = boundary_fraction(CH1_DIR, CH1_SEEDS, STEADY_STATE_MIN_GEN)
    print(f"端効果: frac K_self boundary={frac_self_ch1:.4f}  frac K_out boundary={frac_out_ch1:.4f}  n={n_ch1}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    positions = list(range(len(K_VALUES) + 1))
    labels = [str(k) for k in K_VALUES] + ["99\n(ch.1)"]
    rng = np.random.default_rng(0)

    # eigenvalues
    ax = axes[0]
    eig1_all = [results_a[k][0] for k in K_VALUES] + [eig1_ch1]
    eig2_all = [results_a[k][1] for k in K_VALUES] + [eig2_ch1]
    for pos, (e1, e2) in enumerate(zip(eig1_all, eig2_all)):
        ax.scatter(np.full(len(e1), pos) - 0.15, e1, s=14, color="#2a78d6", alpha=0.5)
        ax.scatter(np.full(len(e2), pos) + 0.15, e2, s=14, color="#eb6834", alpha=0.5)
        ax.plot([pos - 0.15], [np.nanmedian(e1)], marker="_", markersize=20, color="black")
        ax.plot([pos + 0.15], [np.nanmedian(e2)], marker="_", markersize=20, color="black")
    ax.axhline(0, color="#8f8f88", linewidth=1, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k")
    ax.set_ylabel("eigenvalue (per-seed median)")
    ax.set_title("γ-matrix eigenvalues (blue=larger, orange=smaller)")
    ax.spines[["top", "right"]].set_visible(False)

    # angle
    ax = axes[1]
    angle_all = [results_a[k][2] for k in K_VALUES] + [angle_ch1]
    parts = ax.violinplot(angle_all, positions=positions, widths=0.7, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1baf7a")
        pc.set_alpha(0.5)
    parts["cmedians"].set_color("black")
    for pos, vals in zip(positions, angle_all):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10, color="#8f8f88", alpha=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k")
    ax.set_ylabel("major-axis angle from K_self axis (deg, [0,180))")
    ax.set_title("γ-matrix principal axis orientation")
    ax.spines[["top", "right"]].set_visible(False)

    # boundary fractions
    ax = axes[2]
    frac_self_all = [results_b[k][0] for k in K_VALUES] + [frac_self_ch1]
    frac_out_all = [results_b[k][1] for k in K_VALUES] + [frac_out_ch1]
    ax.plot(positions, frac_self_all, marker="o", color="#eb6834", label="K_self")
    ax.plot(positions, frac_out_all, marker="o", color="#2a78d6", label="K_out")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k")
    ax.set_ylabel(f"fraction with value <{BOUNDARY_LOW} or >{BOUNDARY_HIGH}")
    ax.set_title("Boundary-sticking fraction (pooled)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Step 2(a,b,c): canonical gamma-matrix analysis + boundary check", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "selection_gradients_step2_canonical_boundary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
