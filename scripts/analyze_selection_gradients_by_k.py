"""第二章 精密kスイープ: 選択勾配の追加解析(既存9点で先行、k=64/80/99完了後に同一処理を再実行)。

個体レベルのK_self・K_out・利得は既存の individual_log.csv に世代ごと
(gen%100==0)・全個体分保存されていることを確認済み(generation,
agent_index, k_out, k_self, memory_depth, payoff の6列)。

各シード・各世代(gen>=2000)で以下を計算し、シードごとに中央値をとって
から k ごとに要約する(既存の周辺相関の集計方法と揃える: seeds=n=50が
単位):

1. 集団内 Spearman ρ(K_self, K_out)
2. 偏相関(ランクベース、線形回帰残差法 -- 本プロジェクトの既存の慣例と
   同じ方法): ρ(K_out, payoff | K_self), ρ(K_self, payoff | K_out)
3. Lande-Arnold回帰(Phillips & Arnold 1989の規約): 相対適応度
   w = payoff / その世代の平均payoff を、世代内で標準化した
   z1=K_self, z2=K_out とその二乗項・交互作用項に回帰:
       w ~ 1 + z1 + z2 + z1^2 + z2^2 + z1*z2
   β1, β2 は z1, z2 の生の回帰係数(方向性選択勾配)。
   γ11, γ22 は z1^2, z2^2 の生の回帰係数を2倍したもの(定型の補正)。
   γ12 は z1*z2 の生の回帰係数(そのまま、相関選択勾配)。

各統計量について、kごとに50シードの中央値の分布を one-sample
Wilcoxon(vs 0)で検定し、9点(k)にわたってHolm補正する(step0と同じ
方法)。

既存の周辺相関(analyze_opponent_count_precision_sweep.py の結果)と
並べて表・図で報告する。解釈はしない。

Usage:
    python3 scripts/analyze_selection_gradients_by_k.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]
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


def rank_partial_corr(x, y, z):
    """Rank-based partial correlation of x,y controlling for z, via the
    linear-regression-residual method on ranks (this project's existing
    convention, e.g. plot_payoff_colored_scatter.py)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    bx, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    by_, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)
    resid_x = rx - A @ bx
    resid_y = ry - A @ by_
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return np.nan
    return np.corrcoef(resid_x, resid_y)[0, 1]


def lande_arnold(k_self, k_out, payoff):
    """Returns (beta1, beta2, gamma11, gamma22, gamma12) for one
    generation's population. z1=K_self, z2=K_out standardized within
    the generation; w = relative fitness (payoff / mean payoff)."""
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, beta1, beta2, g11_raw, g22_raw, g12 = coef
    return beta1, beta2, 2 * g11_raw, 2 * g22_raw, g12


def per_generation_stats(by_gen, min_gen):
    """Returns dict of lists (one value per qualifying generation) for
    all 8 statistics."""
    out = {name: [] for name in
           ["rho_kself_kout", "partial_kout_given_kself", "partial_kself_given_kout",
            "beta1", "beta2", "gamma11", "gamma22", "gamma12"]}
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        k_self, k_out, payoff = g["k_self"], g["k_out"], g["payoff"]
        if len(set(k_self)) < 2 or len(set(k_out)) < 2 or len(set(payoff)) < 2:
            continue
        rho, _ = spearmanr(k_self, k_out)
        out["rho_kself_kout"].append(rho)
        out["partial_kout_given_kself"].append(rank_partial_corr(k_out, payoff, k_self))
        out["partial_kself_given_kout"].append(rank_partial_corr(k_self, payoff, k_out))
        b1, b2, g11, g22, g12 = lande_arnold(k_self, k_out, payoff)
        out["beta1"].append(b1)
        out["beta2"].append(b2)
        out["gamma11"].append(g11)
        out["gamma22"].append(g22)
        out["gamma12"].append(g12)
    return out


def per_seed_medians_all_stats(sweep_dir, seeds, min_gen):
    stat_names = ["rho_kself_kout", "partial_kout_given_kself", "partial_kself_given_kout",
                  "beta1", "beta2", "gamma11", "gamma22", "gamma12"]
    result = {name: [] for name in stat_names}
    for seed in seeds:
        by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
        stats = per_generation_stats(by_gen, min_gen)
        for name in stat_names:
            vals = [v for v in stats[name] if not np.isnan(v)]
            result[name].append(np.median(vals) if vals else np.nan)
    return {name: np.array(v) for name, v in result.items()}


def rank_biserial_one_sample(values):
    values = values[~np.isnan(values)]
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    stat, p = wilcoxon(values)
    r = 1 - (2 * stat) / (n * (n + 1))
    return r, p


def holm_correction(p_values):
    p_values = np.asarray(p_values)
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


def main():
    print(f"computing selection-gradient statistics for k={K_VALUES} (gen>={STEADY_STATE_MIN_GEN}, n=50 seeds each)")
    print("(individual-level K_self/K_out/payoff confirmed present in individual_log.csv, logged every 100 gen)\n")

    data = {k: per_seed_medians_all_stats(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN) for k in K_VALUES}

    stat_labels = {
        "rho_kself_kout": "ρ(K_self, K_out)",
        "partial_kout_given_kself": "partial ρ(K_out, payoff | K_self)",
        "partial_kself_given_kout": "partial ρ(K_self, payoff | K_out)",
        "beta1": "β1 (linear, K_self)",
        "beta2": "β2 (linear, K_out)",
        "gamma11": "γ11 (quadratic, K_self)",
        "gamma22": "γ22 (quadratic, K_out)",
        "gamma12": "γ12 (correlational, K_self×K_out)",
    }

    all_results = {}
    for stat_name, label in stat_labels.items():
        print(f"=== {label} ===")
        raw_p, r_vals, medians = [], [], []
        for k in K_VALUES:
            r, p = rank_biserial_one_sample(data[k][stat_name])
            raw_p.append(p)
            r_vals.append(r)
            medians.append(np.nanmedian(data[k][stat_name]))
        holm_p = holm_correction(np.array(raw_p))
        print(f"{'k':>4}  {'median':>9}  {'r':>7}  {'raw p':>9}  {'Holm p':>9}  {'sig':>6}")
        for k, med, r, p, hp in zip(K_VALUES, medians, r_vals, raw_p, holm_p):
            sig = "***" if hp < 0.001 else "**" if hp < 0.01 else "*" if hp < 0.05 else "n.s."
            print(f"{k:>4}  {med:>+9.4f}  {r:>7.3f}  {p:>9.4f}  {hp:>9.4f}  {sig:>6}")
        print()
        all_results[stat_name] = {"medians": medians, "raw_p": raw_p, "holm_p": holm_p, "r": r_vals}

    # figure: one row per statistic, violin across k (per-seed medians), Holm-sig annotated
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.flatten()
    positions = list(range(len(K_VALUES)))
    rng = np.random.default_rng(0)

    for ax, (stat_name, label) in zip(axes, stat_labels.items()):
        vals_by_k = [data[k][stat_name][~np.isnan(data[k][stat_name])] for k in K_VALUES]
        parts = ax.violinplot(vals_by_k, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#2a78d6")
            pc.set_alpha(0.5)
        parts["cmedians"].set_color("black")
        for pos, vals in zip(positions, vals_by_k):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=8, color="#8f8f88", alpha=0.4, zorder=3)
        y_top = max(np.percentile(v, 97) for v in vals_by_k)
        y_span = y_top - min(np.percentile(v, 3) for v in vals_by_k)
        for pos, hp in zip(positions, all_results[stat_name]["holm_p"]):
            sig = "***" if hp < 0.001 else "**" if hp < 0.01 else "*" if hp < 0.05 else "n.s."
            ax.text(pos, y_top + 0.05 * y_span, sig, ha="center", fontsize=8,
                    color="#1baf7a" if sig != "n.s." else "#8f8f88")
        ax.axhline(0, color="#8f8f88", linewidth=1.0, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k")
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Selection gradients vs opponent count k (per-seed median, n=50, gen≥2000, Holm-corrected)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig_path = OUT_DIR / "selection_gradients_by_k.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
