"""精密kスイープ 全12点(k=4,6,8,12,16,24,32,40,50,64,80,99)最終解析。

k=64,80,99はローカル実行の延長ラン(n=50シード×5000世代、既存9点と
同一パイプライン・同一パラメータ)で完了。既存9点と合わせて全12点を
同じ条件(gen>=2000、シード中央値、n=50)で解析する。

「k=99」は今回の同一パイプライン版を指す。第一章k=99(well-mixed)は
12点の検定・補正には含めず、別枠の参照点として扱う。

A. 周辺相関(ステップ2):
  1. 全12点の表(K_out・K_self): per-seed Spearman ρの中央値・IQR・
     Wilcoxon p(vs 0)・Holm補正p(12点)・効果量r
  2. 二直線検定(breakpoint regression): 横軸 log k、縦軸 per-seed
     K_out ρ(600点=12k×50シード、各(k,seed)を1標本として長形式で
     回帰)。ブレークポイントは、隣接するk間の11通りの分割点のうち、
     左右2本のOLS直線の残差平方和(SSR)の合計を最小にするものとして
     決定する(Simonsohn 2018の「Robin Hood」アルゴリズムは反復的に
     分割点を1点ずつ移動させてSSRを最小化する手続きだが、本データは
     x軸がk=12点しかないため、取りうる分割点が11通りしかなく、その
     反復アルゴリズムが収束する先の分割点と、全11通りを総当たりして
     SSR最小を選ぶ本実装の結果は数学的に同じになる)。
     左右の傾き・p値(scipy.stats.linregress)を報告。ブレークポイント
     位置はシード単位ブートストラップ(1000回、k毎に50シードを復元
     抽出)で95%区間推定。「逆U字」と判定するのは、左が正・有意かつ
     右が負・有意な場合のみ。
  3. 新k=99 vs 第一章k=99: Mann-Whitney U + 効果量(rank-biserial的な
     r=1-2U/(n1*n2))、K_out・K_self両方。
  4. 図: 12点バイオリン+個別点+中央値+Holm補正後有意記号。第一章k=99は
     右端に灰色の参照点として別枠表示(検定・補正には含めない)。

B. 選択勾配(既存9点の解析を12点に拡張):
  5. 偏相関 ρ(K_out,payoff|K_self), ρ(K_self,payoff|K_out), ρ(K_self,K_out)
  6. Lande-Arnold係数 β1,β2,γ11,γ22,γ12(Phillips & Arnold規約)。
     Wilcoxon(vs 0) + Holm補正(12点)。
  7. シード内平均γ行列の固有分解(固有値2つ・主軸角度)、シード間
     中央値・IQR、全シード平均行列の固有値ブートストラップ95%CI。
  8. 適応度頂上 z*=-Γ^-1β(全シード平均版+ブートストラップ95%CI、
     K_self・K_out成分)。
  9. β2のk依存性: per-seed β2をlog kに回帰した傾き・p値(単調傾向、
     600点長形式)。あわせてA-2と同じ二直線検定をβ2にも適用。

各項目に第一章k=99を参照列として付す。解釈はせず、データが示す形の
記述のみで停止する。

Usage:
    python3 scripts/analyze_full_12point_sweep.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import linregress, mannwhitneyu, rankdata, spearmanr, wilcoxon

SCAN_ROOT = Path("scripts/output/opponent_count_precision_sweep")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50, 64, 80, 99]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000

CH1_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
CH1_SEEDS = list(range(50))
CH1_LABEL = "99(ch.1)"

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- shared

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
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    bx, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    by_, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)
    resid_x = rx - A @ bx
    resid_y = ry - A @ by_
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return np.nan
    return np.corrcoef(resid_x, resid_y)[0, 1]


def lande_arnold_coeffs(k_self, k_out, payoff):
    """beta1,beta2,gamma11,gamma22,gamma12 -- gamma11/22 doubled, gamma12 raw
    (Phillips & Arnold 1989)."""
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


def mannwhitney_effect(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * stat) / (len(x) * len(y))
    return stat, p, r


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


def eig_large_small_angle(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])) % 180
    return eigvals[0], eigvals[1], angle


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


# --------------------------------------------------------- per-seed pass

def compute_seed_summary(path, min_gen):
    """Single pass over one seed's individual_log.csv. Returns a dict with
    everything every later section needs for this seed, so each file is
    read exactly once regardless of how many sections use it."""
    by_gen = load_individual_log(path)

    per_gen = {
        "rho_kout_payoff": [], "rho_kself_payoff": [], "rho_kself_kout": [],
        "partial_kout_given_kself": [], "partial_kself_given_kout": [],
        "beta1": [], "beta2": [], "gamma11": [], "gamma22": [], "gamma12": [],
    }
    gamma_matrices = []  # for seed-mean-before-eigendecomposition (section 7/8)

    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        k_self, k_out, payoff = g["k_self"], g["k_out"], g["payoff"]
        if len(set(k_self)) < 2 or len(set(k_out)) < 2 or len(set(payoff)) < 2:
            continue

        rho_out, _ = spearmanr(k_out, payoff)
        rho_self, _ = spearmanr(k_self, payoff)
        rho_ks_ko, _ = spearmanr(k_self, k_out)
        per_gen["rho_kout_payoff"].append(rho_out)
        per_gen["rho_kself_payoff"].append(rho_self)
        per_gen["rho_kself_kout"].append(rho_ks_ko)
        per_gen["partial_kout_given_kself"].append(rank_partial_corr(k_out, payoff, k_self))
        per_gen["partial_kself_given_kout"].append(rank_partial_corr(k_self, payoff, k_out))

        b1, b2, g11, g22, g12 = lande_arnold_coeffs(k_self, k_out, payoff)
        per_gen["beta1"].append(b1)
        per_gen["beta2"].append(b2)
        per_gen["gamma11"].append(g11)
        per_gen["gamma22"].append(g22)
        per_gen["gamma12"].append(g12)
        gamma_matrices.append(np.array([[g11, g12], [g12, g22]]))

    out = {}
    for name, vals in per_gen.items():
        vals = [v for v in vals if not np.isnan(v)]
        out[name] = np.median(vals) if vals else np.nan

    if gamma_matrices:
        out["beta_mean"] = np.array([np.mean(per_gen["beta1"]), np.mean(per_gen["beta2"])])
        out["gamma_mean"] = np.mean(gamma_matrices, axis=0)
    else:
        out["beta_mean"] = np.full(2, np.nan)
        out["gamma_mean"] = np.full((2, 2), np.nan)

    return out


def load_all(sweep_dir, seeds, min_gen):
    """List of per-seed summary dicts (one per seed, in seed order)."""
    return [compute_seed_summary(Path(sweep_dir) / f"seed_{s}_individual_log.csv", min_gen) for s in seeds]


def stat_array(summaries, name):
    return np.array([s[name] for s in summaries])


# --------------------------------------------------------- A1: marginal table

def section_A1(data, ch1_data):
    print("=" * 100)
    print("A-1. 全12点: K_out・K_self 周辺相関(per-seed Spearman ρ vs payoff, gen>=2000, n=50)")
    print("=" * 100)

    for channel, key in [("K_out", "rho_kout_payoff"), ("K_self", "rho_kself_payoff")]:
        print(f"\n--- {channel} ---")
        raw_p, r_vals, medians, iqrs = [], [], [], []
        for k in K_VALUES:
            vals = stat_array(data[k], key)
            r, p = rank_biserial_one_sample(vals)
            raw_p.append(p)
            r_vals.append(r)
            medians.append(np.nanmedian(vals))
            iqrs.append((np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)))
        holm_p = holm_correction(np.array(raw_p))

        print(f"{'k':>4}  {'median':>9}  {'IQR':>20}  {'r':>7}  {'raw p':>9}  {'Holm p':>9}  {'sig':>5}")
        for k, med, iqr, r, p, hp in zip(K_VALUES, medians, iqrs, r_vals, raw_p, holm_p):
            print(f"{k:>4}  {med:>+9.4f}  [{iqr[0]:+.3f},{iqr[1]:+.3f}]  {r:>7.3f}  {p:>9.4f}  {hp:>9.4f}  {sig_stars(hp):>5}")

        ch1_vals = stat_array(ch1_data, key)
        r1, p1 = rank_biserial_one_sample(ch1_vals)
        print(f"{'99(ch.1)':>4}  {np.nanmedian(ch1_vals):>+9.4f}  "
              f"[{np.nanpercentile(ch1_vals,25):+.3f},{np.nanpercentile(ch1_vals,75):+.3f}]  "
              f"{r1:>7.3f}  {p1:>9.4f}  {'(ref, 補正対象外)':>9}")


# --------------------------------------------------------- A2 / B9: two-line test

def build_long_format(data, key):
    """(x=log(k), y=per-seed value) for all 12 k's x 50 seeds -- one row
    per (k, seed), 600 rows total. Does not include chapter-1 k=99."""
    xs, ys, ks = [], [], []
    for k in K_VALUES:
        vals = stat_array(data[k], key)
        for v in vals:
            if np.isnan(v):
                continue
            xs.append(np.log(k))
            ys.append(v)
            ks.append(k)
    return np.array(xs), np.array(ys), np.array(ks)


def two_line_breakpoint(xs, ys, unique_logk):
    """Exhaustive search over the len(unique_logk)-1 possible splits between
    adjacent k-levels; picks the split minimizing total SSR of the two OLS
    fits. Returns dict with breakpoint (midpoint in log k of the two
    k-levels at the split), left/right slope+p, and which k's fall on
    each side."""
    sorted_logk = np.sort(unique_logk)
    best = None
    for i in range(1, len(sorted_logk)):
        split = (sorted_logk[i - 1] + sorted_logk[i]) / 2
        left_mask = xs <= split
        right_mask = ~left_mask
        if left_mask.sum() < 3 or right_mask.sum() < 3:
            continue
        if len(np.unique(xs[left_mask])) < 2 or len(np.unique(xs[right_mask])) < 2:
            continue  # linregress needs >=2 distinct k-levels per side
        lr_left = linregress(xs[left_mask], ys[left_mask])
        lr_right = linregress(xs[right_mask], ys[right_mask])
        ssr_left = np.sum((ys[left_mask] - (lr_left.intercept + lr_left.slope * xs[left_mask])) ** 2)
        ssr_right = np.sum((ys[right_mask] - (lr_right.intercept + lr_right.slope * xs[right_mask])) ** 2)
        total_ssr = ssr_left + ssr_right
        if best is None or total_ssr < best["ssr"]:
            best = dict(
                ssr=total_ssr, split=split,
                slope_left=lr_left.slope, p_left=lr_left.pvalue,
                slope_right=lr_right.slope, p_right=lr_right.pvalue,
                n_left=int(left_mask.sum()), n_right=int(right_mask.sum()),
            )
    return best


def bootstrap_breakpoint(data, key, n_bootstrap=N_BOOTSTRAP):
    boot_splits = []
    unique_logk = np.log(np.array(K_VALUES, dtype=float))
    per_k_vals = {k: stat_array(data[k], key) for k in K_VALUES}
    per_k_vals = {k: v[~np.isnan(v)] for k, v in per_k_vals.items()}
    for _ in range(n_bootstrap):
        xs_b, ys_b = [], []
        for k in K_VALUES:
            vals = per_k_vals[k]
            if len(vals) == 0:
                continue
            idx = RNG.integers(0, len(vals), size=len(vals))
            resampled = vals[idx]
            xs_b.extend([np.log(k)] * len(resampled))
            ys_b.extend(resampled)
        xs_b, ys_b = np.array(xs_b), np.array(ys_b)
        result = two_line_breakpoint(xs_b, ys_b, unique_logk)
        if result is not None:
            boot_splits.append(result["split"])
    boot_splits = np.array(boot_splits)
    return boot_splits


def section_two_line_test(data, label, key):
    print(f"\n--- 二直線検定(breakpoint regression): {label} ---")
    xs, ys, ks = build_long_format(data, key)
    unique_logk = np.log(np.array(K_VALUES, dtype=float))
    result = two_line_breakpoint(xs, ys, unique_logk)
    if result is None:
        print("  データ不足で分割点を決定できませんでした。")
        return None

    boot_splits = bootstrap_breakpoint(data, key)
    ci = (np.percentile(boot_splits, 2.5), np.percentile(boot_splits, 97.5)) if len(boot_splits) else (np.nan, np.nan)

    bp_k = np.exp(result["split"])
    ci_k = (np.exp(ci[0]), np.exp(ci[1]))

    print(f"  breakpoint (log k) = {result['split']:.4f}  (k ≈ {bp_k:.2f})")
    print(f"  95% CI (log k) = [{ci[0]:.4f}, {ci[1]:.4f}]  (k ≈ [{ci_k[0]:.2f}, {ci_k[1]:.2f}])")
    print(f"  left  slope = {result['slope_left']:+.5f}  p = {result['p_left']:.4f}  (n={result['n_left']})")
    print(f"  right slope = {result['slope_right']:+.5f}  p = {result['p_right']:.4f}  (n={result['n_right']})")

    is_inverted_u = (
        result["slope_left"] > 0 and result["p_left"] < 0.05
        and result["slope_right"] < 0 and result["p_right"] < 0.05
    )
    print(f"  逆U字の判定基準(左正かつ有意、右負かつ有意)を満たすか: {is_inverted_u}")
    return result


# --------------------------------------------------------- A3: new vs ch1 k=99

def section_A3(data, ch1_data):
    print("\n" + "=" * 100)
    print("A-3. 新k=99(同一パイプライン) vs 第一章k=99: Mann-Whitney U")
    print("=" * 100)
    for channel, key in [("K_out", "rho_kout_payoff"), ("K_self", "rho_kself_payoff")]:
        new_vals = stat_array(data[99], key)
        ch1_vals = stat_array(ch1_data, key)
        stat, p, r = mannwhitney_effect(new_vals, ch1_vals)
        print(f"  {channel}: new k=99 median={np.nanmedian(new_vals):+.4f}  "
              f"ch.1 k=99 median={np.nanmedian(ch1_vals):+.4f}  U={stat:.1f}  p={p:.4f}  r={r:+.3f}")


# --------------------------------------------------------- A4: figure

def section_A4_figure(data, ch1_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    KOUT_COLOR = "#2a78d6"
    KSELF_COLOR = "#eb6834"
    MUTED_INK = "#8f8f88"
    SIG_COLOR = "#1baf7a"

    all_k_labels = [str(k) for k in K_VALUES] + ["99\n(ch.1)"]
    positions = list(range(len(all_k_labels)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    for ax, channel, key, color in [
        (axes[0], "K_out", "rho_kout_payoff", KOUT_COLOR),
        (axes[1], "K_self", "rho_kself_payoff", KSELF_COLOR),
    ]:
        all_vals = [stat_array(data[k], key) for k in K_VALUES] + [stat_array(ch1_data, key)]
        parts = ax.violinplot(all_vals, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            c = color if i < len(K_VALUES) else MUTED_INK
            pc.set_facecolor(c)
            pc.set_alpha(0.55)
            pc.set_edgecolor(c)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        rng = np.random.default_rng(0)
        for pos, vals in zip(positions, all_vals):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=10,
                       color=MUTED_INK, alpha=0.4, edgecolor="none", zorder=3)

        # Holm correction across the 12 points only (ch.1 excluded)
        raw_p = [rank_biserial_one_sample(stat_array(data[k], key))[1] for k in K_VALUES]
        holm_p = holm_correction(np.array(raw_p))
        ch1_p = rank_biserial_one_sample(stat_array(ch1_data, key))[1]

        y_top = max(np.nanpercentile(v, 97) for v in all_vals)
        y_bottom = min(np.nanpercentile(v, 3) for v in all_vals)
        y_span = y_top - y_bottom
        for pos, hp in zip(positions[:-1], holm_p):
            stars = sig_stars(hp)
            ax.text(pos, y_top + 0.05 * y_span, stars, ha="center", fontsize=9,
                    color=SIG_COLOR if stars not in ("", "n.s.") else MUTED_INK)
        ax.text(positions[-1], y_top + 0.05 * y_span, f"(ref)\np={ch1_p:.3f}", ha="center", fontsize=8, color=MUTED_INK)

        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(all_k_labels)
        ax.set_xlabel("k (opponent count)")
        ax.set_ylabel(f"Spearman ρ({channel}, payoff)\n(per-seed median, n=50)")
        ax.set_title(f"{channel} vs opponent count k (12 points, Holm-corrected; ch.1 k=99 = gray reference)")
        ax.set_ylim(y_bottom - 0.05 * y_span, y_top + 0.22 * y_span)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Precision opponent-count sweep -- FULL 12-point result (n=50 seeds x 5000 gen per k)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "opponent_count_precision_sweep_12point_final.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


# --------------------------------------------------------- B5/B6: gradients table

def section_B5_B6(data, ch1_data):
    print("\n" + "=" * 100)
    print("B-5/6. 偏相関 + Lande-Arnold係数: 全12点(gen>=2000, n=50, Holm補正12点)")
    print("=" * 100)

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

    for key, label in stat_labels.items():
        print(f"\n--- {label} ---")
        raw_p, r_vals, medians = [], [], []
        for k in K_VALUES:
            vals = stat_array(data[k], key)
            r, p = rank_biserial_one_sample(vals)
            raw_p.append(p)
            r_vals.append(r)
            medians.append(np.nanmedian(vals))
        holm_p = holm_correction(np.array(raw_p))
        print(f"{'k':>4}  {'median':>9}  {'r':>7}  {'raw p':>9}  {'Holm p':>9}  {'sig':>5}")
        for k, med, r, p, hp in zip(K_VALUES, medians, r_vals, raw_p, holm_p):
            print(f"{k:>4}  {med:>+9.4f}  {r:>7.3f}  {p:>9.4f}  {hp:>9.4f}  {sig_stars(hp):>5}")
        ch1_vals = stat_array(ch1_data, key)
        r1, p1 = rank_biserial_one_sample(ch1_vals)
        print(f"{'99(ch.1)':>4}  {np.nanmedian(ch1_vals):>+9.4f}  {r1:>7.3f}  {p1:>9.4f}  {'(ref, 補正対象外)':>9}")


# --------------------------------------------------------- B7: gamma eigen

def section_B7(data, ch1_data):
    print("\n" + "=" * 100)
    print("B-7. シード内平均γ行列の固有分解 + ブートストラップ95%CI(全12点+ch.1)")
    print("=" * 100)

    groups = [(str(k), data[k]) for k in K_VALUES] + [(CH1_LABEL, ch1_data)]

    print(f"\n{'k':>8}  {'eig_large med':>13}  {'IQR':>20}  {'eig_small med':>13}  {'IQR':>20}  "
          f"{'angle med':>9}  {'IQR':>18}")

    boot_results = {}
    for label, summaries in groups:
        matrices = np.array([s["gamma_mean"] for s in summaries])
        valid = matrices[~np.isnan(matrices).any(axis=(1, 2))]
        eig_large, eig_small, angle = [], [], []
        for m in valid:
            el, es, a = eig_large_small_angle(m)
            eig_large.append(el)
            eig_small.append(es)
            angle.append(a)
        eig_large, eig_small, angle = np.array(eig_large), np.array(eig_small), np.array(angle)

        iqr_l = f"[{np.percentile(eig_large,25):+.4f},{np.percentile(eig_large,75):+.4f}]"
        iqr_s = f"[{np.percentile(eig_small,25):+.4f},{np.percentile(eig_small,75):+.4f}]"
        iqr_a = f"[{np.percentile(angle,25):.1f},{np.percentile(angle,75):.1f}]"
        print(f"{label:>8}  {np.median(eig_large):>+13.4f}  {iqr_l:>20}  {np.median(eig_small):>+13.4f}  "
              f"{iqr_s:>20}  {np.median(angle):>9.1f}  {iqr_a:>18}")

        n_seeds = len(valid)
        boot_large, boot_small = [], []
        for _ in range(N_BOOTSTRAP):
            idx = RNG.integers(0, n_seeds, size=n_seeds)
            resampled_mean = valid[idx].mean(axis=0)
            el, es, _ = eig_large_small_angle(resampled_mean)
            boot_large.append(el)
            boot_small.append(es)
        boot_large, boot_small = np.array(boot_large), np.array(boot_small)
        grand_mean_matrix = valid.mean(axis=0)
        grand_large, grand_small, _ = eig_large_small_angle(grand_mean_matrix)
        boot_results[label] = dict(
            grand_large=grand_large, grand_small=grand_small,
            ci_large=(np.percentile(boot_large, 2.5), np.percentile(boot_large, 97.5)),
            ci_small=(np.percentile(boot_small, 2.5), np.percentile(boot_small, 97.5)),
        )

    print(f"\n--- ブートストラップ95%CI(シード単位、n={N_BOOTSTRAP}回、全シード平均γ行列の固有値) ---")
    print(f"{'k':>8}  {'eig_large (grand)':>18}  {'95% CI':>24}  {'ゼロをまたぐか':>14}  |  "
          f"{'eig_small (grand)':>18}  {'95% CI':>24}  {'ゼロをまたぐか':>14}")
    for label, r in boot_results.items():
        straddles_l = "またぐ" if r["ci_large"][0] < 0 < r["ci_large"][1] else "またがない"
        straddles_s = "またぐ" if r["ci_small"][0] < 0 < r["ci_small"][1] else "またがない"
        ci_l = f"[{r['ci_large'][0]:+.4f},{r['ci_large'][1]:+.4f}]"
        ci_s = f"[{r['ci_small'][0]:+.4f},{r['ci_small'][1]:+.4f}]"
        print(f"{label:>8}  {r['grand_large']:>+18.4f}  {ci_l:>24}  {straddles_l:>14}  |  "
              f"{r['grand_small']:>+18.4f}  {ci_s:>24}  {straddles_s:>14}")


# --------------------------------------------------------- B8: fitness peak

def fitness_peak_for_group(summaries):
    z1_list, z2_list, norm_list = [], [], []
    valid_betas, valid_gammas = [], []
    n_excluded, n_no_data = 0, 0
    for s in summaries:
        beta, gamma = s["beta_mean"], s["gamma_mean"]
        if np.isnan(beta).any():
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
        valid_betas.append(beta)
        valid_gammas.append(gamma)
    return dict(
        z1=np.array(z1_list), z2=np.array(z2_list), norm=np.array(norm_list),
        n_excluded=n_excluded, n_no_data=n_no_data, n_total=len(summaries),
        valid_betas=np.array(valid_betas), valid_gammas=np.array(valid_gammas),
    )


def bootstrap_grand_zstar(valid_betas, valid_gammas, n_bootstrap=N_BOOTSTRAP):
    n = len(valid_betas)
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


def section_B8(data, ch1_data):
    print("\n" + "=" * 100)
    print("B-8. 適応度頂上 z* = -Γ^-1β(全12点+ch.1)")
    print("=" * 100)

    groups = [(str(k), data[k]) for k in K_VALUES] + [(CH1_LABEL, ch1_data)]
    results = {}
    print(f"\n{'k':>8}  {'n_excl':>7}  {'n_valid':>7}  {'z*_K_self med':>14}  {'IQR':>18}  {'p':>8}  |  "
          f"{'z*_K_out med':>13}  {'IQR':>18}  {'p':>8}")
    for label, summaries in groups:
        r = fitness_peak_for_group(summaries)
        results[label] = r
        _, p1 = rank_biserial_one_sample(r["z1"])
        _, p2 = rank_biserial_one_sample(r["z2"])
        iqr1 = f"[{np.percentile(r['z1'],25):+.3f},{np.percentile(r['z1'],75):+.3f}]" if len(r["z1"]) else "-"
        iqr2 = f"[{np.percentile(r['z2'],25):+.3f},{np.percentile(r['z2'],75):+.3f}]" if len(r["z2"]) else "-"
        print(f"{label:>8}  {r['n_excluded']:>7}  {len(r['z1']):>7}  "
              f"{np.median(r['z1']):>+14.3f}  {iqr1:>18}  {p1:>8.4f}  |  "
              f"{np.median(r['z2']):>+13.3f}  {iqr2:>18}  {p2:>8.4f}")

    print(f"\n--- 全シード平均版z*のブートストラップ95%CI(n={N_BOOTSTRAP}回) ---")
    print(f"{'k':>8}  {'z*_K_self (grand)':>18}  {'95% CI':>22}  {'z*_K_out (grand)':>17}  {'95% CI':>22}  {'skipped':>8}")
    for label, r in results.items():
        if len(r["valid_betas"]) == 0:
            print(f"{label:>8}  (有効シードなし)")
            continue
        grand_beta = r["valid_betas"].mean(axis=0)
        grand_gamma = r["valid_gammas"].mean(axis=0)
        grand_z = -np.linalg.solve(grand_gamma, grand_beta)
        boot_z1, boot_z2, n_skipped = bootstrap_grand_zstar(r["valid_betas"], r["valid_gammas"])
        ci1 = (np.percentile(boot_z1, 2.5), np.percentile(boot_z1, 97.5)) if len(boot_z1) else (np.nan, np.nan)
        ci2 = (np.percentile(boot_z2, 2.5), np.percentile(boot_z2, 97.5)) if len(boot_z2) else (np.nan, np.nan)
        print(f"{label:>8}  {grand_z[0]:>+18.3f}  [{ci1[0]:+.3f},{ci1[1]:+.3f}]"
              f"{'':>8}  {grand_z[1]:>+17.3f}  [{ci2[0]:+.3f},{ci2[1]:+.3f}]{'':>8}  {n_skipped:>8}")


# --------------------------------------------------------- B9: beta2 vs log k

def section_B9(data, ch1_data):
    print("\n" + "=" * 100)
    print("B-9. β2のk依存性(per-seed β2 を log k に回帰、12点×50シード長形式)")
    print("=" * 100)

    xs, ys, _ = build_long_format(data, "beta2")
    lr = linregress(xs, ys)
    print(f"\n単調傾向(全600点の単純線形回帰): slope={lr.slope:+.5f}  p={lr.pvalue:.4f}  r={lr.rvalue:+.3f}")

    ch1_vals = stat_array(ch1_data, "beta2")
    print(f"(参考: 第一章k=99のβ2中央値={np.nanmedian(ch1_vals):+.4f}、この回帰には含めていない)")

    section_two_line_test(data, "β2 (linear K_out selection gradient)", "beta2")


# --------------------------------------------------------- main

def main():
    print(f"loading individual-level data for k={K_VALUES} (gen>={STEADY_STATE_MIN_GEN}, n=50 seeds each)...")
    data = {k: load_all(SCAN_ROOT / f"k{k}", SEEDS, STEADY_STATE_MIN_GEN) for k in K_VALUES}
    ch1_data = load_all(CH1_DIR, CH1_SEEDS, STEADY_STATE_MIN_GEN)
    print("done.\n")

    section_A1(data, ch1_data)
    section_two_line_test(data, "K_out marginal correlation", "rho_kout_payoff")
    section_A3(data, ch1_data)
    section_A4_figure(data, ch1_data)

    section_B5_B6(data, ch1_data)
    section_B7(data, ch1_data)
    section_B8(data, ch1_data)
    section_B9(data, ch1_data)

    print("\n解釈はせず、ここで停止します。")


if __name__ == "__main__":
    main()
