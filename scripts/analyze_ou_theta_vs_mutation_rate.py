"""Does point_mutation_rate move the OU mean-reversion rate theta --
the fingerprint that replaced the retired ~77-gen period
(analyze_ou_fit_diagnostic.py)?

Reuses the already-completed point-mutation screening run (3 levels x
20 seeds x 3000 gen, scripts/output/point_mutation_sweep_screening/) --
no new simulation. For each (level, seed, channel), fits the same
discrete OU/AR(1) model over window gen[1000,3000):

    x(t+1) - x(t) = -theta*(x(t) - mu) + sigma*eps(t)

via OLS regression of x(t+1) on x(t) (phi = 1-theta).

If Spearman(point_mutation_rate, theta) shows a real trend, the
point-mutation sweep has a new target -- not a period, but how tightly
the process is confined. If theta doesn't move (matching the earlier
null result for beta and DFA alpha in
analyze_hurst_vs_mutation_rate.py), the point-mutation exploration is
done: three independent detrend-free estimators (beta, DFA alpha,
theta) would all have failed to find a point_mutation_rate effect.

Outputs:
    scripts/output/ou_theta_vs_mutation_rate/ou_theta_vs_mutation_rate.csv
    scripts/output/ou_theta_vs_mutation_rate/theta_vs_mutation_rate.png

Usage:
    python3 scripts/analyze_ou_theta_vs_mutation_rate.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/point_mutation_sweep_screening")
OUT_DIR = Path("scripts/output/ou_theta_vs_mutation_rate")
WINDOW_RANGE = (1000, 3000)

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"


def level_dirname(rate: float) -> str:
    return f"pmr_{rate:.3g}"


def load_manifest(sweep_dir: Path) -> dict:
    with open(sweep_dir / "manifest.json") as f:
        return json.load(f)


def load_cell(sweep_dir: Path, rate: float, seed: int):
    path = sweep_dir / level_dirname(rate) / f"seed_{seed}.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def window_mask(gens, lo, hi):
    return (gens >= lo) & (gens < hi)


def fit_ou(x):
    x_t = x[:-1]
    x_tp1 = x[1:]
    phi, c = np.polyfit(x_t, x_tp1, 1)
    theta = 1.0 - phi
    mu = c / theta if theta != 0 else float("nan")
    resid = x_tp1 - (phi * x_t + c)
    sigma = resid.std(ddof=2)
    return theta, mu, sigma


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(SWEEP_DIR)
    levels = manifest["point_mutation_rate_levels"]
    seeds = manifest["seeds"]

    print(f"sweep dir: {SWEEP_DIR}")
    print(f"levels: {levels}, seeds: {len(seeds)}, window: gen{WINDOW_RANGE}")

    rows = []
    for rate in levels:
        for seed in seeds:
            gens, kout, kself = load_cell(SWEEP_DIR, rate, seed)
            mask = window_mask(gens, *WINDOW_RANGE)
            for channel_name, series in [("K_out", kout[mask]), ("K_self", kself[mask])]:
                theta, mu, sigma = fit_ou(series)
                rows.append({"point_mutation_rate": rate, "seed": seed, "channel": channel_name,
                             "theta": theta, "mu": mu, "sigma": sigma})

    csv_path = OUT_DIR / "ou_theta_vs_mutation_rate.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point_mutation_rate", "seed", "channel", "theta", "mu", "sigma"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {csv_path} ({len(rows)} rows)")

    from scipy.stats import kruskal, spearmanr

    agg = {}
    print(f"\n=== theta by point_mutation_rate (window gen{WINDOW_RANGE}) ===")
    for rate in levels:
        for channel_name in ["K_out", "K_self"]:
            sub = [r for r in rows if r["point_mutation_rate"] == rate and r["channel"] == channel_name]
            thetas = np.array([r["theta"] for r in sub])
            agg[(rate, channel_name)] = thetas
            q1, med, q3 = np.percentile(thetas, [25, 50, 75])
            print(f"pmr={rate:<5} {channel_name:<7} theta median={med:.5f} IQR=[{q1:.5f},{q3:.5f}]")

    print("\n=== Trend tests: point_mutation_rate vs theta ===")
    trend_results = {}
    for channel_name in ["K_out", "K_self"]:
        rates_arr = np.array([r["point_mutation_rate"] for r in rows if r["channel"] == channel_name])
        thetas_arr = np.array([r["theta"] for r in rows if r["channel"] == channel_name])
        rho, p_spearman = spearmanr(rates_arr, thetas_arr)
        groups = [agg[(rate, channel_name)] for rate in levels]
        h_stat, p_kw = kruskal(*groups)
        trend_results[channel_name] = {"spearman_rho": rho, "spearman_p": p_spearman, "kruskal_h": h_stat, "kruskal_p": p_kw}
        print(f"{channel_name}: Spearman(point_mutation_rate, theta) rho={rho:.3f} p={p_spearman:.4f}   "
              f"Kruskal-Wallis H={h_stat:.3f} p={p_kw:.4f}")
        for rate in levels:
            v = agg[(rate, channel_name)]
            print(f"    pmr={rate}: n={len(v)} median={np.median(v):.5f} mean={v.mean():.5f} std={v.std():.5f}")

    with open(OUT_DIR / "trend_tests.json", "w") as f:
        json.dump(trend_results, f, indent=2)
    print(f"\nsaved {OUT_DIR / 'trend_tests.json'}")

    # --- plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for channel_name, color, marker in [("K_out", KOUT_COLOR, "o"), ("K_self", KSELF_COLOR, "s")]:
        medians = [np.median(agg[(rate, channel_name)]) for rate in levels]
        q1s = [np.percentile(agg[(rate, channel_name)], 25) for rate in levels]
        q3s = [np.percentile(agg[(rate, channel_name)], 75) for rate in levels]
        lo_err = [m - q for m, q in zip(medians, q1s)]
        hi_err = [q - m for q, m in zip(q3s, medians)]
        tr = trend_results[channel_name]
        ax.errorbar(levels, medians, yerr=[lo_err, hi_err], marker=marker, color=color, markersize=8,
                    linewidth=1.8, capsize=4,
                    label=f"{channel_name} (Spearman ρ={tr['spearman_rho']:.2f}, p={tr['spearman_p']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([str(r) for r in levels])
    ax.set_xlabel("point_mutation_rate")
    ax.set_ylabel("θ (OU mean-reversion rate), median ± IQR")
    ax.set_title(f"Does mutation rate change confinement strength θ? (window gen{WINDOW_RANGE}, n={len(seeds)} seeds/level)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = OUT_DIR / "theta_vs_mutation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
