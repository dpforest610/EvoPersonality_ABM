"""Re-asks the point-mutation-rate question in terms of "how red" the
spectrum is, instead of chasing a peak period that turned out to track
the detrend window rather than the system
(analyze_detrend_window_diagnostic.py).

Reuses the already-completed point-mutation screening run (3 levels x
20 seeds x 3000 gen, scripts/output/point_mutation_sweep_screening/) --
no new simulation. For each (level, seed, channel), computes the same
two detrend-free "color" estimators as analyze_spectral_exponent_hurst.py
over the window gen[1000,3000):

  beta      - spectral slope from the raw (undetrended) Welch PSD,
              period band [20,300] gen: S(f) ~ f^-beta.
  dfa_alpha - DFA scaling exponent (detrends locally at each scale by
              construction, so it doesn't beg the same question as a
              single global detrend would).

If point_mutation_rate shifts beta/alpha, the point-mutation sweep
still has a real target -- just "how red" the process is, not a period.
If it doesn't, mutation rate doesn't touch this axis either.

Outputs:
    scripts/output/hurst_vs_mutation_rate/hurst_vs_mutation_rate.csv
    scripts/output/hurst_vs_mutation_rate/beta_and_alpha_vs_mutation_rate.png

Usage:
    python3 scripts/analyze_hurst_vs_mutation_rate.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/point_mutation_sweep_screening")
OUT_DIR = Path("scripts/output/hurst_vs_mutation_rate")
WINDOW_RANGE = (1000, 3000)
NPERSEG = 512
PERIOD_MIN, PERIOD_MAX = 20.0, 300.0

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


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


def welch_psd_raw(x, nperseg):
    from scipy.signal import welch

    nperseg = min(nperseg, len(x))
    freqs, power = welch(x, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    return freqs, power


def spectral_slope_beta(freqs, power, period_min, period_max):
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= period_min) & (periods <= period_max) & (freqs > 0) & (power > 0)
    log_f = np.log10(freqs[mask])
    log_p = np.log10(power[mask])
    slope, _ = np.polyfit(log_f, log_p, 1)
    return -slope


def dfa_alpha(x, min_box=10, max_box_frac=4):
    n = len(x)
    profile = np.cumsum(x - x.mean())
    max_box = n // max_box_frac
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    box_sizes = box_sizes[box_sizes >= 4]

    log_s, log_f = [], []
    for s in box_sizes:
        n_seg = n // s
        if n_seg < 2:
            continue
        t = np.arange(s)
        variances = []
        for v in range(n_seg):
            seg = profile[v * s : (v + 1) * s]
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            variances.append(np.mean((seg - trend) ** 2))
        for v in range(n_seg):
            seg = profile[n - (v + 1) * s : n - v * s]
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            variances.append(np.mean((seg - trend) ** 2))
        f_s = np.sqrt(np.mean(variances))
        if f_s > 0:
            log_s.append(np.log10(s))
            log_f.append(np.log10(f_s))

    slope, _ = np.polyfit(log_s, log_f, 1)
    return slope


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
                freqs, power = welch_psd_raw(series, NPERSEG)
                beta = spectral_slope_beta(freqs, power, PERIOD_MIN, PERIOD_MAX)
                alpha = dfa_alpha(series)
                rows.append(
                    {"point_mutation_rate": rate, "seed": seed, "channel": channel_name, "beta": beta, "dfa_alpha": alpha}
                )

    csv_path = OUT_DIR / "hurst_vs_mutation_rate.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point_mutation_rate", "seed", "channel", "beta", "dfa_alpha"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {csv_path} ({len(rows)} rows)")

    from scipy.stats import kruskal, spearmanr

    agg = {}
    print(f"\n=== beta / dfa_alpha by point_mutation_rate (window gen{WINDOW_RANGE}) ===")
    for rate in levels:
        for channel_name in ["K_out", "K_self"]:
            sub = [r for r in rows if r["point_mutation_rate"] == rate and r["channel"] == channel_name]
            betas = np.array([r["beta"] for r in sub])
            alphas = np.array([r["dfa_alpha"] for r in sub])
            agg[(rate, channel_name)] = {"betas": betas, "alphas": alphas}
            bq1, bmed, bq3 = np.percentile(betas, [25, 50, 75])
            aq1, amed, aq3 = np.percentile(alphas, [25, 50, 75])
            print(f"pmr={rate:<5} {channel_name:<7} beta median={bmed:6.3f} IQR=[{bq1:6.3f},{bq3:6.3f}]   "
                  f"alpha median={amed:6.3f} IQR=[{aq1:6.3f},{aq3:6.3f}]")

    print("\n=== Trend tests: point_mutation_rate vs beta / dfa_alpha ===")
    trend_results = {}
    for metric in ["beta", "dfa_alpha"]:
        for channel_name in ["K_out", "K_self"]:
            rates_arr = np.array([r["point_mutation_rate"] for r in rows if r["channel"] == channel_name])
            vals_arr = np.array([r[metric] for r in rows if r["channel"] == channel_name])
            rho, p_spearman = spearmanr(rates_arr, vals_arr)
            groups = [agg[(rate, channel_name)]["betas" if metric == "beta" else "alphas"] for rate in levels]
            h_stat, p_kw = kruskal(*groups)
            trend_results[f"{metric}_{channel_name}"] = {
                "spearman_rho": rho, "spearman_p": p_spearman, "kruskal_h": h_stat, "kruskal_p": p_kw,
            }
            print(f"{metric:<10} {channel_name:<7} Spearman rho={rho:6.3f} p={p_spearman:.4f}   "
                  f"Kruskal-Wallis H={h_stat:6.3f} p={p_kw:.4f}")

    with open(OUT_DIR / "trend_tests.json", "w") as f:
        json.dump(trend_results, f, indent=2)
    print(f"\nsaved {OUT_DIR / 'trend_tests.json'}")

    # --- plot: beta and alpha vs mutation rate, median +- IQR, per channel ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_beta, ax_alpha) = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, metric, key, ylabel in [
        (ax_beta, "beta", "betas", "spectral slope β"),
        (ax_alpha, "dfa_alpha", "alphas", "DFA exponent α"),
    ]:
        for channel_name, color, marker in [("K_out", KOUT_COLOR, "o"), ("K_self", KSELF_COLOR, "s")]:
            medians = [np.median(agg[(rate, channel_name)][key]) for rate in levels]
            q1s = [np.percentile(agg[(rate, channel_name)][key], 25) for rate in levels]
            q3s = [np.percentile(agg[(rate, channel_name)][key], 75) for rate in levels]
            lo_err = [m - q for m, q in zip(medians, q1s)]
            hi_err = [q - m for q, m in zip(q3s, medians)]
            tr = trend_results[f"{metric}_{channel_name}"]
            ax.errorbar(levels, medians, yerr=[lo_err, hi_err], marker=marker, color=color, markersize=8,
                        linewidth=1.8, capsize=4,
                        label=f"{channel_name} (Spearman ρ={tr['spearman_rho']:.2f}, p={tr['spearman_p']:.3f})")
        ax.set_xscale("log")
        ax.set_xticks(levels)
        ax.set_xticklabels([str(r) for r in levels])
        ax.set_xlabel("point_mutation_rate")
        ax.set_ylabel(f"{ylabel}, median ± IQR")
        ax.set_title(f"{ylabel} vs. point_mutation_rate")
        ax.legend(fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Does mutation rate change how 'red' the spectrum is? (window gen{WINDOW_RANGE}, n={len(seeds)} seeds/level)", y=1.03)
    fig.tight_layout()
    fig_path = OUT_DIR / "beta_and_alpha_vs_mutation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
