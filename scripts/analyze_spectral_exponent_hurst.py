"""Name the "color" of the K_out/K_self spectrum, without hunting for a
peak and without depending on any detrend-window choice.

The detrend-window diagnostic (analyze_detrend_window_diagnostic.py)
showed the ~77-gen PSD "peak" tracks whatever moving-average detrend
window is used, and the raw spectrum just rises monotonically with
period -- i.e. some flavor of colored noise, not a quasi-oscillation.
This script names which flavor, using two independent, detrend-free
estimators, plus a stationarity test:

1. Spectral slope beta: Welch PSD of the RAW series (no moving-average
   detrend -- only Welch's own per-segment constant/mean removal, which
   is unavoidable bookkeeping, not the trend-removal artifact being
   avoided here), log-log linear fit of power vs. frequency restricted
   to the period band [period-min, period-max] (default 20-300 gen).
   S(f) ~ f^-beta. beta~0 = white noise, beta~1 = 1/f (pink), beta~2 =
   Brownian motion / random walk.

2. DFA scaling exponent alpha (Detrended Fluctuation Analysis, Peng et
   al. 1994): integrates the (mean-centered) series into a profile,
   measures RMS fluctuation of locally-detrended segments across a
   range of segment sizes, and fits the log-log scaling. DFA already
   detrends locally at each scale by construction, so it works
   correctly for a non-stationary series where a single global
   detrend-then-PSD would beg the question. alpha~0.5 = uncorrelated
   (white), alpha~1 = 1/f boundary, alpha~1.5 = Brownian motion /
   random walk. The two estimators should roughly agree via
   beta = 2*alpha - 1 if both are seeing the same "color"; this script
   reports both independently so any disagreement is visible rather
   than papered over.

3. Augmented Dickey-Fuller test (statsmodels) on the same raw window:
   tests the unit-root null. Failing to reject supports "non-stationary
   drift", not "stationary quasi-oscillation".

All three run on the existing 50-seed, 5000-generation confirmatory
coupling sweep (scripts/output/coupling_seed_sweep_5000gen/), window
gen[2000,5000) -- no new simulation, no moving-average detrend.

Outputs:
    scripts/output/spectral_exponent_hurst/spectral_exponent_hurst.csv
    scripts/output/spectral_exponent_hurst/beta_and_dfa_distributions.png

Usage:
    python3 scripts/analyze_spectral_exponent_hurst.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
OUT_DIR = Path("scripts/output/spectral_exponent_hurst")
SEEDS = list(range(50))
WINDOW_RANGE = (2000, 5000)
NPERSEG = 512
PERIOD_MIN, PERIOD_MAX = 20.0, 300.0
ADF_ALPHA = 0.05

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def load_full_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
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
    """No moving-average detrend. Welch's own 'constant' detrend just
    removes each segment's mean -- required bookkeeping, not the
    trend-removal artifact under investigation."""
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
    slope, intercept = np.polyfit(log_f, log_p, 1)
    residuals = log_p - (slope * log_f + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -slope, r_squared


def dfa_alpha(x, min_box=10, max_box_frac=4):
    """Detrended Fluctuation Analysis. Returns the scaling exponent
    alpha from a log-log fit of RMS fluctuation vs. segment size,
    using linearly-detrended, non-overlapping segments read from both
    ends of the profile (standard DFA practice for using all samples)."""
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


def adf_test(x):
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(x, autolag="AIC", result_object=False)
    stat, pvalue = result[0], result[1]
    return stat, pvalue


def interpret(value, anchors):
    """anchors: list of (value, label) sorted ascending; picks nearest label for display."""
    closest = min(anchors, key=lambda a: abs(a[0] - value))
    return closest[1]


BETA_ANCHORS = [(0.0, "white"), (1.0, "1/f (pink)"), (2.0, "Brownian/random-walk")]
ALPHA_ANCHORS = [(0.5, "white/uncorrelated"), (1.0, "1/f boundary"), (1.5, "Brownian/random-walk")]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        gens, kout, kself = load_full_series(seed)
        mask = window_mask(gens, *WINDOW_RANGE)
        for channel_name, series in [("K_out", kout[mask]), ("K_self", kself[mask])]:
            freqs, power = welch_psd_raw(series, NPERSEG)
            beta, r2 = spectral_slope_beta(freqs, power, PERIOD_MIN, PERIOD_MAX)
            alpha = dfa_alpha(series)
            adf_stat, adf_p = adf_test(series)
            rows.append(
                {
                    "seed": seed,
                    "channel": channel_name,
                    "beta": beta,
                    "beta_r2": r2,
                    "dfa_alpha": alpha,
                    "beta_from_alpha": 2 * alpha - 1,
                    "adf_stat": adf_stat,
                    "adf_pvalue": adf_p,
                    "reject_unit_root_at_0.05": adf_p < ADF_ALPHA,
                }
            )

    csv_path = OUT_DIR / "spectral_exponent_hurst.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    print(f"\n=== Spectral slope beta and DFA alpha, window gen{WINDOW_RANGE}, n={len(SEEDS)} seeds ===")
    print("(beta: S(f)~f^-beta, from raw undetrended PSD, period band "
          f"[{PERIOD_MIN:.0f},{PERIOD_MAX:.0f}] gen. alpha: DFA scaling exponent.)")
    for channel_name in ["K_out", "K_self"]:
        sub = [r for r in rows if r["channel"] == channel_name]
        betas = np.array([r["beta"] for r in sub])
        alphas = np.array([r["dfa_alpha"] for r in sub])
        b_q1, b_med, b_q3 = np.percentile(betas, [25, 50, 75])
        a_q1, a_med, a_q3 = np.percentile(alphas, [25, 50, 75])
        print(f"\n{channel_name}:")
        print(f"  beta:  median={b_med:.3f}  IQR=[{b_q1:.3f},{b_q3:.3f}]  -> nearest anchor: {interpret(b_med, BETA_ANCHORS)}")
        print(f"  alpha: median={a_med:.3f}  IQR=[{a_q1:.3f},{a_q3:.3f}]  -> nearest anchor: {interpret(a_med, ALPHA_ANCHORS)}")
        print(f"  cross-check: 2*alpha-1 median = {2*a_med-1:.3f} vs. beta median = {b_med:.3f} "
              f"({'consistent' if abs((2*a_med-1) - b_med) < 0.4 else 'DISAGREE'})")

        n_reject = sum(r["reject_unit_root_at_0.05"] for r in sub)
        print(f"  ADF unit-root test: {n_reject}/{len(sub)} seeds reject unit root at alpha={ADF_ALPHA} "
              f"-> {'mostly stationary' if n_reject > len(sub)/2 else 'mostly NON-stationary (unit root not rejected)'}")

    # --- plot: seed distributions for beta and alpha ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, (ax_beta, ax_alpha) = plt.subplots(1, 2, figsize=(11, 6))
    for ax, key, anchors, title in [
        (ax_beta, "beta", BETA_ANCHORS, "Spectral slope β (S(f) ~ f^-β)"),
        (ax_alpha, "dfa_alpha", ALPHA_ANCHORS, "DFA scaling exponent α"),
    ]:
        for i, (channel_name, color) in enumerate([("K_out", KOUT_COLOR), ("K_self", KSELF_COLOR)]):
            vals = np.array([r[key] for r in rows if r["channel"] == channel_name])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color, alpha=0.5, edgecolor="none")
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.errorbar([i], [med], yerr=[[med - q1], [q3 - med]], marker="o", color=color,
                        markersize=10, linewidth=2.2, capsize=5, zorder=5)
        for val, label in anchors:
            ax.axhline(val, color=MUTED_INK, linewidth=0.8, linestyle="--")
            ax.annotate(label, (1.55, val), fontsize=7.5, color=MUTED_INK, va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["K_out", "K_self"])
        ax.set_xlim(-0.5, 2.3)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    ax_beta.set_ylabel("value (dots = seeds, marker = median ± IQR)")
    fig.suptitle(f"What color is the raw K_out/K_self spectrum? (window gen{WINDOW_RANGE}, n={len(SEEDS)} seeds, no detrend)", y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "beta_and_dfa_distributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
