"""Check whether the 5-seed coupling sweep's three phase-structure
findings hold up across all 50 seeds
(scripts/output/coupling_seed_sweep_5000gen/seed_<seed>_timeseries.csv):

  (1) cross-correlation: period ~70-90 generations, negative trough
      (analyze_kout_kself_ccf.py)
  (2) PSD: a broad hump around ~80 generations, not a sharp spike
      (analyze_kout_kself_psd.py)
  (3) detrended mean radius from the origin ~0.07, common across seeds
      (plot_kout_kself_phase_plane.py / the sweep's own checkpoint stat)

Pure re-analysis of already-saved data -- no new simulation. With 50
seeds, individual per-seed lines would be unreadable overlaid with
distinct hues (that's a job for identity-coloring, and identity isn't
the point here), so each figure instead shows all 50 as a thin muted
"cloud" behind a bold mean line -- color used for its actual job
(spread vs. central tendency), not as a 50-way categorical legend.

Usage:
    python3 scripts/analyze_50seed_confirmation.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
OUT_DIR = Path("scripts/output/50seed_confirmation")
SEEDS = list(range(50))
DETREND_WINDOW = 75
MAX_LAG = 500
NPERSEG = 1024
LANDMARK_PERIOD = 80

CLOUD_COLOR = "#8f8f88"  # muted ink: the role is "spread", not identity
MEAN_COLOR = "#2a78d6"  # categorical slot 1: the role is "central tendency"
SELF_COLOR = "#eb6834"  # categorical slot 2, only used to pair with mean_color for K_self
MUTED_INK = "#8f8f88"


def load_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def detrend(x, window):
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    return x[offset : offset + len(trend)] - trend


def cross_correlation(x_ref, y_shift, max_lag):
    n = len(x_ref)
    lags = np.arange(-max_lag, max_lag + 1)
    ccf = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            a = x_ref[: n - lag] if lag > 0 else x_ref
            b = y_shift[lag:]
        else:
            a = x_ref[-lag:]
            b = y_shift[: n + lag]
        if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
            continue
        ccf[i] = np.corrcoef(a, b)[0, 1]
    return lags, ccf


def trough_stats(lags, ccf, min_distance=50):
    from scipy.signal import find_peaks

    valid = ~np.isnan(ccf)
    global_idx = np.nanargmin(ccf)
    trough_lag = int(lags[global_idx])
    trough_value = float(ccf[global_idx])

    peak_idx, _ = find_peaks(-np.where(valid, ccf, 0.0), distance=min_distance, prominence=0.03)
    trough_lags = np.sort(lags[peak_idx])
    period = float(np.median(np.diff(trough_lags))) if len(trough_lags) >= 2 else float("nan")
    return trough_lag, trough_value, period


def welch_psd(x, nperseg):
    from scipy.signal import welch

    freqs, power = welch(x, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    return freqs, power


def psd_peak_stats(freqs, power, period_min=10.0, period_max=1000.0):
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    in_band = (periods >= period_min) & (periods <= period_max)
    band_idx = np.where(in_band)[0]
    peak_local = np.argmax(power[band_idx])
    peak_idx = band_idx[peak_local]
    peak_period = periods[peak_idx]
    peak_power = power[peak_idx]
    half_power = peak_power / 2.0
    lo_bound, hi_bound = band_idx[0], band_idx[-1]

    i = peak_idx
    while i > lo_bound and power[i] >= half_power:
        i -= 1
    freq_low = np.interp(half_power, [power[i], power[i + 1]], [freqs[i], freqs[i + 1]]) if i < peak_idx else freqs[lo_bound]

    j = peak_idx
    while j < hi_bound and power[j] >= half_power:
        j += 1
    freq_high = np.interp(half_power, [power[j], power[j - 1]], [freqs[j], freqs[j - 1]]) if j > peak_idx else freqs[hi_bound]

    fwhm = freq_high - freq_low
    q_factor = freqs[peak_idx] / fwhm if fwhm > 0 else np.inf
    return peak_period, q_factor


def detrended_mean_radius(kout, kself, window):
    kout_dt = detrend(kout, window)
    kself_dt = detrend(kself, window)
    radius = np.sqrt(kout_dt**2 + kself_dt**2)
    return float(radius.mean())


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {path}")


def summarize(values, label):
    values = np.array(values, dtype=float)
    valid = values[~np.isnan(values)]
    print(
        f"{label}: n={len(valid)}/{len(values)} valid, "
        f"mean={valid.mean():.3f} median={np.median(valid):.3f} "
        f"std={valid.std():.3f} min={valid.min():.3f} max={valid.max():.3f}"
    )
    return valid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ccf_raw_list, ccf_dt_list = [], []
    psd_kout_list, psd_kself_list = [], []
    ccf_rows, psd_rows, radius_rows = [], [], []
    radii = []

    for seed in SEEDS:
        gens, kout, kself = load_series(seed)
        kout_dt = detrend(kout, DETREND_WINDOW)
        kself_dt = detrend(kself, DETREND_WINDOW)

        lags_raw, ccf_raw = cross_correlation(kself, kout, MAX_LAG)
        lags_dt, ccf_dt = cross_correlation(kself_dt, kout_dt, MAX_LAG)
        ccf_raw_list.append(ccf_raw)
        ccf_dt_list.append(ccf_dt)

        raw_lag, raw_val, raw_period = trough_stats(lags_raw, ccf_raw)
        dt_lag, dt_val, dt_period = trough_stats(lags_dt, ccf_dt)
        ccf_rows.append([seed, "raw", raw_lag, f"{raw_val:.4f}", raw_period])
        ccf_rows.append([seed, "detrended", dt_lag, f"{dt_val:.4f}", dt_period])

        freqs_out, power_out = welch_psd(kout_dt, NPERSEG)
        freqs_self, power_self = welch_psd(kself_dt, NPERSEG)
        psd_kout_list.append((freqs_out, power_out))
        psd_kself_list.append((freqs_self, power_self))

        period_out, q_out = psd_peak_stats(freqs_out, power_out)
        period_self, q_self = psd_peak_stats(freqs_self, power_self)
        psd_rows.append([seed, "K_out", f"{period_out:.1f}", f"{q_out:.2f}"])
        psd_rows.append([seed, "K_self", f"{period_self:.1f}", f"{q_self:.2f}"])

        r = detrended_mean_radius(kout, kself, DETREND_WINDOW)
        radii.append(r)
        radius_rows.append([seed, f"{r:.4f}"])

    # --- (1) CCF summary ---
    print("\n=== (1) Cross-correlation: period ~70-90gen, negative trough ===")
    dt_lags = [r[2] for r in ccf_rows if r[1] == "detrended"]
    dt_vals = [float(r[3]) for r in ccf_rows if r[1] == "detrended"]
    dt_periods = [r[4] for r in ccf_rows if r[1] == "detrended"]
    summarize(dt_lags, "detrended trough lag")
    summarize(dt_vals, "detrended trough value")
    summarize(dt_periods, "detrended estimated period")
    n_negative = sum(1 for v in dt_vals if v < 0)
    print(f"negative-trough fraction: {n_negative}/{len(dt_vals)} = {n_negative/len(dt_vals):.0%}")
    write_csv(OUT_DIR / "ccf_per_seed.csv", ["seed", "preprocessing", "trough_lag", "trough_value", "estimated_period"], ccf_rows)

    # --- (2) PSD summary ---
    print("\n=== (2) PSD: broad hump ~80gen (Q~1), not a sharp spike ===")
    kout_periods = [float(r[2]) for r in psd_rows if r[1] == "K_out"]
    kout_qs = [float(r[3]) for r in psd_rows if r[1] == "K_out"]
    kself_periods = [float(r[2]) for r in psd_rows if r[1] == "K_self"]
    kself_qs = [float(r[3]) for r in psd_rows if r[1] == "K_self"]
    summarize(kout_periods, "K_out peak period")
    summarize(kout_qs, "K_out Q-factor")
    summarize(kself_periods, "K_self peak period")
    summarize(kself_qs, "K_self Q-factor")
    write_csv(OUT_DIR / "psd_per_seed.csv", ["seed", "channel", "peak_period", "Q_factor"], psd_rows)

    # --- (3) amplitude summary ---
    print("\n=== (3) Detrended mean radius from origin ~0.07, common across seeds ===")
    summarize(radii, "mean radius")
    write_csv(OUT_DIR / "radius_per_seed.csv", ["seed", "mean_radius"], radius_rows)

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) CCF cloud + mean
    fig, (ax_raw, ax_dt) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, lags, ccf_list, title in [
        (ax_raw, lags_raw, ccf_raw_list, "Raw series"),
        (ax_dt, lags_dt, ccf_dt_list, "Detrended residual"),
    ]:
        for ccf in ccf_list:
            ax.plot(lags, ccf, color=CLOUD_COLOR, linewidth=0.5, alpha=0.25)
        mean_ccf = np.nanmean(np.vstack(ccf_list), axis=0)
        ax.plot(lags, mean_ccf, color=MEAN_COLOR, linewidth=2.5, label="mean across 50 seeds")
        ax.axvline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.axhline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.set_xlabel("lag tau (generations)  [tau>0: K_out lags K_self]")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    ax_raw.set_ylabel("corr(K_self[t], K_out[t+tau])")
    ax_raw.legend(fontsize=8, loc="upper right")
    fig.suptitle("Cross-correlation of mean K_out and mean K_self, all 50 seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ccf_50seed.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'ccf_50seed.png'}")

    # (2) PSD cloud + mean
    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, psd_list, title in [(ax_kout, psd_kout_list, "K_out"), (ax_kself, psd_kself_list, "K_self")]:
        for freqs, power in psd_list:
            with np.errstate(divide="ignore"):
                periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
            mask = (periods >= 10) & (periods <= 1000)
            ax.plot(periods[mask], power[mask], color=CLOUD_COLOR, linewidth=0.5, alpha=0.25)
        mean_power = np.nanmean(np.vstack([p for _, p in psd_list]), axis=0)
        freqs0 = psd_list[0][0]
        with np.errstate(divide="ignore"):
            periods0 = np.where(freqs0 > 0, 1.0 / freqs0, np.inf)
        mask0 = (periods0 >= 10) & (periods0 <= 1000)
        ax.plot(periods0[mask0], mean_power[mask0], color=MEAN_COLOR, linewidth=2.5, label="mean across 50 seeds")
        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 1000)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD (Welch, nperseg={NPERSEG})")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("power")
    ax_kout.legend(fontsize=8, loc="upper left")
    fig.suptitle("PSD of mean K_out / K_self, all 50 seeds", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "psd_50seed.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'psd_50seed.png'}")

    # (3) amplitude histogram
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.hist(radii, bins=15, color=MEAN_COLOR, alpha=0.85, edgecolor="white")
    mean_r, std_r = np.mean(radii), np.std(radii)
    ax.axvline(mean_r, color=MUTED_INK, linewidth=1.5, linestyle="--", label=f"mean={mean_r:.4f}")
    ax.axvline(mean_r - std_r, color=MUTED_INK, linewidth=0.8, linestyle=":")
    ax.axvline(mean_r + std_r, color=MUTED_INK, linewidth=0.8, linestyle=":", label=f"±1 s.d.={std_r:.4f}")
    ax.set_xlabel("detrended mean radius from origin")
    ax.set_ylabel("# seeds")
    ax.set_title("Distribution of detrended mean radius, 50 seeds")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radius_50seed_histogram.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'radius_50seed_histogram.png'}")


if __name__ == "__main__":
    main()
