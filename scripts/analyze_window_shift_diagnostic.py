"""Diagnostic 2: is long-period PSD power a leftover of the memory-depth
"settling down" transient, or an intrinsic (depth-independent) red
spectrum?

Uses the existing 5000-generation, 50-seed confirmatory coupling sweep
(scripts/output/coupling_seed_sweep_5000gen/) -- no new simulation.
detrend window is fixed at 200 generations (deliberately the same
"too-wide" setting diagnosed in the point-mutation screening run, where
it let long-period power dominate) and nperseg=512 is fixed; only the
analysis window is varied: [2000, 5000) vs [3000, 5000).

Read: if sliding the window later (dropping generations 2000-3000, so
depth has had longer to settle) reduces long-period power relative to
short-period power, that power was drift from the still-settling
period. If the long/short power ratio holds steady across the shift,
the long-period content is an intrinsic feature of the spectrum, not a
settling artifact.

Outputs:
    scripts/output/window_shift_diagnostic/psd_overlay_window_shift.png
    scripts/output/window_shift_diagnostic/window_shift_diagnostic.csv
        (one row per seed x channel x window: peak_period,
        long_short_power_ratio)

Usage:
    python3 scripts/analyze_window_shift_diagnostic.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
OUT_DIR = Path("scripts/output/window_shift_diagnostic")
SEEDS = list(range(50))
DETREND_WINDOW = 200
NPERSEG = 512
PERIOD_MIN, PERIOD_MAX = 20.0, 300.0
PLOT_PERIOD_MIN, PLOT_PERIOD_MAX = 10.0, 1000.0
LANDMARK_PERIOD = 77
SHORT_BAND = (20.0, 100.0)
LONG_BAND = (150.0, 300.0)

WINDOWS = [("gen[2000,5000)", (2000, 5000)), ("gen[3000,5000)", (3000, 5000))]
WINDOW_COLORS = {"gen[2000,5000)": "#2a78d6", "gen[3000,5000)": "#eb6834"}
MUTED_INK = "#8f8f88"


def load_full_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, kout, kself, depth = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
            depth.append(float(row["mean_depth"]))
    return np.array(gens), np.array(kout), np.array(kself), np.array(depth)


def window_mask(gens, lo, hi):
    return (gens >= lo) & (gens < hi)


def detrend(x, window):
    if len(x) <= window:
        return x - x.mean()
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    return x[offset : offset + len(trend)] - trend


def welch_psd(x, nperseg):
    from scipy.signal import welch

    nperseg = min(nperseg, len(x))
    freqs, power = welch(x, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    return freqs, power


def peak_period(freqs, power, period_min, period_max):
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    in_band = (periods >= period_min) & (periods <= period_max)
    band_idx = np.where(in_band)[0]
    if len(band_idx) == 0:
        return float("nan")
    peak_idx = band_idx[np.argmax(power[band_idx])]
    return periods[peak_idx]


def band_power(freqs, power, lo, hi):
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= lo) & (periods < hi)
    return power[mask].sum()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    mean_psd = {label: {"K_out": [], "K_self": []} for label, _ in WINDOWS}
    freqs_ref = None
    depth_at_start = {}  # label -> list of mean_depth right at window start, per seed

    for seed in SEEDS:
        gens, kout, kself, depth = load_full_series(seed)
        for label, (lo, hi) in WINDOWS:
            mask = window_mask(gens, lo, hi)
            depth_at_start.setdefault(label, []).append(depth[mask][0])
            for channel_name, series in [("K_out", kout[mask]), ("K_self", kself[mask])]:
                dt = detrend(series, DETREND_WINDOW)
                freqs, power = welch_psd(dt, NPERSEG)
                freqs_ref = freqs
                mean_psd[label][channel_name].append(power)
                period = peak_period(freqs, power, PERIOD_MIN, PERIOD_MAX)
                short_p = band_power(freqs, power, *SHORT_BAND)
                long_p = band_power(freqs, power, *LONG_BAND)
                ratio = long_p / short_p if short_p > 0 else float("nan")
                rows.append(
                    {
                        "window": label,
                        "seed": seed,
                        "channel": channel_name,
                        "peak_period": period,
                        "long_short_power_ratio": ratio,
                    }
                )

    csv_path = OUT_DIR / "window_shift_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "seed", "channel", "peak_period", "long_short_power_ratio"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    print(f"\n=== Mean depth at window start (settling check) ===")
    for label, _ in WINDOWS:
        d = np.array(depth_at_start[label])
        print(f"{label}: mean_depth at window start = {d.mean():.2f} (std {d.std():.2f})")

    print(f"\n=== Peak period, by window (detrend_window={DETREND_WINDOW}, nperseg={NPERSEG}) ===")
    for label, _ in WINDOWS:
        for channel_name in ["K_out", "K_self"]:
            periods = np.array([r["peak_period"] for r in rows if r["window"] == label and r["channel"] == channel_name])
            q1, med, q3 = np.percentile(periods, [25, 50, 75])
            print(f"{label:<16} {channel_name:<7} median={med:6.2f} IQR=[{q1:6.2f},{q3:6.2f}]")

    print(f"\n=== long/short power ratio (long={LONG_BAND}, short={SHORT_BAND} gen), by window ===")
    from scipy.stats import wilcoxon

    for channel_name in ["K_out", "K_self"]:
        ratios_a = np.array([r["long_short_power_ratio"] for r in rows if r["window"] == "gen[2000,5000)" and r["channel"] == channel_name])
        ratios_b = np.array([r["long_short_power_ratio"] for r in rows if r["window"] == "gen[3000,5000)" and r["channel"] == channel_name])
        print(f"{channel_name}: gen[2000,5000) median={np.median(ratios_a):.3f}  "
              f"gen[3000,5000) median={np.median(ratios_b):.3f}")
        stat, p = wilcoxon(ratios_a, ratios_b)
        direction = "decreased" if np.median(ratios_b) < np.median(ratios_a) else "did not decrease"
        print(f"  paired Wilcoxon signed-rank (same 50 seeds, A vs B): stat={stat:.2f}, p={p:.4f} "
              f"-- ratio {direction} when the window moved later")

    # --- plot: overlaid mean PSD per window ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.errstate(divide="ignore"):
        periods_ref = np.where(freqs_ref > 0, 1.0 / freqs_ref, np.inf)
    plot_mask = (periods_ref >= PLOT_PERIOD_MIN) & (periods_ref <= PLOT_PERIOD_MAX)

    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, channel_name, title in [(ax_kout, "K_out", "K_out"), (ax_kself, "K_self", "K_self")]:
        for label, _ in WINDOWS:
            mean_power = np.nanmean(np.vstack(mean_psd[label][channel_name]), axis=0)
            ax.plot(periods_ref[plot_mask], mean_power[plot_mask], color=WINDOW_COLORS[label], linewidth=2.2, label=label)
        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.axvspan(*LONG_BAND, color=WINDOW_COLORS["gen[2000,5000)"], alpha=0.05, label=f"long band {LONG_BAND}")
        ax.axvspan(*SHORT_BAND, color=MUTED_INK, alpha=0.05, label=f"short band {SHORT_BAND}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(PLOT_PERIOD_MIN, PLOT_PERIOD_MAX)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD, mean across {len(SEEDS)} seeds (detrend={DETREND_WINDOW})")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("power")
    ax_kout.legend(fontsize=8, loc="upper left")
    fig.suptitle("Does long-period power fade when the window moves past the depth-settling phase?", y=1.03)
    fig.tight_layout()
    fig_path = OUT_DIR / "psd_overlay_window_shift.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
