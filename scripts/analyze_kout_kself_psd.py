"""Power spectral density of mean K_out and mean K_self, per seed, over
the 5000-generation coupling sweep (scripts/output/coupling_seed_sweep_5000gen/).

Answers whether the ~80-generation rhythm found by
analyze_kout_kself_ccf.py is a sharp-peaked quasi-periodicity, a broad
colored-noise hump, or no characteristic scale at all (flat/red).

Preprocessing: same centered-moving-average detrend as the CCF and
phase-plane scripts (raw series carry a slow drift that would otherwise
dominate the low-frequency end and distort the spectrum), computed on
the detrended residual via Welch's method (scipy.signal.welch, Hann
window, per-segment constant detrend) rather than a single FFT, so a
single 5000-point series still gives a smooth-enough spectrum.

Axes: x is period (generations) = 1/frequency, not frequency, log-scaled,
with a marker at ~80 generations; y is power on a log scale, since sharp
peaks and broad humps are both easier to tell apart in log-power.

Usage:
    python3 scripts/analyze_kout_kself_psd.py --nperseg 512
    python3 scripts/analyze_kout_kself_psd.py --nperseg 1024
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MUTED_INK = "#8f8f88"
LANDMARK_PERIOD = 80


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen")
    p.add_argument("--seed-list", type=str, default="42,1,2,3,4")
    p.add_argument("--detrend-window", type=int, default=75, help="centered moving-average window")
    p.add_argument("--nperseg", type=int, default=1024, help="Welch segment length (512 or 1024)")
    p.add_argument("--period-min", type=float, default=10.0)
    p.add_argument("--period-max", type=float, default=1000.0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def load_series(sweep_dir: Path, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    path = sweep_dir / f"seed_{seed}_timeseries.csv"
    kout, kself = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(kout), np.array(kself)


def detrend_series(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average detrend, 'valid' convolution (no edge padding bias)."""
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    return x[offset : offset + len(trend)] - trend


def welch_psd(x: np.ndarray, nperseg: int):
    from scipy.signal import welch

    freqs, power = welch(x, fs=1.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    return freqs, power


def peak_stats(freqs: np.ndarray, power: np.ndarray, period_min: float, period_max: float):
    """Peak period (max power within [period_min, period_max]) and its
    quality factor Q = peak_freq / FWHM, the standard sharpness measure
    for a resonance: Q >> 1 means a narrow spike (quasi-periodic), Q ~ 1
    means a broad hump comparable in width to its own center frequency
    (colored noise with a preferred scale but no sharp resonance).

    (A naive peak/background-average ratio was tried first and rejected:
    on a log-power axis a single broad hump can still tower over a wide
    band's *average* simply because the band's far tails are low, which
    would call a broad hump "sharp" -- exactly the distinction this
    script exists to make. FWHM in frequency space does not have that
    problem.)
    """
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
    if power[i] < half_power and i < peak_idx:
        freq_low = np.interp(half_power, [power[i], power[i + 1]], [freqs[i], freqs[i + 1]])
    else:
        freq_low = freqs[lo_bound]

    j = peak_idx
    while j < hi_bound and power[j] >= half_power:
        j += 1
    if power[j] < half_power and j > peak_idx:
        freq_high = np.interp(half_power, [power[j], power[j - 1]], [freqs[j], freqs[j - 1]])
    else:
        freq_high = freqs[hi_bound]

    fwhm = freq_high - freq_low
    q_factor = freqs[peak_idx] / fwhm if fwhm > 0 else np.inf
    hit_band_edge = (i == lo_bound and power[lo_bound] >= half_power) or (
        j == hi_bound and power[hi_bound] >= half_power
    )

    return peak_period, peak_power, q_factor, hit_band_edge


def print_table(rows):
    header = f"{'seed':>4} {'channel':>7} {'nperseg':>7} {'peak_period':>11} {'Q_factor':>9} {'note':>12}"
    print(header)
    print("-" * len(header))
    for seed, channel, nperseg, period, q, hit_edge in rows:
        note = "band-limited" if hit_edge else ""
        print(f"{seed:>4} {channel:>7} {nperseg:>7} {period:>11.1f} {q:>9.2f} {note:>12}")
    print(
        "\n(Q = peak_freq / FWHM: Q>>1 = sharp/quasi-periodic, Q~1 = broad hump, "
        "'band-limited' = half-power width hit the edge of the 10-1000gen window)"
    )


def plot(all_data, nperseg: int, period_min: float, period_max: float, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, key, title in [(ax_kout, "kout", "K_out"), (ax_kself, "kself", "K_self")]:
        for i, (seed, series) in enumerate(all_data):
            freqs, power = series[key]
            with np.errstate(divide="ignore"):
                periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
            mask = (periods >= period_min) & (periods <= period_max)
            ax.plot(periods[mask], power[mask], color=SEED_COLORS[i % len(SEED_COLORS)], linewidth=1.3, alpha=0.85, label=f"seed={seed}")

        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(period_min, period_max)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD (Welch, nperseg={nperseg})")
        ax.spines[["top", "right"]].set_visible(False)

    ax_kout.set_ylabel("power")
    ax_kout.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved plot to {out_path}")


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    out_path = Path(args.out) if args.out else Path(f"scripts/output/kout_kself_psd_nperseg{args.nperseg}.png")

    all_data = []
    table_rows = []
    for seed in seeds:
        kout, kself = load_series(sweep_dir, seed)
        kout_dt = detrend_series(kout, args.detrend_window)
        kself_dt = detrend_series(kself, args.detrend_window)

        freqs_out, power_out = welch_psd(kout_dt, args.nperseg)
        freqs_self, power_self = welch_psd(kself_dt, args.nperseg)

        all_data.append((seed, {"kout": (freqs_out, power_out), "kself": (freqs_self, power_self)}))

        for channel, freqs, power in [("K_out", freqs_out, power_out), ("K_self", freqs_self, power_self)]:
            period, _, q_factor, hit_edge = peak_stats(freqs, power, args.period_min, args.period_max)
            table_rows.append((seed, channel, args.nperseg, period, q_factor, hit_edge))

    print_table(table_rows)
    plot(all_data, args.nperseg, args.period_min, args.period_max, out_path)


if __name__ == "__main__":
    main()
