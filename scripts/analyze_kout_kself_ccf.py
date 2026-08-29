"""Cross-correlation between mean K_out and mean K_self across the
5000-generation, 5-seed coupling sweep
(scripts/output/coupling_seed_sweep_5000gen/seed_<seed>_timeseries.csv).

For each seed, K_self is the reference series and K_out is the series
shifted in lag (§ convention below), on both the raw series and a
detrended residual (raw minus a centered moving average, to strip the
slow common drift and isolate the faster back-and-forth).

Lag convention: CCF(tau) = corr(K_self[t], K_out[t+tau]).
  - CCF peaks/troughs at tau > 0 => K_out *lags* K_self (K_self moves
    first, K_out follows tau generations later).
  - tau < 0 => K_out *leads* K_self.

Usage:
    python3 scripts/analyze_kout_kself_ccf.py
    python3 scripts/analyze_kout_kself_ccf.py --max-lag 500 --detrend-window 75
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# HBES dataviz palette: categorical slots 1-5, one per seed (lines/small
# multiples validate on the adjacent pairlist for the full 8-hue order).
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MEAN_COLOR = "#3a3a35"  # dark ink, distinct role from the per-seed hues
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen"
    )
    p.add_argument("--seed-list", type=str, default="42,1,2,3,4")
    p.add_argument("--max-lag", type=int, default=500, help="generations, each direction")
    p.add_argument("--detrend-window", type=int, default=75, help="centered moving-average window")
    p.add_argument(
        "--min-trough-distance", type=int, default=50,
        help="min lag spacing (generations) between detected local minima, for period estimation",
    )
    p.add_argument("--out", type=str, default="scripts/output/kout_kself_ccf.png")
    return p.parse_args()


def load_series(sweep_dir: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = sweep_dir / f"seed_{seed}_timeseries.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def detrend(gens: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Centered moving-average detrend. 'valid' convolution (no edge
    padding) so the trend estimate is never biased by phantom zeros at
    the boundary; both the raw and detrended series are trimmed to the
    same window-centered range so they cover identical generations."""
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    x_trimmed = x[offset : offset + len(trend)]
    gens_trimmed = gens[offset : offset + len(trend)]
    residual = x_trimmed - trend
    return gens_trimmed, residual


def cross_correlation(x_ref: np.ndarray, y_shift: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """CCF(tau) = corr(x_ref[t], y_shift[t+tau]) for tau in [-max_lag, max_lag]."""
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


def trough_stats(lags: np.ndarray, ccf: np.ndarray, min_distance: int):
    """Global trough (lag, depth) plus a period estimate from the median
    spacing between all sufficiently-prominent local minima."""
    from scipy.signal import find_peaks

    valid = ~np.isnan(ccf)
    global_idx = np.nanargmin(ccf)
    trough_lag = int(lags[global_idx])
    trough_value = float(ccf[global_idx])

    peak_idx, _ = find_peaks(-np.where(valid, ccf, 0.0), distance=max(1, min_distance), prominence=0.03)
    trough_lags = np.sort(lags[peak_idx])
    if len(trough_lags) >= 2:
        spacings = np.diff(trough_lags)
        period = float(np.median(spacings))
    else:
        period = float("nan")

    return trough_lag, trough_value, period, trough_lags


def print_table(rows):
    header = f"{'seed':>4} {'trough_lag':>11} {'trough_ccf':>11} {'est_period':>11}  {'preprocessing':>13}"
    print(header)
    print("-" * len(header))
    for seed, kind, lag, val, period in rows:
        period_str = f"{period:.0f}" if not np.isnan(period) else "N/A"
        print(f"{seed:>4} {lag:>11} {val:>11.3f} {period_str:>11}  {kind:>13}")


def plot(all_data, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_raw, ax_detrend) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, key, title in [(ax_raw, "raw", "Raw series"), (ax_detrend, "detrend", "Detrended residual")]:
        curves = []
        for i, (seed, series) in enumerate(all_data):
            lags, ccf = series[key]
            ax.plot(lags, ccf, color=SEED_COLORS[i % len(SEED_COLORS)], linewidth=1.2, alpha=0.85, label=f"seed={seed}")
            curves.append(ccf)
        mean_ccf = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(lags, mean_ccf, color=MEAN_COLOR, linewidth=2.5, label="mean across seeds")

        ax.axvline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.axhline(0, color=MUTED_INK, linewidth=0.8, linestyle="--")
        ax.set_xlabel("lag tau (generations)  [tau>0: K_out lags K_self]")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    ax_raw.set_ylabel("corr(K_self[t], K_out[t+tau])")
    ax_raw.legend(fontsize=8, loc="upper right")
    fig.suptitle("Cross-correlation of mean K_out and mean K_self, per seed", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved plot to {out_path}")


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]

    all_data = []
    table_rows = []
    for seed in seeds:
        gens, kout, kself = load_series(sweep_dir, seed)

        lags_raw, ccf_raw = cross_correlation(kself, kout, args.max_lag)
        _, kout_dt = detrend(gens, kout, args.detrend_window)
        _, kself_dt = detrend(gens, kself, args.detrend_window)
        lags_dt, ccf_dt = cross_correlation(kself_dt, kout_dt, args.max_lag)

        all_data.append((seed, {"raw": (lags_raw, ccf_raw), "detrend": (lags_dt, ccf_dt)}))

        for key, lags, ccf, label in [("raw", lags_raw, ccf_raw, "raw"), ("detrend", lags_dt, ccf_dt, "detrended")]:
            trough_lag, trough_value, period, _ = trough_stats(lags, ccf, args.min_trough_distance)
            table_rows.append((seed, label, trough_lag, trough_value, period))

    print_table(table_rows)
    plot(all_data, Path(args.out))


if __name__ == "__main__":
    main()
