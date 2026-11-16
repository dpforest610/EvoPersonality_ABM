"""Analyze phase for the point-mutation-rate sweep.

Question: does point_mutation_rate set the ~77-generation K_out/K_self
quasi-period? Prediction: raising point_mutation_rate shrinks the period.

Reads ONLY the per-cell CSVs already saved by run_point_mutation_sweep.py
(scripts/output/point_mutation_sweep/pmr_<rate>/seed_<seed>.csv, or the
_smoke variant) plus that run's manifest.json for the level/seed list. No
simulation happens here -- this script can be re-run with different
--window-start/--window-end/--detrend-window/--nperseg/--period-min/
--period-max without recomputing anything upstream.

Measurement (identical across all levels, by construction -- the window,
detrend window, and nperseg are single fixed values applied uniformly, so
a period difference between levels can't be an artifact of measuring them
differently):
  1. Fixed window generation in [window_start, window_end) -- default
     [2000, 5000), the steady-state region -- same window for every level.
  2. Centered moving-average detrend (window=200 generations, default).
  3. Welch PSD (nperseg=512 default, same for every level).
  4. Peak period search restricted to the band [period_min, period_max]
     generations (default [20, 300]).
  5. Per (level, channel): median/Q1/Q3 of peak period across seeds, and
     mean memory depth within the window.
  6. Trend tests: Spearman(point_mutation_rate, peak_period) and
     Kruskal-Wallis across the level groups, per channel. Reported with
     coefficient + p-value + the underlying per-level distributions --
     not collapsed to a bare significant/non-significant verdict.

Outputs (under --out-dir, default scripts/output/point_mutation_sweep_analysis):
  1. point_mutation_sweep_periods.csv -- tidy, one row per (level, seed, channel)
  2. period_vs_mutation_rate.png -- median + IQR vs. point_mutation_rate,
     with a 77-generation reference line
  3. psd_overlay_by_level.png -- mean PSD per level, overlaid, period axis
     (log-log)

Usage:
    python3 scripts/analyze_point_mutation_sweep.py
    python3 scripts/analyze_point_mutation_sweep.py --sweep-dir scripts/output/point_mutation_sweep_smoke \
        --out-dir scripts/output/point_mutation_sweep_analysis_smoke \
        --window-start 0 --window-end 1000 --nperseg 128 --detrend-window 20
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LANDMARK_PERIOD = 77

LEVEL_COLORS = {
    # categorical slots, assigned by identity (which level), not magnitude
    0: "#2a78d6",  # slot 1
    1: "#eb6834",  # slot 2
    2: "#1baf7a",  # slot 3
    3: "#eda100",  # slot 4
    4: "#e87ba4",  # slot 5
}
MUTED_INK = "#8f8f88"


def level_dirname(rate: float) -> str:
    return f"pmr_{rate:.3g}"


def load_manifest(sweep_dir: Path) -> dict:
    with open(sweep_dir / "manifest.json") as f:
        return json.load(f)


def load_cell(sweep_dir: Path, rate: float, seed: int):
    path = sweep_dir / level_dirname(rate) / f"seed_{seed}.csv"
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default="scripts/output/point_mutation_sweep")
    p.add_argument("--out-dir", type=str, default="scripts/output/point_mutation_sweep_analysis")
    p.add_argument("--window-start", type=int, default=2000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--detrend-window", type=int, default=200)
    p.add_argument("--nperseg", type=int, default=512)
    p.add_argument("--period-min", type=float, default=20.0)
    p.add_argument("--period-max", type=float, default=300.0)
    return p.parse_args()


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(sweep_dir)
    levels = manifest["point_mutation_rate_levels"]
    seeds = manifest["seeds"]

    print(f"sweep dir: {sweep_dir}")
    print(f"levels: {levels}")
    print(f"seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"window: gen [{args.window_start}, {args.window_end}) -- fixed across all levels")
    print(f"detrend window: {args.detrend_window}, nperseg: {args.nperseg}, "
          f"period band: [{args.period_min}, {args.period_max}]")

    tidy_rows = []  # level, seed, channel, peak_period, mean_depth_window
    # for PSD overlay: per level, per channel, list of (freqs, power) to average
    psd_by_level = {rate: {"K_out": [], "K_self": []} for rate in levels}
    psd_freqs_ref = None

    for rate in levels:
        for seed in seeds:
            gens, kout, kself, depth = load_cell(sweep_dir, rate, seed)
            mask = window_mask(gens, args.window_start, args.window_end)
            depth_mean = depth[mask].mean()

            for channel_name, series in [("K_out", kout), ("K_self", kself)]:
                windowed = series[mask]
                dt = detrend(windowed, args.detrend_window)
                freqs, power = welch_psd(dt, args.nperseg)
                psd_freqs_ref = freqs
                psd_by_level[rate][channel_name].append(power)
                period = peak_period(freqs, power, args.period_min, args.period_max)
                tidy_rows.append(
                    {
                        "point_mutation_rate": rate,
                        "seed": seed,
                        "channel": channel_name,
                        "peak_period": period,
                        "mean_depth_window": depth_mean,
                    }
                )

    tidy_path = out_dir / "point_mutation_sweep_periods.csv"
    with open(tidy_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point_mutation_rate", "seed", "channel", "peak_period", "mean_depth_window"])
        w.writeheader()
        w.writerows(tidy_rows)
    print(f"\nsaved {tidy_path} ({len(tidy_rows)} rows)")

    # --- aggregate per (level, channel) ---
    from scipy.stats import kruskal, spearmanr

    print("\n=== Per (level, channel): peak period median/IQR, mean depth ===")
    agg = {}  # (rate, channel) -> dict
    for rate in levels:
        for channel_name in ["K_out", "K_self"]:
            periods = np.array(
                [r["peak_period"] for r in tidy_rows if r["point_mutation_rate"] == rate and r["channel"] == channel_name]
            )
            depths = np.array(
                [r["mean_depth_window"] for r in tidy_rows if r["point_mutation_rate"] == rate and r["channel"] == channel_name]
            )
            valid = periods[~np.isnan(periods)]
            q1, med, q3 = np.percentile(valid, [25, 50, 75])
            agg[(rate, channel_name)] = {
                "median": med, "q1": q1, "q3": q3,
                "n": len(valid), "n_total": len(periods),
                "mean_depth": depths.mean(),
                "values": valid,
            }
            print(
                f"pmr={rate:<5} {channel_name:<7} n={len(valid)}/{len(periods)} "
                f"median={med:.2f} IQR=[{q1:.2f}, {q3:.2f}] mean_depth={depths.mean():.2f}"
            )

    print("\n=== Trend tests: point_mutation_rate vs peak_period ===")
    trend_results = {}
    for channel_name in ["K_out", "K_self"]:
        rates_arr = np.array([r["point_mutation_rate"] for r in tidy_rows if r["channel"] == channel_name])
        periods_arr = np.array([r["peak_period"] for r in tidy_rows if r["channel"] == channel_name])
        valid = ~np.isnan(periods_arr)
        rho, p_spearman = spearmanr(rates_arr[valid], periods_arr[valid])

        groups = [agg[(rate, channel_name)]["values"] for rate in levels]
        h_stat, p_kw = kruskal(*groups)

        trend_results[channel_name] = {
            "spearman_rho": rho, "spearman_p": p_spearman,
            "kruskal_h": h_stat, "kruskal_p": p_kw,
        }
        print(f"\n{channel_name}:")
        print(f"  Spearman(point_mutation_rate, peak_period): rho={rho:.3f}, p={p_spearman:.4f}, n={valid.sum()}")
        print(f"  Kruskal-Wallis across {len(levels)} level groups: H={h_stat:.3f}, p={p_kw:.4f}")
        for rate in levels:
            v = agg[(rate, channel_name)]["values"]
            print(f"    pmr={rate}: n={len(v)} median={np.median(v):.2f} "
                  f"mean={v.mean():.2f} std={v.std():.2f} min={v.min():.2f} max={v.max():.2f}")

    with open(out_dir / "trend_tests.json", "w") as f:
        json.dump(trend_results, f, indent=2)
    print(f"\nsaved {out_dir / 'trend_tests.json'}")

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 2: period vs mutation rate, median + IQR, per channel
    fig, ax = plt.subplots(figsize=(7.5, 6))
    channel_markers = {"K_out": "o", "K_self": "s"}
    channel_colors = {"K_out": "#2a78d6", "K_self": "#eb6834"}
    for channel_name in ["K_out", "K_self"]:
        medians = [agg[(rate, channel_name)]["median"] for rate in levels]
        q1s = [agg[(rate, channel_name)]["q1"] for rate in levels]
        q3s = [agg[(rate, channel_name)]["q3"] for rate in levels]
        lo_err = [m - q for m, q in zip(medians, q1s)]
        hi_err = [q - m for q, m in zip(q3s, medians)]
        tr = trend_results[channel_name]
        ax.errorbar(
            levels, medians, yerr=[lo_err, hi_err],
            marker=channel_markers[channel_name], color=channel_colors[channel_name],
            markersize=8, linewidth=1.8, capsize=4,
            label=f"{channel_name} (Spearman ρ={tr['spearman_rho']:.2f}, p={tr['spearman_p']:.3f})",
        )
    ax.axhline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD}-gen reference")
    ax.set_xscale("log")
    ax.set_xticks(levels)
    ax.set_xticklabels([str(r) for r in levels])
    ax.set_xlabel("point_mutation_rate")
    ax.set_ylabel("PSD peak period (generations), median ± IQR")
    ax.set_title(f"K_out / K_self quasi-period vs. point_mutation_rate\n"
                 f"(window gen [{args.window_start},{args.window_end}), fixed across levels; n={len(seeds)} seeds/level)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "period_vs_mutation_rate.png", dpi=150, bbox_inches="tight")
    print(f"saved {out_dir / 'period_vs_mutation_rate.png'}")

    # Figure 3: mean PSD per level, overlaid, period axis, log-log
    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    with np.errstate(divide="ignore"):
        periods_ref = np.where(psd_freqs_ref > 0, 1.0 / psd_freqs_ref, np.inf)
    plot_mask = (periods_ref >= 10) & (periods_ref <= 1000)
    for ax, channel_name, title in [(ax_kout, "K_out", "K_out"), (ax_kself, "K_self", "K_self")]:
        for i, rate in enumerate(levels):
            mean_psd = np.nanmean(np.vstack(psd_by_level[rate][channel_name]), axis=0)
            color = LEVEL_COLORS.get(i, MUTED_INK)
            ax.plot(periods_ref[plot_mask], mean_psd[plot_mask], color=color, linewidth=2.2, label=f"pmr={rate}")
        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.axvspan(args.period_min, args.period_max, color=MUTED_INK, alpha=0.06, label="search band")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 1000)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD, mean across {len(seeds)} seeds/level")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("power")
    ax_kout.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Mean PSD by point_mutation_rate level (window gen [{args.window_start},{args.window_end}), "
                 f"nperseg={args.nperseg})", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "psd_overlay_by_level.png", dpi=150, bbox_inches="tight")
    print(f"saved {out_dir / 'psd_overlay_by_level.png'}")


if __name__ == "__main__":
    main()
