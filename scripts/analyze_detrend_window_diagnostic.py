"""Diagnostic 1: is the ~77-generation PSD peak a real characteristic
scale of the system, or an artifact of the moving-average detrend
window used to remove slow drift before computing the PSD?

Uses the existing 5000-generation, 50-seed confirmatory coupling sweep
(scripts/output/coupling_seed_sweep_5000gen/) -- no new simulation.
Analysis window is fixed at generation [2000, 5000) (the steady-state
region) and nperseg=512 is fixed; only the detrend window is varied
over {none, 400, 300, 200, 150, 100, 75, 50} generations.

Read: if the peak period stays near ~77 gen regardless of the detrend
window, that's a genuine intrinsic scale. If it drifts to longer
periods as the window widens -- or the local hump near 77 disappears
into a monotonically rising "red" spectrum (visible as the peak search
hitting the band edge at period_max instead of a true interior local
max) -- that's the detrend window creating an artifact, not measuring
one.

Outputs:
    scripts/output/detrend_window_diagnostic/peak_period_vs_detrend_window.png
    scripts/output/detrend_window_diagnostic/psd_overlay_by_detrend_window.png
    scripts/output/detrend_window_diagnostic/detrend_window_diagnostic.csv
        (one row per seed x channel x detrend_window: peak_period, Q,
        hit_band_edge)

Usage:
    python3 scripts/analyze_detrend_window_diagnostic.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
OUT_DIR = Path("scripts/output/detrend_window_diagnostic")
SEEDS = list(range(50))
WINDOW_RANGE = (2000, 5000)
NPERSEG = 512
PERIOD_MIN, PERIOD_MAX = 20.0, 300.0
PLOT_PERIOD_MIN, PLOT_PERIOD_MAX = 10.0, 1000.0
LANDMARK_PERIOD = 77

# Ordered narrow (aggressive detrend) -> wide -> none (no detrend at all).
# None sits at the "infinite window" end of the same progression.
DETREND_WINDOWS = [50, 75, 100, 150, 200, 300, 400, None]

MUTED_INK = "#8f8f88"
KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
# Sequential ramp (magnitude = detrend window width), light -> dark.
SEQ_LIGHT = "#cde2fb"
SEQ_DARK = "#0b3d91"


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


def detrend(x, window):
    """None = no moving-average detrend at all (only overall mean removed)."""
    if window is None:
        return x - x.mean()
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


def peak_stats(freqs, power, period_min, period_max):
    """Peak period + Q-factor (peak_freq / FWHM) + whether the half-power
    width search ran off the edge of the search band (a sign the "peak"
    is really just where a monotonic rise got truncated, not a true
    local maximum)."""
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    in_band = (periods >= period_min) & (periods <= period_max)
    band_idx = np.where(in_band)[0]
    if len(band_idx) == 0:
        return float("nan"), float("nan"), float("nan"), True
    peak_idx = band_idx[np.argmax(power[band_idx])]
    peak_period = periods[peak_idx]
    peak_power = power[peak_idx]
    half_power = peak_power / 2.0
    lo_bound, hi_bound = band_idx[0], band_idx[-1]

    i = peak_idx
    while i > lo_bound and power[i] >= half_power:
        i -= 1
    freq_low = freqs[i] if power[i] < half_power else freqs[lo_bound]

    j = peak_idx
    while j < hi_bound and power[j] >= half_power:
        j += 1
    freq_high = freqs[j] if power[j] < half_power else freqs[hi_bound]

    fwhm = freq_high - freq_low
    q_factor = freqs[peak_idx] / fwhm if fwhm > 0 else np.inf
    hit_band_edge = peak_idx in (lo_bound, hi_bound) or (
        (i == lo_bound and power[lo_bound] >= half_power) or (j == hi_bound and power[hi_bound] >= half_power)
    )
    return peak_period, peak_power, q_factor, hit_band_edge


def label_for(window):
    return "none" if window is None else str(window)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # rows: detrend_window_label, seed, channel, peak_period, q_factor, hit_band_edge
    rows = []
    # mean_psd[window][channel] -> list of power arrays (one per seed), plus the shared freqs
    mean_psd = {label_for(w): {"K_out": [], "K_self": []} for w in DETREND_WINDOWS}
    freqs_ref = None

    for seed in SEEDS:
        gens, kout, kself = load_full_series(seed)
        mask = window_mask(gens, *WINDOW_RANGE)
        kout_w = kout[mask]
        kself_w = kself[mask]

        for window in DETREND_WINDOWS:
            label = label_for(window)
            for channel_name, series in [("K_out", kout_w), ("K_self", kself_w)]:
                dt = detrend(series, window)
                freqs, power = welch_psd(dt, NPERSEG)
                freqs_ref = freqs
                mean_psd[label][channel_name].append(power)
                period, _, q, hit_edge = peak_stats(freqs, power, PERIOD_MIN, PERIOD_MAX)
                rows.append(
                    {
                        "detrend_window": label,
                        "seed": seed,
                        "channel": channel_name,
                        "peak_period": period,
                        "q_factor": q,
                        "hit_band_edge": hit_edge,
                    }
                )

    csv_path = OUT_DIR / "detrend_window_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detrend_window", "seed", "channel", "peak_period", "q_factor", "hit_band_edge"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    print(f"\n=== Peak period vs detrend window (window gen {WINDOW_RANGE}, nperseg={NPERSEG}) ===")
    agg = {}
    for window in DETREND_WINDOWS:
        label = label_for(window)
        for channel_name in ["K_out", "K_self"]:
            sub = [r for r in rows if r["detrend_window"] == label and r["channel"] == channel_name]
            periods = np.array([r["peak_period"] for r in sub])
            edge_frac = np.mean([r["hit_band_edge"] for r in sub])
            q1, med, q3 = np.percentile(periods, [25, 50, 75])
            agg[(label, channel_name)] = {"median": med, "q1": q1, "q3": q3, "edge_frac": edge_frac}
            print(
                f"detrend_window={label:<5} {channel_name:<7} median={med:6.2f} "
                f"IQR=[{q1:6.2f},{q3:6.2f}]  hit_band_edge={edge_frac*100:5.1f}% of seeds"
            )

    # --- Figure (a): peak period vs detrend window ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_ordered = [label_for(w) for w in DETREND_WINDOWS]
    x = np.arange(len(labels_ordered))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for channel_name, color, marker in [("K_out", KOUT_COLOR, "o"), ("K_self", KSELF_COLOR, "s")]:
        medians = [agg[(lbl, channel_name)]["median"] for lbl in labels_ordered]
        q1s = [agg[(lbl, channel_name)]["q1"] for lbl in labels_ordered]
        q3s = [agg[(lbl, channel_name)]["q3"] for lbl in labels_ordered]
        lo_err = [m - q for m, q in zip(medians, q1s)]
        hi_err = [q - m for q, m in zip(q3s, medians)]
        ax.errorbar(x, medians, yerr=[lo_err, hi_err], marker=marker, color=color, markersize=8,
                    linewidth=1.8, capsize=4, label=channel_name)
        for xi, lbl in zip(x, labels_ordered):
            frac = agg[(lbl, channel_name)]["edge_frac"]
            if frac >= 0.5:
                ax.annotate(f"{frac*100:.0f}% edge", (xi, agg[(lbl, channel_name)]["median"]),
                            textcoords="offset points", xytext=(0, 10 if channel_name == "K_out" else -16),
                            fontsize=7, color=color, ha="center")

    ax.axhline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD}-gen reference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered)
    ax.set_xlabel("detrend window (generations); 'none' = no moving-average detrend")
    ax.set_ylabel("PSD peak period (generations), median ± IQR")
    ax.set_title(f"Does the peak period depend on the detrend window?\n"
                 f"(window gen {WINDOW_RANGE}, nperseg={NPERSEG}, n={len(SEEDS)} seeds; "
                 f"labels show % of seeds where the 'peak' just hit the search-band edge)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_a_path = OUT_DIR / "peak_period_vs_detrend_window.png"
    fig.savefig(fig_a_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_a_path}")

    # --- Figure (b): overlaid mean PSD per detrend window ---
    n = len(DETREND_WINDOWS)
    cmap_colors = []
    for i in range(n):
        t = i / (n - 1)
        # simple linear interpolation between SEQ_LIGHT and SEQ_DARK in hex
        c1 = np.array([int(SEQ_LIGHT[j:j+2], 16) for j in (1, 3, 5)])
        c2 = np.array([int(SEQ_DARK[j:j+2], 16) for j in (1, 3, 5)])
        c = (c1 + t * (c2 - c1)).astype(int)
        cmap_colors.append(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}")

    with np.errstate(divide="ignore"):
        periods_ref = np.where(freqs_ref > 0, 1.0 / freqs_ref, np.inf)
    plot_mask = (periods_ref >= PLOT_PERIOD_MIN) & (periods_ref <= PLOT_PERIOD_MAX)

    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, channel_name, title in [(ax_kout, "K_out", "K_out"), (ax_kself, "K_self", "K_self")]:
        for i, window in enumerate(DETREND_WINDOWS):
            label = label_for(window)
            mean_power = np.nanmean(np.vstack(mean_psd[label][channel_name]), axis=0)
            ax.plot(periods_ref[plot_mask], mean_power[plot_mask], color=cmap_colors[i], linewidth=2.0,
                    label=f"detrend={label}")
        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.axvspan(PERIOD_MIN, PERIOD_MAX, color=MUTED_INK, alpha=0.06)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(PLOT_PERIOD_MIN, PLOT_PERIOD_MAX)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD, mean across {len(SEEDS)} seeds")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("power")
    ax_kout.legend(fontsize=7, loc="upper left", ncol=1)
    fig.suptitle(f"Does the {LANDMARK_PERIOD}-gen hump survive across detrend windows, "
                 f"or melt into a monotonic rise? (window gen {WINDOW_RANGE})", y=1.03)
    fig.tight_layout()
    fig_b_path = OUT_DIR / "psd_overlay_by_detrend_window.png"
    fig.savefig(fig_b_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_b_path}")


if __name__ == "__main__":
    main()
