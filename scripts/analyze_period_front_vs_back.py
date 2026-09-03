"""Does the ~77-generation K_out/K_self period hold steady across the
run, or does it shift as memory depth deepens?

Splits each of the 50 sweep seeds into an early window (generation
0-999, where memory depth is still shallow and rising -- see
plot_memory_depth_evolution_50seed.py) and a late window (generation
2000-4999, where mean depth has leveled off), computes each half's PSD
peak period independently (Welch, same nperseg for both halves so the
comparison isn't confounded by different frequency resolution), and
checks (a) whether the peak period differs between halves and (b)
whether a seed's period shift correlates with its memory-depth shift.

Pure re-analysis of already-saved timeseries.csv -- no new simulation.

Usage:
    python3 scripts/analyze_period_front_vs_back.py
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
NPERSEG = 512
LANDMARK_PERIOD = 80
FRONT_RANGE = (0, 1000)
BACK_RANGE = (2000, 5000)

FRONT_COLOR = "#eb6834"  # categorical slot 2: identity = "early window"
BACK_COLOR = "#2a78d6"  # categorical slot 1: identity = "late window"
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


def peak_period(freqs, power, period_min=10.0, period_max=1000.0):
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    in_band = (periods >= period_min) & (periods <= period_max)
    band_idx = np.where(in_band)[0]
    if len(band_idx) == 0:
        return float("nan")
    peak_idx = band_idx[np.argmax(power[band_idx])]
    return periods[peak_idx]


def summarize(values, label):
    values = np.array(values, dtype=float)
    valid = values[~np.isnan(values)]
    print(
        f"{label}: n={len(valid)}/{len(values)} mean={valid.mean():.2f} "
        f"median={np.median(valid):.2f} std={valid.std():.2f} "
        f"min={valid.min():.2f} max={valid.max():.2f}"
    )
    return valid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []  # seed, front_period_kout, back_period_kout, front_period_kself, back_period_kself, front_depth, back_depth
    front_psd_kout, back_psd_kout = [], []
    front_psd_kself, back_psd_kself = [], []
    front_freqs = back_freqs = None

    for seed in SEEDS:
        gens, kout, kself, depth = load_full_series(seed)
        fmask = window_mask(gens, *FRONT_RANGE)
        bmask = window_mask(gens, *BACK_RANGE)

        f_kout_dt = detrend(kout[fmask], DETREND_WINDOW)
        f_kself_dt = detrend(kself[fmask], DETREND_WINDOW)
        b_kout_dt = detrend(kout[bmask], DETREND_WINDOW)
        b_kself_dt = detrend(kself[bmask], DETREND_WINDOW)

        ff, fp_kout = welch_psd(f_kout_dt, NPERSEG)
        _, fp_kself = welch_psd(f_kself_dt, NPERSEG)
        bf, bp_kout = welch_psd(b_kout_dt, NPERSEG)
        _, bp_kself = welch_psd(b_kself_dt, NPERSEG)
        front_freqs, back_freqs = ff, bf

        front_psd_kout.append(fp_kout)
        back_psd_kout.append(bp_kout)
        front_psd_kself.append(fp_kself)
        back_psd_kself.append(bp_kself)

        front_period_kout = peak_period(ff, fp_kout)
        back_period_kout = peak_period(bf, bp_kout)
        front_period_kself = peak_period(ff, fp_kself)
        back_period_kself = peak_period(bf, bp_kself)

        front_depth = depth[fmask].mean()
        back_depth = depth[bmask].mean()

        rows.append(
            [seed, front_period_kout, back_period_kout, front_period_kself, back_period_kself, front_depth, back_depth]
        )

    header = ["seed", "front_period_kout", "back_period_kout", "front_period_kself", "back_period_kself", "front_mean_depth", "back_mean_depth"]
    with open(OUT_DIR / "period_front_vs_back_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {OUT_DIR / 'period_front_vs_back_per_seed.csv'}")

    front_kout = [r[1] for r in rows]
    back_kout = [r[2] for r in rows]
    front_kself = [r[3] for r in rows]
    back_kself = [r[4] for r in rows]
    front_depth = [r[5] for r in rows]
    back_depth = [r[6] for r in rows]

    print("\n=== Peak period: front (gen 0-1000) vs back (gen 2000-5000) ===")
    summarize(front_kout, "K_out front period")
    summarize(back_kout, "K_out back period")
    summarize(front_kself, "K_self front period")
    summarize(back_kself, "K_self back period")

    print("\n=== Memory depth: front vs back ===")
    summarize(front_depth, "front mean depth")
    summarize(back_depth, "back mean depth")

    delta_period_kout = np.array(back_kout) - np.array(front_kout)
    delta_period_kself = np.array(back_kself) - np.array(front_kself)
    delta_depth = np.array(back_depth) - np.array(front_depth)
    summarize(delta_period_kout, "Delta period (K_out, back-front)")
    summarize(delta_period_kself, "Delta period (K_self, back-front)")
    summarize(delta_depth, "Delta mean depth (back-front)")

    corr_kout = np.corrcoef(delta_depth, delta_period_kout)[0, 1]
    corr_kself = np.corrcoef(delta_depth, delta_period_kself)[0, 1]
    print(f"\ncorr(Delta depth, Delta period K_out) = {corr_kout:.3f}")
    print(f"corr(Delta depth, Delta period K_self) = {corr_kself:.3f}")

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: front vs back mean PSD, K_out | K_self
    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, front_list, back_list, title in [
        (ax_kout, front_psd_kout, back_psd_kout, "K_out"),
        (ax_kself, front_psd_kself, back_psd_kself, "K_self"),
    ]:
        with np.errstate(divide="ignore"):
            fperiods = np.where(front_freqs > 0, 1.0 / front_freqs, np.inf)
            bperiods = np.where(back_freqs > 0, 1.0 / back_freqs, np.inf)
        fmask_plot = (fperiods >= 10) & (fperiods <= 1000)
        bmask_plot = (bperiods >= 10) & (bperiods <= 1000)

        mean_front = np.nanmean(np.vstack(front_list), axis=0)
        mean_back = np.nanmean(np.vstack(back_list), axis=0)
        ax.plot(fperiods[fmask_plot], mean_front[fmask_plot], color=FRONT_COLOR, linewidth=2.2, label="early (gen 0-1000)")
        ax.plot(bperiods[bmask_plot], mean_back[bmask_plot], color=BACK_COLOR, linewidth=2.2, label="late (gen 2000-5000)")
        ax.axvline(LANDMARK_PERIOD, color=MUTED_INK, linewidth=1.0, linestyle="--", label=f"{LANDMARK_PERIOD} gen")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 1000)
        ax.set_xlabel("period (generations)")
        ax.set_title(f"{title} PSD: early vs. late window")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("power (mean across 50 seeds)")
    ax_kout.legend(fontsize=8, loc="upper left")
    fig.suptitle("Does the PSD peak move as memory deepens? (nperseg=512 both windows)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "psd_front_vs_back.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'psd_front_vs_back.png'}")

    # Figure 2: per-seed period scatter, front vs back
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(front_kout, back_kout, s=28, color=FRONT_COLOR, alpha=0.75, label="K_out", edgecolor="white", linewidth=0.4)
    ax.scatter(front_kself, back_kself, s=28, color=BACK_COLOR, alpha=0.75, label="K_self", edgecolor="white", linewidth=0.4)
    lims = [10, 250]
    ax.plot(lims, lims, color=MUTED_INK, linewidth=1.0, linestyle="--", label="y = x (no change)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("peak period, early window (gen)")
    ax.set_ylabel("peak period, late window (gen)")
    ax.set_title("Per-seed peak period: early vs. late")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "period_front_vs_back_scatter.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'period_front_vs_back_scatter.png'}")

    # Figure 3: Delta depth vs Delta period
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(delta_depth, delta_period_kout, s=28, color=FRONT_COLOR, alpha=0.75, label=f"K_out (corr={corr_kout:.2f})", edgecolor="white", linewidth=0.4)
    ax.scatter(delta_depth, delta_period_kself, s=28, color=BACK_COLOR, alpha=0.75, label=f"K_self (corr={corr_kself:.2f})", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
    ax.axvline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
    ax.set_xlabel("Δ mean memory depth (late - early)")
    ax.set_ylabel("Δ peak period (late - early, generations)")
    ax.set_title("Does deepening memory move the period? (per seed)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_vs_period_shift.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'depth_vs_period_shift.png'}")


if __name__ == "__main__":
    main()
