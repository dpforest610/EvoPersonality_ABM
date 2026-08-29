"""Phase-plane trajectories of (K_self, K_out) for the 5000-generation,
5-seed coupling sweep (scripts/output/coupling_seed_sweep_5000gen/).

Plots K_self (x) against K_out (y) as a trajectory colored by generation
(old=light/cool, new=dark/warm), for the raw series and for a detrended
residual (raw minus the same centered moving average used in
analyze_kout_kself_ccf.py, to strip the slow drift so a genuine limit
cycle -- if one exists -- shows up as a closed loop around the origin
rather than a wandering path). Small multiples, one column per seed, raw
on top and detrended below, so drift-vs-genuine-cycle is a direct
visual comparison.

The line is decimated for rendering (kept at full resolution for the
detrend computation) so ~5000 points per seed don't smear into a solid
blob; default stride keeps ~10 points per estimated ~80-generation
cycle, per the CCF analysis.

Usage:
    python3 scripts/plot_kout_kself_phase_plane.py
    python3 scripts/plot_kout_kself_phase_plane.py --stride 8 --detrend-window 75
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# HBES dataviz palette: a light-cool -> dark-warm gradient for time, so
# rotation direction and lap count read at a glance (a deliberate
# two-hue sequential-for-time encoding, not a diverging/polarity one).
TIME_COLOR_START = "#cde2fb"  # lightest sequential blue (oldest)
TIME_COLOR_END = "#d95926"  # dark-mode categorical orange (newest)
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen")
    p.add_argument("--seed-list", type=str, default="42,1,2,3,4")
    p.add_argument("--detrend-window", type=int, default=75, help="centered moving-average window")
    p.add_argument("--stride", type=int, default=8, help="plot every Nth generation (~1/10 of the ~80-gen period)")
    p.add_argument("--out", type=str, default="scripts/output/kout_kself_phase_plane.png")
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
    """Centered moving-average detrend, 'valid' convolution (no edge
    padding bias); trims to the window-centered range."""
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    x_trimmed = x[offset : offset + len(trend)]
    gens_trimmed = gens[offset : offset + len(trend)]
    return gens_trimmed, x_trimmed - trend


def plot_trajectory(ax, gens, kself, kout, stride, cmap, norm):
    from matplotlib.collections import LineCollection

    g = gens[::stride]
    x = kself[::stride]
    y = kout[::stride]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = cmap(norm(g[:-1]))
    lc = LineCollection(segments, colors=seg_colors, linewidths=0.9, alpha=0.85)
    ax.add_collection(lc)
    ax.scatter([x[0]], [y[0]], s=18, color=cmap(norm(g[0])), zorder=3, edgecolor="white", linewidth=0.4)
    ax.scatter([x[-1]], [y[-1]], s=18, color=cmap(norm(g[-1])), zorder=3, edgecolor="white", linewidth=0.4)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("time", [TIME_COLOR_START, TIME_COLOR_END])

    n = len(seeds)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7.2))

    for col, seed in enumerate(seeds):
        gens, kout, kself = load_series(sweep_dir, seed)
        gens_dt, kout_dt = detrend(gens, kout, args.detrend_window)
        _, kself_dt = detrend(gens, kself, args.detrend_window)

        norm_raw = Normalize(vmin=gens.min(), vmax=gens.max())
        ax_raw = axes[0, col]
        plot_trajectory(ax_raw, gens, kself, kout, args.stride, cmap, norm_raw)
        ax_raw.set_xlim(-0.02, 1.02)
        ax_raw.set_ylim(-0.02, 1.02)
        ax_raw.set_title(f"seed={seed}", fontsize=10)
        ax_raw.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax_raw.set_ylabel("raw\nK_out")
        ax_raw.set_xlabel("K_self")

        norm_dt = Normalize(vmin=gens_dt.min(), vmax=gens_dt.max())
        ax_dt = axes[1, col]
        plot_trajectory(ax_dt, gens_dt, kself_dt, kout_dt, args.stride, cmap, norm_dt)
        lim = max(np.abs(kself_dt).max(), np.abs(kout_dt).max()) * 1.15
        ax_dt.set_xlim(-lim, lim)
        ax_dt.set_ylim(-lim, lim)
        ax_dt.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax_dt.axvline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax_dt.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax_dt.set_ylabel("detrended residual\nK_out")
        ax_dt.set_xlabel("K_self")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08, aspect=40)
    cbar.set_label("generation: old (light/cool) -> new (dark/warm)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["gen 0", "gen 5000"])

    fig.suptitle("(K_self, K_out) phase-plane trajectory per seed: raw vs. detrended", y=0.98)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
