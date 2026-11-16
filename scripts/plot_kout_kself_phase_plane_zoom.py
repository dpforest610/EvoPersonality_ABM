"""Short-window zoom-in companion to plot_kout_kself_phase_plane.py.

The full 5000-generation trajectory (~65 laps at the ~80-generation
period found by analyze_kout_kself_ccf.py) is too dense to show
individual loops, rotation direction, or lap count in one static image.
This plots a short window (default: generations 0-800, roughly 10
cycles) per seed, at full time resolution, so the loop shape and
winding direction are actually traceable.

Usage:
    python3 scripts/plot_kout_kself_phase_plane_zoom.py
    python3 scripts/plot_kout_kself_phase_plane_zoom.py --start 2000 --span 800
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIME_COLOR_START = "#cde2fb"
TIME_COLOR_END = "#d95926"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen")
    p.add_argument("--seed-list", type=str, default="42,1,2,3,4")
    p.add_argument("--detrend-window", type=int, default=75)
    p.add_argument("--start", type=int, default=0, help="first generation of the zoom window")
    p.add_argument("--span", type=int, default=800, help="generations covered by the zoom window")
    p.add_argument("--out", type=str, default="scripts/output/kout_kself_phase_plane_zoom.png")
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
    kernel = np.ones(window) / window
    trend = np.convolve(x, kernel, mode="valid")
    offset = (window - 1) // 2
    x_trimmed = x[offset : offset + len(trend)]
    gens_trimmed = gens[offset : offset + len(trend)]
    return gens_trimmed, x_trimmed - trend


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("time", [TIME_COLOR_START, TIME_COLOR_END])

    n = len(seeds)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds):
        gens, kout, kself = load_series(sweep_dir, seed)
        gens_dt, kout_dt = detrend(gens, kout, args.detrend_window)
        _, kself_dt = detrend(gens, kself, args.detrend_window)

        mask = (gens_dt >= args.start) & (gens_dt < args.start + args.span)
        g, x, y = gens_dt[mask], kself_dt[mask], kout_dt[mask]

        norm = Normalize(vmin=g.min(), vmax=g.max())
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=cmap(norm(g[:-1])), linewidths=1.6, alpha=0.95)
        ax.add_collection(lc)
        ax.scatter(x, y, s=10, c=g, cmap=cmap, norm=norm, zorder=3, edgecolor="white", linewidth=0.3)
        ax.scatter([x[0]], [y[0]], s=60, facecolor="none", edgecolor=cmap(norm(g[0])), linewidth=1.5, zorder=4, label="start")
        ax.scatter([x[-1]], [y[-1]], s=60, marker="s", facecolor="none", edgecolor=cmap(norm(g[-1])), linewidth=1.5, zorder=4, label="end")

        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax.axvline(0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax.set_aspect("equal")
        ax.set_title(f"seed={seed}", fontsize=10)
        ax.set_xlabel("K_self (detrended)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("K_out (detrended)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Detrended (K_self, K_out) trajectory, generations {args.start}-{args.start + args.span}"
        " (old=light/cool -> new=dark/warm)",
        y=1.03,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
