"""Simple (K_self, K_out) trajectory plot, colored by generation.

X = K_self, Y = K_out, raw (undetrended) series, one point/line per
generation, for a single seed from the 50-seed sweep. Pure re-analysis
of already-saved data -- no new simulation.

Usage:
    python3 scripts/plot_kout_kself_raw_trajectory.py
    python3 scripts/plot_kout_kself_raw_trajectory.py --seed 7 --stride 4
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
TIME_COLOR_START = "#cde2fb"  # oldest: light/cool
TIME_COLOR_END = "#d95926"  # newest: dark/warm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=8, help="plot every Nth generation")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def load_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(f"scripts/output/kout_kself_raw_trajectory_seed{args.seed}.png")

    gens, kout, kself = load_series(args.seed)
    g = gens[:: args.stride]
    x = kself[:: args.stride]
    y = kout[:: args.stride]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("time", [TIME_COLOR_START, TIME_COLOR_END])
    norm = Normalize(vmin=g.min(), vmax=g.max())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=cmap(norm(g[:-1])), linewidths=0.9, alpha=0.85)
    ax.add_collection(lc)
    ax.scatter([x[0]], [y[0]], s=40, color=cmap(norm(g[0])), zorder=3, edgecolor="white", linewidth=0.6, label=f"gen {g[0]}")
    ax.scatter([x[-1]], [y[-1]], s=40, marker="s", color=cmap(norm(g[-1])), zorder=3, edgecolor="white", linewidth=0.6, label=f"gen {g[-1]}")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("K_self")
    ax.set_ylabel("K_out")
    ax.set_title(f"(K_self, K_out) trajectory over generations, seed={args.seed}")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("generation")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
