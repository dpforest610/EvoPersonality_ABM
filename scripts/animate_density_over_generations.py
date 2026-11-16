"""Animation 2 -- the moving version of the density-heatmap result,
on top of a faded copy of Figure B's structure (generative diagonals +
four-corner dynamical labels).

For every logged generation (every 100 gen, 0..4900), scatters that
generation's pooled 50 seeds x 100 agents = 5000 (K_out, K_self) points.
Axes fixed to [0,1]x[0,1], generation number annotated per frame.

Pure re-analysis of the already-saved individual logs
(scripts/output/selection_vs_drift_sweep_with_depth/seed_<n>_individual_log.csv)
-- no new simulation.

Usage:
    python3 scripts/animate_density_over_generations.py --last-frame-only   # gen=4900 still only
    python3 scripts/animate_density_over_generations.py --animate           # full GIF
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
OUT_DIR = Path("scripts/output/figures")
SEEDS = list(range(50))

POINT_COLOR = "#2a78d6"
DIAG_COLOR = "#eb6834"
CORNER_COLOR = "#1baf7a"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--last-frame-only", action="store_true")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def load_all_generations(seed: int):
    path = SWEEP_DIR / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
    return by_gen


def draw_figure_b_background(ax):
    """Faded copy of Figure B's diagonals + corner labels, as fixed context."""
    ax.plot([0, 1], [0, 1], color=DIAG_COLOR, linewidth=1.6, alpha=0.35, zorder=1)
    ax.plot([0, 1], [1, 0], color=DIAG_COLOR, linewidth=1.6, alpha=0.35, zorder=1)
    corner_labels = [
        (0.03, 0.03, "LOW/LOW\n(\"fearful\"-like)", "left", "bottom"),
        (0.97, 0.03, "HIGH out, LOW self\n(\"preoccupied\"-like)", "right", "bottom"),
        (0.03, 0.97, "LOW out, HIGH self\n(\"dismissive\"-like)", "left", "top"),
        (0.97, 0.97, "HIGH/HIGH\n(\"secure\"-like)", "right", "top"),
    ]
    for x, y, label, ha, va in corner_labels:
        ax.text(x, y, label, fontsize=7, ha=ha, va=va, color=MUTED_INK, alpha=0.75, zorder=1)


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all 50 seeds' individual logs...")
    per_seed_by_gen = {seed: load_all_generations(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())
    print(f"{len(generations)} generations: {generations[0]}..{generations[-1]}")

    def points_at(gen):
        kout, kself = [], []
        for seed in SEEDS:
            g = per_seed_by_gen[seed][gen]
            kout.extend(g["k_out"])
            kself.extend(g["k_self"])
        return np.array(kout), np.array(kself)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, gen):
        ax.clear()
        draw_figure_b_background(ax)
        kout, kself = points_at(gen)
        ax.scatter(kout, kself, s=6, color=POINT_COLOR, alpha=0.25, edgecolor="none", zorder=3)
        ax.axhline(0.5, color=MUTED_INK, linewidth=0.5, linestyle=":", zorder=1)
        ax.axvline(0.5, color=MUTED_INK, linewidth=0.5, linestyle=":", zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("K_out")
        ax.set_ylabel("K_self")
        ax.set_title(f"Population distribution over generations (generation {gen})")
        ax.spines[["top", "right"]].set_visible(False)

    if args.last_frame_only or not args.animate:
        fig, ax = plt.subplots(figsize=(7, 6.8))
        draw(ax, generations[-1])
        fig.tight_layout()
        fig_path = OUT_DIR / "animation2_density_final_frame.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")

    if args.animate:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(7, 6.8))

        def update(frame_idx):
            draw(ax, generations[frame_idx])

        anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
        gif_path = OUT_DIR / "animation2_density_over_generations.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {gif_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
