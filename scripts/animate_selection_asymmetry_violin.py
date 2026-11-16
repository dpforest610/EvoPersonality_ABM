"""Animated version of the violin-over-generations figure: instead of
binning ~500-generation chunks into static side-by-side violins, this
shows ONE generation's actual 50-seed distribution per frame, growing
across the run -- the moving picture of "how much do the seeds agree
at this point in time," to go with Animation 1's moving median line.

For every logged generation (every 100 gen, 0..4900), computes Spearman
rho(individual K, individual payoff) within that generation, once per
seed (n=50) -- same per-(seed,generation) values as
plot_selection_asymmetry_violin_over_generations.py and
animate_selection_asymmetry_over_time.py, just not binned here since
each animation frame already is one point in time. Plots K_out and
K_self as two violins side by side, y-axis fixed across all frames for
comparability, generation number annotated per frame.

Pure re-analysis of the already-saved individual logs -- no new
simulation.

Usage:
    python3 scripts/animate_selection_asymmetry_violin.py --last-frame-only   # gen=4900 still only
    python3 scripts/animate_selection_asymmetry_violin.py --animate           # full GIF
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

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--last-frame-only", action="store_true")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--fps", type=int, default=4)
    return p.parse_args()


def load_all_generations(seed: int):
    path = SWEEP_DIR / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scipy.stats import spearmanr

    print("Loading all 50 seeds' individual logs and computing per-(seed,generation) correlations...")
    per_seed_by_gen = {seed: load_all_generations(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())

    rho_by_gen = {"K_out": [], "K_self": []}  # list (per generation) of arrays (length 50, one per seed)
    for gen in generations:
        for channel_name, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for seed in SEEDS:
                g = per_seed_by_gen[seed][gen]
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            rho_by_gen[channel_name].append(np.array(rhos))

    all_vals = np.concatenate(rho_by_gen["K_out"] + rho_by_gen["K_self"])
    ylim = (np.percentile(all_vals, 0.5) - 0.05, np.percentile(all_vals, 99.5) + 0.05)
    print(f"{len(generations)} generations: {generations[0]}..{generations[-1]}")
    print(f"y-axis fixed to [{ylim[0]:.3f}, {ylim[1]:.3f}] across all frames")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = [0, 1]
    colors = [KOUT_COLOR, KSELF_COLOR]
    labels = ["K_out", "K_self"]

    def draw(ax, frame_idx):
        ax.clear()
        gen = generations[frame_idx]
        data = [rho_by_gen["K_out"][frame_idx], rho_by_gen["K_self"][frame_idx]]
        parts = ax.violinplot(data, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        parts["cmedians"].set_color(MUTED_INK)
        parts["cmedians"].set_linewidth(2)

        rng = np.random.default_rng(0)
        for pos, vals, color in zip(positions, data, colors):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=14, color=color, alpha=0.5, edgecolor="none", zorder=3)

        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*ylim)
        ax.set_ylabel("Spearman ρ(individual K, individual payoff)\n(n=50 seeds, this generation)")
        ax.set_title(f"Selection asymmetry: distribution across 50 seeds (generation {gen})")
        ax.spines[["top", "right"]].set_visible(False)

    if args.last_frame_only or not args.animate:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        draw(ax, len(generations) - 1)
        fig.tight_layout()
        fig_path = OUT_DIR / "animation3_selection_asymmetry_violin_final_frame.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")

    if args.animate:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(6.5, 6))

        def update(frame_idx):
            draw(ax, frame_idx)

        anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
        gif_path = OUT_DIR / "animation3_selection_asymmetry_violin.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {gif_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
