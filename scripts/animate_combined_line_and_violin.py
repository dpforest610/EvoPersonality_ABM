"""Combined animation: Animation 1's growing median line (left) synced
frame-by-frame with Animation 3's per-generation violin distribution
(right) -- so the median trend and the seed-to-seed spread behind it are
visible together, without any per-frame significance test (no p-value
annotation anywhere; the violin's median tick is a descriptive summary,
not a statistical test).

Produces TWO versions, same as Animation 1's raw/smoothed pair (both
kept, neither replaces the other):

  - RAW: left panel is the raw (unsmoothed) median-rho line, so its
    current-frame value always equals the right panel's violin median
    tick exactly, with no smoothing offset between the two panels.
  - SMOOTHED: left panel is the 5-snapshot-point (~500-generation)
    centered moving average, matching Animation 1's smoothed variant,
    for a presentation-friendlier trend line. Here the line's
    current-frame value will NOT exactly equal the right panel's violin
    median (smoothing pools neighboring generations while the violin
    shows one generation's raw distribution) -- expected, not a bug.

Right panel (both versions): that generation's actual 50-seed
distribution of Spearman rho, K_out vs K_self side by side, y-axis
fixed across all frames (same method as
animate_selection_asymmetry_violin.py).

Pure re-analysis of the already-saved individual logs -- no new
simulation.

Usage:
    python3 scripts/animate_combined_line_and_violin.py --last-frame-only   # gen=4900 stills only
    python3 scripts/animate_combined_line_and_violin.py --animate           # full GIFs (raw + smoothed)
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
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--smooth-window", type=int, default=5,
                   help="centered moving-average window, in snapshot POINTS not raw generations "
                        "(snapshots are spaced 100 gen apart, so window=5 spans ~500 generations)")
    return p.parse_args()


def moving_average(values, window):
    """Centered moving average, 'valid' convolution (matches this project's detrend convention,
    same as animate_selection_asymmetry_over_time.py). Returns (smoothed_values, index_offset)."""
    if window <= 1 or len(values) <= window:
        return list(values), 0
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    offset = (window - 1) // 2
    return list(smoothed), offset


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

    # rho_by_gen[channel][frame_idx] = array of 50 per-seed rho values at that generation
    rho_by_gen = {"K_out": [], "K_self": []}
    median_rho = {"K_out": [], "K_self": []}
    for gen in generations:
        for channel_name, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for seed in SEEDS:
                g = per_seed_by_gen[seed][gen]
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            rhos = np.array(rhos)
            rho_by_gen[channel_name].append(rhos)
            median_rho[channel_name].append(np.median(rhos))

    print(f"{len(generations)} generations: {generations[0]}..{generations[-1]}")
    print(f"final (gen={generations[-1]}): K_out median rho={median_rho['K_out'][-1]:+.3f}  "
          f"K_self median rho={median_rho['K_self'][-1]:+.3f}")

    all_vals = np.concatenate(rho_by_gen["K_out"] + rho_by_gen["K_self"])
    violin_ylim = (np.percentile(all_vals, 0.5) - 0.05, np.percentile(all_vals, 99.5) + 0.05)
    line_ylim = (-0.15, 0.45)
    print(f"violin panel y-axis fixed to [{violin_ylim[0]:.3f}, {violin_ylim[1]:.3f}]")

    smoothed_gen = {}
    smoothed_rho = {}
    for channel in ["K_out", "K_self"]:
        smoothed, offset = moving_average(median_rho[channel], args.smooth_window)
        smoothed_rho[channel] = smoothed
        smoothed_gen[channel] = generations[offset: offset + len(smoothed)]
    print(f"smoothed (window={args.smooth_window} snapshot-points, ~{args.smooth_window*100} generations): "
          f"final K_out rho={smoothed_rho['K_out'][-1]:+.3f}  K_self rho={smoothed_rho['K_self'][-1]:+.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = [0, 1]
    colors = [KOUT_COLOR, KSELF_COLOR]
    labels = ["K_out", "K_self"]

    def draw(ax_line, ax_violin, frame_idx, smoothed: bool):
        gen = generations[frame_idx]

        ax_line.clear()
        if smoothed:
            for channel, color in [("K_out", KOUT_COLOR), ("K_self", KSELF_COLOR)]:
                sg = [x for x in smoothed_gen[channel] if x <= gen]
                sr = smoothed_rho[channel][: len(sg)]
                ax_line.plot(sg, sr, color=color, linewidth=2.6, label=channel)
            line_title = f"Median trend (smoothed, {args.smooth_window}-point moving average)"
        else:
            g = generations[: frame_idx + 1]
            ax_line.plot(g, median_rho["K_out"][: frame_idx + 1], color=KOUT_COLOR, linewidth=2.2, label="K_out")
            ax_line.plot(g, median_rho["K_self"][: frame_idx + 1], color=KSELF_COLOR, linewidth=2.2, label="K_self")
            line_title = "Median trend (raw, unsmoothed)"
        ax_line.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", label="zero (neutral)")
        ax_line.set_xlim(generations[0], generations[-1])
        ax_line.set_ylim(*line_ylim)
        ax_line.set_xlabel("generation")
        ax_line.set_ylabel("median across 50 seeds of\nSpearman ρ(individual K, individual payoff)")
        ax_line.set_title(line_title)
        ax_line.legend(fontsize=8, loc="upper left")
        ax_line.spines[["top", "right"]].set_visible(False)

        ax_violin.clear()
        data = [rho_by_gen["K_out"][frame_idx], rho_by_gen["K_self"][frame_idx]]
        parts = ax_violin.violinplot(data, positions=positions, widths=0.7, showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)
        parts["cmedians"].set_color(MUTED_INK)
        parts["cmedians"].set_linewidth(2)

        rng = np.random.default_rng(0)
        for pos, vals, color in zip(positions, data, colors):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax_violin.scatter(np.full(len(vals), pos) + jitter, vals, s=14, color=color, alpha=0.5,
                               edgecolor="none", zorder=3)

        ax_violin.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(labels, fontsize=12)
        ax_violin.set_xlim(-0.6, 1.6)
        ax_violin.set_ylim(*violin_ylim)
        ax_violin.set_ylabel("Spearman ρ (n=50 seeds, this generation)")
        ax_violin.set_title("Distribution across 50 seeds, this generation")
        ax_violin.spines[["top", "right"]].set_visible(False)

        fig = ax_line.figure
        fig.suptitle(f"Selection asymmetry: trend + spread (generation {gen})", fontsize=13, y=1.02)

    for smoothed, suffix in [(False, ""), (True, "_smoothed")]:
        if args.last_frame_only or not args.animate:
            fig, (ax_line, ax_violin) = plt.subplots(1, 2, figsize=(13, 5.8))
            draw(ax_line, ax_violin, len(generations) - 1, smoothed)
            fig.tight_layout()
            fig_path = OUT_DIR / f"animation4_combined_line_and_violin_final_frame{suffix}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"saved {fig_path}")

        if args.animate:
            from matplotlib.animation import FuncAnimation, PillowWriter

            fig, (ax_line, ax_violin) = plt.subplots(1, 2, figsize=(13, 5.8))

            def update(frame_idx, smoothed=smoothed):
                draw(ax_line, ax_violin, frame_idx, smoothed)

            anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
            gif_path = OUT_DIR / f"animation4_combined_line_and_violin{suffix}.gif"
            anim.save(str(gif_path), writer=PillowWriter(fps=args.fps))
            plt.close(fig)
            print(f"saved {gif_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
