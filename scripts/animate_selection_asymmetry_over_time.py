"""Animation 1 -- the moving version of Figure A: how the K_out/K_self
selection asymmetry develops over generations, not just its final
value.

For every logged generation (every 100 gen, 0..4900) and every seed,
computes Spearman rho(individual K, individual payoff) within that
generation's 100 agents, then takes the MEDIAN across the 50 seeds for
that generation -- one point per generation per channel. Plots K_out
(blue) and K_self (orange) as two lines that grow frame by frame, zero
line marked.

Pure re-analysis of the already-saved individual logs
(scripts/output/selection_vs_drift_sweep_with_depth/seed_<n>_individual_log.csv)
-- no new simulation. Unlike Figure A / analyze_selection_vs_drift_correlation.py
(which restricted to the steady-state region gen>=2000), this covers
the FULL 0..4900 range, since the point here is to see the transient
approach, not just the steady-state value.

Usage:
    python3 scripts/animate_selection_asymmetry_over_time.py --last-frame-only   # gen=4900 still only
    python3 scripts/animate_selection_asymmetry_over_time.py --animate           # full GIF
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
    p.add_argument("--last-frame-only", action="store_true", help="save only the final (gen=4900) still")
    p.add_argument("--animate", action="store_true", help="render the full GIF")
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def load_all_generations(seed: int):
    """Returns {generation: {'k_out':[...], 'k_self':[...], 'payoff':[...]}}."""
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

    print("Loading all 50 seeds' individual logs and computing per-generation correlations...")
    per_seed_by_gen = {seed: load_all_generations(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())

    median_rho = {"K_out": [], "K_self": []}
    for gen in generations:
        for channel_name, key in [("K_out", "k_out"), ("K_self", "k_self")]:
            rhos = []
            for seed in SEEDS:
                g = per_seed_by_gen[seed][gen]
                rho, _ = spearmanr(g[key], g["payoff"])
                rhos.append(rho)
            median_rho[channel_name].append(np.median(rhos))

    print(f"{len(generations)} generations: {generations[0]}..{generations[-1]}")
    print(f"final (gen={generations[-1]}): K_out median rho={median_rho['K_out'][-1]:+.3f}  "
          f"K_self median rho={median_rho['K_self'][-1]:+.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, up_to_idx):
        ax.clear()
        g = generations[: up_to_idx + 1]
        ax.plot(g, median_rho["K_out"][: up_to_idx + 1], color=KOUT_COLOR, linewidth=2.2, label="K_out")
        ax.plot(g, median_rho["K_self"][: up_to_idx + 1], color=KSELF_COLOR, linewidth=2.2, label="K_self")
        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", label="zero (neutral)")
        ax.set_xlim(generations[0], generations[-1])
        ax.set_ylim(-0.15, 0.45)
        ax.set_xlabel("generation")
        ax.set_ylabel("median across 50 seeds of\nSpearman ρ(individual K, individual payoff)")
        ax.set_title(f"Selection asymmetry over time (generation {g[-1]})")
        ax.legend(fontsize=9, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    if args.last_frame_only or not args.animate:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw(ax, len(generations) - 1)
        fig.tight_layout()
        fig_path = OUT_DIR / "animation1_selection_asymmetry_final_frame.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")

    if args.animate:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(8, 5.5))

        def update(frame_idx):
            draw(ax, frame_idx)

        anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
        gif_path = OUT_DIR / "animation1_selection_asymmetry_over_time.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {gif_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
