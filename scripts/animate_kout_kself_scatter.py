"""Animate the (K_out, K_self) individual-agent point cloud over
generations, for one seed, using only the already-saved per-individual
logs from run_selection_vs_drift_sweep.py -- no new simulation.

Reads scripts/output/selection_vs_drift_sweep_with_depth/seed_<n>_individual_log.csv
(100 agents x 1 row per logged generation, every 100 gen by default),
and animates one frame per logged generation: a scatter of that
generation's 100 (K_out, K_self) points, axes fixed to [0,1] x [0,1] so
the cloud's movement/shape change is directly comparable frame to
frame, with the generation number annotated on each frame.

Output format: GIF via matplotlib's PillowWriter (no ffmpeg dependency
in this environment; ffmpeg was checked and is not installed here).

Usage:
    python3 scripts/animate_kout_kself_scatter.py
    python3 scripts/animate_kout_kself_scatter.py --seed 7 --fps 4
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SWEEP_DIR = "scripts/output/selection_vs_drift_sweep_with_depth"
POINT_COLOR = "#2a78d6"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def load_snapshots(sweep_dir: Path, seed: int):
    path = sweep_dir / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
    return dict(sorted(by_gen.items()))


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else Path(f"scripts/output/kout_kself_scatter_animation_seed{args.seed}.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_gen = load_snapshots(sweep_dir, args.seed)
    generations = list(by_gen.keys())
    print(f"seed={args.seed}: {len(generations)} snapshots, gen {generations[0]}..{generations[-1]}, "
          f"{len(by_gen[generations[0]]['k_out'])} agents/snapshot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    scatter = ax.scatter([], [], s=28, color=POINT_COLOR, alpha=0.7, edgecolor="white", linewidth=0.4)
    gen_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, fontsize=13, va="top", ha="left")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("K_out")
    ax.set_ylabel("K_self")
    ax.set_title(f"(K_out, K_self) per-agent point cloud over generations, seed={args.seed}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color=MUTED_INK, linewidth=0.5, linestyle=":")
    ax.axvline(0.5, color=MUTED_INK, linewidth=0.5, linestyle=":")
    fig.tight_layout()

    def update(frame_idx):
        gen = generations[frame_idx]
        pts = np.column_stack([by_gen[gen]["k_out"], by_gen[gen]["k_self"]])
        scatter.set_offsets(pts)
        gen_text.set_text(f"generation {gen}")
        return scatter, gen_text

    anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"saved {out_path} ({len(generations)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
