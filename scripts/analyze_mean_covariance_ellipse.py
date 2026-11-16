"""Shape statistics of the (K_out, K_self) point cloud, averaged across
all 50 seeds, instead of eyeballing one seed's wandering cloud.

Pure re-analysis of the already-saved individual logs from
run_selection_vs_drift_sweep.py (scripts/output/
selection_vs_drift_sweep_with_depth/seed_<n>_individual_log.csv) -- no
new simulation.

For each logged generation: for each of the 50 seeds, compute the 2x2
covariance matrix of that seed's 100-agent point cloud, then average
the 50 covariance matrices elementwise. The resulting "mean covariance"
describes the typical shape/orientation of a single seed's cloud (not
the spread of seed-level means), plotted as a 1-sigma ellipse (semi-axis
lengths = sqrt(eigenvalues) of the averaged covariance, along its
eigenvectors) centered at the grand mean position across all seeds and
agents. The raw pooled points (50 seeds x 100 agents, semi-transparent)
are drawn underneath for context.

Two coordinate systems:
  task 1 (raw):      x=K_out,               y=K_self
  task 2 (rotated):  x=K_out+K_self (sum),  y=K_self-K_out (difference)

If the cloud is tightly confined in one direction and free in the
other, task 2's axis-aligned sum/difference view should show that as a
vertical or horizontal squeeze rather than a tilted ellipse -- this
script reports the ellipse's tilt angle explicitly for both coordinate
systems so that read doesn't rely on eyeballing alone.

Outputs:
    scripts/output/mean_covariance_ellipse/ellipse_raw_gen<G>.png       (task 1 stills)
    scripts/output/mean_covariance_ellipse/ellipse_rotated_gen<G>.png   (task 2 stills)
    scripts/output/mean_covariance_ellipse/ellipse_params_by_generation.csv
    scripts/output/mean_covariance_ellipse/ellipse_animation_raw.gif       (--animate)
    scripts/output/mean_covariance_ellipse/ellipse_animation_rotated.gif  (--animate)

Usage:
    python3 scripts/analyze_mean_covariance_ellipse.py                     # stills only, gen 2000/3000/4000
    python3 scripts/analyze_mean_covariance_ellipse.py --stills 1000,2500  # different still generations
    python3 scripts/analyze_mean_covariance_ellipse.py --animate           # also render both GIFs
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SWEEP_DIR = "scripts/output/selection_vs_drift_sweep_with_depth"
OUT_DIR = Path("scripts/output/mean_covariance_ellipse")
SEEDS = list(range(50))

POINT_COLOR = "#8f8f88"
ELLIPSE_COLOR = "#2a78d6"
CENTER_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--stills", type=str, default="2000,3000,4000")
    p.add_argument("--animate", action="store_true", help="also render the full GIF animations")
    p.add_argument("--fps", type=int, default=6)
    return p.parse_args()


def load_all_seeds(sweep_dir: Path, seeds):
    """Returns {generation: {seed: (k_out_array, k_self_array)}}."""
    by_gen = defaultdict(dict)
    for seed in seeds:
        path = sweep_dir / f"seed_{seed}_individual_log.csv"
        per_gen = defaultdict(lambda: {"k_out": [], "k_self": []})
        with open(path) as f:
            for row in csv.DictReader(f):
                gen = int(row["generation"])
                g = per_gen[gen]
                g["k_out"].append(float(row["k_out"]))
                g["k_self"].append(float(row["k_self"]))
        for gen, g in per_gen.items():
            by_gen[gen][seed] = (np.array(g["k_out"]), np.array(g["k_self"]))
    return dict(sorted(by_gen.items()))


def to_rotated(k_out, k_self):
    """x=sum, y=difference."""
    return k_out + k_self, k_self - k_out


def mean_covariance_and_center(per_seed_xy, seeds):
    """per_seed_xy: {seed: (x_array, y_array)}. Returns (mean_cov 2x2,
    center (cx,cy), all_x, all_y) pooling all seeds' points."""
    covs = []
    all_x, all_y = [], []
    for seed in seeds:
        x, y = per_seed_xy[seed]
        cov = np.cov(np.vstack([x, y]))  # ddof=1 by default
        covs.append(cov)
        all_x.append(x)
        all_y.append(y)
    mean_cov = np.mean(covs, axis=0)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    center = (all_x.mean(), all_y.mean())
    return mean_cov, center, all_x, all_y


def ellipse_params(cov):
    """Returns (width, height, angle_degrees) for a 1-sigma ellipse:
    semi-axis lengths = sqrt(eigenvalues), full width/height = 2x that."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2 * np.sqrt(max(eigvals[0], 0))
    height = 2 * np.sqrt(max(eigvals[1], 0))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def draw_frame(ax, all_x, all_y, center, cov, xlim, ylim, xlabel, ylabel, gen, title):
    from matplotlib.patches import Ellipse

    ax.clear()
    ax.scatter(all_x, all_y, s=4, color=POINT_COLOR, alpha=0.15, edgecolor="none")
    width, height, angle = ellipse_params(cov)
    ell = Ellipse(center, width, height, angle=angle, facecolor="none", edgecolor=ELLIPSE_COLOR, linewidth=2.5)
    ax.add_patch(ell)
    ax.scatter([center[0]], [center[1]], s=50, color=CENTER_COLOR, zorder=5, marker="+")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\ngeneration {gen}  (ellipse tilt={angle:+.1f}°, axes={width:.3f}x{height:.3f})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal" if xlim[1] - xlim[0] == ylim[1] - ylim[0] else "auto")
    return width, height, angle


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    still_gens = [int(g) for g in args.stills.split(",")]

    print("Loading all 50 seeds' individual logs...")
    by_gen = load_all_seeds(sweep_dir, SEEDS)
    generations = list(by_gen.keys())
    print(f"{len(generations)} logged generations: {generations[0]}..{generations[-1]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    raw_frames = {}
    rot_frames = {}

    for gen in generations:
        per_seed_raw = by_gen[gen]
        raw_cov, raw_center, raw_x, raw_y = mean_covariance_and_center(per_seed_raw, SEEDS)

        per_seed_rot = {s: to_rotated(*per_seed_raw[s]) for s in SEEDS}
        rot_cov, rot_center, rot_x, rot_y = mean_covariance_and_center(per_seed_rot, SEEDS)

        raw_frames[gen] = (raw_x, raw_y, raw_center, raw_cov)
        rot_frames[gen] = (rot_x, rot_y, rot_center, rot_cov)

        rw, rh, ra = ellipse_params(raw_cov)
        tw, th, ta = ellipse_params(rot_cov)
        rows.append(
            {
                "generation": gen,
                "raw_center_kout": raw_center[0], "raw_center_kself": raw_center[1],
                "raw_ellipse_width": rw, "raw_ellipse_height": rh, "raw_ellipse_angle_deg": ra,
                "raw_cov_kout_kout": raw_cov[0, 0], "raw_cov_kself_kself": raw_cov[1, 1], "raw_cov_kout_kself": raw_cov[0, 1],
                "rot_center_sum": rot_center[0], "rot_center_diff": rot_center[1],
                "rot_ellipse_width": tw, "rot_ellipse_height": th, "rot_ellipse_angle_deg": ta,
                "rot_cov_sum_sum": rot_cov[0, 0], "rot_cov_diff_diff": rot_cov[1, 1], "rot_cov_sum_diff": rot_cov[0, 1],
            }
        )

    csv_path = out_dir / "ellipse_params_by_generation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    print("\n=== Mean-covariance ellipse params at the requested still generations ===")
    for r in rows:
        if r["generation"] in still_gens:
            print(f"gen={r['generation']}:")
            print(f"  raw (K_out, K_self):   center=({r['raw_center_kout']:.3f},{r['raw_center_kself']:.3f})  "
                  f"axes={r['raw_ellipse_width']:.4f}x{r['raw_ellipse_height']:.4f}  angle={r['raw_ellipse_angle_deg']:+.1f}°  "
                  f"cov(K_out,K_self)={r['raw_cov_kout_kself']:+.5f}")
            print(f"  rotated (sum, diff):   center=({r['rot_center_sum']:.3f},{r['rot_center_diff']:.3f})  "
                  f"axes={r['rot_ellipse_width']:.4f}x{r['rot_ellipse_height']:.4f}  angle={r['rot_ellipse_angle_deg']:+.1f}°  "
                  f"cov(sum,diff)={r['rot_cov_sum_diff']:+.5f}")

    # --- static images ---
    for gen in still_gens:
        if gen not in raw_frames:
            print(f"WARNING: generation {gen} not a logged snapshot, skipping")
            continue
        x, y, center, cov = raw_frames[gen]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        draw_frame(ax, x, y, center, cov, (-0.02, 1.02), (-0.02, 1.02), "K_out", "K_self", gen,
                   "Task 1: mean covariance ellipse, raw (K_out, K_self)")
        fig.tight_layout()
        p = out_dir / f"ellipse_raw_gen{gen}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}")

        x, y, center, cov = rot_frames[gen]
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        draw_frame(ax, x, y, center, cov, (-0.05, 2.05), (-1.05, 1.05), "K_out + K_self (sum)", "K_self - K_out (difference)", gen,
                   "Task 2: mean covariance ellipse, rotated (sum, difference)")
        ax.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle=":")
        fig.tight_layout()
        p = out_dir / f"ellipse_rotated_gen{gen}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}")

    if not args.animate:
        return

    # --- animations ---
    from matplotlib.animation import FuncAnimation, PillowWriter

    for name, frames, xlim, ylim, xlabel, ylabel, title in [
        ("raw", raw_frames, (-0.02, 1.02), (-0.02, 1.02), "K_out", "K_self", "Task 1: mean covariance ellipse, raw (K_out, K_self)"),
        ("rotated", rot_frames, (-0.05, 2.05), (-1.05, 1.05), "K_out + K_self (sum)", "K_self - K_out (difference)", "Task 2: mean covariance ellipse, rotated (sum, difference)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6.5))

        def update(frame_idx, frames=frames, xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel, title=title):
            gen = generations[frame_idx]
            x, y, center, cov = frames[gen]
            draw_frame(ax, x, y, center, cov, xlim, ylim, xlabel, ylabel, gen, title)
            if name == "rotated":
                ax.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle=":")

        anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
        out_path = out_dir / f"ellipse_animation_{name}.gif"
        anim.save(str(out_path), writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {out_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
