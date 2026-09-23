"""Animation 2, redesigned -- centroid trajectory, not the raw 5000-point
scatter (which stays close to uniform at every single generation, since
the basin structure only becomes visible once you pool many generations
-- see the static density heatmap. Animating it would just look like
"nothing changes." The population's average position moving toward the
basin over time is the actual story here, and that story is visible
generation by generation).

For every logged generation (every 100 gen, 0..4900), computes ONE
point: the mean (K_out, K_self) over all 50 seeds x 100 agents pooled
at that generation. Plots the line connecting these 50 points as a
trajectory growing frame by frame, over a faded copy of Figure B's
diagonals and four-corner labels as fixed context.

Misreading guard: the centroid's move from (0.50,0.50) to ~(0.55,0.58)
is small relative to the [0,1]x[0,1] plane, but a lone trajectory line
can still visually read as "the population marches somewhere" if
there's nothing on screen to compare it against. So every frame ALSO
draws that generation's actual spread: the faint pooled 5000-point
scatter, plus a 1-sigma ellipse from the covariance averaged across the
50 seeds' own 100-agent clouds (same method as
analyze_mean_covariance_ellipse.py) centered on that frame's centroid.
The ellipse stays wide and roughly circular for the whole run -- the
same "single broad basin" finding as the static density heatmap, now
visible in the same frame as the centroid's much smaller wander, so the
two readings ("centroid barely moves, near center" / "population stays
spread across a wide single basin") can't be mistaken for each other.

Second panel: a paired violin of the 50 seeds' own mean K_out vs mean
K_self at that generation (from the aggregate timeseries CSV, same
values used by plot_mean_trajectory_wandering.py), with thin lines
connecting each seed's paired (K_out, K_self) point, plus a paired
Wilcoxon signed-rank test on the 50 (mean_K_self - mean_K_out)
differences -- statistic, p-value, and the median difference -- so the
question "are K_out and K_self different at this generation" has an
actual number attached to it, not just a visual impression. Seeds
(n=50), not raw individuals (n=5000), are the test's unit -- consistent
with every other statistical test in this project, since agents within
one seed aren't independent replicates for a population-level question.

Pure re-analysis of the already-saved individual logs and aggregate
timeseries (scripts/output/selection_vs_drift_sweep_with_depth/) -- no
new simulation.

Usage:
    python3 scripts/animate_centroid_trajectory.py --last-frame-only   # gen=4900 still only
    python3 scripts/animate_centroid_trajectory.py --animate           # full GIF
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

TRAJ_COLOR = "#2a78d6"
START_COLOR = "#8f8f88"
END_COLOR = "#eb6834"
DIAG_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"

# from analyze_ou_fit_diagnostic.py / the density-heatmap seed time-averages
BASIN_CENTER = (0.53, 0.61)


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


def load_seed_means(seed: int):
    """Per-generation (mean_k_out, mean_k_self) from the aggregate timeseries CSV."""
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    by_gen = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_gen[int(row["generation"])] = (float(row["mean_k_out"]), float(row["mean_k_self"]))
    return by_gen


def ellipse_params(cov):
    """1-sigma ellipse (width, height, angle_degrees) from a 2x2 covariance matrix --
    same method as analyze_mean_covariance_ellipse.py."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2 * np.sqrt(max(eigvals[0], 0))
    height = 2 * np.sqrt(max(eigvals[1], 0))
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def spread_at(per_seed_by_gen, gen):
    """All pooled points at this generation, plus the mean covariance
    (averaged across the 50 seeds' own 100-agent clouds) for the 1-sigma ellipse."""
    all_kout, all_kself = [], []
    covs = []
    for seed in SEEDS:
        g = per_seed_by_gen[seed][gen]
        k_out = np.array(g["k_out"])
        k_self = np.array(g["k_self"])
        all_kout.append(k_out)
        all_kself.append(k_self)
        covs.append(np.cov(np.vstack([k_out, k_self])))
    mean_cov = np.mean(covs, axis=0)
    return np.concatenate(all_kout), np.concatenate(all_kself), mean_cov


def draw_figure_b_background(ax):
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
    ax.scatter([BASIN_CENTER[0]], [BASIN_CENTER[1]], s=180, facecolor="none", edgecolor=MUTED_INK,
               linewidth=1.4, linestyle=(0, (3, 2)), zorder=2)
    ax.text(BASIN_CENTER[0], BASIN_CENTER[1] - 0.06, "basin center\n(density heatmap)",
            fontsize=7, ha="center", va="top", color=MUTED_INK)


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scipy.stats import wilcoxon

    print("Loading all 50 seeds' individual logs and aggregate timeseries...")
    per_seed_by_gen = {seed: load_all_generations(seed) for seed in SEEDS}
    seed_means_by_gen = {seed: load_seed_means(seed) for seed in SEEDS}
    generations = sorted(next(iter(per_seed_by_gen.values())).keys())
    print(f"{len(generations)} generations: {generations[0]}..{generations[-1]}")

    centroid_kout, centroid_kself = [], []
    spread_by_gen = {}
    # per-generation: 50 seed-mean K_out values, 50 seed-mean K_self values (paired by seed),
    # plus the paired Wilcoxon test on their differences
    paired_by_gen = {}
    for gen in generations:
        kout, kself, mean_cov = spread_at(per_seed_by_gen, gen)
        centroid_kout.append(np.mean(kout))
        centroid_kself.append(np.mean(kself))
        spread_by_gen[gen] = (kout, kself, mean_cov)

        seed_kout = np.array([seed_means_by_gen[seed][gen][0] for seed in SEEDS])
        seed_kself = np.array([seed_means_by_gen[seed][gen][1] for seed in SEEDS])
        diffs = seed_kself - seed_kout
        if np.allclose(diffs, 0):
            stat, pval = np.nan, 1.0
        else:
            stat, pval = wilcoxon(seed_kself, seed_kout)
        paired_by_gen[gen] = (seed_kout, seed_kself, stat, pval, np.median(diffs))

    print(f"paired Wilcoxon (K_self vs K_out, n=50 seed means) at gen={generations[-1]}: "
          f"stat={paired_by_gen[generations[-1]][2]:.1f} p={paired_by_gen[generations[-1]][3]:.4f} "
          f"median diff={paired_by_gen[generations[-1]][4]:+.3f}")

    ellipse_w0, ellipse_h0, _ = ellipse_params(spread_by_gen[generations[0]][2])
    ellipse_wN, ellipse_hN, _ = ellipse_params(spread_by_gen[generations[-1]][2])
    print(f"1-sigma ellipse axes at gen={generations[0]}: {ellipse_w0:.3f} x {ellipse_h0:.3f}")
    print(f"1-sigma ellipse axes at gen={generations[-1]}: {ellipse_wN:.3f} x {ellipse_hN:.3f}  "
          f"(for scale: the centroid moved only {((centroid_kout[-1]-centroid_kout[0])**2+(centroid_kself[-1]-centroid_kself[0])**2)**0.5:.3f} total)")
    print(f"centroid at gen={generations[0]}: (K_out={centroid_kout[0]:.3f}, K_self={centroid_kself[0]:.3f})")
    print(f"centroid at gen={generations[-1]}: (K_out={centroid_kout[-1]:.3f}, K_self={centroid_kself[-1]:.3f})")
    print(f"basin center reference (from density heatmap, gen>=2000 seed time-averages): "
          f"(K_out={BASIN_CENTER[0]}, K_self={BASIN_CENTER[1]})")
    dist_final = ((centroid_kout[-1] - BASIN_CENTER[0]) ** 2 + (centroid_kself[-1] - BASIN_CENTER[1]) ** 2) ** 0.5
    print(f"distance from final centroid to basin center: {dist_final:.4f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.patches import Ellipse

    def draw_trajectory_panel(ax, up_to_idx):
        ax.clear()
        gen = generations[up_to_idx]
        kout_pts, kself_pts, mean_cov = spread_by_gen[gen]

        # this frame's actual spread FIRST (so the centroid's small wander reads
        # against the population's much wider, roughly-constant extent)
        ax.scatter(kout_pts, kself_pts, s=5, color=MUTED_INK, alpha=0.12, edgecolor="none", zorder=2)
        cx, cy = centroid_kout[up_to_idx], centroid_kself[up_to_idx]
        width, height, angle = ellipse_params(mean_cov)
        ell = Ellipse((cx, cy), width, height, angle=angle, facecolor="none",
                      edgecolor=TRAJ_COLOR, linewidth=1.8, alpha=0.7, zorder=3,
                      label="this generation's 1σ spread (avg. per-seed)")
        ax.add_patch(ell)

        draw_figure_b_background(ax)

        x = centroid_kout[: up_to_idx + 1]
        y = centroid_kself[: up_to_idx + 1]
        ax.plot(x, y, color=TRAJ_COLOR, linewidth=2.0, zorder=4, label="centroid trajectory")
        ax.scatter([x[0]], [y[0]], s=60, color=START_COLOR, edgecolor="white", linewidth=1, zorder=5,
                   label=f"generation {generations[0]}")
        ax.scatter([x[-1]], [y[-1]], s=80, color=END_COLOR, edgecolor="white", linewidth=1, zorder=5,
                   label=f"generation {gen}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("K_out")
        ax.set_ylabel("K_self")
        ax.set_title(f"Centroid trajectory + spread (generation {gen})", fontsize=11)
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)

    rng = np.random.default_rng(0)

    def draw_violin_panel(ax, up_to_idx):
        ax.clear()
        gen = generations[up_to_idx]
        seed_kout, seed_kself, stat, pval, med_diff = paired_by_gen[gen]

        parts = ax.violinplot([seed_kout, seed_kself], positions=[0, 1], widths=0.6,
                               showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], [TRAJ_COLOR, END_COLOR]):
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
            pc.set_edgecolor(color)
        parts["cmedians"].set_color(MUTED_INK)
        parts["cmedians"].set_linewidth(2)

        jitter = rng.uniform(-0.06, 0.06, size=len(SEEDS))
        for i in range(len(SEEDS)):
            ax.plot([0 + jitter[i], 1 + jitter[i]], [seed_kout[i], seed_kself[i]],
                    color=MUTED_INK, linewidth=0.5, alpha=0.35, zorder=2)
        ax.scatter(np.zeros(len(SEEDS)) + jitter, seed_kout, s=16, color=TRAJ_COLOR, alpha=0.7, zorder=3)
        ax.scatter(np.ones(len(SEEDS)) + jitter, seed_kself, s=16, color=END_COLOR, alpha=0.7, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["K_out", "K_self"], fontsize=11)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("mean K (per seed, n=50)")
        p_label = "p<0.0001" if pval < 0.0001 else f"p={pval:.4f}"
        ax.set_title(f"Paired seed means (generation {gen})\n"
                     f"Wilcoxon signed-rank: stat={stat:.0f}, {p_label}\n"
                     f"median(K_self−K_out)={med_diff:+.3f}", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)

    def draw(fig, ax_left, ax_right, up_to_idx):
        draw_trajectory_panel(ax_left, up_to_idx)
        draw_violin_panel(ax_right, up_to_idx)

    if args.last_frame_only or not args.animate:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.8))
        draw(fig, ax_left, ax_right, len(generations) - 1)
        fig.tight_layout()
        fig_path = OUT_DIR / "animation2_centroid_trajectory_final_frame.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")

    if args.animate:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.8))

        def update(frame_idx):
            draw(fig, ax_left, ax_right, frame_idx)

        anim = FuncAnimation(fig, update, frames=len(generations), interval=1000 / args.fps, blit=False)
        gif_path = OUT_DIR / "animation2_centroid_trajectory.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"saved {gif_path} ({len(generations)} frames)")


if __name__ == "__main__":
    main()
