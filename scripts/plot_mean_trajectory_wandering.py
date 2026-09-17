"""Visualize the OU-theta asymmetry (K_out theta=0.0093 < K_self
theta=0.0153, from analyze_ou_fit_diagnostic.py) as a shape, not a
number: does each seed's population-mean (K_out, K_self) trajectory
actually wander farther along K_out than along K_self? And in the same
45-degree-rotated coordinates used for the point-cloud diagnostics
(sum=K_out+K_self, difference=K_self-K_out): does the "squarish" raw
wander turn into a rectangle that's long along sum and short along
difference, i.e. does rotating onto the two generative axes make the
asymmetry read straight off the plot's own horizontal/vertical extent
instead of an eyeballed tilt?

For each of the 50 seeds, plots the line connecting (mean_K_out(t),
mean_K_self(t)) for every generation t >= 2000 (the steady-state
region), all 50 seeds overlaid at low alpha, in two coordinate systems:

    raw:      x=mean_K_out,                    y=mean_K_self
    rotated:  x=mean_K_out+mean_K_self (sum),  y=mean_K_self-mean_K_out (difference)

Uses the per-generation AGGREGATE timeseries
(scripts/output/selection_vs_drift_sweep_with_depth/seed_<n>_timeseries.csv,
already saved every generation, not just the individual log's
every-100-gen snapshots) for a smooth, fully-resolved trajectory rather
than only the ~30 sparse points the individual log would give -- pure
re-analysis either way, no new simulation.

Equal aspect ratio is used deliberately in both panels: on unequal
axes, an anisotropic wander shape can look however you scale it, so
equal aspect is the only way either plot's shape actually shows the
real asymmetry rather than an axis-scaling artifact.

Alongside the plots, reports each seed's per-axis wander (std and
range of its own gen>=2000 trajectory, in both coordinate systems) and
the OU-theory prediction for the raw ratio (stationary variance ~
sigma^2 / (2*theta)) as a numeric cross-check on whatever the picture
shows.

Outputs:
    scripts/output/mean_trajectory_wandering/trajectory_wandering_raw.png
    scripts/output/mean_trajectory_wandering/trajectory_wandering_rotated.png
    scripts/output/mean_trajectory_wandering/wander_stats_by_seed.csv

Usage:
    python3 scripts/plot_mean_trajectory_wandering.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
OUT_DIR = Path("scripts/output/mean_trajectory_wandering")
SEEDS = list(range(50))
GEN_MIN = 2000

# from analyze_ou_fit_diagnostic.py (median fit over the earlier 50-seed confirmatory sweep)
OU_THETA = {"K_out": 0.00932, "K_self": 0.01527}
OU_SIGMA = {"K_out": 0.02384, "K_self": 0.02389}

TRAJ_COLOR = "#2a78d6"
MUTED_INK = "#8f8f88"


def load_trajectory(seed: int, gen_min: int):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            if gen < gen_min:
                continue
            gens.append(gen)
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectories = {}
    rotated_trajectories = {}
    for seed in SEEDS:
        gens, kout, kself = load_trajectory(seed, GEN_MIN)
        trajectories[seed] = (kout, kself)
        sum_ = kout + kself
        diff_ = kself - kout
        rotated_trajectories[seed] = (sum_, diff_)
        rows.append(
            {
                "seed": seed,
                "kout_std": kout.std(ddof=1), "kout_range": kout.max() - kout.min(),
                "kself_std": kself.std(ddof=1), "kself_range": kself.max() - kself.min(),
                "sum_std": sum_.std(ddof=1), "sum_range": sum_.max() - sum_.min(),
                "diff_std": diff_.std(ddof=1), "diff_range": diff_.max() - diff_.min(),
            }
        )

    csv_path = OUT_DIR / "wander_stats_by_seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    kout_stds = np.array([r["kout_std"] for r in rows])
    kself_stds = np.array([r["kself_std"] for r in rows])
    kout_ranges = np.array([r["kout_range"] for r in rows])
    kself_ranges = np.array([r["kself_range"] for r in rows])

    print(f"\n=== Wander of the population-mean trajectory, gen>={GEN_MIN}, n={len(SEEDS)} seeds ===")
    print(f"K_out:  std median={np.median(kout_stds):.4f} IQR=[{np.percentile(kout_stds,25):.4f},{np.percentile(kout_stds,75):.4f}]   "
          f"range median={np.median(kout_ranges):.4f}")
    print(f"K_self: std median={np.median(kself_stds):.4f} IQR=[{np.percentile(kself_stds,25):.4f},{np.percentile(kself_stds,75):.4f}]   "
          f"range median={np.median(kself_ranges):.4f}")
    ratio_std = np.median(kout_stds) / np.median(kself_stds)
    ratio_range = np.median(kout_ranges) / np.median(kself_ranges)
    print(f"K_out / K_self ratio: std={ratio_std:.3f}  range={ratio_range:.3f}")

    from scipy.stats import wilcoxon

    stat, p = wilcoxon(kout_stds, kself_stds)
    n_out_bigger = int((kout_stds > kself_stds).sum())
    print(f"paired Wilcoxon (per seed, K_out std vs K_self std): stat={stat:.1f} p={p:.4f}  "
          f"({n_out_bigger}/{len(SEEDS)} seeds have K_out std > K_self std)")

    predicted_var_ratio = (OU_SIGMA["K_out"] ** 2 / OU_THETA["K_out"]) / (OU_SIGMA["K_self"] ** 2 / OU_THETA["K_self"])
    predicted_std_ratio = predicted_var_ratio ** 0.5
    print(f"\nOU-theory prediction (stationary var ~ sigma^2/(2*theta), theta/sigma from the pre-existing OU fit):")
    print(f"  predicted K_out/K_self std ratio = {predicted_std_ratio:.3f}   (observed: {ratio_std:.3f})")

    sum_stds = np.array([r["sum_std"] for r in rows])
    diff_stds = np.array([r["diff_std"] for r in rows])
    sum_ranges = np.array([r["sum_range"] for r in rows])
    diff_ranges = np.array([r["diff_range"] for r in rows])
    print(f"\n=== Rotated (sum, difference) coordinates ===")
    print(f"sum (K_out+K_self):        std median={np.median(sum_stds):.4f} IQR=[{np.percentile(sum_stds,25):.4f},{np.percentile(sum_stds,75):.4f}]   "
          f"range median={np.median(sum_ranges):.4f}")
    print(f"difference (K_self-K_out): std median={np.median(diff_stds):.4f} IQR=[{np.percentile(diff_stds,25):.4f},{np.percentile(diff_stds,75):.4f}]   "
          f"range median={np.median(diff_ranges):.4f}")
    ratio_rot_std = np.median(sum_stds) / np.median(diff_stds)
    print(f"sum / difference ratio: std={ratio_rot_std:.3f}  range={np.median(sum_ranges)/np.median(diff_ranges):.3f}")
    stat_rot, p_rot = wilcoxon(sum_stds, diff_stds)
    n_sum_bigger = int((sum_stds > diff_stds).sum())
    print(f"paired Wilcoxon (per seed, sum std vs difference std): stat={stat_rot:.1f} p={p_rot:.4f}  "
          f"({n_sum_bigger}/{len(SEEDS)} seeds have sum std > difference std)")

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for seed in SEEDS:
        kout, kself = trajectories[seed]
        ax.plot(kout, kself, color=TRAJ_COLOR, linewidth=0.6, alpha=0.25)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("mean K_out")
    ax.set_ylabel("mean K_self")
    ax.set_title(f"Population-mean (K_out, K_self) trajectory, gen>={GEN_MIN}, all {len(SEEDS)} seeds overlaid\n"
                 f"(equal aspect -- shape here is the real wander, not an axis-scaling artifact)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = OUT_DIR / "trajectory_wandering_raw.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for seed in SEEDS:
        sum_, diff_ = rotated_trajectories[seed]
        ax.plot(sum_, diff_, color=TRAJ_COLOR, linewidth=0.6, alpha=0.25)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED_INK, linewidth=0.6, linestyle=":")
    ax.set_xlabel("mean K_out + mean K_self (sum)")
    ax.set_ylabel("mean K_self - mean K_out (difference)")
    ax.set_title(f"Same trajectories, rotated 45°: sum vs. difference, gen>={GEN_MIN}, all {len(SEEDS)} seeds\n"
                 f"(equal aspect; sum/difference std ratio = {ratio_rot_std:.2f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = OUT_DIR / "trajectory_wandering_rotated.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
