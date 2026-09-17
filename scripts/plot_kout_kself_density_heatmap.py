"""Does an attachment quadrant structure emerge from evolution, as a
distribution -- not a hypothesis? Directly answers the design doc's
(HBES2026_ABM骨子_二成分K.md §5) question of whether the (K_out, K_self)
plane develops 4 peaks (one per quadrant), 2 peaks, or 1 peak (or none).

Pools all 50 seeds' individual logs at every steady-state snapshot
(gen >= 2000, every 100 gen) -- 50 seeds x 30 snapshots x 100 agents =
150,000 (K_out, K_self) points -- and bins them into a 2D density
heatmap over the full [0,1]x[0,1] plane. Overlaid on top: each seed's
own time-average position (mean K_out, mean K_self, averaged over every
generation gen>=2000 from the aggregate timeseries, not just the
individual-log snapshots) -- 50 points showing where each independent
run's population actually settled, on average, over the long run.

Pure re-analysis of already-saved data (scripts/output/
selection_vs_drift_sweep_with_depth/) -- no new simulation.

Also reports the density mass in each of the four quadrants
(K_out/K_self each split at 0.5) as a numeric complement to the
heatmap's visual peak structure.

Outputs:
    scripts/output/kout_kself_density_heatmap/density_heatmap.png
    scripts/output/kout_kself_density_heatmap/quadrant_mass.csv

Usage:
    python3 scripts/plot_kout_kself_density_heatmap.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/selection_vs_drift_sweep_with_depth")
OUT_DIR = Path("scripts/output/kout_kself_density_heatmap")
SEEDS = list(range(50))
GEN_MIN = 2000
BINS = 50

SEED_MEAN_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def load_individual_points(seed: int, gen_min: int):
    path = SWEEP_DIR / f"seed_{seed}_individual_log.csv"
    k_out, k_self = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["generation"]) < gen_min:
                continue
            k_out.append(float(row["k_out"]))
            k_self.append(float(row["k_self"]))
    return np.array(k_out), np.array(k_self)


def load_seed_time_average(seed: int, gen_min: int):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    k_out, k_self = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["generation"]) < gen_min:
                continue
            k_out.append(float(row["mean_k_out"]))
            k_self.append(float(row["mean_k_self"]))
    return float(np.mean(k_out)), float(np.mean(k_self))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_kout, all_kself = [], []
    seed_means = []
    for seed in SEEDS:
        k_out, k_self = load_individual_points(seed, GEN_MIN)
        all_kout.append(k_out)
        all_kself.append(k_self)
        seed_means.append(load_seed_time_average(seed, GEN_MIN))
    all_kout = np.concatenate(all_kout)
    all_kself = np.concatenate(all_kself)
    seed_means = np.array(seed_means)  # (50, 2): columns = mean_k_out, mean_k_self

    print(f"pooled points: {len(all_kout)} ({len(SEEDS)} seeds x snapshots gen>={GEN_MIN} x 100 agents)")

    # --- quadrant mass ---
    quad_rows = []
    for kout_high, kout_label in [(False, "K_out<0.5"), (True, "K_out>=0.5")]:
        for kself_high, kself_label in [(False, "K_self<0.5"), (True, "K_self>=0.5")]:
            kout_mask = (all_kout >= 0.5) if kout_high else (all_kout < 0.5)
            kself_mask = (all_kself >= 0.5) if kself_high else (all_kself < 0.5)
            frac = (kout_mask & kself_mask).mean()
            quad_rows.append({"quadrant": f"{kout_label}, {kself_label}", "fraction_of_points": frac})

    csv_path = OUT_DIR / "quadrant_mass.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quadrant", "fraction_of_points"])
        w.writeheader()
        w.writerows(quad_rows)
    print(f"\nsaved {csv_path}")
    print("=== Quadrant mass (fraction of all pooled points, gen>=2000) ===")
    for r in quad_rows:
        print(f"  {r['quadrant']:<28} {r['fraction_of_points']*100:5.1f}%")

    # --- 2D histogram peak ---
    hist, xedges, yedges = np.histogram2d(all_kout, all_kself, bins=BINS, range=[[0, 1], [0, 1]])
    peak_idx = np.unravel_index(np.argmax(hist), hist.shape)
    peak_kout = (xedges[peak_idx[0]] + xedges[peak_idx[0] + 1]) / 2
    peak_kself = (yedges[peak_idx[1]] + yedges[peak_idx[1] + 1]) / 2
    print(f"\nsingle densest bin: (K_out={peak_kout:.3f}, K_self={peak_kself:.3f}), "
          f"{hist[peak_idx]/len(all_kout)*100:.3f}% of points in that one bin "
          f"(uniform would be {100/BINS**2:.3f}%)")

    print(f"\nseed time-average positions (mean over gen>={GEN_MIN}):")
    print(f"  mean K_out:  median={np.median(seed_means[:,0]):.3f}  IQR=[{np.percentile(seed_means[:,0],25):.3f},{np.percentile(seed_means[:,0],75):.3f}]")
    print(f"  mean K_self: median={np.median(seed_means[:,1]):.3f}  IQR=[{np.percentile(seed_means[:,1],25):.3f},{np.percentile(seed_means[:,1],75):.3f}]")

    # --- plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    h = ax.hist2d(all_kout, all_kself, bins=BINS, range=[[0, 1], [0, 1]], cmap="viridis")
    ax.scatter(seed_means[:, 0], seed_means[:, 1], s=45, color=SEED_MEAN_COLOR, edgecolor="white",
               linewidth=0.8, zorder=5, label=f"seed time-average (n={len(SEEDS)})")
    ax.axhline(0.5, color=MUTED_INK, linewidth=0.6, linestyle=":")
    ax.axvline(0.5, color=MUTED_INK, linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("K_out")
    ax.set_ylabel("K_self")
    ax.set_title(f"Does an attachment quadrant structure emerge?\n"
                 f"({len(SEEDS)} seeds x snapshots gen>={GEN_MIN}, {len(all_kout)} agent-points pooled)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("agent count per bin")
    fig.tight_layout()
    fig_path = OUT_DIR / "density_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
