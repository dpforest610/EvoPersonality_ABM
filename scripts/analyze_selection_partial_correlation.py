"""Does the K_self-payoff correlation survive controlling for memory
depth, or was it riding on the co-evolving Lindgren table?

analyze_selection_vs_drift_correlation.py found a within-generation
zero-order correlation: K_self correlates positively with payoff
(median Spearman rho +0.31, consistent across seeds), K_out does not
(median +0.02, indistinguishable from zero). But K and memory depth
co-evolve under the same fitness, so that K_self result could be a
confound: if depth alone drives payoff and K_self happens to correlate
with depth, K_self would look selected on without being the actual
target.

This script computes, per (seed, generation) snapshot in the
steady-state region (gen >= 2000 by default), the partial correlation
of K vs. payoff CONTROLLING FOR memory depth -- both Pearson and
Spearman variants:

    partial_corr(K, payoff | depth) = corr(resid(K ~ depth), resid(payoff ~ depth))

(first-order partial correlation via linear-regression residuals; for
the Spearman variant, ranks are taken first, then residualized -- the
standard rank-based partial correlation). Also reports corr(depth, K)
per seed, directly, as the confound-strength check the request asked
for: if K and depth barely correlate, a big shift between the
zero-order and partial correlation would be surprising; if they
correlate strongly, it explains why partialling could matter.

Read:
  - If K_self's positive partial correlation survives (stays clearly
    above zero), memory depth doesn't explain it away -- K_self itself
    looks selected on.
  - If it collapses toward zero once depth is controlled for, the
    zero-order result was riding on depth.

Uses only the existing per-individual logs from
run_selection_vs_drift_sweep.py's depth-augmented output
(scripts/output/selection_vs_drift_sweep_with_depth/) -- no new
simulation beyond what that run already produced.

Outputs:
    scripts/output/selection_partial_correlation/partial_correlation_by_snapshot.csv
    scripts/output/selection_partial_correlation/depth_confound_by_seed.csv
    scripts/output/selection_partial_correlation/partial_vs_zero_order_correlation.png
    scripts/output/selection_partial_correlation/depth_confound_strength.png

Usage:
    python3 scripts/analyze_selection_partial_correlation.py
    python3 scripts/analyze_selection_partial_correlation.py --sweep-dir scripts/output/selection_vs_drift_sweep_with_depth_smoke --gen-min 0
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SWEEP_DIR = "scripts/output/selection_vs_drift_sweep_with_depth"
OUT_DIR = Path("scripts/output/selection_partial_correlation")
SEEDS = list(range(50))

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--gen-min", type=int, default=2000, help="steady-state cutoff (inclusive)")
    p.add_argument("--seeds", type=str, default=None, help="comma-separated seed list override (default: 0..49)")
    return p.parse_args()


def load_snapshots(sweep_dir: Path, seed: int, gen_min: int):
    path = sweep_dir / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "depth": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            if gen < gen_min:
                continue
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["depth"].append(float(row["memory_depth"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def partial_corr(x, y, z, rank_based):
    """First-order partial correlation of x,y controlling for z, via
    linear-regression residuals. If rank_based, ranks are substituted
    for x/y/z first (the standard way to get a Spearman-flavored
    partial correlation)."""
    from scipy.stats import pearsonr, rankdata

    if rank_based:
        x, y, z = rankdata(x), rankdata(y), rankdata(z)

    def residual(a, b):
        # a ~ b: a - (slope*b + intercept)
        if np.std(b) == 0:
            return a - a.mean()
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)

    resid_x = residual(x, z)
    resid_y = residual(y, z)
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return float("nan")
    r, _ = pearsonr(resid_x, resid_y)
    return r


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    from scipy.stats import pearsonr, spearmanr

    snapshot_rows = []
    confound_rows = []

    for seed in seeds:
        by_gen = load_snapshots(sweep_dir, seed, args.gen_min)
        all_depth, all_kout, all_kself = [], [], []

        for gen, g in sorted(by_gen.items()):
            k_out = np.array(g["k_out"])
            k_self = np.array(g["k_self"])
            depth = np.array(g["depth"])
            payoff = np.array(g["payoff"])
            all_depth.extend(depth)
            all_kout.extend(k_out)
            all_kself.extend(k_self)

            for channel_name, k_vals in [("K_out", k_out), ("K_self", k_self)]:
                zero_pear, _ = pearsonr(k_vals, payoff)
                zero_spear, _ = spearmanr(k_vals, payoff)
                part_pear = partial_corr(k_vals, payoff, depth, rank_based=False)
                part_spear = partial_corr(k_vals, payoff, depth, rank_based=True)
                snapshot_rows.append(
                    {
                        "seed": seed,
                        "generation": gen,
                        "channel": channel_name,
                        "zero_order_pearson": zero_pear,
                        "zero_order_spearman": zero_spear,
                        "partial_pearson_given_depth": part_pear,
                        "partial_spearman_given_depth": part_spear,
                    }
                )

        # per-seed confound strength: corr(depth, K) pooled across this seed's steady-state snapshots
        all_depth = np.array(all_depth)
        all_kout = np.array(all_kout)
        all_kself = np.array(all_kself)
        r_depth_kout, _ = pearsonr(all_depth, all_kout)
        rho_depth_kout, _ = spearmanr(all_depth, all_kout)
        r_depth_kself, _ = pearsonr(all_depth, all_kself)
        rho_depth_kself, _ = spearmanr(all_depth, all_kself)
        confound_rows.append(
            {
                "seed": seed,
                "pearson_depth_kout": r_depth_kout,
                "spearman_depth_kout": rho_depth_kout,
                "pearson_depth_kself": r_depth_kself,
                "spearman_depth_kself": rho_depth_kself,
            }
        )

    snap_csv = out_dir / "partial_correlation_by_snapshot.csv"
    with open(snap_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snapshot_rows[0].keys()))
        w.writeheader()
        w.writerows(snapshot_rows)
    print(f"saved {snap_csv} ({len(snapshot_rows)} rows)")

    confound_csv = out_dir / "depth_confound_by_seed.csv"
    with open(confound_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(confound_rows[0].keys()))
        w.writeheader()
        w.writerows(confound_rows)
    print(f"saved {confound_csv} ({len(confound_rows)} rows)")

    from scipy.stats import wilcoxon

    print(f"\n=== Confound strength: corr(memory_depth, K), pooled per seed, gen>={args.gen_min} ===")
    for channel_name, pear_key, spear_key in [
        ("K_out", "pearson_depth_kout", "spearman_depth_kout"),
        ("K_self", "pearson_depth_kself", "spearman_depth_kself"),
    ]:
        pear = np.array([r[pear_key] for r in confound_rows])
        spear = np.array([r[spear_key] for r in confound_rows])
        print(f"{channel_name}: Pearson median={np.median(pear):+.3f} IQR=[{np.percentile(pear,25):+.3f},{np.percentile(pear,75):+.3f}]   "
              f"Spearman median={np.median(spear):+.3f} IQR=[{np.percentile(spear,25):+.3f},{np.percentile(spear,75):+.3f}]")

    print(f"\n=== Zero-order vs. partial (controlling for depth) K-payoff correlation, gen>={args.gen_min} ===")
    for channel_name in ["K_out", "K_self"]:
        sub = [r for r in snapshot_rows if r["channel"] == channel_name]
        for label, key in [
            ("zero-order Pearson", "zero_order_pearson"),
            ("partial Pearson | depth", "partial_pearson_given_depth"),
            ("zero-order Spearman", "zero_order_spearman"),
            ("partial Spearman | depth", "partial_spearman_given_depth"),
        ]:
            vals = np.array([r[key] for r in sub])
            vals = vals[~np.isnan(vals)]
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            frac_pos = (vals > 0).mean()
            print(f"{channel_name:<7} {label:<26} median={med:+.4f}  IQR=[{q1:+.4f},{q3:+.4f}]  {frac_pos*100:5.1f}% positive")

        # per-seed median partial correlation, Wilcoxon vs 0 (Spearman variant, matching the main analysis's primary metric)
        per_seed_partial_spear = [
            np.nanmedian([r["partial_spearman_given_depth"] for r in sub if r["seed"] == s]) for s in seeds
        ]
        per_seed_partial_spear = [v for v in per_seed_partial_spear if not np.isnan(v)]
        w_stat, p_val = wilcoxon(per_seed_partial_spear)
        n_pos = sum(1 for v in per_seed_partial_spear if v > 0)
        print(f"  per-seed median partial Spearman (|depth): {n_pos}/{len(per_seed_partial_spear)} seeds positive; "
              f"Wilcoxon signed-rank vs 0: stat={w_stat:.1f} p={p_val:.4f}")
        print()

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: zero-order vs partial Spearman, distributions, per channel
    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, channel_name, color in [(ax_kout, "K_out", KOUT_COLOR), (ax_kself, "K_self", KSELF_COLOR)]:
        sub = [r for r in snapshot_rows if r["channel"] == channel_name]
        zero_vals = np.array([r["zero_order_spearman"] for r in sub])
        part_vals = np.array([r["partial_spearman_given_depth"] for r in sub])
        part_vals = part_vals[~np.isnan(part_vals)]
        ax.hist(zero_vals, bins=bins, color=color, alpha=0.35, label=f"zero-order (median={np.median(zero_vals):+.3f})")
        ax.hist(part_vals, bins=bins, color=color, alpha=0.85, histtype="step", linewidth=2.2,
                label=f"partial | depth (median={np.median(part_vals):+.3f})")
        ax.axvline(0, color=MUTED_INK, linewidth=1.2, linestyle="--")
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ(K, payoff), within generation")
        ax.set_title(f"{channel_name}: zero-order vs. partial (controlling for memory depth)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("count (seed x generation snapshots)")
    fig.suptitle("Does the K-payoff correlation survive controlling for memory depth?", y=1.02)
    fig.tight_layout()
    fig1_path = out_dir / "partial_vs_zero_order_correlation.png"
    fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig1_path}")

    # Figure 2: confound strength, corr(depth, K) per seed, K_out vs K_self
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rng = np.random.default_rng(0)
    for i, (channel_name, color, key) in enumerate(
        [("K_out", KOUT_COLOR, "spearman_depth_kout"), ("K_self", KSELF_COLOR, "spearman_depth_kself")]
    ):
        vals = np.array([r[key] for r in confound_rows])
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, color=color, alpha=0.6, edgecolor="none")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.errorbar([i], [med], yerr=[[med - q1], [q3 - med]], marker="o", color=color,
                    markersize=10, linewidth=2.2, capsize=5, zorder=5)
    ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["K_out", "K_self"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Spearman ρ(memory_depth, K), per seed")
    ax.set_title(f"How strongly does K correlate with memory depth? (confound strength, gen>={args.gen_min})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig2_path = out_dir / "depth_confound_strength.png"
    fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig2_path}")


if __name__ == "__main__":
    main()
