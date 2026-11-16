"""Selection or drift? Correlate individual K_out/K_self with individual
payoff, within each generation, using the per-individual logs from
run_selection_vs_drift_sweep.py (50 seeds x 5000 gen, logged every 100
gen, 100 agents per snapshot).

For each (seed, logged generation) in the steady-state region
(gen >= 2000 by default), computes both Pearson r and Spearman rho
between that snapshot's 100 (K_out, payoff) pairs and 100
(K_self, payoff) pairs. If K were under consistent positive selection,
these correlations should sit reliably above zero across seeds and
generations; if K drifts like a (nearly) neutral trait, they should
scatter around zero with no consistent sign.

Important framing (do not over-read a single number):
  - This is a within-generation COVARIANCE, not a causal estimate --
    the Lindgren table (memory depth) evolves alongside K and could
    confound the K-payoff relationship.
  - A weak correlation does not by itself mean "no selection": if
    selection already narrowed the population's K variance, the
    remaining cross-sectional spread is small and correlations computed
    on it are naturally noisy/attenuated regardless of whether
    selection is or was acting. That's why this script also reports
    each snapshot's std(K) alongside its correlation, and plots one
    against the other, rather than reporting correlation alone.
  - Whether this result is consistent with the pre-departure finding
    that OU sigma (~0.024) is ~5x larger than the naive
    independent-mutation-only prediction (k_mutation_std/sqrt(N) ~
    0.005) is a synthesis question for discussion, not something this
    script decides on its own.

Uses only the existing per-individual logs (scripts/output/
selection_vs_drift_sweep/seed_<n>_individual_log.csv) -- no new
simulation.

Outputs:
    scripts/output/selection_vs_drift_correlation/correlation_by_snapshot.csv
        (one row per seed x generation x channel: pearson_r,
        spearman_rho, std_k, mean_payoff, n)
    scripts/output/selection_vs_drift_correlation/correlation_distributions.png
    scripts/output/selection_vs_drift_correlation/correlation_vs_std_k.png

Usage:
    python3 scripts/analyze_selection_vs_drift_correlation.py
    python3 scripts/analyze_selection_vs_drift_correlation.py --gen-min 2000
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/selection_vs_drift_sweep")
OUT_DIR = Path("scripts/output/selection_vs_drift_correlation")
SEEDS = list(range(50))

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"

# from the pre-departure OU fit / naive-mutation-noise back-of-envelope
K_MUTATION_STD = 0.05
POPULATION_SIZE = 100
NAIVE_SIGMA_PREDICTION = K_MUTATION_STD / (POPULATION_SIZE ** 0.5)
MEASURED_OU_SIGMA = {"K_out": 0.02384, "K_self": 0.02389}  # medians from analyze_ou_fit_diagnostic.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default=str(SWEEP_DIR))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--gen-min", type=int, default=2000, help="steady-state cutoff (inclusive)")
    return p.parse_args()


def load_snapshots(sweep_dir: Path, seed: int, gen_min: int):
    """Returns {generation: {'k_out': [...], 'k_self': [...], 'payoff': [...]}}, ordered by agent_index."""
    path = sweep_dir / f"seed_{seed}_individual_log.csv"
    by_gen = defaultdict(lambda: {"agent_index": [], "k_out": [], "k_self": [], "payoff": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            if gen < gen_min:
                continue
            g = by_gen[gen]
            g["agent_index"].append(int(row["agent_index"]))
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
    return by_gen


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scipy.stats import pearsonr, spearmanr, wilcoxon

    rows = []
    for seed in SEEDS:
        by_gen = load_snapshots(sweep_dir, seed, args.gen_min)
        for gen, g in sorted(by_gen.items()):
            k_out = np.array(g["k_out"])
            k_self = np.array(g["k_self"])
            payoff = np.array(g["payoff"])
            n = len(payoff)

            for channel_name, k_vals in [("K_out", k_out), ("K_self", k_self)]:
                pear_r, _ = pearsonr(k_vals, payoff)
                spear_rho, _ = spearmanr(k_vals, payoff)
                rows.append(
                    {
                        "seed": seed,
                        "generation": gen,
                        "channel": channel_name,
                        "pearson_r": pear_r,
                        "spearman_rho": spear_rho,
                        "std_k": k_vals.std(ddof=1),
                        "mean_payoff": payoff.mean(),
                        "n": n,
                    }
                )

    csv_path = out_dir / "correlation_by_snapshot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows: {len(SEEDS)} seeds x snapshots >= gen{args.gen_min} x 2 channels)")

    print(f"\n=== Correlation(individual K, individual payoff), within-generation, gen>={args.gen_min} ===")
    for channel_name in ["K_out", "K_self"]:
        sub = [r for r in rows if r["channel"] == channel_name]
        pear = np.array([r["pearson_r"] for r in sub])
        spear = np.array([r["spearman_rho"] for r in sub])
        std_k = np.array([r["std_k"] for r in sub])

        pq1, pmed, pq3 = np.percentile(pear, [25, 50, 75])
        sq1, smed, sq3 = np.percentile(spear, [25, 50, 75])
        frac_pos_pear = (pear > 0).mean()
        frac_pos_spear = (spear > 0).mean()

        print(f"\n{channel_name} (n={len(sub)} seed x generation snapshots, {sub[0]['n']} agents each):")
        print(f"  Pearson r:    median={pmed:+.4f}  IQR=[{pq1:+.4f},{pq3:+.4f}]  {frac_pos_pear*100:.1f}% of snapshots positive")
        print(f"  Spearman rho: median={smed:+.4f}  IQR=[{sq1:+.4f},{sq3:+.4f}]  {frac_pos_spear*100:.1f}% of snapshots positive")
        print(f"  std(K) at these snapshots: median={np.median(std_k):.4f}  IQR=[{np.percentile(std_k,25):.4f},{np.percentile(std_k,75):.4f}]")

        # per-seed median correlation -- each seed as one independent replicate
        per_seed_pear = [np.median([r["pearson_r"] for r in sub if r["seed"] == s]) for s in SEEDS]
        per_seed_spear = [np.median([r["spearman_rho"] for r in sub if r["seed"] == s]) for s in SEEDS]
        w_pear, p_pear = wilcoxon(per_seed_pear)
        w_spear, p_spear = wilcoxon(per_seed_spear)
        n_pos_seeds_pear = sum(1 for v in per_seed_pear if v > 0)
        n_pos_seeds_spear = sum(1 for v in per_seed_spear if v > 0)
        print(f"  per-seed median Pearson r: {n_pos_seeds_pear}/{len(SEEDS)} seeds positive; "
              f"Wilcoxon signed-rank vs 0: stat={w_pear:.1f} p={p_pear:.4f}")
        print(f"  per-seed median Spearman rho: {n_pos_seeds_spear}/{len(SEEDS)} seeds positive; "
              f"Wilcoxon signed-rank vs 0: stat={w_spear:.1f} p={p_spear:.4f}")

    print(f"\n=== Cross-check against the pre-departure OU-sigma amplification puzzle ===")
    print(f"naive prediction (independent mutation only, k_mutation_std/sqrt(N)): {NAIVE_SIGMA_PREDICTION:.4f}")
    for channel_name in ["K_out", "K_self"]:
        measured = MEASURED_OU_SIGMA[channel_name]
        print(f"{channel_name}: measured OU sigma = {measured:.4f} ({measured/NAIVE_SIGMA_PREDICTION:.1f}x naive prediction)")
    print("(see this script's docstring: whether the correlation result above is consistent with\n"
          " selection vs. reproductive-sampling drift as the source of that amplification is for\n"
          " discussion, not decided here.)")

    # --- plots ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: histograms of Pearson r and Spearman rho, K_out vs K_self
    fig, (ax_pear, ax_spear) = plt.subplots(1, 2, figsize=(13, 5.5))
    bins = np.linspace(-1, 1, 41)
    for ax, key, title in [(ax_pear, "pearson_r", "Pearson r"), (ax_spear, "spearman_rho", "Spearman ρ")]:
        for channel_name, color in [("K_out", KOUT_COLOR), ("K_self", KSELF_COLOR)]:
            vals = [r[key] for r in rows if r["channel"] == channel_name]
            med = np.median(vals)
            ax.hist(vals, bins=bins, color=color, alpha=0.5, label=f"{channel_name} (median={med:+.3f})")
            ax.axvline(med, color=color, linewidth=1.5, linestyle="-")
        ax.axvline(0, color=MUTED_INK, linewidth=1.2, linestyle="--", label="zero (neutral)")
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{title}(individual K, individual payoff), within generation")
        ax.set_ylabel("count (seed x generation snapshots)")
        ax.set_title(f"{title}: distribution across {len(SEEDS)} seeds x gen>={args.gen_min} snapshots")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Is K under selection (correlation > 0) or drifting near-neutrally (correlation ~ 0)?", y=1.02)
    fig.tight_layout()
    fig_path = out_dir / "correlation_distributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {fig_path}")

    # Figure 2: correlation vs std(K) -- does a narrow snapshot attenuate the correlation?
    fig, (ax_kout, ax_kself) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, channel_name, color in [(ax_kout, "K_out", KOUT_COLOR), (ax_kself, "K_self", KSELF_COLOR)]:
        sub = [r for r in rows if r["channel"] == channel_name]
        std_k = [r["std_k"] for r in sub]
        spear = [r["spearman_rho"] for r in sub]
        ax.scatter(std_k, spear, s=14, color=color, alpha=0.4, edgecolor="none")
        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--")
        ax.set_xlabel(f"std({channel_name}) within that snapshot")
        ax.set_title(f"{channel_name}: correlation vs. cross-sectional spread")
        ax.spines[["top", "right"]].set_visible(False)
    ax_kout.set_ylabel("Spearman ρ(K, payoff), within generation")
    fig.suptitle("Does a narrower K spread attenuate the K-payoff correlation? (caution #2)", y=1.02)
    fig.tight_layout()
    fig_path2 = out_dir / "correlation_vs_std_k.png"
    fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path2}")


if __name__ == "__main__":
    main()
