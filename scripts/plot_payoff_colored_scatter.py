"""Visualize the selection asymmetry (K_self shows a within-generation
positive correlation with payoff, K_out doesn't --
analyze_selection_vs_drift_correlation.py) as a color gradient, not
just a coefficient: does payoff visibly brighten along the K_self axis
while staying flat along K_out?

Pools all 50 seeds' 100 agents at one representative steady-state
generation (default 3000) from the already-saved individual logs
(scripts/output/selection_vs_drift_sweep_with_depth/seed_<n>_individual_log.csv,
this generation is one of the every-100-gen snapshots) -- no new
simulation -- and scatters all 5000 (K_out, K_self) points colored by
that individual's own payoff (viridis, sequential -- payoff is a
magnitude, not a category).

Read: a color gradient running vertically (along K_self) with little
horizontal structure (along K_out) would match "K_self selected,
K_out neutral"; a gradient in both or neither directions would not.

CAUTION -- pooling raw payoff across 50 seeds is a Simpson's-paradox
risk: seeds differ in their overall payoff level and K distribution,
so pooling raw values can dilute or even flip the true within-seed
pattern (this script found exactly that: pooled-raw Spearman rho(K_out,
payoff)=-0.08 and rho(K_self,payoff)=+0.16, both much weaker/different
than the earlier per-seed-then-aggregated medians of ~0.00 and ~+0.31
from analyze_selection_vs_drift_correlation.py). So this script plots
BOTH: the naive pooled-raw-payoff version, and a within-seed-standardized
version (each seed's 100 payoffs converted to a z-score before pooling,
which removes between-seed level/scale differences and isolates the
same within-generation covariance the correlation analysis measured).
The standardized version is the one that actually matches the earlier
correlation numbers -- treat the raw version as illustrative only.

Outputs:
    scripts/output/payoff_colored_scatter/payoff_scatter_gen<G>_raw.png
    scripts/output/payoff_colored_scatter/payoff_scatter_gen<G>_within_seed_standardized.png

Usage:
    python3 scripts/plot_payoff_colored_scatter.py
    python3 scripts/plot_payoff_colored_scatter.py --gen 4000
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_SWEEP_DIR = "scripts/output/selection_vs_drift_sweep_with_depth"
OUT_DIR = Path("scripts/output/payoff_colored_scatter")
SEEDS = list(range(50))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=str, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--gen", type=int, default=3000)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def load_generation(sweep_dir: Path, seed: int, gen: int):
    path = sweep_dir / f"seed_{seed}_individual_log.csv"
    k_out, k_self, payoff = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["generation"]) != gen:
                continue
            k_out.append(float(row["k_out"]))
            k_self.append(float(row["k_self"]))
            payoff.append(float(row["payoff"]))
    return np.array(k_out), np.array(k_self), np.array(payoff)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_kout, all_kself, all_payoff, all_payoff_z = [], [], [], []
    for seed in SEEDS:
        k_out, k_self, payoff = load_generation(sweep_dir, seed, args.gen)
        all_kout.append(k_out)
        all_kself.append(k_self)
        all_payoff.append(payoff)
        all_payoff_z.append((payoff - payoff.mean()) / payoff.std(ddof=1))
    all_kout = np.concatenate(all_kout)
    all_kself = np.concatenate(all_kself)
    all_payoff = np.concatenate(all_payoff)
    all_payoff_z = np.concatenate(all_payoff_z)
    print(f"gen={args.gen}: {len(all_kout)} points ({len(SEEDS)} seeds x 100 agents)")
    print(f"payoff range: [{all_payoff.min():.3f}, {all_payoff.max():.3f}], mean={all_payoff.mean():.3f}")

    from scipy.stats import spearmanr

    rho_out, p_out = spearmanr(all_kout, all_payoff)
    rho_self, p_self = spearmanr(all_kself, all_payoff)
    print(f"pooled-raw Spearman rho(K_out, payoff)  = {rho_out:+.3f}  p={p_out:.2e}")
    print(f"pooled-raw Spearman rho(K_self, payoff) = {rho_self:+.3f}  p={p_self:.2e}")

    rho_out_z, p_out_z = spearmanr(all_kout, all_payoff_z)
    rho_self_z, p_self_z = spearmanr(all_kself, all_payoff_z)
    print(f"within-seed-standardized Spearman rho(K_out, payoff)  = {rho_out_z:+.3f}  p={p_out_z:.2e}")
    print(f"within-seed-standardized Spearman rho(K_self, payoff) = {rho_self_z:+.3f}  p={p_self_z:.2e}")
    print("(compare to analyze_selection_vs_drift_correlation.py's per-seed-then-aggregated "
          "medians: K_out ~+0.02, K_self ~+0.31 -- the within-seed-standardized version here "
          "should track those, the pooled-raw version above should not)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for values, label, suffix, rho_out_show, rho_self_show in [
        (all_payoff, "payoff (raw, pooled across seeds)", "raw", rho_out, rho_self),
        (all_payoff_z, "payoff (z-score within each seed)", "within_seed_standardized", rho_out_z, rho_self_z),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        order = np.argsort(values)  # draw high-payoff points last so they aren't buried
        # robust color scaling: a few extreme low-payoff outliers (e.g. agents that got
        # exploited hard by a defector) otherwise compress the range for everyone else
        vmin, vmax = np.percentile(values, [2, 98])
        sc = ax.scatter(all_kout[order], all_kself[order], c=values[order], s=10, cmap="viridis",
                         alpha=0.65, edgecolor="none", vmin=vmin, vmax=vmax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("K_out")
        ax.set_ylabel("K_self")
        ax.set_title(f"Individual {label} over (K_out, K_self), generation {args.gen}\n"
                     f"({len(SEEDS)} seeds x 100 agents pooled; Spearman ρ(K_out,payoff)={rho_out_show:+.2f}, "
                     f"ρ(K_self,payoff)={rho_self_show:+.2f})")
        ax.spines[["top", "right"]].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)
        fig.tight_layout()
        fig_path = out_dir / f"payoff_scatter_gen{args.gen}_{suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
