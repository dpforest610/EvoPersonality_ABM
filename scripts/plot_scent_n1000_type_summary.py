"""Addition C (docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md): the heatbugs-
style 4-type summary charts (type share, cumulative payoff-deviation),
computed from ALL 30 production seeds per condition (M0/M1/M2) instead of
the single-seed prototype (heatbugs_style_reference.py). Uses the FIXED
pooled-median thresholds from scripts/compute_scent_n1000_type_thresholds.py
(run that first if scripts/output/scent_spatial_sweep_n1000/analysis/type_thresholds.json
doesn't exist yet).

Exploratory/descriptive only -- no significance tests. Each quantity is
shown as the across-seed mean with a 95% seed-level bootstrap band.

Production individual logs are committed only every 100 generations
(gen>=2000: 2000, 2100, ..., 4900 -- 30 snapshots/seed), so these charts
are necessarily on that same 30-point grid, not a dense per-generation
trace (that needs the Addition-A dense log, once the Mac re-run exists).

Usage:
    python3 scripts/compute_scent_n1000_type_thresholds.py   # if not already run
    python3 scripts/plot_scent_n1000_type_summary.py
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path("scripts/output/scent_spatial_sweep_n1000")
MACHINE = "doeRyanoMacBook-Neo.local"
CONDITIONS = [("none", "M0: no movement"), ("random", "M1: random movement"), ("scent", "M2: scent-following")]
SEEDS = list(range(30))
STEADY_STATE_MIN_GEN = 2000
OUT_DIR = Path("docs/figures/scent_n1000")
THRESHOLDS_JSON = ROOT / "analysis" / "type_thresholds.json"
N_BOOTSTRAP = 2000

TYPES = [
    ("high K_self / high K_out", "#FFFFFF", "#999999"),   # (name, fill, line-safe-on-white)
    ("high K_self / low K_out", "#1E90FF", "#1E90FF"),
    ("low K_self / high K_out", "#32CD32", "#32CD32"),
    ("low K_self / low K_out", "#FF4FD8", "#FF4FD8"),
]


def load_thresholds():
    with open(THRESHOLDS_JSON) as f:
        d = json.load(f)
    return d["threshold_kself"], d["threshold_kout"]


def tcode(k_self, k_out, ts, to):
    hs = k_self >= ts
    ho = k_out >= to
    return np.where(hs & ho, 0, np.where(hs & ~ho, 1, np.where(~hs & ho, 2, 3)))


def seed_trajectory(path, ts, to):
    """Returns (gens, share[n_gens,4] in %, deviation[n_gens,4]) for one
    seed: at each gen>=STEADY_STATE_MIN_GEN snapshot, the 4 types' share
    of the population and (type mean payoff - population mean payoff)."""
    by_gen = {}
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            if gen < STEADY_STATE_MIN_GEN:
                continue
            by_gen.setdefault(gen, []).append((float(row["k_self"]), float(row["k_out"]), float(row["payoff"])))
    gens = sorted(by_gen)
    share = np.zeros((len(gens), 4))
    dev = np.zeros((len(gens), 4))
    for i, g in enumerate(gens):
        arr = np.array(by_gen[g])
        k_self, k_out, payoff = arr[:, 0], arr[:, 1], arr[:, 2]
        t = tcode(k_self, k_out, ts, to)
        pop_mean = payoff.mean()
        for k in range(4):
            m = t == k
            share[i, k] = m.mean() * 100
            dev[i, k] = (payoff[m].mean() - pop_mean) if m.any() else 0.0
    return gens, share, dev


def bootstrap_band(per_seed_arr, n_boot=N_BOOTSTRAP, seed=0):
    """per_seed_arr: (n_seeds, n_gens, 4). Returns (mean, lo, hi), each
    (n_gens, 4): across-seed mean and a 95% seed-level bootstrap band."""
    rng = np.random.default_rng(seed)
    n_seeds = per_seed_arr.shape[0]
    mean = per_seed_arr.mean(axis=0)
    boot_means = np.empty((n_boot,) + mean.shape)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = per_seed_arr[idx].mean(axis=0)
    lo = np.percentile(boot_means, 2.5, axis=0)
    hi = np.percentile(boot_means, 97.5, axis=0)
    return mean, lo, hi


def compute_all():
    ts, to = load_thresholds()
    print(f"using fixed thresholds: K_self>={ts:.7f}, K_out>={to:.7f}")
    out = {}
    for cond, _ in CONDITIONS:
        shares, devs = [], []
        gens_ref = None
        for seed in SEEDS:
            path = ROOT / cond / MACHINE / f"seed_{seed}_individual_log.csv.gz"
            gens, share, dev = seed_trajectory(path, ts, to)
            if gens_ref is None:
                gens_ref = gens
            assert gens == gens_ref, f"{cond} seed {seed}: generation grid mismatch"
            shares.append(share)
            devs.append(dev)
        shares = np.array(shares)          # (30, n_gens, 4)
        cum_devs = np.cumsum(np.array(devs), axis=1)  # cumulative sum over the sampled generations
        share_mean, share_lo, share_hi = bootstrap_band(shares)
        cum_mean, cum_lo, cum_hi = bootstrap_band(cum_devs)
        out[cond] = dict(gens=gens_ref, share_mean=share_mean, share_lo=share_lo, share_hi=share_hi,
                          cum_mean=cum_mean, cum_lo=cum_lo, cum_hi=cum_hi)
        print(f"{cond}: {len(gens_ref)} generations x 30 seeds")
    return out, ts, to


def _style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_figure(data, ts, to, key_mean, key_lo, key_hi, ylabel, title, out_stem, pct=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant, figsize, dpi, fontscale in [("print", (12, 4.6), 300, 1.0), ("slide", (16, 9), 150, 1.35)]:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, facecolor="white")
        for ax, (cond, label) in zip(axes, CONDITIONS):
            d = data[cond]
            gens = d["gens"]
            for ti, (name, fill, line_color) in enumerate(TYPES):
                ax.plot(gens, d[key_mean][:, ti], color=line_color, linewidth=2.0 * fontscale, label=name)
                ax.fill_between(gens, d[key_lo][:, ti], d[key_hi][:, ti], color=line_color, alpha=0.18, linewidth=0)
            ax.set_title(label, fontsize=11 * fontscale)
            ax.set_xlabel("generation", fontsize=9 * fontscale)
            if not pct:
                ax.axhline(0, color="#ccc", linewidth=0.8, zorder=0)
            _style_ax(ax)
            ax.tick_params(labelsize=8 * fontscale)
        axes[0].set_ylabel(ylabel, fontsize=10 * fontscale)
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8 * fontscale)
        fig.suptitle(title, fontsize=13 * fontscale, y=1.02)
        fig.text(0.5, -0.04,
                  f"N=1000, gen>=2000, mean +/- 95% seed-level bootstrap band (n=30 seeds, {N_BOOTSTRAP} resamples).\n"
                  f"Types = fixed median split pooled across all 3 conditions x 30 seeds, gen>=2000 "
                  f"(K_self >= {ts:.4f}, K_out >= {to:.4f}). Exploratory/descriptive -- no significance test.\n"
                  "The population is one continuous cloud, not four natural groups; the split is a forced median cut.",
                  ha="center", fontsize=7.5 * fontscale, color="#333")
        fig.tight_layout()
        out_path = OUT_DIR / f"{out_stem}_{variant}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, ts, to = compute_all()
    plot_figure(data, ts, to, "share_mean", "share_lo", "share_hi",
                ylabel="share of agents (%)", title="Type share vs. generation (N=1000)",
                out_stem="type_share_summary", pct=True)
    plot_figure(data, ts, to, "cum_mean", "cum_lo", "cum_hi",
                ylabel="running sum of payoff above population mean",
                title="Cumulative type payoff advantage vs. generation (N=1000)",
                out_stem="type_cumulative_deviation_summary")


if __name__ == "__main__":
    main()
