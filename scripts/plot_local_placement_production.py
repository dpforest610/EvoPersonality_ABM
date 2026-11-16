"""Local-placement production figures: violin plots of beta2 (primary
endpoint), beta_self(K_out), beta_ctx(K_out), and cooperation_rate,
local vs shuffle(replay) side by side per k=4,8,16,32. gen>=2000,
per-seed values, n=50.

Usage:
    python3 scripts/plot_local_placement_production.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path("scripts/output/local_placement_sweep")
SHUFFLE_ROOT = Path("scripts/output/local_placement_sweep_shuffle_replay")
OUT_DIR = Path("scripts/output/figures")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))
STEADY_STATE_MIN_GEN = 2000

LOCAL_COLOR = "#2a78d6"
SHUFFLE_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def load_individual_log(path):
    by_gen = defaultdict(lambda: {"k_out": [], "k_self": [], "payoff": [],
                                   "opp_k_self": [], "opp_k_out": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            g = by_gen[gen]
            g["k_out"].append(float(row["k_out"]))
            g["k_self"].append(float(row["k_self"]))
            g["payoff"].append(float(row["payoff"]))
            g["opp_k_self"].append(float(row["opponent_mean_k_self"]))
            g["opp_k_out"].append(float(row["opponent_mean_k_out"]))
    return by_gen


def own_trait_beta2(k_self, k_out, payoff):
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[2]


def context_betas(k_self, k_out, opp_k_self, opp_k_out, payoff):
    def standardize(v):
        v = np.asarray(v)
        sd = np.std(v)
        return (v - np.mean(v)) / sd if sd > 0 else np.zeros_like(v)
    z1_self, z2_self = standardize(k_self), standardize(k_out)
    z1_ctx, z2_ctx = standardize(opp_k_self), standardize(opp_k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1_self), z1_self, z2_self, z1_ctx, z2_ctx])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[2], coef[4]  # beta_self(K_out), beta_ctx(K_out)


def seed_medians(sweep_dir, seed, min_gen):
    by_gen = load_individual_log(Path(sweep_dir) / f"seed_{seed}_individual_log.csv")
    beta2s, beta_self_kouts, beta_ctx_kouts = [], [], []
    for gen, g in by_gen.items():
        if gen < min_gen:
            continue
        if (len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2
                or len(set(g["opp_k_self"])) < 2 or len(set(g["opp_k_out"])) < 2):
            continue
        beta2s.append(own_trait_beta2(g["k_self"], g["k_out"], g["payoff"]))
        bs, bc = context_betas(g["k_self"], g["k_out"], g["opp_k_self"], g["opp_k_out"], g["payoff"])
        beta_self_kouts.append(bs)
        beta_ctx_kouts.append(bc)
    return dict(
        beta2=np.median(beta2s) if beta2s else np.nan,
        beta_self_kout=np.median(beta_self_kouts) if beta_self_kouts else np.nan,
        beta_ctx_kout=np.median(beta_ctx_kouts) if beta_ctx_kouts else np.nan,
    )


def seed_mean_cooperation(sweep_dir, seed, min_gen):
    rows = list(csv.DictReader(open(Path(sweep_dir) / f"seed_{seed}_timeseries.csv")))
    vals = [float(r["cooperation_rate"]) for r in rows if int(r["generation"]) >= min_gen and r["cooperation_rate"] != ""]
    return np.mean(vals) if vals else np.nan


def main():
    print("computing per-seed values (this reads every individual_log.csv -- may take a minute)...", flush=True)
    data = {}
    for root_label, root in [("local", LOCAL_ROOT), ("shuffle", SHUFFLE_ROOT)]:
        for k in K_VALUES:
            beta2s, beta_selfs, beta_ctxs, coops = [], [], [], []
            for s in SEEDS:
                m = seed_medians(root / f"k{k}", s, STEADY_STATE_MIN_GEN)
                beta2s.append(m["beta2"])
                beta_selfs.append(m["beta_self_kout"])
                beta_ctxs.append(m["beta_ctx_kout"])
                coops.append(seed_mean_cooperation(root / f"k{k}", s, STEADY_STATE_MIN_GEN))
            data[(root_label, k)] = dict(
                beta2=np.array(beta2s), beta_self_kout=np.array(beta_selfs),
                beta_ctx_kout=np.array(beta_ctxs), cooperation_rate=np.array(coops),
            )
    print("done.", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("beta2", "beta2 (own-trait-only, K_out) -- PRIMARY ENDPOINT"),
        ("beta_self_kout", "beta_self(K_out) (context model)"),
        ("beta_ctx_kout", "beta_ctx(K_out) (context model)"),
        ("cooperation_rate", "cooperation rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for ax, (key, title) in zip(axes, panels):
        positions = []
        all_vals = []
        colors = []
        labels = []
        pos = 0
        for k in K_VALUES:
            for cond, color in [("local", LOCAL_COLOR), ("shuffle", SHUFFLE_COLOR)]:
                all_vals.append(data[(cond, k)][key])
                positions.append(pos)
                colors.append(color)
                labels.append(f"{k}\n{cond}")
                pos += 1
            pos += 0.5  # gap between k groups

        parts = ax.violinplot(all_vals, positions=positions, widths=0.8, showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.55)
            pc.set_edgecolor(c)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        for p, vals, c in zip(positions, all_vals, colors):
            jitter = rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), p) + jitter, vals, s=8, color=MUTED_INK, alpha=0.35, zorder=3)

        ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Local-placement production: local vs shuffle(replay), k=4,8,16,32 "
                 "(gen>=2000, per-seed median, n=50)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig_path = OUT_DIR / "local_placement_production_violins.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
