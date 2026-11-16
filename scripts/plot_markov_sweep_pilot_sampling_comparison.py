"""Trend figures comparing sampling schemes (A) within-seed vs (B)
cross-lineage, from scripts/analyze_markov_sweep_pilot_sampling_comparison.py's
output -- requested 2026-10-17, exploratory/pilot-scoped (n=10 only).

One PNG per main-set opponent: columns are the two schemes (A, B), rows
are metric families, each panel overlaying both noise conditions
(0.01 solid, 0.0 dashed).

Usage:
    python3 scripts/plot_markov_sweep_pilot_sampling_comparison.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "IPAGothic"

IN_PATH = Path("scripts/output/markov_sweep_pilot_sampling_comparison.csv")
OUT_DIR = Path("scripts/output/figures")
CONDITIONS = ["noise_rate=0.01", "noise_rate=0.0"]
CONDITION_STYLE = {"noise_rate=0.01": "-", "noise_rate=0.0": "--"}
SCHEMES = ["A_within_seed", "B_cross_seed"]
SCHEME_LABEL = {"A_within_seed": "(A) 集団内抽出", "B_cross_seed": "(B) 系統間抽出"}
OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual"]
# these two opponents' vocabularies are small enough that the edge-Jaccard>=.30
# "present" set is identical across all 3 human groups -- reported as a caveat
# directly on the figure rather than silently showing 3 overlapping lines.
JACCARD_DEGENERATE_OPPONENTS = {"TFT", "AllD"}


def load_rows():
    with open(IN_PATH, newline="") as f:
        return list(csv.DictReader(f))


def get(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", "nan") else float("nan")


def plot_opponent(rows, opponent, out_path):
    fig, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True)
    for col, scheme in enumerate(SCHEMES):
        sub = [r for r in rows if r["opponent"] == opponent and r["scheme"] == scheme]
        by_cond_age = defaultdict(dict)
        for r in sub:
            by_cond_age[r["condition"]][int(r["age"])] = r

        def series(condition, key):
            ages = sorted(by_cond_age[condition])
            return ages, [get(by_cond_age[condition][a], key) for a in ages]

        ax = axes[0, col]
        for condition in CONDITIONS:
            ages, vals = series(condition, "mean_recovery")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:red", label=f"recovery, {condition}")
            ages, vals = series(condition, "mean_descent")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:blue", label=f"descent, {condition}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{opponent}, {SCHEME_LABEL[scheme]}\nwidest-path threshold")
        ax.legend(fontsize=6)

        ax = axes[1, col]
        for condition in CONDITIONS:
            ages, vals = series(condition, "rate_bottom3_present")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:red", label=f"底底底, {condition}")
            ages, vals = series(condition, "rate_top3_present")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:green", label=f"和和和, {condition}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("rate of key node presence")
        ax.legend(fontsize=6)

        ax = axes[2, col]
        for condition in CONDITIONS:
            ages, vals = series(condition, "rate_recovery_undefined")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:red", label=f"recovery undefined, {condition}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("rate of recovery widest-path undefined")
        ax.legend(fontsize=6)

        ax = axes[3, col]
        for condition in CONDITIONS:
            ages, vals = series(condition, "mean_largest_scc_at_40")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:purple", label=condition)
        ax.set_title("largest strongly-connected component (j>=.40)")
        ax.legend(fontsize=6)

        ax = axes[4, col]
        for condition in CONDITIONS:
            for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
                ages, vals = series(condition, f"mean_corr_{group}")
                ax.plot(ages, vals, CONDITION_STYLE[condition], color=color, label=f"{group}, {condition}", alpha=0.8)
        ax.set_title("correlation vs human group network")
        ax.legend(fontsize=5, ncol=2)

        ax = axes[5, col]
        if opponent in JACCARD_DEGENERATE_OPPONENTS:
            ax.text(0.5, 0.5, "edge-Jaccard degenerate for this opponent\n"
                               "(small vocabulary -> same edge-presence set\n"
                               "for all 3 human groups at threshold .30);\n"
                               "use correlation above instead.",
                    ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            for condition in CONDITIONS:
                for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
                    ages, vals = series(condition, f"mean_jaccard_{group}")
                    ax.plot(ages, vals, CONDITION_STYLE[condition], color=color, label=f"{group}, {condition}", alpha=0.8)
            ax.legend(fontsize=5, ncol=2)
        ax.set_title("edge-set Jaccard (>=.30) vs human group network")

        ax = axes[6, col]
        for condition in CONDITIONS:
            ages, vals = series(condition, "mean_n_distinct_genotypes")
            ax.plot(ages, vals, CONDITION_STYLE[condition], color="tab:orange", label=f"n_distinct_genotypes, {condition}")
        ax2 = ax.twinx()
        for condition in CONDITIONS:
            ages, vals = series(condition, "mean_mean_pairwise_jaccard_distance")
            ax2.plot(ages, vals, CONDITION_STYLE[condition], color="tab:cyan", label=f"pairwise Jaccard dist, {condition}")
        ax.set_title("group diversity: distinct genotypes (orange) / pairwise 3-gram-set distance (cyan)")
        ax.set_xlabel("generation")
        ax.legend(fontsize=5, loc="upper left")
        ax2.legend(fontsize=5, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    rows = load_rows()
    for opponent in OPPONENTS:
        plot_opponent(rows, opponent, OUT_DIR / f"markov_sweep_pilot_sampling_comparison_{opponent}.png")


if __name__ == "__main__":
    main()
