"""Scheme (A) within-seed vs (B) cross-lineage comparison figures --
preregistration section 6 item 1, descriptive/non-confirmatory. Reads
only the aggregated descriptive CSV produced by
run_markov_confirmatory_analysis.py (no raw probe data needed).

Usage:
    python3 scripts/plot_markov_confirmatory_sampling_comparison.py \\
        --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \\
        --role primary --out-dir scripts/output/figures
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "IPAGothic"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual", "GradualAppeasement", "WSLS"]
SCHEME_LABEL = {"A": "(A) 集団内抽出", "B": "(B) 系統間抽出 [主分析]"}


def load_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean_ignoring_nan(values):
    clean = [v for v in values if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def series_for(rows, opponent, scheme, group_size, key):
    by_age = defaultdict(list)
    for row in rows:
        if row["opponent"] != opponent or row["scheme"] != scheme or int(row["group_size"]) != group_size:
            continue
        by_age[int(row["age"])].append(get(row, key))
    ages = sorted(by_age)
    return ages, [mean_ignoring_nan(by_age[a]) for a in ages]


def plot_opponent(rows, opponent, group_size, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    ax = axes[0]
    for scheme, color in [("A", "tab:orange"), ("B", "tab:blue")]:
        ages, vals = series_for(rows, opponent, scheme, group_size, "mean_n_distinct_genotypes")
        if ages:
            ax.plot(ages, vals, label=SCHEME_LABEL[scheme], color=color)
    ax.set_title(f"{opponent} (group_size={group_size})\ngroup diversity: n_distinct_genotypes")
    ax.legend(fontsize=8)

    ax = axes[1]
    for scheme, color in [("A", "tab:orange"), ("B", "tab:blue")]:
        ages, vals = series_for(rows, opponent, scheme, group_size, "mean_mean_pairwise_jaccard_distance")
        if ages:
            ax.plot(ages, vals, label=SCHEME_LABEL[scheme], color=color)
    ax.set_title("group diversity: mean pairwise 3-gram-set Jaccard distance")
    ax.legend(fontsize=8)

    ax = axes[2]
    for scheme, style in [("A", "--"), ("B", "-")]:
        for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
            ages, vals = series_for(rows, opponent, scheme, group_size, f"mean_corr_{group}")
            if ages:
                ax.plot(ages, vals, style, color=color, label=f"{group}, {SCHEME_LABEL[scheme]}", alpha=0.85)
    ax.set_title("correlation vs human group network (solid=B, dashed=A)")
    ax.set_xlabel("generation")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descriptive-csv", type=str, required=True)
    p.add_argument("--role", type=str, required=True)
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    rows = load_rows(Path(args.descriptive_csv))
    out_dir = Path(args.out_dir)
    for opponent in OPPONENTS:
        plot_opponent(rows, opponent, args.group_size, out_dir / f"markov_confirmatory_sampling_comparison_{args.role}_{opponent}.png")


if __name__ == "__main__":
    main()
