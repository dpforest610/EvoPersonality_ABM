"""Descriptive trend figures from run_markov_confirmatory_analysis.py's
output -- preregistration section 6 items 2, 3, 7 (recovery/descent
widest path with human Table 8/15 reference lines, Table 7 j*/largest
SCC/Table 12 cycle-count-sweep, extended-set opponents' trends with no
human comparison). ALL of this is descriptive/non-confirmatory -- see
docs/PREREGISTRATION_markov_structure_evolution.md section 6; the
confirmatory H1-H3 results live in markov_confirmatory_results_<role>.json,
not here.

Reads only the aggregated descriptive CSV (no raw probe data needed),
so this can run anywhere, including this cloud session, once the Mac
has produced results_dir/markov_confirmatory_descriptive_<role>.csv.

Usage:
    python3 scripts/plot_markov_confirmatory_trends.py \\
        --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \\
        --role primary --out-dir scripts/output/figures
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "IPAGothic"

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_recovery import EXPECTED_DESCENT_TFT, EXPECTED_RECOVERY

OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual", "GradualAppeasement", "WSLS"]
# this script's group_size:human_group convention (production default; pass
# --group-size to use a different (test) size if the CSV was built with overrides)
HUMAN_LABEL_OF = {"TFT": "TFT", "Random": "RAN", "Gradual": "GRA"}  # recovery/descent reference only exists for these 3
JACCARD_DEGENERATE_OPPONENTS = {"TFT", "AllD"}
UNDEFINED_FLAG_THRESHOLD = 0.5  # preregistration section 6 item 2: >50% undefined -> "推定不能"


def load_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cycle_threshold_columns(rows):
    cols = sorted(
        (float(c.rsplit("_", 1)[1]) for row in rows for c in row if c.startswith("mean_cycle_count_")),
    )
    return sorted(set(cols))


def mean_jstar_columns(rows):
    return sorted({c for row in rows for c in row if c.startswith("mean_jstar_")})


def get(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def aggregate(rows, opponent, group_size):
    by_age = defaultdict(lambda: defaultdict(list))
    jstar_cols = mean_jstar_columns(rows)
    for row in rows:
        if row["scheme"] != "B" or row["opponent"] != opponent or int(row["group_size"]) != group_size:
            continue
        age = int(row["age"])
        for key in (
            "mean_recovery", "mean_descent", "rate_bottom3_present", "rate_top3_present",
            "rate_recovery_undefined", "rate_descent_undefined", "mean_largest_scc_at_40",
        ):
            by_age[age][key].append(get(row, key))
        jvals = [get(row, c) for c in jstar_cols if row.get(c, "") not in ("", None)]
        jvals = [v for v in jvals if v == v]  # drop NaN
        if jvals:
            by_age[age]["mean_jstar_all"].append(sum(jvals) / len(jvals))
        for group in ("Secure", "Mix", "Fearful"):
            for metric in ("corr", "jaccard"):
                key = f"mean_{metric}_{group}"
                if row.get(key, "") not in ("", None):
                    by_age[age][key].append(get(row, key))
        for t in cycle_threshold_columns(rows):
            key = f"mean_cycle_count_{t}"
            if row.get(key, "") not in ("", None):
                by_age[age][f"cycle_{t}"].append(get(row, key))
    return by_age


def mean_ignoring_nan(values):
    clean = [v for v in values if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def plot_opponent(rows, opponent, group_size, out_path):
    by_age = aggregate(rows, opponent, group_size)
    ages = sorted(by_age)
    if not ages:
        print(f"no scheme-B rows for {opponent} at group_size={group_size}, skipping")
        return

    def series(key):
        return [mean_ignoring_nan(by_age[a][key]) for a in ages]

    cycle_thresholds = cycle_threshold_columns(rows)
    n_rows = 7 + (1 if cycle_thresholds else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows))

    human_label = HUMAN_LABEL_OF.get(opponent)

    ax = axes[0]
    ax.plot(ages, series("mean_recovery"), label="recovery (底底底→和和和)", color="tab:red")
    ax.plot(ages, series("mean_descent"), label="descent (和和和→底底底)", color="tab:blue")
    if human_label:
        for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
            key = (human_label, group)
            if key in EXPECTED_RECOVERY:
                ax.axhline(EXPECTED_RECOVERY[key], color=color, linestyle=":", alpha=0.5, linewidth=1)
        if human_label == "TFT":
            for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
                ax.axhline(EXPECTED_DESCENT_TFT[group], color=color, linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{opponent} (group_size={group_size})\nwidest-path threshold (dotted=human recovery, dashed=human descent[TFT only])")
    ax.legend(fontsize=7)

    ax = axes[1]
    undefined_recovery = series("rate_recovery_undefined")
    undefined_descent = series("rate_descent_undefined")
    ax.plot(ages, undefined_recovery, label="recovery undefined", color="tab:red")
    ax.plot(ages, undefined_descent, label="descent undefined", color="tab:blue")
    ax.axhline(UNDEFINED_FLAG_THRESHOLD, color="black", linestyle="--", linewidth=0.8, label="推定不能 threshold (>50%)")
    flagged_recovery = [a for a, v in zip(ages, undefined_recovery) if v > UNDEFINED_FLAG_THRESHOLD]
    flagged_descent = [a for a, v in zip(ages, undefined_descent) if v > UNDEFINED_FLAG_THRESHOLD]
    for a in flagged_recovery:
        ax.axvspan(a - 10, a + 10, color="tab:red", alpha=0.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"rate of widest-path undefined (shaded = 推定不能, n_flagged_recovery={len(flagged_recovery)}, n_flagged_descent={len(flagged_descent)})")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(ages, series("rate_bottom3_present"), label="底底底 present", color="tab:red")
    ax.plot(ages, series("rate_top3_present"), label="和和和 present", color="tab:green")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("rate of key node presence")
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(ages, series("mean_largest_scc_at_40"), color="tab:purple")
    ax.set_title("largest strongly-connected component (j>=.40, Table 10)")

    ax = axes[4]
    ax.plot(ages, series("mean_jstar_all"), color="tab:orange")
    ax.set_title("mean Table-7 j* across named circuits")

    # correlation is the PRIMARY similarity metric for every opponent (never degenerate)
    ax = axes[5]
    for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
        ax.plot(ages, series(f"mean_corr_{group}"), label=f"corr {group}", color=color)
    ax.legend(fontsize=7)
    ax.set_title("correlation vs human group network (primary similarity metric)")

    # edge-set Jaccard is a SECONDARY/reference metric only -- degenerate (all 3 human
    # groups' lines coincide) for TFT/AllD's small vocabularies (user note, 2026-10-17)
    ax = axes[6]
    if opponent in JACCARD_DEGENERATE_OPPONENTS:
        ax.text(0.5, 0.5, "edge-Jaccard degenerate for this opponent (small\nvocabulary -> identical edge-presence set across\nall 3 human groups at threshold .30); correlation\nabove is the metric to use for this opponent",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
            ax.plot(ages, series(f"mean_jaccard_{group}"), label=f"jaccard {group}", color=color)
        ax.legend(fontsize=7)
    ax.set_title("edge-set Jaccard (>=.30) vs human group network (reference only)")
    ax.set_xlabel("generation")

    if cycle_thresholds:
        ax = axes[7]
        window_avg = [mean_ignoring_nan([v for a in ages for v in by_age[a][f"cycle_{t}"]]) for t in cycle_thresholds]
        ax.plot(cycle_thresholds, window_avg, marker="o", color="tab:brown")
        ax.set_xlabel("Jaccard threshold")
        ax.set_title("Table 12: mean simple-cycle count (len<=6) vs threshold, averaged over all ages shown")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def write_undefined_rate_report(rows, group_size, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "opponent", "age", "rate_bottom3_present", "rate_top3_present",
            "rate_recovery_undefined", "rate_descent_undefined", "flagged_as_推定不能_recovery", "flagged_as_推定不能_descent",
        ])
        for opponent in OPPONENTS:
            by_age = aggregate(rows, opponent, group_size)
            for age in sorted(by_age):
                r_undef = mean_ignoring_nan(by_age[age]["rate_recovery_undefined"])
                d_undef = mean_ignoring_nan(by_age[age]["rate_descent_undefined"])
                w.writerow([
                    opponent, age,
                    f"{mean_ignoring_nan(by_age[age]['rate_bottom3_present']):.4f}",
                    f"{mean_ignoring_nan(by_age[age]['rate_top3_present']):.4f}",
                    f"{r_undef:.4f}", f"{d_undef:.4f}",
                    r_undef > UNDEFINED_FLAG_THRESHOLD, d_undef > UNDEFINED_FLAG_THRESHOLD,
                ])
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descriptive-csv", type=str, required=True)
    p.add_argument("--role", type=str, required=True)
    p.add_argument("--group-size", type=int, default=56, help="which group_size's rows to plot (default 56=Secure's size; recovery/descent/scc/jstar trends are similar across sizes, pick one for the main figure)")
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    rows = load_rows(Path(args.descriptive_csv))
    out_dir = Path(args.out_dir)
    for opponent in OPPONENTS:
        plot_opponent(rows, opponent, args.group_size, out_dir / f"markov_confirmatory_trends_{args.role}_{opponent}.png")
    write_undefined_rate_report(
        rows, args.group_size,
        Path(args.descriptive_csv).parent / f"markov_confirmatory_undefined_rates_{args.role}.csv",
    )


if __name__ == "__main__":
    main()
