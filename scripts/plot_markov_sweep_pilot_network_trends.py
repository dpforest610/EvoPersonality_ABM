"""Generation-on-x-axis trend figures from the pilot's Markov-network
analysis (scripts/analyze_markov_sweep_pilot_network.py's output) --
design doc sections 3.3/7.2, exploratory/pilot-scoped.

One PNG per main-set opponent (TFT/AllC/AllD/Random/Gradual), each with
noise_rate=0.01 and noise_rate=0.0 side by side (columns) and one row
per metric family: widest-path recovery/descent thresholds, rate of
底底底/和和和 presence, rate of widest-path undefined, largest j>=.40
SCC, mean Table-7 j* across the opponent's named circuits, and
similarity (correlation, edge-Jaccard) to each human group.

Usage:
    python3 scripts/plot_markov_sweep_pilot_network_trends.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "IPAGothic"  # renders 底/和/搾/惨 in titles/legends

IN_PATH = Path("scripts/output/markov_sweep_pilot_network.csv")
OUT_DIR = Path("scripts/output/figures")
CONDITIONS = ["noise_rate=0.01", "noise_rate=0.0"]
OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual"]
GROUP_SIZE = 56  # representative group size for the main trend figure (Secure's size)


def load_rows():
    with open(IN_PATH, newline="") as f:
        return list(csv.DictReader(f))


def mean_jstar_columns(rows):
    cols = set()
    for row in rows:
        for k in row:
            if k.startswith("mean_jstar_"):
                cols.add(k)
    return sorted(cols)


def aggregate(rows, opponent, group_size):
    """{age: {metric: (mean_across_seeds, values)}} for one opponent+group_size,
    at the fixed GROUP_SIZE, across all 10 seeds."""
    jstar_cols = mean_jstar_columns(rows)
    by_age = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["opponent"] != opponent or int(row["group_size"]) != group_size:
            continue
        age = int(row["age"])
        by_age[age]["mean_recovery"].append(float(row["mean_recovery"]) if row["mean_recovery"] != "" else None)
        by_age[age]["mean_descent"].append(float(row["mean_descent"]) if row["mean_descent"] != "" else None)
        by_age[age]["rate_bottom3_present"].append(float(row["rate_bottom3_present"]))
        by_age[age]["rate_top3_present"].append(float(row["rate_top3_present"]))
        by_age[age]["rate_recovery_undefined"].append(float(row["rate_recovery_undefined"]))
        by_age[age]["rate_descent_undefined"].append(float(row["rate_descent_undefined"]))
        by_age[age]["mean_largest_scc_at_40"].append(float(row["mean_largest_scc_at_40"]))
        jstar_vals = [float(row[c]) for c in jstar_cols if row.get(c, "") != ""]
        if jstar_vals:
            by_age[age]["mean_jstar_all"].append(statistics.fmean(jstar_vals))
        for group in ("Secure", "Mix", "Fearful"):
            corr_key, jacc_key = f"mean_corr_{group}", f"mean_jaccard_{group}"
            if row.get(corr_key, "") != "":
                by_age[age][corr_key].append(float(row[corr_key]))
            if row.get(jacc_key, "") != "":
                by_age[age][jacc_key].append(float(row[jacc_key]))
    return by_age


def mean_ignoring_none(values):
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else float("nan")


def plot_opponent(rows, opponent, out_path):
    fig, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True)
    for col, condition in enumerate(CONDITIONS):
        cond_rows = [r for r in rows if r["condition"] == condition]
        by_age = aggregate(cond_rows, opponent, GROUP_SIZE)
        ages = sorted(by_age)

        def series(metric):
            return [mean_ignoring_none(by_age[a][metric]) for a in ages]

        ax = axes[0, col]
        ax.plot(ages, series("mean_recovery"), label="recovery (底底底→和和和)", color="tab:red")
        ax.plot(ages, series("mean_descent"), label="descent (和和和→底底底)", color="tab:blue")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{opponent}, {condition}\nwidest-path threshold")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.plot(ages, series("rate_bottom3_present"), label="底底底 present", color="tab:red")
        ax.plot(ages, series("rate_top3_present"), label="和和和 present", color="tab:green")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("rate of key node presence")
        ax.legend(fontsize=7)

        ax = axes[2, col]
        ax.plot(ages, series("rate_recovery_undefined"), label="recovery undefined", color="tab:red")
        ax.plot(ages, series("rate_descent_undefined"), label="descent undefined", color="tab:blue")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("rate of widest-path undefined (None)")
        ax.legend(fontsize=7)

        ax = axes[3, col]
        ax.plot(ages, series("mean_largest_scc_at_40"), color="tab:purple")
        ax.set_title("largest strongly-connected component (j>=.40)")

        ax = axes[4, col]
        ax.plot(ages, series("mean_jstar_all"), color="tab:orange")
        ax.set_title("mean Table-7 j* across named circuits")

        ax = axes[5, col]
        for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
            ax.plot(ages, series(f"mean_corr_{group}"), label=group, color=color)
        ax.set_title("correlation vs human group network")
        ax.legend(fontsize=7)

        ax = axes[6, col]
        for group, color in [("Secure", "tab:blue"), ("Mix", "tab:gray"), ("Fearful", "tab:red")]:
            ax.plot(ages, series(f"mean_jaccard_{group}"), label=group, color=color)
        ax.set_title("edge-set Jaccard (>=.30) vs human group network")
        ax.set_xlabel("generation")
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def write_undefined_rate_report(rows, out_path):
    """Table: condition x opponent x age -> rate_recovery_undefined,
    rate_descent_undefined, rate_bottom3_present, rate_top3_present,
    averaged across seeds at GROUP_SIZE -- the explicit reporting
    requirement from the user's 2026-10-16 request."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "opponent", "age", "rate_bottom3_present", "rate_top3_present",
            "rate_recovery_undefined", "rate_descent_undefined",
        ])
        for condition in CONDITIONS:
            cond_rows = [r for r in rows if r["condition"] == condition]
            for opponent in OPPONENTS:
                by_age = aggregate(cond_rows, opponent, GROUP_SIZE)
                for age in sorted(by_age):
                    w.writerow([
                        condition, opponent, age,
                        f"{mean_ignoring_none(by_age[age]['rate_bottom3_present']):.4f}",
                        f"{mean_ignoring_none(by_age[age]['rate_top3_present']):.4f}",
                        f"{mean_ignoring_none(by_age[age]['rate_recovery_undefined']):.4f}",
                        f"{mean_ignoring_none(by_age[age]['rate_descent_undefined']):.4f}",
                    ])
    print(f"saved {out_path}")


def main():
    rows = load_rows()
    for opponent in OPPONENTS:
        plot_opponent(rows, opponent, OUT_DIR / f"markov_sweep_pilot_network_trends_{opponent}.png")
    write_undefined_rate_report(rows, Path("scripts/output/markov_sweep_pilot_network_undefined_rates.csv"))


if __name__ == "__main__":
    main()
