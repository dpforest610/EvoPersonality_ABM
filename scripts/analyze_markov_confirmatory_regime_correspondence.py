"""Memory depth / payoff / regime-switch correspondence with the network
metrics -- preregistration section 6 item 4, descriptive/non-confirmatory.

Reads generation_stats.csv (committed, lightweight -- one row per
generation per seed, written by run_lindgren_markov_sweep.py; no raw
probe data needed) and the aggregated descriptive CSV from
run_markov_confirmatory_analysis.py, and overlays: fraction of seeds in
the cooperation regime, mean memory depth, mean payoff, and a network
metric trend (TFT recovery/descent widest-path threshold and mean
correlation to the human groups, by default), all on the same
generation axis -- so the two can be visually compared. No formal
statistical correspondence test is run here (that would need its own
design); this is the descriptive overlay the preregistration asks for.

Regime classification and switch-counting reuse
scripts/analyze_markov_sweep_pilot_transitions.py's definition
(COOP_THRESHOLD=2.0, the midpoint of the PD's defect/cooperate
payoffs) for continuity with the pilot's own regime-switch analysis.

Usage:
    python3 scripts/analyze_markov_confirmatory_regime_correspondence.py \\
        --generation-stats-dir scripts/output/markov_sweep_production_noise01 \\
        --descriptive-csv scripts/output/markov_confirmatory_production_noise01/markov_confirmatory_descriptive_primary.csv \\
        --role primary --seeds 70 --seed-offset 100 --out-dir scripts/output/figures
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

COOP_THRESHOLD = 2.0  # matches scripts/analyze_markov_sweep_pilot_transitions.py


def load_generation_stats(path: Path):
    with open(path, newline="") as f:
        return [(int(row["generation"]), float(row["mean_depth"]), float(row["mean_payoff"])) for row in csv.DictReader(f)]


def regime_switches(rows):
    regimes = ["coop" if p >= COOP_THRESHOLD else "defect" for _, _, p in rows]
    switches = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            switches.append(rows[i][0])
    return regimes, switches


def get(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean_ignoring_nan(values):
    clean = [v for v in values if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def load_network_series(descriptive_csv: Path, opponent: str, group_size: int):
    by_age = defaultdict(lambda: defaultdict(list))
    with open(descriptive_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["opponent"] != opponent or row["scheme"] != "B" or int(row["group_size"]) != group_size:
                continue
            age = int(row["age"])
            for key in ("mean_recovery", "mean_descent", "mean_corr_Secure", "mean_corr_Mix", "mean_corr_Fearful"):
                by_age[age][key].append(get(row, key))
    ages = sorted(by_age)
    return ages, {
        key: [mean_ignoring_nan(by_age[a][key]) for a in ages]
        for key in ("mean_recovery", "mean_descent", "mean_corr_Secure", "mean_corr_Mix", "mean_corr_Fearful")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generation-stats-dir", type=str, required=True)
    p.add_argument("--descriptive-csv", type=str, required=True)
    p.add_argument("--role", type=str, required=True)
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--network-opponent", type=str, default="TFT")
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    gs_dir = Path(args.generation_stats_dir)
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    all_rows = {}
    for seed in seeds:
        path = gs_dir / f"seed_{seed}_generation_stats.csv"
        if path.exists():
            all_rows[seed] = load_generation_stats(path)

    if not all_rows:
        print(f"no generation_stats.csv files found under {gs_dir}", flush=True)
        return

    generations = sorted({g for rows in all_rows.values() for g, _, _ in rows})
    depth_by_gen = defaultdict(list)
    payoff_by_gen = defaultdict(list)
    regime_by_gen_seed = {}
    total_switches = 0
    seeds_with_switch = 0

    for seed, rows in all_rows.items():
        for g, depth, payoff in rows:
            depth_by_gen[g].append(depth)
            payoff_by_gen[g].append(payoff)
        regimes, switches = regime_switches(rows)
        if switches:
            seeds_with_switch += 1
            total_switches += len(switches)
        for (g, _, _), regime in zip(rows, regimes):
            regime_by_gen_seed[(seed, g)] = regime

    frac_coop = [
        sum(1 for seed in all_rows if regime_by_gen_seed.get((seed, g)) == "coop") / len(all_rows)
        for g in generations
    ]
    mean_depth = [mean_ignoring_nan(depth_by_gen[g]) for g in generations]
    mean_payoff = [mean_ignoring_nan(payoff_by_gen[g]) for g in generations]

    print(
        f"{seeds_with_switch}/{len(all_rows)} seeds had >=1 regime switch; "
        f"total switches={total_switches}", flush=True,
    )

    net_ages, net_series = load_network_series(Path(args.descriptive_csv), args.network_opponent, args.group_size)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    ax = axes[0]
    ax.plot(generations, frac_coop, color="tab:green")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"fraction of seeds in cooperation regime (mean payoff >= {COOP_THRESHOLD})\n"
                 f"{seeds_with_switch}/{len(all_rows)} seeds had >=1 switch, total switches={total_switches}")

    ax = axes[1]
    ax.plot(generations, mean_depth, color="tab:blue")
    ax.set_title("mean memory depth across seeds")

    ax = axes[2]
    ax.plot(generations, mean_payoff, color="tab:red")
    ax.axhline(COOP_THRESHOLD, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("mean payoff across seeds")

    ax = axes[3]
    ax.plot(net_ages, net_series["mean_recovery"], label="recovery (底底底→和和和)", color="tab:red")
    ax.plot(net_ages, net_series["mean_descent"], label="descent (和和和→底底底)", color="tab:blue")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{args.network_opponent} widest-path threshold (scheme B, group_size={args.group_size})")
    ax.set_xlabel("generation")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"markov_confirmatory_regime_correspondence_{args.role}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}", flush=True)

    summary_path = out_dir.parent / f"markov_confirmatory_regime_summary_{args.role}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "frac_coop", "mean_depth", "mean_payoff"])
        for g, fc, md, mp in zip(generations, frac_coop, mean_depth, mean_payoff):
            w.writerow([g, f"{fc:.4f}", f"{md:.4f}", f"{mp:.4f}"])
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
