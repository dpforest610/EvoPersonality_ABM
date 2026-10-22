"""Exploratory, outside preregistration (docs/EXPLORATORY_selfplay_recovery.md
section 6.4): decomposes the change, across generations, in the
population-level "recovered within free_rounds" rate (1 - never_recovered_
rate) for each opponent (TFT/SELF/POP/WSLS) into:

  - a WITHIN-group effect: each depth group's own recovery rate changing
    over generations, holding each group's population share fixed.
  - a COMPOSITION effect: the population's share across the 3
    effective-memory-depth groups (depthLow/depthMid/depthHigh) shifting
    over generations, holding each group's own recovery rate fixed.

Uses the exact (no-residual) midpoint/chain decomposition: for every
consecutive pair of probed generations (t_prev, t_cur), with w_g the
group's share of genotype-repeats and r_g its recovery rate,

    R(t) = sum_g w_g(t) * r_g(t)
    R(t_cur) - R(t_prev) = within_step + composition_step   (exactly)
    within_step       = sum_g  mean(w_g(t_prev), w_g(t_cur)) * (r_g(t_cur) - r_g(t_prev))
    composition_step  = sum_g (w_g(t_cur) - w_g(t_prev)) * mean(r_g(t_prev), r_g(t_cur))

Summing within_step/composition_step over every consecutive pair in the
probed window gives an EXACT decomposition of the total change
R(last) - R(first), regardless of whether the trend is monotonic (unlike
a naive endpoints-only decomposition, which the chain sum reduces to
only if the trend happens to be monotonic in both w and r together).

This is purely a DESCRIPTIVE decomposition of population-level averages.
It is NOT a causal claim: a depth group having a different recovery rate
than another is a CORRELATION across evolved individuals (deeper- or
shallower-memory individuals happening to recover differently), not
evidence that memory depth itself causes the recovery difference --
other correlated traits (e.g. which fixed points the lookup table
settles into) could be the real driver. See section 6.4's caveat.

Also plots a comparison figure for SELF: never_recovered_rate by depth
group over generations (3 solid lines), with the pooled (non-depth-
split) TFT never_recovered_rate overlaid as one faint reference line.

Usage:
    python3 scripts/analyze_selfplay_recovery_by_depth.py \\
        --by-depth-csv scripts/output/figures/markov_selfplay_recovery_time_by_depth.csv \\
        --pooled-csv scripts/output/figures/markov_selfplay_recovery_time.csv \\
        --out-dir scripts/output/figures
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "IPAGothic"

OPPONENTS = ["TFT", "SELF", "POP", "WSLS"]
DEPTH_GROUPS = ["depthLow", "depthMid", "depthHigh"]
GROUP_COLOR = {"depthLow": "tab:blue", "depthMid": "tab:orange", "depthHigh": "tab:red"}
GROUP_LABEL_JA = {"depthLow": "深さ浅(depthLow)", "depthMid": "深さ中(depthMid)", "depthHigh": "深さ深(depthHigh)"}


def load_by_depth(path: Path) -> Dict[str, Dict[int, Dict[str, dict]]]:
    """{opponent: {age: {depth_group: row_dict}}}"""
    out: Dict[str, Dict[int, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["opponent"]][int(row["age"])][row["depth_group"]] = row
    return out


def load_pooled(path: Path) -> Dict[str, Dict[int, dict]]:
    """{opponent: {age: row_dict}}"""
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["opponent"]][int(row["age"])] = row
    return out


def weights_and_rates(by_age: Dict[int, Dict[str, dict]], age: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    rows = by_age[age]
    total_n = sum(int(rows[g]["n"]) for g in DEPTH_GROUPS)
    weights = {g: int(rows[g]["n"]) / total_n for g in DEPTH_GROUPS}
    rates = {g: 1 - float(rows[g]["never_recovered_rate"]) for g in DEPTH_GROUPS}
    return weights, rates


def decompose(by_age: Dict[int, Dict[str, dict]]) -> dict:
    """Chained within/composition decomposition of R(last)-R(first) across
    every consecutive pair of probed ages, exact (no residual) per step."""
    ages = sorted(by_age)
    within_total = 0.0
    composition_total = 0.0
    w_prev, r_prev = weights_and_rates(by_age, ages[0])
    for age in ages[1:]:
        w_cur, r_cur = weights_and_rates(by_age, age)
        for g in DEPTH_GROUPS:
            w_mid = (w_prev[g] + w_cur[g]) / 2
            r_mid = (r_prev[g] + r_cur[g]) / 2
            within_total += w_mid * (r_cur[g] - r_prev[g])
            composition_total += (w_cur[g] - w_prev[g]) * r_mid
        w_prev, r_prev = w_cur, r_cur

    def pooled_rate(age: int) -> float:
        w, r = weights_and_rates(by_age, age)
        return sum(w[g] * r[g] for g in DEPTH_GROUPS)

    r_first = pooled_rate(ages[0])
    r_last = pooled_rate(ages[-1])
    total = r_last - r_first

    return {
        "age_first": ages[0], "age_last": ages[-1],
        "recovery_rate_first": r_first, "recovery_rate_last": r_last,
        "total_change": total, "within_total": within_total, "composition_total": composition_total,
        "within_pct_of_total": within_total / total * 100 if total else float("nan"),
        "composition_pct_of_total": composition_total / total * 100 if total else float("nan"),
        "residual_check": total - (within_total + composition_total),  # must be ~0 (float rounding only)
    }


def mean_rate_by_group(by_age: Dict[int, Dict[str, dict]]) -> Dict[str, float]:
    """Simple descriptive summary (not the decomposition): each group's own
    never_recovered_rate averaged over every probed age, for reporting the
    shallow/mid/deep ORDERING (correlational, see module docstring)."""
    out = {}
    for g in DEPTH_GROUPS:
        vals = [float(by_age[age][g]["never_recovered_rate"]) for age in by_age]
        out[g] = sum(vals) / len(vals)
    return out


def print_report(by_depth: Dict[str, Dict[int, Dict[str, dict]]]) -> None:
    for opponent in OPPONENTS:
        d = decompose(by_depth[opponent])
        m = mean_rate_by_group(by_depth[opponent])
        print(f"=== {opponent} ===")
        print(f"  recovery rate (1-never_recovered_rate): gen{d['age_first']}={d['recovery_rate_first']:.4f} -> gen{d['age_last']}={d['recovery_rate_last']:.4f} (change={d['total_change']:+.4f})")
        print(f"  within-group contribution:  {d['within_total']:+.4f}  ({d['within_pct_of_total']:+.1f}% of total change)")
        print(f"  composition contribution:   {d['composition_total']:+.4f}  ({d['composition_pct_of_total']:+.1f}% of total change)")
        print(f"  residual (should be ~0):    {d['residual_check']:+.2e}")
        print(f"  mean never_recovered_rate by group (whole window): " + ", ".join(f"{g}={v:.4f}" for g, v in m.items()))
        print()


def plot_self_vs_tft(by_depth: Dict[str, Dict[int, Dict[str, dict]]], pooled: Dict[str, Dict[int, dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    self_ages = sorted(by_depth["SELF"])
    for g in DEPTH_GROUPS:
        ys = [float(by_depth["SELF"][age][g]["never_recovered_rate"]) for age in self_ages]
        ax.plot(self_ages, ys, label=f"SELF, {GROUP_LABEL_JA[g]}", color=GROUP_COLOR[g], linewidth=2)

    tft_ages = sorted(pooled["TFT"])
    tft_ys = [float(pooled["TFT"][age]["never_recovered_rate"]) for age in tft_ages]
    ax.plot(tft_ages, tft_ys, label="TFT(参考、深さ分けなし)", color="gray", alpha=0.35, linewidth=1.5, linestyle="--")

    ax.set_xlabel("世代")
    ax.set_ylabel("戻らなかった割合(never_recovered_rate)")
    ax.set_title("底からの回復:SELF相手(記憶深さ群別)とTFT相手(参考)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--by-depth-csv", type=str, default="scripts/output/figures/markov_selfplay_recovery_time_by_depth.csv")
    p.add_argument("--pooled-csv", type=str, default="scripts/output/figures/markov_selfplay_recovery_time.csv")
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    by_depth = load_by_depth(Path(args.by_depth_csv))
    pooled = load_pooled(Path(args.pooled_csv))

    print_report(by_depth)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_self_vs_tft(by_depth, pooled, out_dir / "markov_selfplay_recovery_self_vs_tft_by_depth.png")


if __name__ == "__main__":
    main()
