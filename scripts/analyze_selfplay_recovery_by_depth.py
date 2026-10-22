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

Also plots two kinds of figures:

  - plot_self_vs_tft (UNCHANGED from before, same output filename
    markov_selfplay_recovery_self_vs_tft_by_depth.png): never_recovered_
    rate by depth group for SELF, with the pooled TFT reference line --
    kept exactly as-is for continuity with earlier results.

  - plot_academic_figure (NEW, section 6.5), one per focal opponent
    (SELF/POP/WSLS), to NEW filenames
    (markov_selfplay_recovery_rate_by_depth_academic_<OPPONENT>.png):
    recovered rate (1 - never_recovered_rate) by depth group, each line
    with a shaded seed-level bootstrap 95% band (resampling the 70 seeds
    WITH REPLACEMENT --n-boot times and recomputing the pooled rate each
    draw, same "resample seeds, not individual duels" convention as the
    rest of this project's seed bootstraps -- see markov_selfplay_
    recovery_time_by_depth_per_seed.csv, needed because the pooled CSVs
    above discard per-seed counts), plus the pooled TFT reference line
    with its own band. Uses Hiragino (checked against the fonts actually
    installed on THIS machine) when available, falling back to IPAGothic
    with a printed warning otherwise -- see docs/MARKOV_STRUCTURE_
    EVOLUTION_MAC_RUNBOOK.md section 14 for regenerating on a Mac to get
    the real Hiragino rendering.

Usage:
    python3 scripts/analyze_selfplay_recovery_by_depth.py \\
        --by-depth-csv scripts/output/figures/markov_selfplay_recovery_time_by_depth.csv \\
        --pooled-csv scripts/output/figures/markov_selfplay_recovery_time.csv \\
        --per-seed-csv scripts/output/figures/markov_selfplay_recovery_time_by_depth_per_seed.csv \\
        --out-dir scripts/output/figures
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

OPPONENTS = ["TFT", "SELF", "POP", "WSLS"]
DEPTH_GROUPS = ["depthLow", "depthMid", "depthHigh"]
GROUP_COLOR = {"depthLow": "tab:blue", "depthMid": "tab:orange", "depthHigh": "tab:red"}
GROUP_LABEL_JA = {"depthLow": "深さ浅(depthLow)", "depthMid": "深さ中(depthMid)", "depthHigh": "深さ深(depthHigh)"}

# section 6.5's legend wording, exactly as requested: deep/mid/shallow/TFT-reference
ACADEMIC_LEGEND_LABEL = {"depthHigh": "記憶が深い", "depthMid": "中くらい", "depthLow": "浅い"}
ACADEMIC_LEGEND_ORDER = ["depthHigh", "depthMid", "depthLow"]  # legend row order requested
TFT_REFERENCE_LABEL = "TFT相手(参考)"

# section 6.6's revised title wording per focal opponent
ACADEMIC_TITLE_LABEL = {"SELF": "仲間(SELF)", "POP": "同じ集団(POP)", "WSLS": "WSLS"}
ACADEMIC_BAND_CAPTION = "帯はseedをまたいだブートストラップの95%範囲"

HIRAGINO_CANDIDATES = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Kaku Gothic Pro"]


def select_japanese_font(available_family_names: set) -> Tuple[str, bool]:
    """(font_family, is_hiragino) -- picks the first installed Hiragino
    variant found on THIS machine, else falls back to IPAGothic (used by
    every other figure script in this project). Pure/testable: takes the
    font name set instead of querying the system itself."""
    for candidate in HIRAGINO_CANDIDATES:
        if candidate in available_family_names:
            return candidate, True
    return "IPAGothic", False


_FONT_FAMILY, _IS_HIRAGINO = select_japanese_font({f.name for f in fm.fontManager.ttflist})
matplotlib.rcParams["font.family"] = _FONT_FAMILY
if not _IS_HIRAGINO:
    print(
        f"WARNING: no Hiragino font found on this machine -- using {_FONT_FAMILY} instead. "
        "Re-run this script on a Mac (which has Hiragino installed) to regenerate the academic "
        "figures with the requested font -- see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 14.",
        file=sys.stderr,
    )


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


def load_per_seed(path: Path) -> Dict[str, Dict[int, Dict[str, Dict[int, Tuple[int, int]]]]]:
    """{opponent: {age: {depth_group: {seed: (n, n_recovered)}}}}"""
    out: Dict[str, Dict[int, Dict[str, Dict[int, Tuple[int, int]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["opponent"]][int(row["age"])][row["depth_group"]][int(row["seed"])] = (int(row["n"]), int(row["n_recovered"]))
    return out


def tft_pooled_per_seed(per_seed: Dict[str, Dict[int, Dict[str, Dict[int, Tuple[int, int]]]]]) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """{age: {seed: (n, n_recovered)}} for TFT, summing ACROSS the 3 depth
    groups within each seed -- reconstructs the same per-seed counts
    plot_self_vs_tft's pooled (non-depth-split) TFT reference line is
    built from, straight from this script's own per-seed data (no need
    to also track export_markov_selfplay_recovery.py's separate,
    non-depth-tagged per-seed counts)."""
    out: Dict[int, Dict[int, Tuple[int, int]]] = defaultdict(dict)
    for age, by_group in per_seed["TFT"].items():
        seed_totals: Dict[int, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        for seed_counts in by_group.values():
            for seed, (n, n_recovered) in seed_counts.items():
                prev_n, prev_r = seed_totals[seed]
                seed_totals[seed] = (prev_n + n, prev_r + n_recovered)
        out[age] = dict(seed_totals)
    return out


def seed_bootstrap_recovered_rate(
    seed_counts: Dict[int, Tuple[int, int]], n_boot: int, rng: random.Random,
) -> Optional[Tuple[float, float]]:
    """(lo95, hi95) of the pooled RECOVERED rate (1-never_recovered_rate)
    under a seed-level bootstrap: resample the available seeds WITH
    REPLACEMENT n_boot times, and for each draw recompute n_recovered/n
    pooled across the resampled seeds. None if fewer than 2 seeds have
    data (a band is not meaningful with 0-1 seeds)."""
    seeds = list(seed_counts)
    if len(seeds) < 2:
        return None
    draws = []
    for _ in range(n_boot):
        resampled = [seed_counts[rng.choice(seeds)] for _ in seeds]
        total_n = sum(n for n, _r in resampled)
        total_r = sum(r for _n, r in resampled)
        draws.append(total_r / total_n if total_n else 0.0)
    draws.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = min(int(0.975 * n_boot), n_boot - 1)
    return draws[lo_idx], draws[hi_idx]


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


def plot_academic_figure(
    opponent: str,
    by_depth: Dict[str, Dict[int, Dict[str, dict]]],
    per_seed: Dict[str, Dict[int, Dict[str, Dict[int, Tuple[int, int]]]]],
    tft_per_seed_pooled: Dict[int, Dict[int, Tuple[int, int]]],
    pooled: Dict[str, Dict[int, dict]],
    n_boot: int,
    rng: random.Random,
    out_path: Path,
) -> None:
    """Section 6.5: recovered rate (not never_recovered_rate) by depth
    group for `opponent`, each line with a shaded seed-bootstrap 95%
    band, plus the pooled TFT reference line (also banded). Legend
    wording and order exactly as requested."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ages = sorted(by_depth[opponent])
    for g in ACADEMIC_LEGEND_ORDER:
        points = [1 - float(by_depth[opponent][age][g]["never_recovered_rate"]) for age in ages]
        los, his = [], []
        for age in ages:
            band = seed_bootstrap_recovered_rate(per_seed[opponent][age][g], n_boot, rng)
            lo, hi = band if band is not None else (points[ages.index(age)], points[ages.index(age)])
            los.append(lo)
            his.append(hi)
        ax.fill_between(ages, los, his, color=GROUP_COLOR[g], alpha=0.15, linewidth=0)
        ax.plot(ages, points, label=ACADEMIC_LEGEND_LABEL[g], color=GROUP_COLOR[g], linewidth=2)

    tft_ages = sorted(pooled["TFT"])
    tft_points = [1 - float(pooled["TFT"][age]["never_recovered_rate"]) for age in tft_ages]
    tft_los, tft_his = [], []
    for age in tft_ages:
        band = seed_bootstrap_recovered_rate(tft_per_seed_pooled[age], n_boot, rng)
        lo, hi = band if band is not None else (tft_points[tft_ages.index(age)],) * 2
        tft_los.append(lo)
        tft_his.append(hi)
    ax.fill_between(tft_ages, tft_los, tft_his, color="gray", alpha=0.12, linewidth=0)
    ax.plot(tft_ages, tft_points, label=TFT_REFERENCE_LABEL, color="gray", alpha=0.5, linewidth=1.5, linestyle="--")

    ax.set_xlabel("世代")
    ax.set_ylabel("20回のうちに和和和へ戻れた割合")
    ax.set_title(f"底からの回復:{ACADEMIC_TITLE_LABEL[opponent]}相手")
    ax.set_ylim(0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    order = ACADEMIC_LEGEND_ORDER + ["_tft"]
    label_order = [ACADEMIC_LEGEND_LABEL[g] for g in ACADEMIC_LEGEND_ORDER] + [TFT_REFERENCE_LABEL]
    ordered = sorted(zip(handles, labels), key=lambda hl: label_order.index(hl[1]))
    ax.legend([h for h, _ in ordered], [l for _, l in ordered], loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.01, ACADEMIC_BAND_CAPTION, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--by-depth-csv", type=str, default="scripts/output/figures/markov_selfplay_recovery_time_by_depth.csv")
    p.add_argument("--pooled-csv", type=str, default="scripts/output/figures/markov_selfplay_recovery_time.csv")
    p.add_argument("--per-seed-csv", type=str, default="scripts/output/figures/markov_selfplay_recovery_time_by_depth_per_seed.csv")
    p.add_argument("--academic-opponents", type=str, default="SELF,POP,WSLS", help="comma-separated focal opponents to make an academic figure for")
    p.add_argument("--n-boot", type=int, default=2000, help="seed-bootstrap draws, matching this project's established confirmatory-analysis convention")
    p.add_argument("--boot-seed", type=str, default="markov_selfplay_recovery_by_depth_bootstrap_v1")
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    by_depth = load_by_depth(Path(args.by_depth_csv))
    pooled = load_pooled(Path(args.pooled_csv))

    print_report(by_depth)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_self_vs_tft(by_depth, pooled, out_dir / "markov_selfplay_recovery_self_vs_tft_by_depth.png")

    per_seed = load_per_seed(Path(args.per_seed_csv))
    tft_per_seed_pooled = tft_pooled_per_seed(per_seed)
    rng = random.Random(args.boot_seed)
    for opponent in args.academic_opponents.split(","):
        plot_academic_figure(
            opponent, by_depth, per_seed, tft_per_seed_pooled, pooled, args.n_boot, rng,
            out_dir / f"markov_selfplay_recovery_rate_by_depth_academic_{opponent}.png",
        )


if __name__ == "__main__":
    main()
