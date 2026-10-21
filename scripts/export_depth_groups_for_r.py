"""Exploratory, outside preregistration: split probed genotypes into 3
EFFECTIVE-memory-depth groups (shallow/mid/deep) instead of the human
study's 3 attachment groups, and write per-opponent 3-gram text files in
exactly the human study's input format, for
docs/r_scripts/GRA_MAR10feb04Fmarkov_network_with_loops_parameterized.R
to draw networks from.

Two phases, so the (large) raw moves files are read only for the small
number of (seed, age) pairs actually needed:

Phase 1 (census only, small/fast): for every (seed, age) in the analysis
window, reconstruct each census genotype's LindgrenTable and compute its
effective_memory_depth (evopersonality_abm.lindgren). Reports the
nominal-vs-effective depth distribution per generation, proposes 3
depth-range cutpoints (each needing >= --group-size distinct seeds
available), and selects --group-size virtual participants per group,
preferring one per distinct seed (falling back to a different age from
an already-used seed only if there are not enough distinct seeds).

Phase 2 (moves, targeted): for the selected participants only, reads
their 3-gram series for each of TFT/AllC/AllD/Random/Gradual, picking one
probe repeat at random per participant (same convention as
evopersonality_abm.markov_apply.resample_virtual_group). Gradual is read
from --gradual-reprobe-dir (the DEVIATION_gradual_rule.md-fixed re-probe,
scripts/reprobe_gradual_rule_fix.py's output), never from --data-dir.

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 10.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import reprobe_gradual_rule_fix as reprobe_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.lindgren import effective_memory_depth  # noqa: E402
from evopersonality_abm.markov_apply import OPPONENT_INFO, load_series_by_genotype_all_opponents  # noqa: E402

GROUP_OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual"]
R_CODE = {opp: OPPONENT_INFO[opp][0] for opp in GROUP_OPPONENTS}  # TFT/ALC/ALD/RAN/GRA
DEPTH_GROUPS = ["Low", "Mid", "High"]


# ------------------------------------------------------------------ Phase 1: census + depth

def collect_census_pool(data_dir: Path, seeds: List[int], ages: List[int]) -> List[dict]:
    pool = []
    for age in ages:
        for seed in seeds:
            cpath = reprobe_mod.census_path(data_dir, seed, age)
            if not cpath.exists():
                continue
            for gid, freq, table in reprobe_mod.load_census_genotypes(cpath):
                pool.append({
                    "seed": seed, "age": age, "gid": gid, "freq": freq,
                    "nominal_depth": table.memory_depth,
                    "effective_depth": effective_memory_depth(table),
                })
    return pool


def write_depth_distribution(pool: List[dict], path: Path) -> None:
    counts: Dict[Tuple[int, str, int], int] = defaultdict(int)
    for row in pool:
        counts[(row["age"], "nominal", row["nominal_depth"])] += row["freq"]
        counts[(row["age"], "effective", row["effective_depth"])] += row["freq"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "depth_kind", "depth", "frequency_weighted_count"])
        for (age, kind, depth), c in sorted(counts.items()):
            w.writerow([age, kind, depth, c])


def propose_cutpoints(pool: List[dict], group_size: int) -> Optional[Tuple[int, int, Tuple[int, int, int]]]:
    """Greedy search over all (d1, d2) splits of the observed effective
    depths into 3 contiguous ranges [min,d1], (d1,d2], (d2,max]; keeps
    only splits where every range has >= group_size DISTINCT seeds
    represented somewhere in the window, and returns the split that
    maximizes the smallest range's seed count (most robust to the actual
    per-group sampling that follows). Returns None if no split works at
    all (caller should report this and let a human lower --group-size or
    adjust --depth-cutpoints manually)."""
    seeds_by_depth: Dict[int, set] = defaultdict(set)
    for row in pool:
        seeds_by_depth[row["effective_depth"]].add(row["seed"])
    depths = sorted(seeds_by_depth)

    def distinct_seed_count(depth_subset: List[int]) -> int:
        s = set()
        for d in depth_subset:
            s |= seeds_by_depth[d]
        return len(s)

    best = None
    for i in range(len(depths) - 1):
        for j in range(i + 1, len(depths) - 1 + 1):
            low, mid, high = depths[: i + 1], depths[i + 1 : j + 1], depths[j + 1 :]
            if not mid or not high:
                continue
            n_low, n_mid, n_high = distinct_seed_count(low), distinct_seed_count(mid), distinct_seed_count(high)
            if n_low >= group_size and n_mid >= group_size and n_high >= group_size:
                score = min(n_low, n_mid, n_high)
                if best is None or score > best[0]:
                    best = (score, depths[i], depths[j], (n_low, n_mid, n_high))
    if best is None:
        return None
    _score, d1, d2, counts = best
    return d1, d2, counts


def depth_ranges_from_cutpoints(pool: List[dict], d1: int, d2: int) -> Dict[str, range]:
    depths = sorted({row["effective_depth"] for row in pool})
    lo, hi = depths[0], depths[-1]
    return {"Low": range(lo, d1 + 1), "Mid": range(d1 + 1, d2 + 1), "High": range(d2 + 1, hi + 1)}


# ------------------------------------------------------------------ Phase 1: selection

def select_group(pool: List[dict], group_size: int, depth_range: range, rng: random.Random) -> List[dict]:
    """group_size candidates with effective_depth in depth_range: one per
    distinct seed where possible (shuffled, frequency-weighted pick within
    a seed's options), falling back to reusing a seed at a different age
    only once distinct seeds are exhausted."""
    candidates = [r for r in pool if r["effective_depth"] in depth_range]
    by_seed: Dict[int, List[dict]] = defaultdict(list)
    for r in candidates:
        by_seed[r["seed"]].append(r)

    seed_order = list(by_seed)
    rng.shuffle(seed_order)

    selected: List[dict] = []
    used_seed_ages = set()
    while len(selected) < group_size:
        progressed = False
        for seed in seed_order:
            if len(selected) >= group_size:
                break
            options = [r for r in by_seed[seed] if (r["seed"], r["age"]) not in used_seed_ages]
            if not options:
                continue
            weights = [r["freq"] for r in options]
            picked = rng.choices(options, weights=weights, k=1)[0]
            selected.append(picked)
            used_seed_ages.add((picked["seed"], picked["age"]))
            progressed = True
        if not progressed:
            raise RuntimeError(
                f"only found {len(selected)}/{group_size} candidates in effective_depth range "
                f"{depth_range.start}-{depth_range.stop - 1}, even allowing seed reuse at different ages"
            )
    return selected


# ------------------------------------------------------------------ Phase 2: series extraction

def extract_series_for_participants(
    data_dir: Path, gradual_reprobe_dir: Path, moves_suffix: str,
    participants: List[dict], rng: random.Random,
) -> Dict[str, Dict[int, str]]:
    """{opponent: {participant_index_in_`participants`: series}}. Reads
    each distinct (seed, age)'s moves file at most once."""
    by_seed_age: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, p in enumerate(participants):
        by_seed_age[(p["seed"], p["age"])].append(i)

    out: Dict[str, Dict[int, str]] = {opp: {} for opp in GROUP_OPPONENTS}
    for (seed, age), indices in by_seed_age.items():
        main_path = data_dir / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        gradual_path = gradual_reprobe_dir / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        main_series = load_series_by_genotype_all_opponents(main_path, main_analysis_rounds=20) if main_path.exists() else {}
        gradual_series = load_series_by_genotype_all_opponents(gradual_path, main_analysis_rounds=20) if gradual_path.exists() else {}

        for i in indices:
            gid = participants[i]["gid"]
            for opponent in GROUP_OPPONENTS:
                source = gradual_series if opponent == "Gradual" else main_series
                options = source.get(opponent, {}).get(gid)
                if options:
                    out[opponent][i] = rng.choice(options)
    return out


# ------------------------------------------------------------------ output

def write_group_files(
    out_dir: Path, groups: Dict[str, List[dict]], series_by_group: Dict[str, Dict[str, Dict[int, str]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for opponent in GROUP_OPPONENTS:
        code = R_CODE[opponent]
        for group_name in DEPTH_GROUPS:
            participants = groups[group_name]
            series_by_index = series_by_group[group_name][opponent]
            lines = [series_by_index[i] for i in range(len(participants)) if i in series_by_index]
            missing = len(participants) - len(lines)
            if missing:
                print(f"warning: {code}_depth{group_name}.txt missing series for {missing}/{len(participants)} participants", flush=True)
            path = out_dir / f"{code}_depth{group_name}.txt"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for line in lines:
                    f.write(line + "\n")
            print(f"wrote {path} ({len(lines)} lines)", flush=True)


def write_participants_csv(out_dir: Path, groups: Dict[str, List[dict]], path_name: str) -> None:
    path = out_dir / path_name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_group", "participant_index", "seed", "age", "genotype_id", "frequency", "nominal_depth", "effective_depth"])
        for group_name in DEPTH_GROUPS:
            for i, p in enumerate(groups[group_name]):
                w.writerow([group_name, i, p["seed"], p["age"], p["gid"], p["freq"], p["nominal_depth"], p["effective_depth"]])
    print(f"wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="production primary dir (scripts/output/markov_sweep_production_noise01)")
    p.add_argument("--gradual-reprobe-dir", type=str, required=True, help="output of scripts/reprobe_gradual_rule_fix.py (fixed Gradual probe data)")
    p.add_argument("--moves-suffix", type=str, default="noise0p01")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--depth-cutpoints", type=str, default=None, help="override the proposed cutpoints, e.g. '2,4' (Low<=2, 3<=Mid<=4, High>=5). Omit to use the auto-proposed split.")
    p.add_argument("--master-seed", type=str, default="markov_depth_groups_v1")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    gradual_reprobe_dir = Path(args.gradual_reprobe_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.window_start, args.window_end + 1, args.probe_interval))

    print(f"Phase 1: scanning census for {len(seeds)} seeds x {len(ages)} ages...", flush=True)
    pool = collect_census_pool(data_dir, seeds, ages)
    print(f"{len(pool)} (seed, age, genotype) census rows collected", flush=True)

    write_depth_distribution(pool, out_dir / "depth_distribution_by_generation.csv")

    if args.depth_cutpoints:
        d1, d2 = (int(x) for x in args.depth_cutpoints.split(","))
        print(f"using manually-specified cutpoints: Low<={d1}, {d1+1}<=Mid<={d2}, High>={d2+1}", flush=True)
    else:
        proposal = propose_cutpoints(pool, args.group_size)
        if proposal is None:
            print(
                f"ERROR: no 3-way split of the observed effective-depth values gives every group "
                f">= {args.group_size} distinct seeds. Lower --group-size, or inspect "
                f"depth_distribution_by_generation.csv and pass --depth-cutpoints manually.",
                flush=True,
            )
            sys.exit(1)
        d1, d2, counts = proposal
        print(f"proposed cutpoints: Low<={d1}, {d1+1}<=Mid<={d2}, High>={d2+1} (distinct seeds available: {counts})", flush=True)

    ranges = depth_ranges_from_cutpoints(pool, d1, d2)

    rng = random.Random(args.master_seed)
    groups: Dict[str, List[dict]] = {}
    for group_name in DEPTH_GROUPS:
        groups[group_name] = select_group(pool, args.group_size, ranges[group_name], rng)
        print(f"selected {len(groups[group_name])} participants for depth group {group_name} "
              f"({len({p['seed'] for p in groups[group_name]})} distinct seeds)", flush=True)

    print("Phase 2: extracting 3-gram series for selected participants...", flush=True)
    series_by_group = {
        group_name: extract_series_for_participants(data_dir, gradual_reprobe_dir, args.moves_suffix, groups[group_name], rng)
        for group_name in DEPTH_GROUPS
    }

    write_group_files(out_dir, groups, series_by_group)
    write_participants_csv(out_dir, groups, "depth_group_participants.csv")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
