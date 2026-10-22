"""Exploratory, outside preregistration (docs/EXPLORATORY_selfplay_recovery.md
section 6): breaks down the "recovery time from the bottom" auxiliary
metric (scripts/export_markov_selfplay_recovery.py section 3) by the SAME
3 effective-memory-depth groups the HTML viewer uses (depthLow/depthMid/
depthHigh, scripts/export_depth_groups_for_r.py's --depth-cutpoints), per
generation, for TFT/SELF/POP/WSLS. Also reports, per generation, the
frequency-weighted fraction of NOMINAL memory-depth-1 genotypes whose
lookup table matches pavlov_table()'s win-stay-lose-shift rule exactly
([1, 0, 0, 1] -- "stay after 和 or 搾, shift after 惨 or 底") -- a plain
census count, no duels needed.

Why a NEW script instead of re-reading export_markov_selfplay_recovery.py's
own age cache (_age_cache_selfplay/): that cache's process_one_age already
POOLS every genotype's recovery-time draws together before caching (via
summarize_recovery_times), discarding each genotype's identity/depth --
there is nothing to regroup by depth in the cached pickles. Recomputing
from the SAME (seed, age) census with the SAME probe.
recovery_time_from_forced_bottom is unavoidable, but it is cheap: unlike
the original script, this one reads ONLY the small census.csv.gz files
(no SELF/POP series-building, no moves.csv.gz, no widest-path/edge/node
bootstrap) -- see section 6's benchmark in the design doc for the
measured cost. No new evolution is run either way; both this script and
the original read the SAME already-completed production census.

Does NOT touch markov_edge_weights.json, markov_node_freq.json,
markov_selfplay_widest_path.csv, markov_selfplay_recovery_time.csv,
markov_selfplay_edges.json, or markov_selfplay_nodes.json. Writes:
  scripts/output/figures/markov_selfplay_recovery_time_by_depth.csv
  scripts/output/figures/markov_depth1_wsls_rule_fraction.csv

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 13.
"""

import argparse
import random
import statistics
import sys
import time
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import export_markov_edge_weights_for_viewer as edge_mod  # noqa: E402
import export_markov_selfplay_recovery as recovery_mod  # noqa: E402
import reprobe_gradual_rule_fix as reprobe_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.lindgren import LindgrenTable, effective_memory_depth, pavlov_table, tit_for_tat_table  # noqa: E402
from evopersonality_abm.probe import recovery_time_from_forced_bottom  # noqa: E402

RECOVERY_TIME_OPPONENTS = ["TFT", "SELF", "POP", "WSLS"]
WSLS_RULE_TABLE = (1, 0, 0, 1)  # pavlov_table().table -- "stay after 和/搾, shift after 惨/底"
assert pavlov_table().table == WSLS_RULE_TABLE  # keep the literal constant honest


def depth_group_lookup(d1: int, d2: int, max_depth: int) -> Dict[int, str]:
    """{effective_depth: group_name} -- flattens edge_mod.depth_ranges's
    3 ranges for an O(1) per-genotype lookup."""
    out: Dict[int, str] = {}
    for name, depth_range in edge_mod.depth_ranges(d1, d2, max_depth).items():
        for d in depth_range:
            out[d] = name
    return out


def is_depth1_wsls_rule(table: LindgrenTable) -> bool:
    """NOMINAL memory_depth==1 (a literal 4-entry lookup table, not the
    viewer's effective-depth grouping) whose table matches pavlov_table()
    bit-for-bit. Prehistory-independent: "same rule" means the same
    decision for every (own_last, opp_last) combination, not the same
    assumed round-0 history."""
    return table.memory_depth == 1 and table.table == WSLS_RULE_TABLE


def wsls_rule_counts_for_seed_age(census_genotypes: List[Tuple[int, int, LindgrenTable]]) -> Tuple[int, int]:
    """(n_depth1, n_wsls_rule), frequency-weighted, for one census."""
    n_depth1 = 0
    n_wsls_rule = 0
    for _gid, freq, table in census_genotypes:
        if table.memory_depth != 1:
            continue
        n_depth1 += freq
        if is_depth1_wsls_rule(table):
            n_wsls_rule += freq
    return n_depth1, n_wsls_rule


def recovery_times_by_depth_for_seed_age(
    census_genotypes: List[Tuple[int, int, LindgrenTable]], probe_repeats: int,
    free_rounds: int, noise_rate: float, rng: random.Random, depth_group_of_depth: Dict[int, str],
) -> Dict[Tuple[str, str], List[Optional[int]]]:
    """Same per-genotype recovery_time_from_forced_bottom draws as
    export_markov_selfplay_recovery.recovery_times_for_seed_age, but keyed
    by (opponent, depth_group) instead of pooled across every genotype.
    depth_group_of_depth maps EFFECTIVE memory depth -> group name
    (depthLow/depthMid/depthHigh); a genotype whose effective depth falls
    outside --max-depth's 3 ranges is skipped entirely (not counted
    towards any group, and its own draws consume no rng -- the POP
    partner draw pool below is unaffected, since POP partners are drawn
    from the FULL population regardless of the partner's own depth)."""
    gids = [gid for gid, _freq, _table in census_genotypes]
    tables = {gid: table for gid, _freq, table in census_genotypes}
    weights = [freq for _gid, freq, _table in census_genotypes]

    out: Dict[Tuple[str, str], List[Optional[int]]] = defaultdict(list)
    tft_table = tit_for_tat_table()
    wsls_table = pavlov_table()

    for gid in gids:
        focal_table = tables[gid]
        group = depth_group_of_depth.get(effective_memory_depth(focal_table))
        if group is None:
            continue

        for opponent, opponent_table in (
            ("TFT", tft_table),
            ("SELF", focal_table),
            ("WSLS", wsls_table),
        ):
            for _ in range(probe_repeats):
                out[(opponent, group)].append(
                    recovery_time_from_forced_bottom(focal_table, opponent_table, rng, free_rounds=free_rounds, noise_rate=noise_rate)
                )
        for _ in range(probe_repeats):
            partner_table = recovery_mod.draw_partner_table(gids, weights, tables, rng)
            out[("POP", group)].append(
                recovery_time_from_forced_bottom(focal_table, partner_table, rng, free_rounds=free_rounds, noise_rate=noise_rate)
            )

    return out


def summarize_recovery_times_by_depth(age: int, pooled: Dict[Tuple[str, str], List[Optional[int]]]) -> List[dict]:
    rows = []
    for (opponent, depth_group), times in pooled.items():
        n = len(times)
        if n == 0:
            continue
        recovered = [t for t in times if t is not None]
        rows.append({
            "opponent": opponent,
            "depth_group": depth_group,
            "age": age,
            "n": n,
            "n_recovered": len(recovered),
            "median_recovery_round": statistics.median(recovered) if recovered else "",
            "never_recovered_rate": round(1 - len(recovered) / n, 4),
        })
    return rows


def process_one_age(
    data_dir_str: str, age: int, seeds: List[int], probe_repeats: int,
    free_rounds: int, noise_rate: float, d1: int, d2: int, max_depth: int, master_seed: str,
) -> dict:
    data_dir = Path(data_dir_str)
    depth_group_of_depth = depth_group_lookup(d1, d2, max_depth)

    pooled_recovery: Dict[Tuple[str, str], List[Optional[int]]] = defaultdict(list)
    total_depth1 = 0
    total_wsls_rule = 0

    for seed in seeds:
        cpath = reprobe_mod.census_path(data_dir, seed, age)
        if not cpath.exists():
            continue
        census = reprobe_mod.load_census_genotypes(cpath)
        if not census:
            continue

        n_depth1, n_wsls_rule = wsls_rule_counts_for_seed_age(census)
        total_depth1 += n_depth1
        total_wsls_rule += n_wsls_rule

        rng = random.Random(f"{master_seed}_{seed}_{age}")
        seed_times = recovery_times_by_depth_for_seed_age(
            census, probe_repeats, free_rounds, noise_rate, rng, depth_group_of_depth,
        )
        for key, times in seed_times.items():
            pooled_recovery[key].extend(times)

    recovery_rows = summarize_recovery_times_by_depth(age, pooled_recovery)
    wsls_rule_row = {
        "age": age,
        "n_depth1": total_depth1,
        "n_wsls_rule": total_wsls_rule,
        "fraction": round(total_wsls_rule / total_depth1, 4) if total_depth1 else "",
    }

    return {"age": age, "recovery_rows": recovery_rows, "wsls_rule_row": wsls_rule_row}


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def run_all_ages(
    data_dir: Path, ages: List[int], seeds: List[int], probe_repeats: int, free_rounds: int,
    noise_rate: float, d1: int, d2: int, max_depth: int, master_seed: str, workers: int, cache_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age, str(data_dir), age, seeds, probe_repeats, free_rounds,
                noise_rate, d1, d2, max_depth, master_seed,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    recovery_rows = []
    wsls_rule_rows = []
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        recovery_rows.extend(result["recovery_rows"])
        wsls_rule_rows.append(result["wsls_rule_row"])

    return recovery_rows, wsls_rule_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="production primary dir (census.csv.gz only -- no moves files are read)")
    p.add_argument("--noise-rate", type=float, default=0.01, help="matches the primary condition's execution noise")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--generations-start", type=int, default=1000)
    p.add_argument("--generations-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--probe-repeats", type=int, default=30)
    p.add_argument("--free-rounds", type=int, default=20, help="free-play rounds after the forced-bottom start, same as --main-analysis-rounds in export_markov_selfplay_recovery.py")
    p.add_argument("--depth-cutpoints", type=str, required=True, help="'D1,D2' -- use the SAME cutpoints as the viewer (export_depth_groups_for_r.py / export_markov_edge_weights_for_viewer.py)")
    p.add_argument("--max-depth", type=int, default=6, help="matches --max-memory-depth of the production evolution run")
    p.add_argument("--master-seed", type=str, default="markov_selfplay_recovery_by_depth_v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.generations_start, args.generations_end + 1, args.probe_interval))
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "_age_cache_selfplay_recovery_by_depth"
    d1, d2 = (int(x) for x in args.depth_cutpoints.split(","))

    recovery_rows, wsls_rule_rows = run_all_ages(
        Path(args.data_dir), ages, seeds, args.probe_repeats, args.free_rounds, args.noise_rate,
        d1, d2, args.max_depth, args.master_seed, args.workers, cache_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    recovery_mod.write_widest_path_csv(out_dir / "markov_selfplay_recovery_time_by_depth.csv", recovery_rows)
    print(f"wrote {out_dir / 'markov_selfplay_recovery_time_by_depth.csv'} ({len(recovery_rows)} rows)", flush=True)

    recovery_mod.write_widest_path_csv(out_dir / "markov_depth1_wsls_rule_fraction.csv", wsls_rule_rows)
    print(f"wrote {out_dir / 'markov_depth1_wsls_rule_fraction.csv'} ({len(wsls_rule_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
