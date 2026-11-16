"""Exploratory, outside preregistration (docs/EXPLORATORY_selfplay_recovery.md):
does an evolved individual lack the ABILITY to recover from 底底底
(mutual defection) to 和和和 (mutual cooperation), or does it have a
recovery mechanism that only works among peers (WSLS-style
win-stay-lose-shift, which never recovers against a strict reciprocator
like TFT but recovers quickly among copies of itself)? Adds 3 new probe
"opponents" (vocab=64, same as Random):

  - SELF: the focal genotype against an independent copy of ITSELF
    (same LindgrenTable, two separate Agents -- probe.run_selfplay_duel).
  - POP: the focal genotype against another individual drawn from the
    SAME (seed, age) population, frequency-weighted, redrawn fresh each
    of the 30 repeats.
  - WSLS: already probed as part of the production run's extended set
    (probe.OPPONENTS["WSLS"]) -- reused from the EXISTING production
    moves data, no new duels.

For each (seed, age), builds {opponent: {seed: (frequencies,
series_by_genotype, identities)}} pools in the exact shape
evopersonality_abm.markov_apply's cross-lineage machinery expects, then
reuses, UNCHANGED:
  - markov_apply.bootstrap_aggregate_with_sampler (-> widest-path
    recovery/descent CSV, same columns as the project's other
    descriptive CSVs -- no human comparison, since SELF/POP/WSLS have no
    human reference, same treatment as GradualAppeasement/WSLS elsewhere)
  - export_markov_edge_weights_for_viewer.average_edges_over_draws /
    node_appearance_rates_over_draws (-> edges/node-rate JSON in the
    SAME schema as markov_edge_weights.json / markov_node_freq.json,
    cross_lineage grouping only, opponent names "SELF"/"POP"/"WSLS")

Also computes a supplementary, exploratory metric OUTSIDE the human
study's own measuring stick (section 3 of the design doc): "recovery
time from the bottom" -- force the first 3 rounds to mutual defection
(底底底), then free-play `--main-analysis-rounds`-worth of rounds (20 by
default) against TFT/SELF/POP/WSLS and record the first round (if any)
with 2 CONSECUTIVE rounds of mutual cooperation
(probe.recovery_time_from_forced_bottom). Reports, per (generation,
opponent), the median round among genotype x repeat draws that did
recover, and the fraction that never recovered within the free-play
window. This is a NEW metric invented for this investigation, not
ported from or validated against the human data -- clearly labeled as
such in its own CSV and in docs/EXPLORATORY_selfplay_recovery.md.

Does NOT read or write scripts/output/figures/markov_edge_weights.json
or markov_node_freq.json. Writes separate files:
  scripts/output/figures/markov_selfplay_widest_path.csv
  scripts/output/figures/markov_selfplay_edges.json
  scripts/output/figures/markov_selfplay_nodes.json
  scripts/output/figures/markov_selfplay_recovery_time.csv

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 12.
"""

import argparse
import csv
import json
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import export_markov_edge_weights_for_viewer as edge_mod  # noqa: E402
import reprobe_gradual_rule_fix as reprobe_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.lindgren import LindgrenTable, pavlov_table, tit_for_tat_table  # noqa: E402
from evopersonality_abm.markov_apply import bootstrap_aggregate_with_sampler, load_series_by_genotype_all_opponents, resample_cross_seed_group  # noqa: E402
from evopersonality_abm.markov_recovery import load_vocab, series_from_moves  # noqa: E402
from evopersonality_abm.probe import recovery_time_from_forced_bottom, run_selfplay_duel  # noqa: E402

SELFPLAY_OPPONENTS = ["SELF", "POP", "WSLS"]
RECOVERY_TIME_OPPONENTS = ["TFT", "SELF", "POP", "WSLS"]
VOCAB = load_vocab("RAN")  # full 64-word vocabulary, same as Random/Gradual


def build_selfplay_series_for_seed_age(
    census_genotypes: List[Tuple[int, int, LindgrenTable]], probe_repeats: int, probe_rounds: int,
    main_analysis_rounds: int, noise_rate: float, rng: random.Random,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """{genotype_id: [series, ...]} for SELF and for POP, one census's worth."""
    gids = [gid for gid, _freq, _table in census_genotypes]
    freqs = {gid: freq for gid, freq, _table in census_genotypes}
    tables = {gid: table for gid, _freq, table in census_genotypes}
    weights = [freqs[g] for g in gids]

    self_series: Dict[int, List[str]] = {gid: [] for gid in gids}
    pop_series: Dict[int, List[str]] = {gid: [] for gid in gids}

    for gid in gids:
        focal_table = tables[gid]
        for _ in range(probe_repeats):
            own_moves, opp_moves = run_selfplay_duel(focal_table, focal_table, rng, rounds=probe_rounds, noise_rate=noise_rate)
            self_series[gid].append(series_from_moves(own_moves, opp_moves, n_rounds=main_analysis_rounds))

        for _ in range(probe_repeats):
            partner_table = draw_partner_table(gids, weights, tables, rng)
            own_moves, opp_moves = run_selfplay_duel(focal_table, partner_table, rng, rounds=probe_rounds, noise_rate=noise_rate)
            pop_series[gid].append(series_from_moves(own_moves, opp_moves, n_rounds=main_analysis_rounds))

    return self_series, pop_series


def draw_partner_table(gids: List[int], weights: List[int], tables: Dict[int, LindgrenTable], rng: random.Random) -> LindgrenTable:
    """One frequency-weighted, with-replacement POP partner draw (may
    coincidentally redraw the focal genotype itself -- representing
    "any population member", same convention as the rest of this
    project's frequency-weighted sampling)."""
    partner_gid = rng.choices(gids, weights=weights, k=1)[0]
    return tables[partner_gid]


def recovery_times_for_seed_age(
    census_genotypes: List[Tuple[int, int, LindgrenTable]], probe_repeats: int,
    free_rounds: int, noise_rate: float, rng: random.Random,
) -> Dict[str, List[Optional[int]]]:
    """{opponent: [recovery_time or None, ...]} pooling every genotype's
    probe_repeats draws, for TFT/SELF/POP/WSLS (section 3, exploratory --
    docs/EXPLORATORY_selfplay_recovery.md). WSLS here is the fixed
    pavlov_table() opponent (not the production probe data -- this
    metric needs a hand-forced initial history that reading pre-recorded
    moves can't provide)."""
    gids = [gid for gid, _freq, _table in census_genotypes]
    tables = {gid: table for gid, _freq, table in census_genotypes}
    weights = [freq for _gid, freq, _table in census_genotypes]

    out: Dict[str, List[Optional[int]]] = {opp: [] for opp in RECOVERY_TIME_OPPONENTS}
    tft_table = tit_for_tat_table()
    wsls_table = pavlov_table()

    for gid in gids:
        focal_table = tables[gid]
        for opponent, opponent_table in (
            ("TFT", tft_table),
            ("SELF", focal_table),
            ("WSLS", wsls_table),
        ):
            for _ in range(probe_repeats):
                out[opponent].append(
                    recovery_time_from_forced_bottom(focal_table, opponent_table, rng, free_rounds=free_rounds, noise_rate=noise_rate)
                )
        for _ in range(probe_repeats):
            partner_table = draw_partner_table(gids, weights, tables, rng)
            out["POP"].append(
                recovery_time_from_forced_bottom(focal_table, partner_table, rng, free_rounds=free_rounds, noise_rate=noise_rate)
            )

    return out


def summarize_recovery_times(age: int, recovery_times_by_opponent: Dict[str, List[Optional[int]]]) -> List[dict]:
    """[{opponent, age, n, n_recovered, median_recovery_round, never_recovered_rate}, ...]"""
    rows = []
    for opponent, times in recovery_times_by_opponent.items():
        n = len(times)
        if n == 0:
            continue
        recovered = [t for t in times if t is not None]
        rows.append({
            "opponent": opponent,
            "age": age,
            "n": n,
            "n_recovered": len(recovered),
            "median_recovery_round": statistics.median(recovered) if recovered else "",
            "never_recovered_rate": round(1 - len(recovered) / n, 4),
        })
    return rows


def build_pools_for_age(
    data_dir: Path, moves_suffix: str, age: int, seeds: List[int],
    probe_repeats: int, probe_rounds: int, main_analysis_rounds: int, noise_rate: float, master_seed: str,
) -> Tuple[Dict[str, Dict[int, tuple]], Dict[str, List[Optional[int]]]]:
    """{opponent: {seed: (frequencies, series_by_genotype, identities)}}
    for SELF/POP/WSLS, plus pooled recovery_time_from_forced_bottom
    samples (section 3, exploratory) for TFT/SELF/POP/WSLS, both at one
    age -- reuses the SAME per-seed census load for both rather than
    reading it twice."""
    pools: Dict[str, Dict[int, tuple]] = {opp: {} for opp in SELFPLAY_OPPONENTS}
    recovery_times: Dict[str, List[Optional[int]]] = {opp: [] for opp in RECOVERY_TIME_OPPONENTS}

    for seed in seeds:
        cpath = reprobe_mod.census_path(data_dir, seed, age)
        if not cpath.exists():
            continue
        census = reprobe_mod.load_census_genotypes(cpath)
        if not census:
            continue
        frequencies = {gid: freq for gid, freq, _table in census}
        identities = {gid: (table.memory_depth, ",".join(map(str, table.table)), ",".join(map(str, table.prehistory))) for gid, _freq, table in census}

        rng = random.Random(f"{master_seed}_{seed}_{age}")
        self_series, pop_series = build_selfplay_series_for_seed_age(
            census, probe_repeats, probe_rounds, main_analysis_rounds, noise_rate, rng,
        )
        pools["SELF"][seed] = (frequencies, self_series, identities)
        pools["POP"][seed] = (frequencies, pop_series, identities)

        moves_path = data_dir / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        if moves_path.exists():
            wsls_series = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=main_analysis_rounds).get("WSLS", {})
            if wsls_series:
                pools["WSLS"][seed] = (frequencies, wsls_series, identities)

        seed_recovery_times = recovery_times_for_seed_age(census, probe_repeats, main_analysis_rounds, noise_rate, rng)
        for opponent, times in seed_recovery_times.items():
            recovery_times[opponent].extend(times)

    return pools, recovery_times


def process_one_age(
    data_dir_str: str, moves_suffix: str, age: int, seeds: List[int],
    probe_repeats: int, probe_rounds: int, main_analysis_rounds: int, noise_rate: float,
    group_size: int, n_draws: int, master_seed: str,
) -> dict:
    data_dir = Path(data_dir_str)
    pools, recovery_times = build_pools_for_age(
        data_dir, moves_suffix, age, seeds, probe_repeats, probe_rounds, main_analysis_rounds, noise_rate, master_seed,
    )

    rng = random.Random(f"{master_seed}_agg_{age}")
    widest_path_rows = []
    edges_by_opponent = {}
    nodes_by_opponent = {}

    for opponent in SELFPLAY_OPPONENTS:
        pool = pools[opponent]
        if len(pool) < group_size:
            continue

        def sample_fn(rng_, pools_=pool, gs=group_size):
            return resample_cross_seed_group(pools_, gs, rng_)

        summary = bootstrap_aggregate_with_sampler(sample_fn, opponent, n_draws, rng, {}, VOCAB)
        widest_path_rows.append({"opponent": opponent, "age": age, **summary})

        edges = edge_mod.average_edges_over_draws(pool, VOCAB, group_size, n_draws, rng)
        if edges is not None:
            edges_by_opponent[opponent] = edge_mod.edges_to_rows(edges)

        rates = edge_mod.node_appearance_rates_over_draws(pool, VOCAB, group_size, n_draws, rng)
        if rates is not None:
            nodes_by_opponent[opponent] = rates

    recovery_time_rows = summarize_recovery_times(age, recovery_times)

    return {
        "age": age, "widest_path_rows": widest_path_rows, "edges_by_opponent": edges_by_opponent,
        "nodes_by_opponent": nodes_by_opponent, "recovery_time_rows": recovery_time_rows,
    }


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def run_all_ages(
    data_dir: Path, moves_suffix: str, ages: List[int], seeds: List[int],
    probe_repeats: int, probe_rounds: int, main_analysis_rounds: int, noise_rate: float,
    group_size: int, n_draws: int, master_seed: str, workers: int, cache_dir: Path,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age, str(data_dir), moves_suffix, age, seeds,
                probe_repeats, probe_rounds, main_analysis_rounds, noise_rate, group_size, n_draws, master_seed,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    widest_path_rows = []
    recovery_time_rows = []
    edges_by_opponent = {opp: {} for opp in SELFPLAY_OPPONENTS}
    nodes_by_opponent = {opp: {} for opp in SELFPLAY_OPPONENTS}
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        widest_path_rows.extend(result["widest_path_rows"])
        recovery_time_rows.extend(result["recovery_time_rows"])
        for opponent, rows in result["edges_by_opponent"].items():
            edges_by_opponent[opponent][age] = rows
        for opponent, rates in result["nodes_by_opponent"].items():
            nodes_by_opponent[opponent][age] = rates

    # Explicit, deterministic row order (not relied-upon insertion order) --
    # every run of this script, on the same data, produces byte-identical
    # CSVs regardless of ProcessPoolExecutor completion order or process
    # launch (string hash randomization affects dict/set iteration order
    # between separate `python3 ...` runs, which previously also leaked
    # into write_widest_path_csv's column order -- see that function).
    widest_path_rows.sort(key=lambda r: (r["opponent"], r["age"]))
    recovery_time_rows.sort(key=lambda r: (r["opponent"], r["age"]))

    return widest_path_rows, recovery_time_rows, edges_by_opponent, nodes_by_opponent


def write_widest_path_csv(path: Path, rows: List[dict]) -> None:
    """Writes `rows` as-is (caller is responsible for sorting them into a
    deterministic order first -- see each script's run_all_ages). The
    column order, however, is this function's own responsibility: it used
    to be `list({k for row in rows for k in row})`, a set comprehension
    whose iteration order depends on Python's per-process string hash
    randomization (PYTHONHASHSEED) -- same row content, shuffled columns
    on every separate `python3 ...` launch. `sorted(...)` fixes that."""
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="production primary dir (scripts/output/markov_sweep_production_noise01) -- WSLS is read from here; SELF/POP are newly simulated from its census")
    p.add_argument("--moves-suffix", type=str, default="noise0p01")
    p.add_argument("--noise-rate", type=float, default=0.01, help="execution noise for SELF/POP duels, matching the primary condition")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--generations-start", type=int, default=1000)
    p.add_argument("--generations-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50, help="use 250 if 50 is too slow (see the runbook's benchmark)")
    p.add_argument("--probe-repeats", type=int, default=30)
    p.add_argument("--probe-rounds", type=int, default=100)
    p.add_argument("--main-analysis-rounds", type=int, default=20)
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--master-seed", type=str, default="markov_selfplay_recovery_v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.generations_start, args.generations_end + 1, args.probe_interval))
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "_age_cache_selfplay"

    widest_path_rows, recovery_time_rows, edges_by_opponent, nodes_by_opponent = run_all_ages(
        Path(args.data_dir), args.moves_suffix, ages, seeds,
        args.probe_repeats, args.probe_rounds, args.main_analysis_rounds, args.noise_rate,
        args.group_size, args.n_draws, args.master_seed, args.workers, cache_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_widest_path_csv(out_dir / "markov_selfplay_widest_path.csv", widest_path_rows)
    print(f"wrote {out_dir / 'markov_selfplay_widest_path.csv'} ({len(widest_path_rows)} rows)", flush=True)

    write_widest_path_csv(out_dir / "markov_selfplay_recovery_time.csv", recovery_time_rows)
    print(f"wrote {out_dir / 'markov_selfplay_recovery_time.csv'} ({len(recovery_time_rows)} rows)", flush=True)

    with open(out_dir / "markov_selfplay_edges.json", "w", encoding="utf-8") as f:
        json.dump({"abm": {opp: {"cross_lineage": edges_by_opponent[opp]} for opp in SELFPLAY_OPPONENTS}}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out_dir / 'markov_selfplay_edges.json'}", flush=True)

    with open(out_dir / "markov_selfplay_nodes.json", "w", encoding="utf-8") as f:
        json.dump({"abm_nodes": {opp: {"cross_lineage": nodes_by_opponent[opp]} for opp in SELFPLAY_OPPONENTS}}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out_dir / 'markov_selfplay_nodes.json'}", flush=True)


if __name__ == "__main__":
    main()
