"""Exploratory, outside preregistration: per-edge average Jaccard weights
for the Jaccard-threshold x generation HTML viewer
(docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 11).

For the primary condition (noise 0.01), every probe age 0-5000 (step 50,
101 timepoints) x 5 opponents (TFT/AllC/AllD/Random/Gradual) x grouping
scheme: cross-lineage (scheme B, 56 people) and the 3 effective-memory-depth
groups from scripts/export_depth_groups_for_r.py (same --depth-cutpoints,
pass whatever that script printed/decided). For each (opponent, age,
grouping), draws 200 cross-lineage-style groups (evopersonality_abm.
markov_apply.resample_cross_seed_group) and averages EVERY edge's Jaccard
across all 200 draws (no threshold applied here -- an edge absent from a
given draw contributes 0 to that draw's average, same convention as the
rest of this project's bootstrap aggregation; the viewer applies its own
threshold interactively). An (age, grouping) with fewer than 56 distinct
seeds available (common for a narrow depth range at a single age, even
though the depth-range cutpoints guarantee >=56 pooled across the whole
window) is simply omitted -- the viewer shows "no data" for it.

Gradual is read from --gradual-reprobe-dir (the DEVIATION_gradual_rule.md
-fixed re-probe), never from --data-dir.

Also reformats --human-weights (docs/human_network_weights.csv, already
in the repo) into the same schema -- one network per human group
(Secure/Mix/Fearful). Human self-loops are NOT included: the raw human
3-gram text is not in this repository, only this precomputed edge table
(which itself has none), so the viewer's human panels will show no
self-loops -- that's a data-availability gap, not a bug.

Output JSON schema (nested to avoid repeating opponent/grouping/age
thousands of times):
{
  "max_depth": 6,
  "depth_cutpoints": [d1, d2],
  "human": {opponent: {group: [[from, to, jaccard], ...]}},
  "abm": {opponent: {grouping: {age: [[from, to, avg_jaccard], ...]}}}
}

Size estimate: a 64-node opponent's 3-gram graph has at most 4 possible
successors per node (append any of the 4 symbols, drop the oldest) plus
a self-loop, so at most ~5*64=320 possible edges; TFT/AllC/AllD have far
smaller 16/8/8-word vocabularies. Upper bound: 4 groupings x 5 opponents
x 101 ages x ~320 possible edges =~ 646k edge-records; in practice most
(age, grouping) combinations will have far fewer edges above 0 average
Jaccard, and many depth-group timepoints will be entirely omitted
(insufficient seeds). Expect roughly 10-30MB for the whole JSON file.

--nodes-only (section 12, exploratory, outside preregistration): instead
of edge weights, writes <out-path's directory>/markov_node_freq.json
with each vocab 3-gram's appearance rate -- the fraction of the 200
drawn virtual participants whose own 3-gram set contains that word at
least once, averaged over all 200 draws (a draw where nobody has the
word contributes 0). Uses the SAME per-age RNG stream and the SAME
cross_lineage-then-3-depth-groupings draw order as the edge-weight
export (process_one_age_nodes mirrors process_one_age exactly), so
running this against the same arguments as a prior (non---nodes-only)
run reproduces the identical 200 draws that produced that run's
edge-weights JSON. Schema:
{"abm_nodes": {opponent: {grouping: {age: {node: rate}}}}}
Does not read, write, or touch the edge-weights JSON at --out-path, and
uses a separate age cache (_age_cache_node_freq/, vs. the edge export's
_age_cache_edge_weights/).

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 11.
"""

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import reanalyze_gradual_rule_fix as reanalyze_mod  # noqa: E402
import run_markov_confirmatory_analysis as rmca  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.lindgren import LindgrenTable, effective_memory_depth  # noqa: E402
from evopersonality_abm.markov_apply import (  # noqa: E402
    OPPONENT_INFO,
    load_human_weights,
    resample_cross_seed_group,
    series_to_word_set,
)
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges  # noqa: E402

GROUP_OPPONENTS = ["TFT", "AllC", "AllD", "Random", "Gradual"]
DEPTH_GROUPINGS = ["depthLow", "depthMid", "depthHigh"]
GROUPINGS = ["cross_lineage"] + DEPTH_GROUPINGS


def effective_depth_of_identity(identity: Tuple[int, str, str]) -> int:
    memory_depth, table_str, prehistory_str = identity
    table = tuple(int(x) for x in table_str.split(",") if x != "")
    prehistory = tuple(int(x) for x in prehistory_str.split(",") if x != "")
    return effective_memory_depth(LindgrenTable(memory_depth, table, prehistory))


def depth_ranges(d1: int, d2: int, max_depth: int) -> Dict[str, range]:
    return {"depthLow": range(0, d1 + 1), "depthMid": range(d1 + 1, d2 + 1), "depthHigh": range(d2 + 1, max_depth + 1)}


def seeds_with_opp_by_opponent_for_age(
    data_dir: Path, gradual_reprobe_dir: Path, moves_suffix: str, probe_noise_rate: float,
    age: int, seeds: List[int],
) -> Dict[str, Dict[int, tuple]]:
    """{opponent: {seed: (frequencies, series_by_genotype, identities)}}
    for all of GROUP_OPPONENTS at one age, reading each distinct moves
    file (main production dir, and the Gradual re-probe dir) only ONCE --
    calling rmca.load_seed_pools_for_age per-opponent instead would
    re-read+re-parse the same multi-opponent moves file redundantly once
    per opponent."""
    main_pools = rmca.load_seed_pools_for_age(data_dir, moves_suffix, age, seeds)
    gradual_pools = reanalyze_mod.load_seed_pools_for_age_reprobe(data_dir, gradual_reprobe_dir, probe_noise_rate, age, seeds)
    out: Dict[str, Dict[int, tuple]] = {}
    for opponent in GROUP_OPPONENTS:
        if opponent == "Gradual":
            out[opponent] = {s: gradual_pools[s][opponent] for s in gradual_pools if opponent in gradual_pools[s]}
        else:
            out[opponent] = {s: main_pools[s][opponent] for s in main_pools if opponent in main_pools[s]}
    return out


def filter_pool_by_depth(seeds_with_opp: Dict[int, tuple], depth_range: range) -> Dict[int, tuple]:
    out = {}
    for seed, (frequencies, series_by_genotype, identities) in seeds_with_opp.items():
        keep = [g for g in frequencies if g in series_by_genotype and effective_depth_of_identity(identities[g]) in depth_range]
        if keep:
            out[seed] = ({g: frequencies[g] for g in keep}, {g: series_by_genotype[g] for g in keep}, identities)
    return out


def average_edges_over_draws(pool: Dict[int, tuple], vocab, group_size: int, n_draws: int, rng: random.Random) -> Optional[Dict[Tuple[str, str], float]]:
    if len(pool) < group_size:
        return None
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    for _ in range(n_draws):
        series, _idents = resample_cross_seed_group(pool, group_size, rng)
        edges = dict(build_edges(series, vocab))
        edges.update({(n, n): w for n, w in self_loop_edges(series, vocab).items()})
        for k, v in edges.items():
            sums[k] += v
    return {k: v / n_draws for k, v in sums.items()}


def edges_to_rows(edges: Dict[Tuple[str, str], float]) -> List[list]:
    return [[a, b, round(v, 4)] for (a, b), v in sorted(edges.items())]


def build_human_section(human_weights_path: Path) -> Dict[str, Dict[str, list]]:
    human_weights = load_human_weights(human_weights_path)
    out: Dict[str, Dict[str, list]] = defaultdict(dict)
    for (label, group), edges in human_weights.items():
        out[label][group] = edges_to_rows(edges)
    return dict(out)


def process_one_age(
    data_dir_str: str, gradual_reprobe_dir_str: str, moves_suffix: str, probe_noise_rate: float,
    age: int, seeds: List[int], d1: int, d2: int, max_depth: int, group_size: int, n_draws: int,
    master_seed: str,
) -> dict:
    data_dir, gradual_reprobe_dir = Path(data_dir_str), Path(gradual_reprobe_dir_str)
    ranges = depth_ranges(d1, d2, max_depth)
    vocabs = {opp: load_vocab(OPPONENT_INFO[opp][1]) for opp in GROUP_OPPONENTS}
    rng = random.Random(f"{master_seed}_{age}")  # one deterministic stream per age -- order-independent, parallelizable

    seeds_with_opp_by_opponent = seeds_with_opp_by_opponent_for_age(
        data_dir, gradual_reprobe_dir, moves_suffix, probe_noise_rate, age, seeds
    )

    by_opponent: Dict[str, Dict[str, Optional[list]]] = {}
    for opponent in GROUP_OPPONENTS:
        seeds_with_opp = seeds_with_opp_by_opponent[opponent]
        vocab = vocabs[opponent]
        rows_by_grouping: Dict[str, Optional[list]] = {}

        edges = average_edges_over_draws(seeds_with_opp, vocab, group_size, n_draws, rng)
        rows_by_grouping["cross_lineage"] = edges_to_rows(edges) if edges is not None else None

        for grouping in DEPTH_GROUPINGS:
            restricted = filter_pool_by_depth(seeds_with_opp, ranges[grouping])
            edges = average_edges_over_draws(restricted, vocab, group_size, n_draws, rng)
            rows_by_grouping[grouping] = edges_to_rows(edges) if edges is not None else None

        by_opponent[opponent] = rows_by_grouping

    return {"age": age, "by_opponent": by_opponent}


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def node_appearance_rates_over_draws(pool: Dict[int, tuple], vocab, group_size: int, n_draws: int, rng: random.Random) -> Optional[Dict[str, float]]:
    """--nodes-only (section 12): for every vocab word, the fraction of
    the group_size drawn virtual participants whose 3-gram set (same
    per-participant parsing as build_edges, via series_to_word_set)
    contains that word at least once, averaged over n_draws draws (a
    draw where nobody has the word contributes 0). Consumes exactly one
    resample_cross_seed_group(pool, group_size, rng) call per draw --
    the same call, with the same arguments, that average_edges_over_draws
    makes -- so calling this in the same position in the same per-age
    rng stream as the edge-weight export reproduces the IDENTICAL draws."""
    if len(pool) < group_size:
        return None
    sums: Dict[str, float] = {word: 0.0 for word in vocab}
    for _ in range(n_draws):
        series, _idents = resample_cross_seed_group(pool, group_size, rng)
        counts: Counter = Counter()
        for s in series:
            counts.update(series_to_word_set(s, vocab))
        for word in vocab:
            sums[word] += counts.get(word, 0) / group_size
    return {word: round(sums[word] / n_draws, 4) for word in vocab}


def process_one_age_nodes(
    data_dir_str: str, gradual_reprobe_dir_str: str, moves_suffix: str, probe_noise_rate: float,
    age: int, seeds: List[int], d1: int, d2: int, max_depth: int, group_size: int, n_draws: int,
    master_seed: str,
) -> dict:
    """Mirrors process_one_age's exact control flow and RNG consumption
    order (same per-age seed f"{master_seed}_{age}", same pool loading,
    same opponent loop, same cross_lineage-then-3-depth-groupings order)
    so that, run with the SAME arguments as the edge-weight export, it
    draws the IDENTICAL 200 cross-lineage groups per (opponent, age,
    grouping) -- just computing each 3-gram's appearance rate from them
    instead of edge Jaccards. Never reads or writes anything related to
    the edge-weights output, and process_one_age itself is untouched, so
    the existing edge-weights JSON's values cannot change."""
    data_dir, gradual_reprobe_dir = Path(data_dir_str), Path(gradual_reprobe_dir_str)
    ranges = depth_ranges(d1, d2, max_depth)
    vocabs = {opp: load_vocab(OPPONENT_INFO[opp][1]) for opp in GROUP_OPPONENTS}
    rng = random.Random(f"{master_seed}_{age}")

    seeds_with_opp_by_opponent = seeds_with_opp_by_opponent_for_age(
        data_dir, gradual_reprobe_dir, moves_suffix, probe_noise_rate, age, seeds
    )

    by_opponent: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    for opponent in GROUP_OPPONENTS:
        seeds_with_opp = seeds_with_opp_by_opponent[opponent]
        vocab = vocabs[opponent]
        rates_by_grouping: Dict[str, Optional[Dict[str, float]]] = {}

        rates_by_grouping["cross_lineage"] = node_appearance_rates_over_draws(seeds_with_opp, vocab, group_size, n_draws, rng)

        for grouping in DEPTH_GROUPINGS:
            restricted = filter_pool_by_depth(seeds_with_opp, ranges[grouping])
            rates_by_grouping[grouping] = node_appearance_rates_over_draws(restricted, vocab, group_size, n_draws, rng)

        by_opponent[opponent] = rates_by_grouping

    return {"age": age, "by_opponent": by_opponent}


def build_abm_nodes_section(
    data_dir: Path, gradual_reprobe_dir: Path, moves_suffix: str, probe_noise_rate: float,
    ages: List[int], seeds: List[int], d1: int, d2: int, max_depth: int,
    group_size: int, n_draws: int, master_seed: str, workers: int, cache_dir: Path,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age_nodes, str(data_dir), str(gradual_reprobe_dir), moves_suffix, probe_noise_rate,
                age, seeds, d1, d2, max_depth, group_size, n_draws, master_seed,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    out: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {opp: {g: {} for g in GROUPINGS} for opp in GROUP_OPPONENTS}
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        for opponent, rates_by_grouping in result["by_opponent"].items():
            for grouping, rates in rates_by_grouping.items():
                if rates is not None:
                    out[opponent][grouping][age] = rates
    return out


def build_abm_section(
    data_dir: Path, gradual_reprobe_dir: Path, moves_suffix: str, probe_noise_rate: float,
    ages: List[int], seeds: List[int], d1: int, d2: int, max_depth: int,
    group_size: int, n_draws: int, master_seed: str, workers: int, cache_dir: Path,
) -> Dict[str, Dict[str, Dict[int, list]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age, str(data_dir), str(gradual_reprobe_dir), moves_suffix, probe_noise_rate,
                age, seeds, d1, d2, max_depth, group_size, n_draws, master_seed,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    out: Dict[str, Dict[str, Dict[int, list]]] = {opp: {g: {} for g in GROUPINGS} for opp in GROUP_OPPONENTS}
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        for opponent, rows_by_grouping in result["by_opponent"].items():
            for grouping, rows in rows_by_grouping.items():
                if rows is not None:
                    out[opponent][grouping][age] = rows
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--gradual-reprobe-dir", type=str, required=True)
    p.add_argument("--moves-suffix", type=str, default="noise0p01")
    p.add_argument("--probe-noise-rate", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--generations-start", type=int, default=0)
    p.add_argument("--generations-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--depth-cutpoints", type=str, required=True, help="'D1,D2' -- use the SAME cutpoints export_depth_groups_for_r.py proposed/used, so the depth groups agree across both exports")
    p.add_argument("--max-depth", type=int, default=6, help="matches --max-memory-depth of the production evolution run")
    p.add_argument("--human-weights", type=str, default="docs/human_network_weights.csv")
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--master-seed", type=str, default="markov_edge_weights_viewer_v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-path", type=str, required=True, help="edge-weights output path; also used to derive the sibling markov_node_freq.json path under --nodes-only")
    p.add_argument(
        "--nodes-only", action="store_true",
        help="compute ONLY node appearance rates (section 12, exploratory) instead of edge weights. "
        "Writes <out-path's directory>/markov_node_freq.json and leaves --out-path itself untouched "
        "(does not read, write, or recompute the edge-weights JSON). Uses the identical per-age RNG "
        "stream and draw order as the edge-weight export, so re-running with this flag against the "
        "same arguments reproduces the exact same 200 cross-lineage draws that produced the "
        "already-written edge-weights JSON.",
    )
    args = p.parse_args()

    d1, d2 = (int(x) for x in args.depth_cutpoints.split(","))
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.generations_start, args.generations_end + 1, args.probe_interval))
    out_path = Path(args.out_path)

    if args.nodes_only:
        node_out_path = out_path.parent / "markov_node_freq.json"
        node_cache_dir = out_path.parent / "_age_cache_node_freq"
        abm_nodes_section = build_abm_nodes_section(
            Path(args.data_dir), Path(args.gradual_reprobe_dir), args.moves_suffix, args.probe_noise_rate,
            ages, seeds, d1, d2, args.max_depth, args.group_size, args.n_draws, args.master_seed,
            args.workers, node_cache_dir,
        )
        node_payload = {"abm_nodes": abm_nodes_section}
        node_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(node_out_path, "w", encoding="utf-8") as f:
            json.dump(node_payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"wrote {node_out_path} ({node_out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return

    cache_dir = out_path.parent / "_age_cache_edge_weights"

    human_section = build_human_section(Path(args.human_weights))
    abm_section = build_abm_section(
        Path(args.data_dir), Path(args.gradual_reprobe_dir), args.moves_suffix, args.probe_noise_rate,
        ages, seeds, d1, d2, args.max_depth, args.group_size, args.n_draws, args.master_seed,
        args.workers, cache_dir,
    )

    payload = {"max_depth": args.max_depth, "depth_cutpoints": [d1, d2], "human": human_section, "abm": abm_section}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
