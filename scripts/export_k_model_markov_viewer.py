"""K-model Markov-network viewer export -- PRODUCTION MODE
(docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 11; type naming convention:
docs/PREREGISTRATION_k_model_markov.md section 1-1, code identifiers
K_Secure/K_Dismissing/K_Preoccupied/K_Fearful).

Sampling scheme (production, scheme B / cross-lineage): every one of the
5 groups (K_Secure, K_Dismissing, K_Preoccupied, K_Fearful, and the
combined Mix) is drawn via evopersonality_abm.k_model_cross_lineage's
draw_cross_lineage_identities -- group_size DISTINCT seeds, one
individual per seed, frequency-weighted within that seed -- the SAME
sampling scheme and the SAME effective-group-size shrinking rule
(docs/DEVIATION_k_model_group_size.md: group_size =
min(human-matched target, seeds that actually carry this type at this
age)) as scripts/run_k_model_confirmatory_analysis.py's own H-A. The 3
groups compared directly to a human group use that group's target size
(K_Secure->56, Mix->46, K_Fearful->69); K_Dismissing/K_Preoccupied use
Mix's target (46, docs/K_MODEL_MARKOV_DESIGN.md section 8-1 point 5 --
no individual human-group size applies to them).

This REPLACES the pilot-only merged-pool sampling
(build_seed_pool_for_type/merge_pools_across_seeds/pool_size, now
removed) that an earlier version of this script used when only 3-5
seeds were available -- see git history before this rewrite for that
version. Reuses evopersonality_abm/markov_recovery.py and markov_apply.py
UNCHANGED (design memo section 6-1).

Per (opponent x grouping x age), writes:
  - edge weights (average Jaccard over --n-draws cross-lineage draws)
    and node appearance rates, in the SAME JSON schema as
    claude/markov-production-analysis's viewer export
    (scripts/export_markov_edge_weights_for_viewer.py):
      edges: {"human": {opp_label: {group: [[from,to,jaccard],...]}},
              "abm": {opponent: {grouping: {age: [[from,to,avg_jaccard],...]}}},
              "caveat": CAVEAT_TEXT}
      nodes: {"abm_nodes": {opponent: {grouping: {age: {word: rate}}}},
              "caveat": CAVEAT_TEXT}
    ("human"/caveat added; no "human_nodes" section, same limitation as
    the Lindgren-side export -- raw human 3-gram text isn't in this
    repository, only the precomputed edge table.)
  - widest-path recovery/descent (markov_apply.bootstrap_aggregate_with_sampler)
  - forced-bottom recovery-time summary (never_recovered_rate, median
    recovery round) -- pooled directly across all probed repeats of that
    (opponent, type, age) with NO resampling, same convention as
    scripts/run_k_model_confirmatory_analysis.py's H-B/H-C point
    estimate (unaffected by the cross-lineage sampling change above,
    which only applies to the network/edge metrics)
  - the effective group size actually used per (grouping, age)
    (docs/DEVIATION_k_model_group_size.md's ha_effective_group_size,
    same rule applied here)

Also reports individuals-per-type-per-generation counts.

Reads raw per-age census/moves/recovery data (Mac-local only, gitignored
in production) -- THIS SCRIPT MUST RUN ON THE MAC where they live for a
production run. Processes ages in parallel (--workers) and caches each
age's raw contribution once to --out-dir/_age_cache/age_<N>.pkl; killing
and rerunning skips ages whose cache already exists (same pattern as
scripts/run_k_model_confirmatory_analysis.py).

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 11 for the full
Mac command and time estimate):
    python3 scripts/export_k_model_markov_viewer.py \\
        --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \\
        --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \\
        --window-start 1000 --window-end 5000 --n-draws 200 --workers 5 \\
        --out-dir scripts/output/coupling_markov_production/viewer
"""

import argparse
import csv
import gzip
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
import run_coupling_markov_pilot as pilot_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.k_model_cross_lineage import (  # noqa: E402
    draw_cross_lineage_identities,
    effective_group_size,
    load_series_first_rounds,
    lookup_series,
    merge_type_membership,
    seeds_with_type,
    type_membership_from_labels,
)
from evopersonality_abm.k_model_probe import FIXED_OPPONENTS, OPPONENTS, SELFPLAY_OPPONENTS  # noqa: E402
from evopersonality_abm.markov_apply import (  # noqa: E402
    bootstrap_aggregate_with_sampler,
    load_human_weights,
)
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges, series_from_moves  # noqa: E402

ALL_OPPONENTS = FIXED_OPPONENTS + SELFPLAY_OPPONENTS  # TFT/AllC/AllD/Random/Gradual/WSLS/SELF/POP
OPPONENT_VOCAB = {name: spec["vocab"] for name, spec in OPPONENTS.items()}
OPPONENT_VOCAB["SELF"] = "RAN"
OPPONENT_VOCAB["POP"] = "RAN"

# human_network_weights.csv's "opponent" column label -- only these 5
# opponents have human data (WSLS/SELF/POP do not: design memo section 3.4).
HUMAN_LABEL = {"TFT": "TFT", "AllC": "ALC", "AllD": "ALD", "Random": "RAN", "Gradual": "GRA"}

TYPES = ["K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"]
GROUPINGS = TYPES + ["Mix"]
TYPE_TO_HUMAN = {"K_Secure": "Secure", "K_Dismissing": "Mix", "K_Preoccupied": "Mix", "K_Fearful": "Fearful"}
# docs/DEVIATION_k_model_group_size.md: the TARGET group size (before the
# effective-group-size shrinking rule). K_Dismissing/K_Preoccupied use
# Mix's target (docs/K_MODEL_MARKOV_DESIGN.md section 8-1 point 5: no
# individual human-group size applies to them).
TARGET_GROUP_SIZE = {"K_Secure": 56, "K_Dismissing": 46, "K_Preoccupied": 46, "K_Fearful": 69, "Mix": 46}
CAVEAT_TEXT = "中央値で無理やり分けた型であり、集団は連続した一つの雲"


def classify_type(k_out: float, k_self: float, threshold_kout: float, threshold_kself: float) -> str:
    """K-安定型(K_Secure)=高K_self・高K_out、K-拒絶型(K_Dismissing)=高K_self・低K_out、
    K-とらわれ型(K_Preoccupied)=低K_self・高K_out、K-恐れ型(K_Fearful)=低K_self・低K_out
    (docs/K_MODEL_MARKOV_DESIGN.md section 8-1 point 5; naming convention:
    docs/PREREGISTRATION_k_model_markov.md section 1-1. Ties go to the "high"
    side, matching numpy.median's own convention of an exact-median value
    landing on neither side specifically -- an arbitrary but harmless choice
    at continuous-valued K)."""
    high_self = k_self >= threshold_kself
    high_out = k_out >= threshold_kout
    if high_self and high_out:
        return "K_Secure"
    if high_self and not high_out:
        return "K_Dismissing"
    if not high_self and high_out:
        return "K_Preoccupied"
    return "K_Fearful"


def load_census_with_k(path: Path) -> List[Tuple[int, int, int, str, str, float, float]]:
    """[(genotype_id, frequency, memory_depth, table, prehistory, k_out, k_self), ...]"""
    out = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            out.append((
                int(row["genotype_id"]), int(row["frequency"]), int(row["memory_depth"]),
                row["table"], row["prehistory"], float(row["k_out"]), float(row["k_self"]),
            ))
    return out


def load_series_by_genotype_all_opponents(moves_path: Path, main_analysis_rounds: int = 20) -> Dict[str, Dict[int, List[str]]]:
    """One pass over a seed_X_probe_age_G_moves.csv.gz file: {opponent:
    {genotype_id: [series, ...]}}. Kept as the naive (read-every-row)
    reference implementation -- production code paths use
    k_model_cross_lineage.load_series_first_rounds instead (skips rows
    past round main_analysis_rounds, ~4x faster); a test checks the two
    agree."""
    own: Dict[Tuple[str, int, int], Dict[int, str]] = defaultdict(dict)
    opp: Dict[Tuple[str, int, int], Dict[int, str]] = defaultdict(dict)
    with gzip.open(moves_path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["opponent"], int(row["genotype_id"]), int(row["repeat"]))
            rnd = int(row["round"])
            own[key][rnd] = row["own_move"]
            opp[key][rnd] = row["opp_move"]

    result: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for (opponent, gid, repeat), own_by_round in own.items():
        max_round = max(own_by_round)
        own_moves = [own_by_round[i] for i in range(1, max_round + 1)]
        opp_moves = [opp[(opponent, gid, repeat)][i] for i in range(1, max_round + 1)]
        series = series_from_moves(own_moves, opp_moves, n_rounds=main_analysis_rounds)
        result[opponent][gid].append(series)
    return {k: dict(v) for k, v in result.items()}


def load_recovery(path: Path) -> List[Tuple[int, str, int, Optional[int]]]:
    """[(genotype_id, opponent, repeat, recovery_round_or_None), ...]"""
    out = []
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            rr = int(row["recovery_round"]) if row["recovery_round"] != "" else None
            out.append((int(row["genotype_id"]), row["opponent"], int(row["repeat"]), rr))
    return out


def census_true_types(census, threshold_kout: float, threshold_kself: float) -> Dict[int, str]:
    return {gid: classify_type(k_out, k_self, threshold_kout, threshold_kself) for gid, _f, _md, _t, _p, k_out, k_self in census}


def discover_ages(data_dir: Path) -> List[int]:
    ages = set()
    for p in data_dir.glob("seed_*_probe_age_*_census.csv.gz"):
        parts = p.name.split("_")
        ages.add(int(parts[parts.index("age") + 1]))
    return sorted(ages)


def edges_to_rows(edges: Dict[Tuple[str, str], float]) -> List[list]:
    return [[a, b, round(v, 4)] for (a, b), v in sorted(edges.items())]


def average_edges_over_draws(
    pools: Dict[int, tuple], vocab, group_size: int, n_draws: int, opponent: str,
    series_by_seed: Dict[int, Dict[str, Dict[int, list]]], rng: random.Random,
) -> Optional[Dict[Tuple[str, str], float]]:
    """Cross-lineage (scheme B): draws group_size DISTINCT seeds' worth of
    individuals n_draws times, averaging every edge's Jaccard across draws
    (an edge absent from a draw contributes 0 -- same convention as
    markov_apply.bootstrap_aggregate)."""
    if group_size == 0:
        return None
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    for _ in range(n_draws):
        identities = draw_cross_lineage_identities(pools, group_size, rng)
        series = lookup_series(identities, series_by_seed, opponent, rng)
        edges = dict(build_edges(series, vocab))
        edges.update({(n, n): w for n, w in self_loop_edges(series, vocab).items()})
        for k, v in edges.items():
            sums[k] += v
    return {k: v / n_draws for k, v in sums.items()}


def node_appearance_rates_over_draws(
    pools: Dict[int, tuple], vocab, group_size: int, n_draws: int, opponent: str,
    series_by_seed: Dict[int, Dict[str, Dict[int, list]]], rng: random.Random,
) -> Optional[Dict[str, float]]:
    if group_size == 0:
        return None
    vocab_sorted = sorted(vocab)
    sums: Dict[str, float] = {word: 0.0 for word in vocab_sorted}
    for _ in range(n_draws):
        identities = draw_cross_lineage_identities(pools, group_size, rng)
        series = lookup_series(identities, series_by_seed, opponent, rng)
        counts: Dict[str, int] = defaultdict(int)
        for s in series:
            for word in set(n for n in (x.strip().rstrip("。") for x in s.split(",")) if n in vocab):
                counts[word] += 1
        for word in vocab_sorted:
            sums[word] += counts.get(word, 0) / group_size
    return {word: round(sums[word] / n_draws, 4) for word in vocab_sorted}


def summarize_recovery(age: int, opponent: str, type_name: str, recovery_rounds: List[Optional[int]]) -> Optional[dict]:
    n = len(recovery_rounds)
    if n == 0:
        return None
    recovered = [r for r in recovery_rounds if r is not None]
    return {
        "age": age, "opponent": opponent, "type": type_name, "n": n,
        "n_recovered": len(recovered),
        "median_recovery_round": statistics.median(recovered) if recovered else "",
        "never_recovered_rate": round(1 - len(recovered) / n, 4),
    }


def build_human_section(human_weights_path: Path) -> Dict[str, Dict[str, list]]:
    """{csv_label: {human_group: edge_rows}} -- static (not age-dependent),
    computed once from docs/human_network_weights.csv."""
    human_weights = load_human_weights(human_weights_path)
    out: Dict[str, Dict[str, list]] = defaultdict(dict)
    for (label, group), edges in human_weights.items():
        out[label][group] = edges_to_rows(edges)
    return dict(out)


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def process_one_age(
    data_dir: Path, age: int, seeds: List[int], threshold_kout: float, threshold_kself: float,
    n_draws: int, master_seed: str,
) -> dict:
    rng = random.Random(f"{master_seed}_{age}")

    per_seed_census: Dict[int, List[Tuple]] = {}
    per_seed_series: Dict[int, Dict[str, Dict[int, List[str]]]] = {}
    per_seed_recovery: Dict[int, List[Tuple[int, str, int, Optional[int]]]] = {}
    for seed in seeds:
        cpath = pilot_mod.census_path(data_dir, seed, age)
        mpath = pilot_mod.moves_path(data_dir, seed, age)
        rpath = pilot_mod.recovery_path(data_dir, seed, age)
        if not (cpath.exists() and mpath.exists() and rpath.exists()):
            continue
        per_seed_census[seed] = load_census_with_k(cpath)
        per_seed_series[seed] = load_series_first_rounds(mpath)
        per_seed_recovery[seed] = load_recovery(rpath)

    true_type_by_seed = {seed: census_true_types(census, threshold_kout, threshold_kself) for seed, census in per_seed_census.items()}

    type_count_rows = []
    for type_name in TYPES:
        total = sum(
            freq for seed, census in per_seed_census.items()
            for gid, freq, _md, _t, _p, _k_out, _k_self in census
            if true_type_by_seed[seed].get(gid) == type_name
        )
        type_count_rows.append({"age": age, "type": type_name, "n_individuals": total})

    base_pools: Dict[str, Dict[int, tuple]] = {}
    for type_name in TYPES:
        base_pools[type_name] = {
            seed: type_membership_from_labels(census, true_type_by_seed[seed], type_name)
            for seed, census in per_seed_census.items()
        }
    base_pools["Mix"] = {
        seed: merge_type_membership(base_pools["K_Dismissing"][seed], base_pools["K_Preoccupied"][seed])
        for seed in per_seed_census
    }

    edges_by_opponent: Dict[str, Dict[str, list]] = {}  # {opponent: {grouping: rows}}
    nodes_by_opponent: Dict[str, Dict[str, dict]] = {}
    widest_path_rows = []
    effective_gs: Dict[str, int] = {}

    for grouping in GROUPINGS:
        pools = base_pools[grouping]
        n_avail = len(seeds_with_type(pools))
        gs = effective_group_size(TARGET_GROUP_SIZE[grouping], n_avail)
        effective_gs[grouping] = gs
        if gs == 0:
            continue

        for opponent in ALL_OPPONENTS:
            vocab = load_vocab(OPPONENT_VOCAB[opponent])

            edges = average_edges_over_draws(pools, vocab, gs, n_draws, opponent, per_seed_series, rng)
            if edges is not None:
                edges_by_opponent.setdefault(opponent, {})[grouping] = edges_to_rows(edges)

            rates = node_appearance_rates_over_draws(pools, vocab, gs, n_draws, opponent, per_seed_series, rng)
            if rates is not None:
                nodes_by_opponent.setdefault(opponent, {})[grouping] = rates

            def sample_fn(rng_, pools_=pools, gs_=gs, opponent_=opponent):
                identities = draw_cross_lineage_identities(pools_, gs_, rng_)
                series = lookup_series(identities, per_seed_series, opponent_, rng_)
                idents = [pools_[s][1][g] for s, g in identities]
                return series, idents

            summary = bootstrap_aggregate_with_sampler(sample_fn, opponent, n_draws, rng, {}, vocab)
            widest_path_rows.append({"opponent": opponent, "grouping": grouping, "age": age, "group_size": gs, **summary})

    # recovery time: pooled directly (no resampling), per type (not Mix -- Mix is
    # a combined NETWORK view only; recovery pooling for K_Dismissing+K_Preoccupied
    # individually already covers that population, a separate "Mix recovery" would
    # double count). UNAFFECTED by the cross-lineage sampling change above.
    recovery_rows = []
    for opponent in ALL_OPPONENTS:
        for type_name in TYPES:
            rounds_for_type: List[Optional[int]] = []
            for seed, recovery in per_seed_recovery.items():
                for gid, opp_name, _repeat, rr in recovery:
                    if opp_name == opponent and true_type_by_seed[seed].get(gid) == type_name:
                        rounds_for_type.append(rr)
            row = summarize_recovery(age, opponent, type_name, rounds_for_type)
            if row is not None:
                recovery_rows.append(row)

    return {
        "age": age, "type_count_rows": type_count_rows, "widest_path_rows": widest_path_rows,
        "recovery_rows": recovery_rows, "edges_by_opponent": edges_by_opponent, "nodes_by_opponent": nodes_by_opponent,
        "effective_group_size": effective_gs,
    }


def run_all_ages(
    data_dir: Path, ages: List[int], seeds: List[int], threshold_kout: float, threshold_kself: float,
    n_draws: int, master_seed: str, workers: int, cache_dir: Path,
) -> List[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_one_age, data_dir, age, seeds, threshold_kout, threshold_kself, n_draws, master_seed): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    results = []
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            results.append(pickle.load(f))
    return results


def write_csv(path: Path, rows: List[dict]) -> None:
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
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--thresholds-json", type=str, required=True)
    p.add_argument("--human-weights-csv", type=str, default=str(REPO_ROOT / "docs" / "human_network_weights.csv"))
    p.add_argument("--window-start", type=int, default=0, help="0 = no lower bound (use every discovered age)")
    p.add_argument("--window-end", type=int, default=10**9, help="large default = no upper bound")
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--master-seed", type=str, default="k_model_markov_viewer_v2")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    with open(args.thresholds_json) as f:
        thresholds = json.load(f)
    threshold_kout = thresholds["threshold_kout"]
    threshold_kself = thresholds["threshold_kself"]

    all_ages = discover_ages(data_dir)
    ages = [a for a in all_ages if args.window_start <= a <= args.window_end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_age_cache"

    results = run_all_ages(data_dir, ages, seeds, threshold_kout, threshold_kself, args.n_draws, args.master_seed, args.workers, cache_dir)

    type_count_rows: List[dict] = []
    widest_path_rows: List[dict] = []
    recovery_rows: List[dict] = []
    effective_group_size_rows: List[dict] = []
    edges_by_opponent: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    nodes_by_opponent: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for result in results:
        age = result["age"]
        type_count_rows.extend(result["type_count_rows"])
        widest_path_rows.extend(result["widest_path_rows"])
        recovery_rows.extend(result["recovery_rows"])
        for grouping, gs in result["effective_group_size"].items():
            effective_group_size_rows.append({"age": age, "grouping": grouping, "effective_group_size": gs})
        for opponent, by_grouping in result["edges_by_opponent"].items():
            for grouping, rows in by_grouping.items():
                edges_by_opponent[opponent][grouping][age] = rows
        for opponent, by_grouping in result["nodes_by_opponent"].items():
            for grouping, rates in by_grouping.items():
                nodes_by_opponent[opponent][grouping][age] = rates

    type_count_rows.sort(key=lambda r: (r["type"], r["age"]))
    widest_path_rows.sort(key=lambda r: (r["opponent"], r["grouping"], r["age"]))
    recovery_rows.sort(key=lambda r: (r["opponent"], r["type"], r["age"]))
    effective_group_size_rows.sort(key=lambda r: (r["grouping"], r["age"]))

    write_csv(out_dir / "k_model_type_counts.csv", type_count_rows)
    write_csv(out_dir / "k_model_widest_path.csv", widest_path_rows)
    write_csv(out_dir / "k_model_recovery_time.csv", recovery_rows)
    write_csv(out_dir / "k_model_effective_group_size.csv", effective_group_size_rows)

    human_section = build_human_section(Path(args.human_weights_csv))

    with open(out_dir / "k_model_edges.json", "w", encoding="utf-8") as f:
        json.dump({"human": human_section, "abm": edges_by_opponent, "caveat": CAVEAT_TEXT}, f, ensure_ascii=False, separators=(",", ":"))
    with open(out_dir / "k_model_nodes.json", "w", encoding="utf-8") as f:
        json.dump({"abm_nodes": nodes_by_opponent, "caveat": CAVEAT_TEXT}, f, ensure_ascii=False, separators=(",", ":"))

    print(f"wrote {out_dir / 'k_model_type_counts.csv'} ({len(type_count_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_widest_path.csv'} ({len(widest_path_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_recovery_time.csv'} ({len(recovery_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_effective_group_size.csv'} ({len(effective_group_size_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_edges.json'}")
    print(f"wrote {out_dir / 'k_model_nodes.json'}")


if __name__ == "__main__":
    main()
