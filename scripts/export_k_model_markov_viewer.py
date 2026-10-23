"""Per-type (白/青/緑/ピンク + combined Mix) Markov-network viewer export
for the K-model pilot -- docs/K_MODEL_MARKOV_DESIGN.md section 8-6.

Reuses evopersonality_abm/markov_recovery.py and markov_apply.py
UNCHANGED (section 6-1 of the design memo: "相手がKモデルの個体でも...
このファイルは変更せずそのまま使えると見込む"). The only new code here
is: classifying each probed genotype into one of the 4 K-model types by
the FIXED pooled thresholds (scripts/compute_k_model_type_thresholds.py),
and building the {seed: (frequencies, series_by_genotype, identities)}
pools markov_apply's resampling functions expect, filtered per type.

Sampling scheme (IMPORTANT, pilot-specific): markov_apply's
resample_cross_seed_group ("系統間抽出") requires >= group_size DISTINCT
SEEDS and is the scheme production Lindgren runs use (56-69 seeds). A
3-5 seed pilot cannot supply that -- so this script instead MERGES all
probed seeds' per-type genotype pools into one combined pool (global,
collision-free genotype ids) and draws from it with
resample_within_seed_group_with_identity (frequency-weighted, WITH
replacement -- same per-draw mechanics as within-seed "集団内抽出", just
applied across the pilot's pooled seeds instead of one seed's own
population). This is NOT the cross-lineage scheme and the two are not
interchangeable -- see docs/K_MODEL_MARKOV_DESIGN.md section 8-7 for the
caveat this implies for interpreting pilot output. A production run with
enough seeds (>= the human group size being compared against) should use
true cross-lineage sampling instead.

Per (opponent x type x age), writes:
  - edge weights (average Jaccard over --n-draws) and node appearance
    rates, in the SAME schema as markov_production_analysis's viewer
    JSON ({type: {age: [[from,to,weight],...]}} / {type: {age: {word: rate}}})
  - widest-path recovery/descent (markov_apply.bootstrap_aggregate_with_sampler)
  - forced-bottom recovery-time summary (never_recovered_rate, median
    recovery round) -- pooled directly across all probed repeats of that
    (opponent, type, age) with NO resampling (same convention as
    export_markov_selfplay_recovery.py's recovery-time metric, which
    also pools directly rather than bootstrapping)

Also reports individuals-per-type-per-generation counts (checking
whether some type has too few members to build a network at all).

Does NOT read or write any pure-Lindgren production output.

Usage:
    python3 scripts/export_k_model_markov_viewer.py \\
        --data-dir scripts/output/coupling_markov_pilot --seeds 4 --seed-offset 100 \\
        --thresholds-json scripts/output/coupling_markov_pilot/analysis/type_thresholds.json \\
        --group-size 20 --n-draws 200 --out-dir scripts/output/coupling_markov_pilot/analysis
"""

import argparse
import csv
import gzip
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import run_coupling_markov_pilot as pilot_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.k_model_probe import FIXED_OPPONENTS, OPPONENTS, SELFPLAY_OPPONENTS  # noqa: E402
from evopersonality_abm.markov_apply import (  # noqa: E402
    HUMAN_GROUP_SIZES,
    resample_within_seed_group_with_identity,
    bootstrap_aggregate_with_sampler,
)
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges, series_from_moves  # noqa: E402

ALL_OPPONENTS = FIXED_OPPONENTS + SELFPLAY_OPPONENTS  # TFT/AllC/AllD/Random/Gradual/WSLS/SELF/POP
OPPONENT_VOCAB = {name: spec["vocab"] for name, spec in OPPONENTS.items()}
OPPONENT_VOCAB["SELF"] = "RAN"
OPPONENT_VOCAB["POP"] = "RAN"

TYPES = ["White", "Blue", "Green", "Pink"]
TYPE_TO_HUMAN = {"White": "Secure", "Blue": "Mix", "Green": "Mix", "Pink": "Fearful"}
# docs/K_MODEL_MARKOV_DESIGN.md section 8-1 point 5: no single human-group
# size applies to Blue/Green individually (they only become "Mix"-sized
# once merged) -- use that merged target size for each anyway, for a
# consistent default (not mandated by Annen's decision, documented here).
TYPE_GROUP_SIZE = {"White": HUMAN_GROUP_SIZES["Secure"], "Blue": HUMAN_GROUP_SIZES["Mix"], "Green": HUMAN_GROUP_SIZES["Mix"], "Pink": HUMAN_GROUP_SIZES["Fearful"]}
CAVEAT_TEXT = "中央値で無理やり分けた型であり、集団は連続した一つの雲"


def classify_type(k_out: float, k_self: float, threshold_kout: float, threshold_kself: float) -> str:
    """白=高K_self・高K_out、青=高K_self・低K_out、緑=低K_self・高K_out、
    ピンク=低K_self・低K_out (docs/K_MODEL_MARKOV_DESIGN.md section 8-1 point 5,
    ties go to the "high" side, matching numpy.median's own convention of
    an exact-median value landing on neither side specifically -- an
    arbitrary but harmless choice at continuous-valued K)."""
    high_self = k_self >= threshold_kself
    high_out = k_out >= threshold_kout
    if high_self and high_out:
        return "White"
    if high_self and not high_out:
        return "Blue"
    if not high_self and high_out:
        return "Green"
    return "Pink"


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
    {genotype_id: [series, ...]}}. Same column schema as pure Lindgren's
    moves files, so this mirrors markov_apply.load_series_by_genotype_all_opponents
    exactly (not imported from there to avoid depending on its probe-age
    moves-suffix convention, which this K-model pilot's filenames don't
    use)."""
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


SeedPool = Tuple[Dict[int, int], Dict[int, List[str]], Dict[int, Tuple[int, str, str]]]


def build_seed_pool_for_type(
    census: List[Tuple[int, int, int, str, str, float, float]],
    series_by_genotype: Dict[int, List[str]],
    type_name: str, threshold_kout: float, threshold_kself: float,
) -> SeedPool:
    """(frequencies, series_by_genotype, identities) for ONE seed's
    genotypes matching `type_name`, in markov_apply's expected pool shape."""
    frequencies: Dict[int, int] = {}
    identities: Dict[int, Tuple[int, str, str]] = {}
    filtered_series: Dict[int, List[str]] = {}
    for gid, freq, memory_depth, table, prehistory, k_out, k_self in census:
        if classify_type(k_out, k_self, threshold_kout, threshold_kself) != type_name:
            continue
        if gid not in series_by_genotype:
            continue
        frequencies[gid] = freq
        identities[gid] = (memory_depth, table, prehistory)
        filtered_series[gid] = series_by_genotype[gid]
    return frequencies, filtered_series, identities


def merge_pools_across_seeds(pools: List[SeedPool]) -> SeedPool:
    """Merges several seeds' pools into one, remapping genotype ids to be
    globally unique (collision-free across seeds) -- see module
    docstring's "pilot-specific sampling scheme" note for why this merged
    pool, rather than markov_apply's distinct-seed cross-lineage scheme,
    is used here."""
    merged_freq: Dict[int, int] = {}
    merged_series: Dict[int, List[str]] = {}
    merged_idents: Dict[int, Tuple[int, str, str]] = {}
    next_id = 0
    for frequencies, series_by_genotype, identities in pools:
        for gid in frequencies:
            merged_freq[next_id] = frequencies[gid]
            merged_series[next_id] = series_by_genotype[gid]
            merged_idents[next_id] = identities[gid]
            next_id += 1
    return merged_freq, merged_series, merged_idents


def pool_size(pool: SeedPool) -> int:
    return len(pool[0])


def average_edges_over_draws(pool: SeedPool, vocab, group_size: int, n_draws: int, rng: random.Random) -> Optional[Dict[Tuple[str, str], float]]:
    frequencies, series_by_genotype, identities = pool
    if pool_size(pool) == 0:
        return None
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    for _ in range(n_draws):
        series, _idents = resample_within_seed_group_with_identity(frequencies, series_by_genotype, identities, group_size, rng)
        edges = dict(build_edges(series, vocab))
        edges.update({(n, n): w for n, w in self_loop_edges(series, vocab).items()})
        for k, v in edges.items():
            sums[k] += v
    return {k: v / n_draws for k, v in sums.items()}


def edges_to_rows(edges: Dict[Tuple[str, str], float]) -> List[list]:
    return [[a, b, round(v, 4)] for (a, b), v in sorted(edges.items())]


def node_appearance_rates_over_draws(pool: SeedPool, vocab, group_size: int, n_draws: int, rng: random.Random) -> Optional[Dict[str, float]]:
    frequencies, series_by_genotype, identities = pool
    if pool_size(pool) == 0:
        return None
    vocab_sorted = sorted(vocab)
    sums: Dict[str, float] = {word: 0.0 for word in vocab_sorted}
    for _ in range(n_draws):
        series, _idents = resample_within_seed_group_with_identity(frequencies, series_by_genotype, identities, group_size, rng)
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


def discover_ages(data_dir: Path) -> List[int]:
    ages = set()
    for p in data_dir.glob("seed_*_probe_age_*_census.csv.gz"):
        parts = p.name.split("_")
        ages.add(int(parts[parts.index("age") + 1]))
    return sorted(ages)


def process_one_age(
    data_dir: Path, age: int, seeds: List[int], threshold_kout: float, threshold_kself: float,
    group_size: int, n_draws: int, master_seed: str,
) -> dict:
    rng = random.Random(f"{master_seed}_agg_{age}")

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
        per_seed_series[seed] = load_series_by_genotype_all_opponents(mpath)
        per_seed_recovery[seed] = load_recovery(rpath)

    type_count_rows = []
    for type_name in TYPES:
        total = 0
        for seed, census in per_seed_census.items():
            total += sum(freq for _gid, freq, _md, _t, _p, k_out, k_self in census if classify_type(k_out, k_self, threshold_kout, threshold_kself) == type_name)
        type_count_rows.append({"age": age, "type": type_name, "n_individuals": total})

    edges_by_grouping: Dict[str, Dict[str, list]] = {}  # {opponent: {grouping: rows}}
    nodes_by_grouping: Dict[str, Dict[str, dict]] = {}
    widest_path_rows = []
    recovery_rows = []

    for opponent in ALL_OPPONENTS:
        vocab = load_vocab(OPPONENT_VOCAB[opponent])
        type_pools: Dict[str, SeedPool] = {}
        for type_name in TYPES:
            seed_pools = []
            for seed in per_seed_census:
                census = per_seed_census[seed]
                series_for_opp = per_seed_series[seed].get(opponent, {})
                seed_pools.append(build_seed_pool_for_type(census, series_for_opp, type_name, threshold_kout, threshold_kself))
            type_pools[type_name] = merge_pools_across_seeds(seed_pools)

        # combined Mix (Blue+Green) for human comparison, decision 5
        type_pools["Mix"] = merge_pools_across_seeds([type_pools["Blue"], type_pools["Green"]])

        for grouping, pool in type_pools.items():
            gs = HUMAN_GROUP_SIZES.get(TYPE_TO_HUMAN.get(grouping, ""), TYPE_GROUP_SIZE.get(grouping, group_size)) if grouping != "Mix" else HUMAN_GROUP_SIZES["Mix"]
            gs = min(gs, group_size) if group_size else gs  # --group-size caps the pilot-feasible size
            if pool_size(pool) == 0:
                continue

            def sample_fn(rng_, pool_=pool, gs_=gs):
                return resample_within_seed_group_with_identity(pool_[0], pool_[1], pool_[2], gs_, rng_)

            summary = bootstrap_aggregate_with_sampler(sample_fn, opponent, n_draws, rng, {}, vocab)
            widest_path_rows.append({"opponent": opponent, "grouping": grouping, "age": age, "group_size": gs, **summary})

            edges = average_edges_over_draws(pool, vocab, gs, n_draws, rng)
            if edges is not None:
                edges_by_grouping.setdefault(opponent, {})[grouping] = edges_to_rows(edges)

            rates = node_appearance_rates_over_draws(pool, vocab, gs, n_draws, rng)
            if rates is not None:
                nodes_by_grouping.setdefault(opponent, {})[grouping] = rates

        # recovery time: pooled directly (no resampling), per type (not Mix -- Mix is
        # a combined NETWORK view only; recovery pooling for Blue+Green individually
        # already covers that population, a separate "Mix recovery" would double count)
        for type_name in TYPES:
            rounds_for_type: List[Optional[int]] = []
            for seed, recovery in per_seed_recovery.items():
                census = per_seed_census[seed]
                type_of_gid = {gid: classify_type(k_out, k_self, threshold_kout, threshold_kself) for gid, _f, _md, _t, _p, k_out, k_self in census}
                for gid, opp_name, _repeat, rr in recovery:
                    if opp_name == opponent and type_of_gid.get(gid) == type_name:
                        rounds_for_type.append(rr)
            row = summarize_recovery(age, opponent, type_name, rounds_for_type)
            if row is not None:
                recovery_rows.append(row)

    return {
        "age": age, "type_count_rows": type_count_rows, "widest_path_rows": widest_path_rows,
        "recovery_rows": recovery_rows, "edges_by_grouping": edges_by_grouping, "nodes_by_grouping": nodes_by_grouping,
    }


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
    p.add_argument("--group-size", type=int, default=0, help="caps every grouping's draw size (0 = no cap, use the human-matched size); set this to the pilot's actual feasible size")
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--master-seed", type=str, default="k_model_markov_viewer_v1")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    with open(args.thresholds_json) as f:
        thresholds = json.load(f)
    threshold_kout = thresholds["threshold_kout"]
    threshold_kself = thresholds["threshold_kself"]

    ages = discover_ages(data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    type_count_rows: List[dict] = []
    widest_path_rows: List[dict] = []
    recovery_rows: List[dict] = []
    edges_by_opponent: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    nodes_by_opponent: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for age in ages:
        result = process_one_age(data_dir, age, seeds, threshold_kout, threshold_kself, args.group_size, args.n_draws, args.master_seed)
        type_count_rows.extend(result["type_count_rows"])
        widest_path_rows.extend(result["widest_path_rows"])
        recovery_rows.extend(result["recovery_rows"])
        for opponent, by_grouping in result["edges_by_grouping"].items():
            for grouping, rows in by_grouping.items():
                edges_by_opponent[opponent][grouping][age] = rows
        for opponent, by_grouping in result["nodes_by_grouping"].items():
            for grouping, rates in by_grouping.items():
                nodes_by_opponent[opponent][grouping][age] = rates
        print(f"age={age} done", flush=True)

    type_count_rows.sort(key=lambda r: (r["type"], r["age"]))
    widest_path_rows.sort(key=lambda r: (r["opponent"], r["grouping"], r["age"]))
    recovery_rows.sort(key=lambda r: (r["opponent"], r["type"], r["age"]))

    write_csv(out_dir / "k_model_type_counts.csv", type_count_rows)
    write_csv(out_dir / "k_model_widest_path.csv", widest_path_rows)
    write_csv(out_dir / "k_model_recovery_time.csv", recovery_rows)

    with open(out_dir / "k_model_edges.json", "w", encoding="utf-8") as f:
        json.dump({"abm": edges_by_opponent, "caveat": CAVEAT_TEXT}, f, ensure_ascii=False, separators=(",", ":"))
    with open(out_dir / "k_model_nodes.json", "w", encoding="utf-8") as f:
        json.dump({"abm_nodes": nodes_by_opponent, "caveat": CAVEAT_TEXT}, f, ensure_ascii=False, separators=(",", ":"))

    print(f"wrote {out_dir / 'k_model_type_counts.csv'} ({len(type_count_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_widest_path.csv'} ({len(widest_path_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_recovery_time.csv'} ({len(recovery_rows)} rows)")
    print(f"wrote {out_dir / 'k_model_edges.json'}")
    print(f"wrote {out_dir / 'k_model_nodes.json'}")


if __name__ == "__main__":
    main()
