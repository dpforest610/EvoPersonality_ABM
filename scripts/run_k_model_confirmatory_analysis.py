"""Confirmatory analysis for the K-model Markov-structure production
study, strictly per docs/PREREGISTRATION_k_model_markov.md sections 4-5
(confirmed commit -- see that document's header).

Reads raw probe data (seed_X_probe_age_G_census.csv.gz / ..._moves.csv.gz /
..._recovery.csv.gz, written by scripts/run_coupling_markov_pilot.py run
at production scale) -- these are Mac-local only (gitignored, never
pushed), so THIS SCRIPT MUST RUN ON THE MAC where they live. It writes
only aggregated results (CSV/JSON tables) to --results-dir, which is what
gets committed and pushed.

Per age in the analysis window (preregistration section 3: gen
1000-5000), computes:
  - H-A's point estimate: for each of 3 types (K_Secure, Mix, K_Fearful)
    and the 5 human opponents (TFT, Random, AllC, AllD, Gradual), the
    scheme-B (cross-lineage) averaged edge-weight network over
    --n-draws-main draws, using a human-group-matched group size
    (56/46/69).
  - H-B/H-C's point estimate: never_recovered_rate (SELF/POP), pooled
    directly per genotype (no draws, no frequency weighting -- same
    convention as scripts/export_k_model_markov_viewer.py's
    summarize_recovery), split by type and (for H-C) by effective-depth
    stratum.
  - The seed-level bootstrap (section 5-2): --n-bootstrap-seed resamples
    of the seed set, giving H-A/H-B/H-C's difference distributions ->
    one-sided p-values -> Holm correction -> verdicts (section 5-3).
  - The label-permutation test (section 5-4): --n-permutations random
    relabelings of each (seed, age) population's type assignment
    (preserving each type's genotype count), recomputing the SAME
    statistic (H-A at --permutation-draws-ha draws per permutation,
    matching the real-label statistic computed at that SAME draw count
    for this test -- section 5-4 step 3; H-B/H-C have no draw step at
    all, so no matching is needed there) to build a null distribution,
    reporting where the true (unpermuted) statistic falls.

Resumable: each age's raw contribution is cached once to
--cache-dir/age_<N>.pkl; killing and rerunning this script skips ages
whose cache already exists (same pattern as
scripts/run_coupling_markov_pilot.py's checkpointing).

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md for the full Mac command):
    venv/bin/python scripts/run_k_model_confirmatory_analysis.py \\
        --data-dir scripts/output/coupling_markov_production \\
        --seeds 70 --seed-offset 200 \\
        --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \\
        --results-dir scripts/output/k_model_confirmatory_production \\
        --workers 5
"""

import argparse
import csv
import gzip
import importlib.util
import json
import pickle
import random
import statistics
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from evopersonality_abm.k_model_confirmatory import (  # noqa: E402
    HA_COMPARISONS,
    HB_COMPARISONS,
    HC_MIN_STRATUM_N,
    HC_STRATA,
    depth_stratum,
    ha_comparison_diff,
    hb_comparison_diff,
    hc_comparison_diff,
    holm_correction,
    one_sided_bootstrap_pvalue,
    permutation_pvalue,
    verdict,
)
from evopersonality_abm.lindgren import LindgrenTable, effective_memory_depth  # noqa: E402
from evopersonality_abm.markov_apply import load_human_weights  # noqa: E402
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges, series_from_moves  # noqa: E402

# Imported the same way scripts/export_k_model_markov_viewer.py imports
# scripts/run_coupling_markov_pilot.py -- a script, not a package module.
_viewer_spec = importlib.util.spec_from_file_location("export_k_model_markov_viewer", SCRIPTS_DIR / "export_k_model_markov_viewer.py")
viewer_mod = importlib.util.module_from_spec(_viewer_spec)
sys.modules["export_k_model_markov_viewer"] = viewer_mod
_viewer_spec.loader.exec_module(viewer_mod)

HUMAN_OPPONENTS = ["TFT", "Random", "AllC", "AllD", "Gradual"]  # concatenation order, preregistration section 4
HA_TYPES = ["K_Secure", "Mix", "K_Fearful"]
HA_GROUP_SIZE = {"K_Secure": 56, "Mix": 46, "K_Fearful": 69}
HUMAN_LABEL = {"TFT": "TFT", "Random": "RAN", "AllC": "ALC", "AllD": "ALD", "Gradual": "GRA"}  # human_network_weights.csv's "opponent" column label
VOCAB_NAME = {"TFT": "TFT", "Random": "RAN", "AllC": "AllC", "AllD": "AllD", "Gradual": "GRA"}  # markov_recovery.load_vocab's name (different spelling for AllC/AllD)


# ------------------------------------------------------------------ loading + pool construction

def load_series_first_rounds(moves_path: Path, main_analysis_rounds: int = 20) -> Dict[str, Dict[int, List[str]]]:
    """Same output as viewer_mod.load_series_by_genotype_all_opponents, but
    rows past round `main_analysis_rounds` are skipped before any parsing:
    series_from_moves only ever reads the first `main_analysis_rounds`
    moves, and the moves file stores 100 rounds per duel, so ~80% of the
    rows were parsed for nothing (tests check the outputs are equal)."""
    own: Dict[Tuple[str, int, int], Dict[int, str]] = defaultdict(dict)
    opp: Dict[Tuple[str, int, int], Dict[int, str]] = defaultdict(dict)
    with gzip.open(moves_path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_gid, i_opp, i_rep, i_rnd = (header.index(c) for c in ("genotype_id", "opponent", "repeat", "round"))
        i_own, i_om = header.index("own_move"), header.index("opp_move")
        for row in reader:
            rnd = int(row[i_rnd])
            if rnd > main_analysis_rounds:
                continue
            key = (row[i_opp], int(row[i_gid]), int(row[i_rep]))
            own[key][rnd] = row[i_own]
            opp[key][rnd] = row[i_om]
    result: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for (opponent, gid, repeat), own_by_round in own.items():
        max_round = max(own_by_round)
        own_moves = [own_by_round[i] for i in range(1, max_round + 1)]
        opp_moves = [opp[(opponent, gid, repeat)][i] for i in range(1, max_round + 1)]
        series = series_from_moves(own_moves, opp_moves, n_rounds=main_analysis_rounds)
        result[opponent][gid].append(series)
    return {k: dict(v) for k, v in result.items()}


def load_one_seed_age(data_dir: Path, seed: int, age: int):
    """(census_with_k, series_by_opponent, recovery_rows) for one (seed,
    age), or None if any of the 3 files is missing."""
    cpath = viewer_mod.pilot_mod.census_path(data_dir, seed, age)
    mpath = viewer_mod.pilot_mod.moves_path(data_dir, seed, age)
    rpath = viewer_mod.pilot_mod.recovery_path(data_dir, seed, age)
    if not (cpath.exists() and mpath.exists() and rpath.exists()):
        return None
    census = viewer_mod.load_census_with_k(cpath)
    series_by_opponent = load_series_first_rounds(mpath)
    recovery = viewer_mod.load_recovery(rpath)
    return census, series_by_opponent, recovery


def census_true_types(census, threshold_kout: float, threshold_kself: float) -> Dict[int, str]:
    return {gid: viewer_mod.classify_type(k_out, k_self, threshold_kout, threshold_kself) for gid, _f, _md, _t, _p, k_out, k_self in census}


def merge_type_membership(members_a: Tuple[Dict[int, int], Dict[int, tuple]], members_b: Tuple[Dict[int, int], Dict[int, tuple]]):
    freq_a, idents_a = members_a
    freq_b, idents_b = members_b
    freq = dict(freq_a)
    freq.update(freq_b)  # gids are disjoint (two different types partition the same census)
    idents = dict(idents_a)
    idents.update(idents_b)
    return freq, idents


def type_membership_from_labels(census, gid_to_type: Dict[int, str], type_name: str) -> Tuple[Dict[int, int], Dict[int, tuple]]:
    """(frequencies, identities) for genotypes labeled `type_name` under
    `gid_to_type` (either the true classify_type labels, or a permuted
    relabeling -- see permute_type_labels)."""
    freq: Dict[int, int] = {}
    idents: Dict[int, tuple] = {}
    for gid, f, md, t, p, _k_out, _k_self in census:
        if gid_to_type.get(gid) != type_name:
            continue
        freq[gid] = f
        idents[gid] = (md, t, p)
    return freq, idents


def permute_type_labels(gid_to_true_type: Dict[int, str], rng: random.Random) -> Dict[int, str]:
    """Random relabeling of each GENOTYPE's type at one (seed, age),
    preserving the overall per-type genotype count exactly (preregistration
    section 5-4 step 1). Operates at the genotype level, unweighted by
    frequency -- consistent with how the true-label statistic itself pools
    recovery records per genotype, unweighted by frequency
    (export_k_model_markov_viewer.summarize_recovery's convention). Since
    K_out/K_self's per-generation Gaussian nudge makes near-total
    per-individual genotype uniqueness the norm (frequency=1 for nearly
    every genotype, design doc section 8-3), this also preserves each
    type's individual head count almost exactly in practice."""
    gids = list(gid_to_true_type.keys())
    labels = [gid_to_true_type[g] for g in gids]
    shuffled = labels[:]
    rng.shuffle(shuffled)
    return dict(zip(gids, shuffled))


def draw_cross_lineage_identities(type_pools: Dict[int, Tuple[Dict[int, int], Dict[int, tuple]]], group_size: int, rng: random.Random) -> List[Tuple[int, int]]:
    """[(seed, gid), ...] of length group_size: group_size DISTINCT seeds,
    one genotype per seed (frequency-weighted) -- same selection as
    evopersonality_abm.markov_apply.resample_cross_seed_group, but returns
    the (seed, gid) identity instead of looking up a specific opponent's
    series, so the SAME draw can be reused across all 5 human opponents
    (preregistration section 4: one virtual group, not independently
    redrawn per opponent)."""
    seeds = [s for s in type_pools if type_pools[s][0]]
    if group_size > len(seeds):
        raise ValueError(f"group_size={group_size} exceeds available seeds={len(seeds)}")
    chosen_seeds = rng.sample(seeds, group_size)
    result = []
    for s in chosen_seeds:
        frequencies, _idents = type_pools[s]
        gids = list(frequencies.keys())
        weights = [frequencies[g] for g in gids]
        g = rng.choices(gids, weights=weights, k=1)[0]
        result.append((s, g))
    return result


def seeds_with_type(type_pools) -> List[int]:
    return [s for s in type_pools if type_pools[s][0]]


def effective_group_size(target: int, n_available: int) -> int:
    """docs/DEVIATION_k_model_group_size.md: when fewer lineages carry a
    type than the human group size, the cross-lineage group uses every
    available lineage (one individual each) instead of failing."""
    return min(target, n_available)


def draw_cross_lineage_identities_from_resampled_seeds(
    type_pools: Dict[int, Tuple[Dict[int, int], Dict[int, tuple]]], resampled_seed_list: Sequence[int], group_size: int, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Seed-bootstrap variant of draw_cross_lineage_identities (section
    5-2): group_size DISTINCT POSITIONS from `resampled_seed_list` (a
    seed id may repeat at different positions, each contributing its own
    independent individual) -- mirrors
    markov_apply.draw_cross_lineage_group_from_resampled_seeds."""
    if group_size > len(resampled_seed_list):
        raise ValueError(f"group_size={group_size} exceeds resampled list length={len(resampled_seed_list)}")
    chosen_positions = rng.sample(range(len(resampled_seed_list)), group_size)
    result = []
    for pos in chosen_positions:
        s = resampled_seed_list[pos]
        frequencies, _idents = type_pools[s]
        gids = list(frequencies.keys())
        weights = [frequencies[g] for g in gids]
        g = rng.choices(gids, weights=weights, k=1)[0]
        result.append((s, g))
    return result


def lookup_series(identities: List[Tuple[int, int]], series_by_seed_opponent: Dict[int, Dict[str, Dict[int, list]]], opponent: str, rng: random.Random) -> List[str]:
    out = []
    for seed, gid in identities:
        options = series_by_seed_opponent.get(seed, {}).get(opponent, {}).get(gid, [])
        out.append(rng.choice(options) if options else "")
    return out


def build_edges_for_series(series_list: List[str], vocab) -> Dict[Tuple[str, str], float]:
    edges = dict(build_edges(series_list, vocab))
    edges.update({(n, n): w for n, w in self_loop_edges(series_list, vocab).items()})
    return edges


def average_edges(identities_draws: List[List[Tuple[int, int]]], series_by_seed_opponent, opponent: str, vocab, rng: random.Random) -> Dict[Tuple[str, str], float]:
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    n = len(identities_draws)
    for identities in identities_draws:
        series = lookup_series(identities, series_by_seed_opponent, opponent, rng)
        for k, v in build_edges_for_series(series, vocab).items():
            sums[k] += v
    return {k: v / n for k, v in sums.items()}


# ------------------------------------------------------------------ H-A: concatenated-vector correlation

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def concatenate_edges(edges_by_opponent: Dict[str, Dict[Tuple[str, str], float]]) -> Dict[Tuple[str, str, str], float]:
    """{(opponent, a, b): weight} across HUMAN_OPPONENTS -- preregistration
    section 4's "人の5相手の網の辺の重みを1本の長いベクトルに連結"."""
    out: Dict[Tuple[str, str, str], float] = {}
    for opponent in HUMAN_OPPONENTS:
        for (a, b), w in edges_by_opponent.get(opponent, {}).items():
            out[(opponent, a, b)] = w
    return out


def correlation_concatenated(type_edges: Dict[Tuple[str, str, str], float], human_edges: Dict[Tuple[str, str, str], float]) -> Optional[float]:
    all_keys = set(type_edges) | set(human_edges)
    if not all_keys:
        return None
    xs = [type_edges.get(k, 0.0) for k in all_keys]
    ys = [human_edges.get(k, 0.0) for k in all_keys]
    return _pearson(xs, ys)


def build_human_concatenated_vectors(human_weights_path: Path) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    human_weights = load_human_weights(human_weights_path)
    result: Dict[str, Dict[Tuple[str, str, str], float]] = {}
    for group in ("Secure", "Mix", "Fearful"):
        edges_by_opponent = {opp: human_weights.get((HUMAN_LABEL[opp], group), {}) for opp in HUMAN_OPPONENTS}
        result[group] = concatenate_edges(edges_by_opponent)
    return result


# ------------------------------------------------------------------ H-B/H-C: recovery pooling

def pool_never_recovered_rate(recovery_rows, gid_to_type: Dict[int, str], type_name: str, opponent: str) -> Optional[Tuple[int, int]]:
    """(n, n_recovered) pooled directly over every recorded repeat of
    every genotype labeled `type_name` -- unweighted by frequency, same
    convention as export_k_model_markov_viewer.summarize_recovery."""
    n = n_recovered = 0
    for gid, opp_name, _repeat, rr in recovery_rows:
        if opp_name != opponent or gid_to_type.get(gid) != type_name:
            continue
        n += 1
        if rr is not None:
            n_recovered += 1
    return (n, n_recovered) if n else None


def _parse_bits(s: str) -> List[int]:
    """Parse a census bit string ("0,1,1"). Memory-depth-0 genotypes have an
    empty prehistory, written as "" -- which must parse to [], not fail."""
    return [int(x) for x in s.split(",") if x.strip() != ""]


def gid_effective_depth(census) -> Dict[int, int]:
    out = {}
    for gid, _f, md, t, p, _k_out, _k_self in census:
        table = LindgrenTable(md, _parse_bits(t), _parse_bits(p))
        out[gid] = effective_memory_depth(table)
    return out



# ------------------------------------------------------------------ recovery-count pre-aggregation
#
# H-B/H-C need, for many relabelings (true labels, 2000 seed-bootstrap
# resamples, 1000 label permutations), the pooled (n, n_recovered) over all
# recovery records of genotypes carrying a given type label. Re-scanning the
# full record list (~1.7M rows per age at 70 seeds) for every relabeling made
# a single age take hours. Instead, count once per (seed, genotype, key) --
# key = ("opp", "SELF"|"POP") or ("stratum", <depth stratum>, SELF records
# only) -- and sum those integer counts. The sums are exactly the same
# integers the per-row scan produced, so every rate is bit-identical.

def recovery_counts_by_gid(recovery_by_seed, depth_by_seed):
    """{seed: {gid: {key: [n, n_recovered]}}}."""
    out = {}
    for seed, rows in recovery_by_seed.items():
        per_gid = {}
        depths = depth_by_seed[seed]
        for gid, opp_name, _repeat, rr in rows:
            keys = []
            if opp_name in ("SELF", "POP"):
                keys.append(("opp", opp_name))
            if opp_name == "SELF" and gid in depths:
                keys.append(("stratum", depth_stratum(depths[gid])))
            if not keys:
                continue
            d = per_gid.setdefault(gid, {})
            hit = 1 if rr is not None else 0
            for k in keys:
                c = d.get(k)
                if c is None:
                    d[k] = [1, hit]
                else:
                    c[0] += 1
                    c[1] += hit
        out[seed] = per_gid
    return out


def recovery_counts_by_label(counts_by_gid, label_by_seed, seed_weights=None):
    """{(type_name, key): (n, n_recovered)} summed over seeds, each seed
    weighted by seed_weights[seed] (default 1; 0/absent weight = skipped)."""
    tot = {}
    for seed, per_gid in counts_by_gid.items():
        w = 1 if seed_weights is None else seed_weights.get(seed, 0)
        if not w:
            continue
        labels = label_by_seed[seed]
        for gid, d in per_gid.items():
            t = labels.get(gid)
            if t is None:
                continue
            for k, (n, nr) in d.items():
                c = tot.get((t, k))
                if c is None:
                    tot[(t, k)] = [w * n, w * nr]
                else:
                    c[0] += w * n
                    c[1] += w * nr
    return {k: (v[0], v[1]) for k, v in tot.items()}


# ------------------------------------------------------------------ per-age worker

def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def process_one_age(
    data_dir_str: str, age: int, seeds: List[int], threshold_kout: float, threshold_kself: float,
    human_weights_path_str: str, n_draws_main: int, seedboot_resamples: List[List[int]],
    n_permutations: int, permutation_draws_ha: int, master_seed: str,
    ha_group_size: Optional[Dict[str, int]] = None,
) -> dict:
    ha_group_size = ha_group_size or HA_GROUP_SIZE
    data_dir = Path(data_dir_str)
    human_vectors = build_human_concatenated_vectors(Path(human_weights_path_str))

    census_by_seed = {}
    series_by_seed_opponent: Dict[int, Dict[str, Dict[int, list]]] = {}
    recovery_by_seed = {}
    for seed in seeds:
        loaded = load_one_seed_age(data_dir, seed, age)
        if loaded is None:
            continue
        census, series_by_opponent, recovery = loaded
        census_by_seed[seed] = census
        series_by_seed_opponent[seed] = series_by_opponent
        recovery_by_seed[seed] = recovery

    true_type_by_seed = {seed: census_true_types(c, threshold_kout, threshold_kself) for seed, c in census_by_seed.items()}
    depth_by_seed = {seed: gid_effective_depth(c) for seed, c in census_by_seed.items()}
    all_seeds_with_data = list(census_by_seed.keys())

    def type_pools_for_labels(label_by_seed: Dict[int, Dict[int, str]], type_name: str) -> Dict[int, Tuple[Dict[int, int], Dict[int, tuple]]]:
        pools = {}
        for seed, census in census_by_seed.items():
            pools[seed] = type_membership_from_labels(census, label_by_seed[seed], type_name)
        return pools

    def ha_type_pools(label_by_seed, type_name) -> Dict[int, Tuple[Dict[int, int], Dict[int, tuple]]]:
        if type_name == "Mix":
            a = type_pools_for_labels(label_by_seed, "K_Dismissing")
            b = type_pools_for_labels(label_by_seed, "K_Preoccupied")
            return {s: merge_type_membership(a[s], b[s]) for s in a}
        return type_pools_for_labels(label_by_seed, type_name)

    # True-label pools never change within an age; build them once.
    true_pools = {t: ha_type_pools(true_type_by_seed, t) for t in HA_TYPES}
    rng_main = random.Random(f"{master_seed}_main_{age}")
    vocabs = {opp: load_vocab(VOCAB_NAME[opp]) for opp in HUMAN_OPPONENTS}

    # --- H-A point estimate: n_draws_main cross-lineage draws, this age ---
    ha_point_edges: Dict[str, Dict[str, Dict[Tuple[str, str], float]]] = {}
    effective_gs: Dict[str, int] = {}
    for type_name in HA_TYPES:
        pools = true_pools[type_name]
        gs = effective_group_size(ha_group_size[type_name], len(seeds_with_type(pools)))
        effective_gs[type_name] = gs
        draws = [draw_cross_lineage_identities(pools, gs, rng_main) for _ in range(n_draws_main)] if gs else []
        ha_point_edges[type_name] = {
            opp: average_edges(draws, series_by_seed_opponent, opp, vocabs[opp], rng_main) for opp in HUMAN_OPPONENTS
        }

    # --- H-B/H-C point estimate: direct pooling, no draws ---
    counts_by_gid = recovery_counts_by_gid(recovery_by_seed, depth_by_seed)
    hb_point: Dict[Tuple[str, str], Tuple[int, int]] = {}
    hc_point: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    tot_true = recovery_counts_by_label(counts_by_gid, true_type_by_seed)
    for type_name in ("K_Secure", "K_Fearful"):
        for opponent in ("SELF", "POP"):
            c = tot_true.get((type_name, ("opp", opponent)))
            if c and c[0]:
                hb_point[(type_name, opponent)] = c
        for stratum in HC_STRATA:
            c = tot_true.get((type_name, ("stratum", stratum)))
            if c and c[0]:
                hc_point[(type_name, stratum)] = c

    # --- seed-level bootstrap (section 5-2) ---
    # Per-iteration values, keyed by iteration index i -- NOT flattened
    # across ages here, since the preregistration's point-estimate/
    # bootstrap convention ("各回で時間窓内の平均...を計算し") requires
    # averaging ONE iteration's value over the window's ages before
    # treating iterations as the bootstrap distribution (see
    # run_all_ages/build_results, which does that averaging). The
    # resampled seed LIST for iteration i is the same across every age
    # (passed in via `seedboot_resamples`, generated once for the whole
    # run) -- only the within-age draw's rng varies by age, since the
    # resampled seed set itself represents one fixed alternate population
    # of lineages, not something that should be redrawn per age.
    ha_seedboot_diffs: Dict[str, Dict[int, float]] = defaultdict(dict)
    hb_seedboot_diffs: Dict[str, Dict[int, float]] = defaultdict(dict)
    hc_seedboot_diffs: Dict[str, Dict[int, float]] = defaultdict(dict)

    for i, resampled_full in enumerate(seedboot_resamples):
        resampled = [s for s in resampled_full if s in census_by_seed]
        if not resampled:
            continue
        rng = random.Random(f"{master_seed}_seedboot_draw_{i}_{age}")

        r_by_type_group: Dict[Tuple[str, str], float] = {}
        for type_name in HA_TYPES:
            pools = true_pools[type_name]
            # Only resampled positions whose lineage carries this type can
            # contribute an individual (a position without one would make
            # rng.choices fail on an empty pool).
            resampled_with_type = [s for s in resampled if pools[s][0]]
            gs = effective_group_size(ha_group_size[type_name], len(resampled_with_type))
            if not gs:
                continue
            identities = draw_cross_lineage_identities_from_resampled_seeds(pools, resampled_with_type, gs, rng)
            edges_by_opp = {opp: build_edges_for_series(lookup_series(identities, series_by_seed_opponent, opp, rng), vocabs[opp]) for opp in HUMAN_OPPONENTS}
            type_vector = concatenate_edges(edges_by_opp)
            for group in ("Secure", "Mix", "Fearful"):
                corr = correlation_concatenated(type_vector, human_vectors[group])
                if corr is not None:
                    r_by_type_group[(type_name, group)] = corr
        for hyp, label, type_, g_same, g_other in HA_COMPARISONS:
            if (type_, g_same) in r_by_type_group and (type_, g_other) in r_by_type_group:
                ha_seedboot_diffs[label][i] = ha_comparison_diff(r_by_type_group, type_, g_same, g_other)

        resampled_counts = defaultdict(int)
        for s in resampled:
            resampled_counts[s] += 1
        tot_boot = recovery_counts_by_label(counts_by_gid, true_type_by_seed, resampled_counts)
        rate_by_type_opp: Dict[Tuple[str, str], float] = {}
        rate_by_type_stratum: Dict[Tuple[str, str], float] = {}
        for type_name in ("K_Secure", "K_Fearful"):
            for opponent in ("SELF", "POP"):
                c = tot_boot.get((type_name, ("opp", opponent)))
                if c and c[0]:
                    rate_by_type_opp[(type_name, opponent)] = 1 - c[1] / c[0]  # never_recovered_rate
            for stratum in HC_STRATA:
                c = tot_boot.get((type_name, ("stratum", stratum)))
                if c and c[0]:
                    rate_by_type_stratum[(type_name, stratum)] = 1 - c[1] / c[0]  # never_recovered_rate
        for hyp, label, opponent in HB_COMPARISONS:
            if ("K_Fearful", opponent) in rate_by_type_opp and ("K_Secure", opponent) in rate_by_type_opp:
                hb_seedboot_diffs[label][i] = hb_comparison_diff(rate_by_type_opp, opponent)
        for stratum in HC_STRATA:
            if ("K_Fearful", stratum) in rate_by_type_stratum and ("K_Secure", stratum) in rate_by_type_stratum:
                hc_seedboot_diffs[stratum][i] = hc_comparison_diff(rate_by_type_stratum, stratum)

    # --- label-permutation null distributions (section 5-4) ---
    # Null values are iteration-indexed (Dict[label][i] = diff), NOT a
    # flat per-age list -- same reason as the seed-bootstrap above: the
    # preregistration's statistic is a WINDOW-AVERAGE per iteration, so
    # build_results must average across ages per iteration index before
    # treating iterations as the null distribution (flattening ages and
    # iterations together would both inflate N and mix together values
    # that were never meant to share a single permutation's realization).
    ha_perm_null: Dict[str, Dict[int, float]] = defaultdict(dict)
    ha_perm_true: Dict[str, float] = {}
    hb_perm_null: Dict[str, Dict[int, float]] = defaultdict(dict)
    hb_perm_true: Dict[str, float] = {}
    hc_perm_null: Dict[str, Dict[int, float]] = defaultdict(dict)
    hc_perm_true: Dict[str, float] = {}

    # H-A: true-label statistic at the SAME draw count as the permutations (section 5-4 step 3)
    rng_true_ha = random.Random(f"{master_seed}_true_ha_{age}")
    r_true: Dict[Tuple[str, str], float] = {}
    for type_name in HA_TYPES:
        pools = true_pools[type_name]
        gs = effective_gs[type_name]
        if not gs:
            continue
        draws = [draw_cross_lineage_identities(pools, gs, rng_true_ha) for _ in range(permutation_draws_ha)]
        edges_by_opp = {opp: average_edges(draws, series_by_seed_opponent, opp, vocabs[opp], rng_true_ha) for opp in HUMAN_OPPONENTS}
        type_vector = concatenate_edges(edges_by_opp)
        for group in ("Secure", "Mix", "Fearful"):
            corr = correlation_concatenated(type_vector, human_vectors[group])
            if corr is not None:
                r_true[(type_name, group)] = corr
    for hyp, label, type_, g_same, g_other in HA_COMPARISONS:
        if (type_, g_same) in r_true and (type_, g_other) in r_true:
            ha_perm_true[label] = ha_comparison_diff(r_true, type_, g_same, g_other)

    # H-B/H-C: true-label statistic (direct pooling, no draw-count issue)
    for hyp, label, opponent in HB_COMPARISONS:
        if (opponent in ("SELF", "POP")) and ("K_Fearful", opponent) in hb_point and ("K_Secure", opponent) in hb_point:
            rate = {t: 1 - hb_point[(t, opponent)][1] / hb_point[(t, opponent)][0] for t in ("K_Secure", "K_Fearful")}  # never_recovered_rate
            hb_perm_true[label] = rate["K_Fearful"] - rate["K_Secure"]
    for stratum in HC_STRATA:
        if ("K_Fearful", stratum) in hc_point and ("K_Secure", stratum) in hc_point:
            rate = {t: 1 - hc_point[(t, stratum)][1] / hc_point[(t, stratum)][0] for t in ("K_Secure", "K_Fearful")}  # never_recovered_rate
            hc_perm_true[stratum] = rate["K_Fearful"] - rate["K_Secure"]

    for i in range(n_permutations):
        rng = random.Random(f"{master_seed}_perm_{i}_{age}")
        permuted_label_by_seed = {seed: permute_type_labels(true_type_by_seed[seed], rng) for seed in all_seeds_with_data}

        r_perm: Dict[Tuple[str, str], float] = {}
        for type_name in HA_TYPES:
            pools = ha_type_pools(permuted_label_by_seed, type_name)
            # Labels are permuted within each (seed, age), so the lineages
            # carrying a type -- and hence the effective group size -- are
            # the same as for the true labels.
            gs = effective_group_size(ha_group_size[type_name], len(seeds_with_type(pools)))
            if not gs:
                continue
            draws = [draw_cross_lineage_identities(pools, gs, rng) for _ in range(permutation_draws_ha)]
            edges_by_opp = {opp: average_edges(draws, series_by_seed_opponent, opp, vocabs[opp], rng) for opp in HUMAN_OPPONENTS}
            type_vector = concatenate_edges(edges_by_opp)
            for group in ("Secure", "Mix", "Fearful"):
                corr = correlation_concatenated(type_vector, human_vectors[group])
                if corr is not None:
                    r_perm[(type_name, group)] = corr
        for hyp, label, type_, g_same, g_other in HA_COMPARISONS:
            if (type_, g_same) in r_perm and (type_, g_other) in r_perm:
                ha_perm_null[label][i] = ha_comparison_diff(r_perm, type_, g_same, g_other)

        tot_perm = recovery_counts_by_label(counts_by_gid, permuted_label_by_seed)
        rate_by_type_opp = {}
        rate_by_type_stratum = {}
        for type_name in ("K_Secure", "K_Fearful"):
            for opponent in ("SELF", "POP"):
                c = tot_perm.get((type_name, ("opp", opponent)))
                if c and c[0]:
                    rate_by_type_opp[(type_name, opponent)] = 1 - c[1] / c[0]  # never_recovered_rate
            for stratum in HC_STRATA:
                c = tot_perm.get((type_name, ("stratum", stratum)))
                if c and c[0]:
                    rate_by_type_stratum[(type_name, stratum)] = 1 - c[1] / c[0]  # never_recovered_rate
        for hyp, label, opponent in HB_COMPARISONS:
            if ("K_Fearful", opponent) in rate_by_type_opp and ("K_Secure", opponent) in rate_by_type_opp:
                hb_perm_null[label][i] = hb_comparison_diff(rate_by_type_opp, opponent)
        for stratum in HC_STRATA:
            if ("K_Fearful", stratum) in rate_by_type_stratum and ("K_Secure", stratum) in rate_by_type_stratum:
                hc_perm_null[stratum][i] = hc_comparison_diff(rate_by_type_stratum, stratum)

    return {
        "age": age,
        "recovery_diff_convention": RECOVERY_DIFF_CONVENTION,
        "ha_point_edges": ha_point_edges,
        "ha_effective_group_size": effective_gs,
        "hb_point": hb_point,
        "hc_point": hc_point,
        "ha_seedboot_diffs": dict(ha_seedboot_diffs),
        "hb_seedboot_diffs": dict(hb_seedboot_diffs),
        "hc_seedboot_diffs": dict(hc_seedboot_diffs),
        "ha_perm_true": dict(ha_perm_true),
        "ha_perm_null": dict(ha_perm_null),
        "hb_perm_true": hb_perm_true,
        "hb_perm_null": dict(hb_perm_null),
        "hc_perm_true": hc_perm_true,
        "hc_perm_null": dict(hc_perm_null),
    }


# ------------------------------------------------------------------ orchestration

def build_seedboot_resamples(seeds: List[int], n_bootstrap_seed: int, master_seed: str) -> List[List[int]]:
    """n_bootstrap_seed resamples of `seeds` (with replacement), generated
    ONCE for the whole run (age-independent) -- preregistration section
    5-2: the resampled seed SET represents one fixed alternate population
    of lineages, so the SAME resample must be reused at every age for a
    given iteration index (only the within-age individual draw varies by
    age; see process_one_age's docstring note on this)."""
    rng = random.Random(f"{master_seed}_seedboot_resample")
    return [[rng.choice(seeds) for _ in range(len(seeds))] for _ in range(n_bootstrap_seed)]


def run_all_ages(
    data_dir: Path, ages: List[int], seeds: List[int], threshold_kout: float, threshold_kself: float,
    human_weights_path: Path, n_draws_main: int, seedboot_resamples: List[List[int]], n_permutations: int,
    permutation_draws_ha: int, master_seed: str, workers: int, cache_dir: Path,
    ha_group_size: Optional[Dict[str, int]] = None,
) -> List[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age, str(data_dir), age, seeds, threshold_kout, threshold_kself,
                str(human_weights_path), n_draws_main, seedboot_resamples, n_permutations,
                permutation_draws_ha, master_seed, ha_group_size,
            ): age
            for age in pending
        }
        failed = []
        for fut in as_completed(futures):
            age = futures[fut]
            # One failing age must not discard the others: log it, keep
            # going, and let every finished age reach the cache.
            try:
                result = fut.result()
            except Exception:
                failed.append(age)
                print(f"age={age} FAILED, elapsed={time.time() - t0:.0f}s", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                continue
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} age(s) failed: {sorted(failed)} -- cached ages are kept; fix and rerun the same command to resume.")

    results = []
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            results.append(pickle.load(f))
    return results


def average_edge_dicts(dicts: List[Dict[Tuple[str, str], float]]) -> Dict[Tuple[str, str], float]:
    sums: Dict[Tuple[str, str], float] = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            sums[k] += v
    return {k: v / len(dicts) for k, v in sums.items()}


def window_average_rate(points: List[Tuple[int, int]]) -> Optional[float]:
    rates = [n_rec / n for n, n_rec in points if n]
    return statistics.fmean(rates) if rates else None


RECOVERY_DIFF_CONVENTION = "never_recovered"


def normalize_cached_result(r: dict) -> dict:
    """Age caches written before the sign fix (no "recovery_diff_convention"
    key) stored H-B/H-C differences as recovered(F) - recovered(S), while
    the preregistered statistic is never_recovered(F) - never_recovered(S)
    -- exactly the negative. Negate those so old caches can be summarized
    without recomputation. (H-A and all counts are unaffected.)"""
    if r.get("recovery_diff_convention") == RECOVERY_DIFF_CONVENTION:
        return r
    r = dict(r)
    for key in ("hb_seedboot_diffs", "hc_seedboot_diffs", "hb_perm_null", "hc_perm_null"):
        r[key] = {label: {i: -v for i, v in d.items()} for label, d in r[key].items()}
    for key in ("hb_perm_true", "hc_perm_true"):
        r[key] = {label: -v for label, v in r[key].items()}
    r["recovery_diff_convention"] = RECOVERY_DIFF_CONVENTION
    return r


def build_results(results: List[dict], human_vectors) -> dict:
    results = [normalize_cached_result(r) for r in results]
    # H-A point estimate: window-average the per-age edge dicts, then concatenate + correlate once
    r_by_type_group: Dict[Tuple[str, str], float] = {}
    for type_name in HA_TYPES:
        edges_by_opp = {opp: average_edge_dicts([r["ha_point_edges"][type_name][opp] for r in results]) for opp in HUMAN_OPPONENTS}
        type_vector = concatenate_edges(edges_by_opp)
        for group in ("Secure", "Mix", "Fearful"):
            corr = correlation_concatenated(type_vector, human_vectors[group])
            if corr is not None:
                r_by_type_group[(type_name, group)] = corr

    ha_point_diffs = {label: ha_comparison_diff(r_by_type_group, type_, g_same, g_other) for hyp, label, type_, g_same, g_other in HA_COMPARISONS if (type_, g_same) in r_by_type_group and (type_, g_other) in r_by_type_group}

    hb_point_rate: Dict[Tuple[str, str], Optional[float]] = {}
    for type_name in ("K_Secure", "K_Fearful"):
        for opponent in ("SELF", "POP"):
            points = [r["hb_point"][(type_name, opponent)] for r in results if (type_name, opponent) in r["hb_point"]]
            hb_point_rate[(type_name, opponent)] = window_average_rate(points)
    hb_point_diffs = {
        # hb_point_rate holds RECOVERED rates (n_recovered / n); the
        # preregistered statistic is never_recovered(F) - never_recovered(S)
        # = recovered(S) - recovered(F).
        label: hb_point_rate[("K_Secure", opponent)] - hb_point_rate[("K_Fearful", opponent)]
        for hyp, label, opponent in HB_COMPARISONS
        if hb_point_rate.get(("K_Fearful", opponent)) is not None and hb_point_rate.get(("K_Secure", opponent)) is not None
    }

    hc_point_rate: Dict[Tuple[str, str], Optional[float]] = {}
    hc_n: Dict[Tuple[str, str], int] = {}
    for type_name in ("K_Secure", "K_Fearful"):
        for stratum in HC_STRATA:
            points = [r["hc_point"][(type_name, stratum)] for r in results if (type_name, stratum) in r["hc_point"]]
            hc_point_rate[(type_name, stratum)] = window_average_rate(points)
            hc_n[(type_name, stratum)] = sum(n for n, _n_rec in points)
    testable_strata = [
        s for s in HC_STRATA
        if hc_n.get(("K_Secure", s), 0) >= HC_MIN_STRATUM_N and hc_n.get(("K_Fearful", s), 0) >= HC_MIN_STRATUM_N
    ]
    hc_point_diffs = {
        s: hc_point_rate[("K_Secure", s)] - hc_point_rate[("K_Fearful", s)]  # never(F) - never(S), see H-B above
        for s in testable_strata
        if hc_point_rate.get(("K_Fearful", s)) is not None and hc_point_rate.get(("K_Secure", s)) is not None
    }

    # seed bootstrap -> one-sided p -> Holm -> verdict. Each age's
    # ..._seedboot_diffs is {label: {iteration_index: diff}} -- average
    # PER ITERATION across the ages where that iteration produced a value
    # for that label, THEN treat the resulting (one value per iteration)
    # list as the bootstrap distribution (preregistration's "各回で時間窓
    # 内の平均...を計算し" -- the iteration, not the age, is the unit of
    # the bootstrap distribution).
    def window_average_per_iteration(per_age_dicts: List[Dict[str, Dict[int, float]]]) -> Dict[str, List[float]]:
        by_label_iter: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for d in per_age_dicts:
            for label, by_iter in d.items():
                for i, v in by_iter.items():
                    by_label_iter[label][i].append(v)
        return {label: [statistics.fmean(vs) for vs in by_iter.values()] for label, by_iter in by_label_iter.items()}

    ha_seedboot_pooled = window_average_per_iteration([r["ha_seedboot_diffs"] for r in results])
    hb_seedboot_pooled = window_average_per_iteration([r["hb_seedboot_diffs"] for r in results])
    hc_seedboot_pooled = window_average_per_iteration([r["hc_seedboot_diffs"] for r in results])

    def pvalues_and_verdict(comparisons_meta, point_diffs, seedboot_pooled):
        rows = []
        by_hyp: Dict[str, List[bool]] = defaultdict(list)
        p_by_hyp: Dict[str, List[float]] = defaultdict(list)
        for meta in comparisons_meta:
            hyp, label = meta[0], meta[1]
            if label not in point_diffs or label not in seedboot_pooled or not seedboot_pooled[label]:
                continue
            p = one_sided_bootstrap_pvalue(seedboot_pooled[label])
            p_by_hyp[hyp].append(p)
            rows.append({"hypothesis": hyp, "comparison": label, "point_diff": point_diffs[label], "p_seedboot": p})
        for hyp, ps in p_by_hyp.items():
            rejects = holm_correction(ps)
            idx = 0
            for row in rows:
                if row["hypothesis"] == hyp:
                    row["significant_holm"] = rejects[idx]
                    idx += 1
        verdicts = {hyp: verdict([r["significant_holm"] for r in rows if r["hypothesis"] == hyp]) for hyp in p_by_hyp}
        return rows, verdicts

    ha_rows, ha_verdicts = pvalues_and_verdict(HA_COMPARISONS, ha_point_diffs, ha_seedboot_pooled)
    hb_rows, hb_verdicts = pvalues_and_verdict(HB_COMPARISONS, hb_point_diffs, hb_seedboot_pooled)
    hc_meta = [("HC1", s) for s in testable_strata]
    hc_rows, hc_verdicts = pvalues_and_verdict(hc_meta, hc_point_diffs, hc_seedboot_pooled)

    # permutation test: the true statistic is window-averaged across ages
    # (one value per age -> fmean over ages); the null distribution is
    # window-averaged PER PERMUTATION ITERATION the same way as the
    # seed-bootstrap above, giving exactly n_permutations null values.
    ha_perm_null_pooled = window_average_per_iteration([r["ha_perm_null"] for r in results])
    hb_perm_null_pooled = window_average_per_iteration([r["hb_perm_null"] for r in results])
    hc_perm_null_pooled = window_average_per_iteration([r["hc_perm_null"] for r in results])

    ha_perm_rows = []
    for label in {l for hyp, l, *_ in HA_COMPARISONS}:
        trues = [r["ha_perm_true"].get(label) for r in results if label in r["ha_perm_true"]]
        nulls = ha_perm_null_pooled.get(label, [])
        if trues and nulls:
            ha_perm_rows.append({"comparison": label, "true_diff_window_avg": statistics.fmean(trues), "p_permutation": permutation_pvalue(statistics.fmean(trues), nulls)})
    hb_perm_rows = []
    for hyp, label, opponent in HB_COMPARISONS:
        trues = [r["hb_perm_true"].get(label) for r in results if label in r["hb_perm_true"]]
        nulls = hb_perm_null_pooled.get(label, [])
        if trues and nulls:
            hb_perm_rows.append({"comparison": label, "true_diff_window_avg": statistics.fmean(trues), "p_permutation": permutation_pvalue(statistics.fmean(trues), nulls)})
    hc_perm_rows = []
    for stratum in testable_strata:
        trues = [r["hc_perm_true"].get(stratum) for r in results if stratum in r["hc_perm_true"]]
        nulls = hc_perm_null_pooled.get(stratum, [])
        if trues and nulls:
            hc_perm_rows.append({"comparison": stratum, "true_diff_window_avg": statistics.fmean(trues), "p_permutation": permutation_pvalue(statistics.fmean(trues), nulls)})

    return {
        "ha": {"rows": ha_rows, "verdicts": ha_verdicts, "permutation": ha_perm_rows},
        "hb": {"rows": hb_rows, "verdicts": hb_verdicts, "permutation": hb_perm_rows},
        "hc": {"rows": hc_rows, "verdicts": hc_verdicts, "permutation": hc_perm_rows, "testable_strata": testable_strata, "untestable_strata": [s for s in HC_STRATA if s not in testable_strata]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--thresholds-json", type=str, required=True)
    p.add_argument("--human-weights-csv", type=str, default=str(REPO_ROOT / "docs" / "human_network_weights.csv"))
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--n-draws-main", type=int, default=1000)
    p.add_argument("--n-bootstrap-seed", type=int, default=2000)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--permutation-draws-ha", type=int, default=50)
    p.add_argument("--master-seed", type=str, default="k_model_confirmatory_v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--group-size-secure", type=int, default=HA_GROUP_SIZE["K_Secure"], help="cross-lineage group size for K_Secure (human Secure group's N; override for a small-scale test run)")
    p.add_argument("--group-size-mix", type=int, default=HA_GROUP_SIZE["Mix"], help="cross-lineage group size for Mix (human Mix group's N)")
    p.add_argument("--group-size-fearful", type=int, default=HA_GROUP_SIZE["K_Fearful"], help="cross-lineage group size for K_Fearful (human Fearful group's N)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    with open(args.thresholds_json) as f:
        thresholds = json.load(f)
    ha_group_size = {"K_Secure": args.group_size_secure, "Mix": args.group_size_mix, "K_Fearful": args.group_size_fearful}

    all_ages = viewer_mod.discover_ages(data_dir)
    ages = [a for a in all_ages if args.window_start <= a <= args.window_end]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / "_age_cache"

    seedboot_resamples = build_seedboot_resamples(seeds, args.n_bootstrap_seed, args.master_seed)
    raw_results = run_all_ages(
        data_dir, ages, seeds, thresholds["threshold_kout"], thresholds["threshold_kself"],
        Path(args.human_weights_csv), args.n_draws_main, seedboot_resamples, args.n_permutations,
        args.permutation_draws_ha, args.master_seed, args.workers, cache_dir, ha_group_size,
    )

    human_vectors = build_human_concatenated_vectors(Path(args.human_weights_csv))
    summary = build_results(raw_results, human_vectors)

    with open(results_dir / "k_model_confirmatory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"wrote {results_dir / 'k_model_confirmatory_summary.json'}")
    print(f"H-A verdicts: {summary['ha']['verdicts']}")
    print(f"H-B verdicts: {summary['hb']['verdicts']}")
    print(f"H-C verdicts: {summary['hc']['verdicts']} (testable strata: {summary['hc']['testable_strata']}, untestable: {summary['hc']['untestable_strata']})")


if __name__ == "__main__":
    main()
