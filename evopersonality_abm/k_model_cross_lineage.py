"""Cross-lineage (scheme B) sampling helpers for the K-model production
analysis -- docs/PREREGISTRATION_k_model_markov.md section 3, with the
effective-group-size shrinking rule decided in
docs/DEVIATION_k_model_group_size.md ("群の人数 = min(人の群の人数, その
類型がいるseedの数)", applied consistently everywhere a cross-lineage
group is drawn).

Shared by scripts/run_k_model_confirmatory_analysis.py (confirmatory
analysis, sections 4-5) and scripts/export_k_model_markov_viewer.py
(production mode, descriptive section 6 / the network viewer) so both use
the exact same sampling logic -- kept in this package module, rather than
defined in one script and imported by the other, because
run_k_model_confirmatory_analysis.py already loads
export_k_model_markov_viewer.py as a script module (via
importlib.util.spec_from_file_location, which re-executes the target file
rather than caching it through sys.modules the way a normal `import`
does), so the reverse direction would re-execute both files into each
other indefinitely.
"""

import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .markov_recovery import series_from_moves

# (frequencies, identities) for genotypes of one type at one seed --
# opponent-independent (membership in a type depends only on K_out/K_self,
# not on which opponent's series you're about to look up).
SeedPool = Tuple[Dict[int, int], Dict[int, tuple]]


def load_series_first_rounds(moves_path: Path, main_analysis_rounds: int = 20) -> Dict[str, Dict[int, List[str]]]:
    """Same output as export_k_model_markov_viewer.load_series_by_genotype_all_opponents,
    but rows past round `main_analysis_rounds` are skipped before any
    parsing: series_from_moves only ever reads the first
    `main_analysis_rounds` moves, and the moves file stores up to 100
    rounds per duel, so most rows would otherwise be parsed for nothing
    (~4x faster at the default 20-of-100)."""
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


def merge_type_membership(members_a: SeedPool, members_b: SeedPool) -> SeedPool:
    """Combine two types' (frequencies, identities) at the SAME seed into
    one -- used for Mix = K_Dismissing + K_Preoccupied. gids are disjoint
    (the two types partition the same census), so a plain dict union is
    exact, no remapping needed."""
    freq_a, idents_a = members_a
    freq_b, idents_b = members_b
    freq = dict(freq_a)
    freq.update(freq_b)
    idents = dict(idents_a)
    idents.update(idents_b)
    return freq, idents


def type_membership_from_labels(census, gid_to_type: Dict[int, str], type_name: str) -> SeedPool:
    """(frequencies, identities) for genotypes labeled `type_name` under
    `gid_to_type` (either the true classify_type labels, or a permuted
    relabeling)."""
    freq: Dict[int, int] = {}
    idents: Dict[int, tuple] = {}
    for gid, f, md, t, p, _k_out, _k_self in census:
        if gid_to_type.get(gid) != type_name:
            continue
        freq[gid] = f
        idents[gid] = (md, t, p)
    return freq, idents


def seeds_with_type(type_pools: Dict[int, SeedPool]) -> List[int]:
    return [s for s in type_pools if type_pools[s][0]]


def effective_group_size(target: int, n_available: int) -> int:
    """docs/DEVIATION_k_model_group_size.md: when fewer lineages carry a
    type than the human group size, the cross-lineage group uses every
    available lineage (one individual each) instead of failing."""
    return min(target, n_available)


def draw_cross_lineage_identities(type_pools: Dict[int, SeedPool], group_size: int, rng: random.Random) -> List[Tuple[int, int]]:
    """[(seed, gid), ...] of length group_size: group_size DISTINCT seeds,
    one genotype per seed (frequency-weighted) -- same selection as
    evopersonality_abm.markov_apply.resample_cross_seed_group, but returns
    the (seed, gid) identity instead of looking up a specific opponent's
    series, so the SAME draw can be reused across every opponent that
    needs it (one virtual group, not independently redrawn per opponent).
    Raises ValueError if group_size exceeds the number of available
    seeds -- callers should pass effective_group_size(target, n_available)
    as group_size, never the raw target, to avoid this."""
    seeds = seeds_with_type(type_pools)
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


def draw_cross_lineage_identities_from_resampled_seeds(
    type_pools: Dict[int, SeedPool], resampled_seed_list: Sequence[int], group_size: int, rng: random.Random,
) -> List[Tuple[int, int]]:
    """Seed-bootstrap variant of draw_cross_lineage_identities: group_size
    DISTINCT POSITIONS from `resampled_seed_list` (a seed id may repeat at
    different positions, each contributing its own independent
    individual) -- mirrors
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
    """For each (seed, gid) identity, draw uniformly one of its recorded
    probe-repeat series against `opponent`."""
    out = []
    for seed, gid in identities:
        options = series_by_seed_opponent.get(seed, {}).get(opponent, {}).get(gid, [])
        out.append(rng.choice(options) if options else "")
    return out
