"""Apply the ported human-study Markov-network metrics
(`evopersonality_abm.markov_recovery`) to this project's own probe data,
by resampling "virtual participants" at the human study's own group
sizes -- see docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md sections 3.3/7.2.

Pipeline, per (seed, probe age, opponent, group_size):
  1. `load_census`/`load_series_by_genotype_all_opponents` read one
     probe timepoint's genotype census + raw move sequences (written by
     scripts/run_lindgren_markov_sweep.py).
  2. `resample_virtual_group` draws a group_size-sized "virtual
     participant" sample, weighted by genotype frequency (each draw
     picks a genotype, then uniformly one of that genotype's recorded
     probe repeats as that virtual participant's 20-round sequence).
  3. `build_edges` (markov_recovery, unchanged human-study definition)
     turns that sample into a 3-gram co-occurrence network.
  4. `network_metrics`/`similarity_to_human` compute the widest-path
     recovery/descent thresholds, Table 7 j* for the opponent's named
     circuits, the largest j>=.40 strongly-connected component, and
     similarity to each human group's own network.
  5. `bootstrap_aggregate` repeats steps 2-4 many times and summarizes
     (mean over bootstraps, plus the rate at which a metric was
     undefined/None or a key node was absent).
"""

import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .markov_recovery import (
    NAMED_CIRCUITS,
    build_edges,
    cycle_jstar,
    load_vocab,
    series_from_moves,
    strongly_connected_components,
    widest_path_threshold,
)

HUMAN_GROUP_SIZES = {"Secure": 56, "Mix": 46, "Fearful": 69}
# probe.py opponent name -> (human_network_weights.csv label, markov_recovery vocab name).
# human label is None for the extended-set opponents (GradualAppeasement, WSLS):
# the human study has no data for them (design doc section 3.4 / preregistration
# section 6 item 7, "網の指標の推移のみを記述する"), so similarity_to_human is
# never meaningfully computed for them (an empty human_weights_by_group is passed).
OPPONENT_INFO = {
    "TFT": ("TFT", "TFT"),
    "AllC": ("ALC", "AllC"),
    "AllD": ("ALD", "AllD"),
    "Random": ("RAN", "RAN"),
    "Gradual": ("GRA", "GRA"),
    "GradualAppeasement": (None, "GRA"),
    "WSLS": (None, "GRA"),
}

# Table 7's named circuits (markov_recovery.NAMED_CIRCUITS) were identified
# for specific opponents only (TFT/GRA/AllC/AllD/RAN) -- NOT simply "every
# opponent sharing that vocabulary". GradualAppeasement and WSLS use the
# same 64-word GRA vocabulary for network CONSTRUCTION, but must not
# silently inherit GRA's circuit list for j* -- this mapping (rather than
# OPPONENT_INFO's vocab_name field) is what network_metrics uses to decide
# which circuits, if any, apply to a given opponent.
CIRCUIT_KEY = {"TFT": "TFT", "AllC": "AllC", "AllD": "AllD", "Random": "RAN", "Gradual": "GRA"}


def load_census(path: Path) -> Dict[int, int]:
    """genotype_id -> frequency, from a seed_X_probe_age_G_census.csv.gz file."""
    with gzip.open(path, "rt", newline="") as f:
        return {int(row["genotype_id"]): int(row["frequency"]) for row in csv.DictReader(f)}


def load_series_by_genotype_all_opponents(
    moves_path: Path, main_analysis_rounds: int = 20
) -> Dict[str, Dict[int, List[str]]]:
    """One pass over a seed_X_probe_age_G_noiseN_moves.csv.gz file (which
    interleaves all opponents): {opponent: {genotype_id: [series, ...]}},
    one series string per (genotype, repeat) via markov_recovery.series_from_moves."""
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


def resample_virtual_group(
    frequencies: Dict[int, int],
    series_by_genotype: Dict[int, List[str]],
    group_size: int,
    rng: random.Random,
) -> List[str]:
    """group_size series strings: draw a genotype weighted by frequency,
    then uniformly one of its recorded probe repeats, `group_size` times
    (with replacement -- design doc section 3.3's "頻度に比例した重み付き抽出")."""
    genotype_ids = [g for g in frequencies if series_by_genotype.get(g)]
    weights = [frequencies[g] for g in genotype_ids]
    chosen = rng.choices(genotype_ids, weights=weights, k=group_size)
    return [rng.choice(series_by_genotype[g]) for g in chosen]


def load_genotype_identities(path: Path) -> Dict[int, Tuple[int, str, str]]:
    """genotype_id -> (memory_depth, table, prehistory) identity tuple,
    from a census.csv.gz file. Unlike genotype_id (only meaningful within
    one seed's own census), this identity tuple is comparable ACROSS
    seeds -- needed for the cross-lineage sampling scheme's diversity
    metrics below (design doc 2026-10-16, "系統間抽出")."""
    with gzip.open(path, "rt", newline="") as f:
        return {
            int(row["genotype_id"]): (int(row["memory_depth"]), row["table"], row["prehistory"])
            for row in csv.DictReader(f)
        }


def resample_within_seed_group_with_identity(
    frequencies: Dict[int, int],
    series_by_genotype: Dict[int, List[str]],
    identities: Dict[int, Tuple[int, str, str]],
    group_size: int,
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[int, str, str]]]:
    """Scheme (A), "集団内抽出" (within-population sampling): the same draw
    as resample_virtual_group, but also returns each draw's genotype
    identity, for group_diversity_metrics below."""
    genotype_ids = [g for g in frequencies if series_by_genotype.get(g)]
    weights = [frequencies[g] for g in genotype_ids]
    chosen = rng.choices(genotype_ids, weights=weights, k=group_size)
    series = [rng.choice(series_by_genotype[g]) for g in chosen]
    idents = [identities[g] for g in chosen]
    return series, idents


SeedPool = Tuple[Dict[int, int], Dict[int, List[str]], Dict[int, Tuple[int, str, str]]]


def draw_cross_lineage_group_from_seed_list(
    seed_pools: Dict[int, SeedPool],
    seed_list: Sequence[int],
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[int, str, str]]]:
    """One independent individual (frequency-weighted within that seed's
    own population) per entry of `seed_list` -- each entry triggers its
    own draw, so a seed appearing more than once in `seed_list` (as
    happens under the seed-level bootstrap below) contributes multiple
    DIFFERENT individuals, never the same one copied -- per the
    preregistration's explicit instruction (section 5): "復元抽出で同じ
    seedが複数回選ばれた場合は、その seedから別々の個体を抽出する"."""
    series: List[str] = []
    idents: List[Tuple[int, str, str]] = []
    for s in seed_list:
        frequencies, series_by_genotype, identities = seed_pools[s]
        genotype_ids = [g for g in frequencies if series_by_genotype.get(g)]
        weights = [frequencies[g] for g in genotype_ids]
        g = rng.choices(genotype_ids, weights=weights, k=1)[0]
        series.append(rng.choice(series_by_genotype[g]))
        idents.append(identities[g])
    return series, idents


def resample_cross_seed_group(
    seed_pools: Dict[int, SeedPool],
    group_size: int,
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[int, str, str]]]:
    """Scheme (B), "系統間抽出" (cross-lineage sampling, design doc
    2026-10-16): draws group_size DISTINCT seeds (without replacement --
    each virtual participant corresponds to one independently-evolved
    lineage, the way each human participant is one person), one
    individual per seed (frequency-weighted within that seed).
    seed_pools: {seed: (frequencies, series_by_genotype, identities)}."""
    seeds = list(seed_pools.keys())
    if group_size > len(seeds):
        raise ValueError(f"group_size={group_size} exceeds available seeds={len(seeds)}")
    chosen_seeds = rng.sample(seeds, group_size)
    return draw_cross_lineage_group_from_seed_list(seed_pools, chosen_seeds, rng)


def seed_bootstrap_resample(seeds: Sequence[int], rng: random.Random) -> List[int]:
    """One seed-level bootstrap resample (preregistration section 5):
    draw len(seeds) seed-ids WITH replacement from `seeds`. The result is
    a list the same length as `seeds`, generally containing repeats and
    omissions -- it represents one alternate "population of lineages"
    for assessing uncertainty from the finite number of evolved seeds."""
    return [rng.choice(seeds) for _ in range(len(seeds))]


def draw_cross_lineage_group_from_resampled_seeds(
    seed_pools: Dict[int, SeedPool],
    resampled_seed_list: Sequence[int],
    group_size: int,
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[int, str, str]]]:
    """Scheme (B) applied to a seed-bootstrap resample: picks group_size
    DISTINCT POSITIONS (not distinct seed VALUES -- a repeated seed-value
    may be picked more than once, at different positions, each
    contributing its own independent individual) from
    `resampled_seed_list`, then draws one individual per picked position
    via draw_cross_lineage_group_from_seed_list."""
    if group_size > len(resampled_seed_list):
        raise ValueError(f"group_size={group_size} exceeds resampled list length={len(resampled_seed_list)}")
    chosen_positions = rng.sample(range(len(resampled_seed_list)), group_size)
    chosen_seeds = [resampled_seed_list[i] for i in chosen_positions]
    return draw_cross_lineage_group_from_seed_list(seed_pools, chosen_seeds, rng)


def series_to_word_set(series: str, vocab) -> frozenset:
    """The SET of vocab-restricted 3-grams in one participant's series
    string -- mirrors build_edges's own per-participant parsing exactly
    (split on ',', strip, rstrip('。'), vocab filter). Used only for the
    diversity metrics below; build_edges itself stays untouched (ported
    verbatim from the human-study reference, see markov_recovery.py)."""
    if not series:
        return frozenset()
    return frozenset(n for n in (x.strip().rstrip("。") for x in series.split(",")) if n in vocab)


def group_diversity_metrics(
    series_list: Sequence[str], idents_list: Sequence[Tuple[int, str, str]], vocab,
) -> Dict[str, float]:
    """n_distinct_genotypes: how many distinct genotype identities are in
    this virtual group. mean_pairwise_jaccard_distance: mean, over every
    pair of members, of 1 - Jaccard(their 3-gram sets) -- design doc
    2026-10-16's "群の中の多様さの指標" (how much individual variation a
    sampling scheme's groups contain -- scheme (A) is expected to be
    lower-diversity than (B), since selection has already converged a
    single seed's population toward few genotypes)."""
    n = len(series_list)
    n_distinct_genotypes = len(set(idents_list))
    word_sets = [series_to_word_set(s, vocab) for s in series_list]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = word_sets[i], word_sets[j]
            union = a | b
            jacc = len(a & b) / len(union) if union else 1.0  # both empty -> identical, distance 0
            distances.append(1 - jacc)
    return {
        "n_distinct_genotypes": n_distinct_genotypes,
        "mean_pairwise_jaccard_distance": sum(distances) / len(distances) if distances else 0.0,
    }


def network_metrics(edges: Dict[Tuple[str, str], float], opponent: str) -> dict:
    """One bootstrap network's metrics: widest-path recovery (底底底->和和和)
    and descent (和和和->底底底), Table 7 j* for each of the opponent's
    named circuits (markov_recovery.NAMED_CIRCUITS), and the largest
    j>=.40 strongly-connected component (Table 10).

    `opponent` is the probe.py name (e.g. "Gradual", "Random") -- NAMED_CIRCUITS
    is keyed by the markov_recovery vocab label instead (e.g. "GRA", "RAN"),
    so this maps through CIRCUIT_KEY before the lookup (NOT OPPONENT_INFO's
    vocab_name field -- that would let GradualAppeasement/WSLS silently
    inherit GRA's circuits just because they share its vocabulary; see
    CIRCUIT_KEY's docstring). Bug fixed 2026-10-16: looking NAMED_CIRCUITS
    up directly by the probe.py name silently returned [] for every
    "Gradual"/"Random" call (no opponent name happens to collide with the
    wrong key by accident), so GRA's and RAN's circuits were never
    actually applied before this fix."""
    nodes = {x for e in edges for x in e}
    circuit_key = CIRCUIT_KEY.get(opponent)
    jstars = {i: cycle_jstar(edges, seq) for i, seq in enumerate(NAMED_CIRCUITS.get(circuit_key, []))}
    sccs = strongly_connected_components(edges, threshold=0.40)
    return {
        "has_bottom3": "底底底" in nodes,
        "has_top3": "和和和" in nodes,
        "recovery": widest_path_threshold(edges, start="底底底", goal="和和和"),
        "descent": widest_path_threshold(edges, start="和和和", goal="底底底"),
        "largest_scc_at_40": max((len(c) for c in sccs), default=0),
        "jstars": jstars,
    }


def load_human_weights(csv_path: Path) -> Dict[Tuple[str, str], Dict[Tuple[str, str], float]]:
    """{(opponent_label, group): {(a, b): jaccard}} from human_network_weights.csv."""
    result: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            result[(row["opponent"], row["group"])][(row["from"], row["to"])] = float(row["jaccard"])
    return dict(result)


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


def similarity_to_human(
    edges: Dict[Tuple[str, str], float],
    human_edges: Dict[Tuple[str, str], float],
    threshold: float = 0.30,
) -> Tuple[Optional[float], Optional[float]]:
    """(pearson_correlation, edge_set_jaccard) between two (a,b)->jaccard
    edge-weight dicts -- design doc section 7.4. Correlation: union of
    node-pairs present in either network, missing weight treated as 0.
    Jaccard: over edges whose weight is >= threshold in each network."""
    all_pairs = set(edges) | set(human_edges)
    if not all_pairs:
        return None, None
    xs = [edges.get(p, 0.0) for p in all_pairs]
    ys = [human_edges.get(p, 0.0) for p in all_pairs]
    corr = _pearson(xs, ys)

    present_a = {p for p, j in edges.items() if j >= threshold}
    present_b = {p for p, j in human_edges.items() if j >= threshold}
    union = present_a | present_b
    jacc = len(present_a & present_b) / len(union) if union else None
    return corr, jacc


def _accumulate_bootstrap_metrics(
    sums: Dict[str, float],
    counts: Dict[str, int],
    edges: Dict[Tuple[str, str], float],
    opponent: str,
    human_weights_by_group: Dict[str, Dict[Tuple[str, str], float]],
) -> Tuple[bool, bool]:
    """One bootstrap draw's network_metrics/similarity_to_human contribution
    to the running sums/counts (shared by bootstrap_aggregate and
    bootstrap_aggregate_with_sampler). Returns (has_bottom3, has_top3)."""
    m = network_metrics(edges, opponent)
    for key in ("recovery", "descent"):
        if m[key] is not None:
            sums[key] += m[key]
            counts[key] += 1
    sums["largest_scc_at_40"] += m["largest_scc_at_40"]
    counts["largest_scc_at_40"] += 1
    for ci, v in m["jstars"].items():
        sums[f"jstar_{ci}"] += v
        counts[f"jstar_{ci}"] += 1
    for group_name, hw in human_weights_by_group.items():
        corr, jacc = similarity_to_human(edges, hw)
        if corr is not None:
            sums[f"corr_{group_name}"] += corr
            counts[f"corr_{group_name}"] += 1
        if jacc is not None:
            sums[f"jaccard_{group_name}"] += jacc
            counts[f"jaccard_{group_name}"] += 1
    return m["has_bottom3"], m["has_top3"]


def _finalize_bootstrap_summary(
    sums: Dict[str, float], counts: Dict[str, int], n_bottom3: int, n_top3: int, n_boot: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n_boot": n_boot,
        "rate_bottom3_present": n_bottom3 / n_boot,
        "rate_top3_present": n_top3 / n_boot,
        "rate_recovery_undefined": 1 - counts["recovery"] / n_boot,
        "rate_descent_undefined": 1 - counts["descent"] / n_boot,
    }
    for key, total in sums.items():
        out[f"mean_{key}"] = total / counts[key] if counts[key] else float("nan")
    return out


def bootstrap_aggregate(
    frequencies: Dict[int, int],
    series_by_genotype: Dict[int, List[str]],
    opponent: str,
    group_size: int,
    n_boot: int,
    rng: random.Random,
    human_weights_by_group: Dict[str, Dict[Tuple[str, str], float]],
) -> Dict[str, float]:
    """Repeat resample+build_edges+metrics n_boot times; summarize as
    means (over the bootstraps where the metric was defined) plus the
    rate at which 底底底/和和和 were absent or the widest path was
    undefined (None) -- design doc section 3.3's reporting requirement."""
    vocab = load_vocab(OPPONENT_INFO[opponent][1])

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    n_bottom3 = n_top3 = 0

    for _ in range(n_boot):
        group = resample_virtual_group(frequencies, series_by_genotype, group_size, rng)
        edges = build_edges(group, vocab)
        has_bottom3, has_top3 = _accumulate_bootstrap_metrics(sums, counts, edges, opponent, human_weights_by_group)
        n_bottom3 += has_bottom3
        n_top3 += has_top3

    return _finalize_bootstrap_summary(sums, counts, n_bottom3, n_top3, n_boot)


def bootstrap_aggregate_with_sampler(
    sample_fn,
    opponent: str,
    n_boot: int,
    rng: random.Random,
    human_weights_by_group: Dict[str, Dict[Tuple[str, str], float]],
    vocab,
    with_diversity: bool = False,
) -> Dict[str, float]:
    """Like bootstrap_aggregate, but draws each bootstrap's virtual group
    via an arbitrary `sample_fn(rng) -> (series_list, idents_list)`
    instead of the fixed within-seed resample_virtual_group -- lets the
    same metrics machinery run over resample_cross_seed_group too
    (design doc 2026-10-16, comparing sampling schemes (A)/(B)). If
    with_diversity, also aggregates group_diversity_metrics
    (mean_n_distinct_genotypes, mean_mean_pairwise_jaccard_distance)."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    n_bottom3 = n_top3 = 0

    for _ in range(n_boot):
        series_list, idents_list = sample_fn(rng)
        edges = build_edges(series_list, vocab)
        has_bottom3, has_top3 = _accumulate_bootstrap_metrics(sums, counts, edges, opponent, human_weights_by_group)
        n_bottom3 += has_bottom3
        n_top3 += has_top3
        if with_diversity:
            div = group_diversity_metrics(series_list, idents_list, vocab)
            sums["n_distinct_genotypes"] += div["n_distinct_genotypes"]
            counts["n_distinct_genotypes"] += 1
            sums["mean_pairwise_jaccard_distance"] += div["mean_pairwise_jaccard_distance"]
            counts["mean_pairwise_jaccard_distance"] += 1

    return _finalize_bootstrap_summary(sums, counts, n_bottom3, n_top3, n_boot)
