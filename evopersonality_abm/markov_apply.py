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
# probe.py opponent name -> (human_network_weights.csv label, markov_recovery vocab name)
OPPONENT_INFO = {
    "TFT": ("TFT", "TFT"),
    "AllC": ("ALC", "AllC"),
    "AllD": ("ALD", "AllD"),
    "Random": ("RAN", "RAN"),
    "Gradual": ("GRA", "GRA"),
}


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


def network_metrics(edges: Dict[Tuple[str, str], float], opponent: str) -> dict:
    """One bootstrap network's metrics: widest-path recovery (底底底->和和和)
    and descent (和和和->底底底), Table 7 j* for each of the opponent's
    named circuits (markov_recovery.NAMED_CIRCUITS; empty for an opponent
    with no named circuits, e.g. Random), and the largest j>=.40
    strongly-connected component (Table 10)."""
    nodes = {x for e in edges for x in e}
    jstars = {i: cycle_jstar(edges, seq) for i, seq in enumerate(NAMED_CIRCUITS.get(opponent, []))}
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
    vocab_name = OPPONENT_INFO[opponent][1]
    vocab = load_vocab(vocab_name)

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    n_bottom3 = n_top3 = 0

    for _ in range(n_boot):
        group = resample_virtual_group(frequencies, series_by_genotype, group_size, rng)
        edges = build_edges(group, vocab)
        m = network_metrics(edges, opponent)
        n_bottom3 += m["has_bottom3"]
        n_top3 += m["has_top3"]
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
