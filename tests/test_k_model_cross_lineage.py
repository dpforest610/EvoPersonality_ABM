import random

from evopersonality_abm.k_model_cross_lineage import (
    draw_cross_lineage_identities,
    draw_cross_lineage_identities_from_resampled_seeds,
    effective_group_size,
    lookup_series,
    merge_type_membership,
    seeds_with_type,
    type_membership_from_labels,
)


def test_effective_group_size_shrinks_to_available():
    assert effective_group_size(69, 63) == 63
    assert effective_group_size(69, 70) == 69
    assert effective_group_size(56, 56) == 56


def test_merge_type_membership_combines_disjoint_gids():
    a = ({0: 1, 1: 2}, {0: ("id0",), 1: ("id1",)})
    b = ({2: 3}, {2: ("id2",)})
    freq, idents = merge_type_membership(a, b)
    assert freq == {0: 1, 1: 2, 2: 3}
    assert idents == {0: ("id0",), 1: ("id1",), 2: ("id2",)}


def test_type_membership_from_labels_filters_by_label():
    census = [
        (0, 2, 1, "0,1,0,1", "1,1", 0.7, 0.7),
        (1, 1, 1, "1,0,1,0", "0,0", 0.3, 0.7),
    ]
    gid_to_type = {0: "K_Secure", 1: "K_Dismissing"}
    freq, idents = type_membership_from_labels(census, gid_to_type, "K_Secure")
    assert freq == {0: 2}
    assert idents == {0: (1, "0,1,0,1", "1,1")}


def test_seeds_with_type_excludes_empty_pools():
    pools = {0: ({1: 1}, {1: ("x",)}), 1: ({}, {})}
    assert seeds_with_type(pools) == [0]


def test_draw_cross_lineage_identities_uses_distinct_seeds():
    type_pools = {s: ({0: 1}, {0: (1, "t", "p")}) for s in range(5)}
    rng = random.Random(0)
    identities = draw_cross_lineage_identities(type_pools, 3, rng)
    seeds_used = [s for s, _g in identities]
    assert len(identities) == 3
    assert len(set(seeds_used)) == 3


def test_draw_cross_lineage_identities_raises_if_too_few_seeds():
    type_pools = {0: ({0: 1}, {0: (1, "t", "p")})}
    rng = random.Random(0)
    try:
        draw_cross_lineage_identities(type_pools, 5, rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_draw_cross_lineage_identities_from_resampled_seeds_allows_repeated_seed_values():
    type_pools = {0: ({0: 1}, {0: (1, "t", "p")}), 1: ({0: 1}, {0: (1, "t", "p")})}
    resampled = [0, 0, 1]  # seed 0 appears twice -- each position still draws its own individual
    rng = random.Random(0)
    identities = draw_cross_lineage_identities_from_resampled_seeds(type_pools, resampled, 2, rng)
    assert len(identities) == 2


def test_lookup_series_draws_from_recorded_options():
    identities = [(0, 1), (1, 2)]
    series_by_seed_opponent = {0: {"TFT": {1: ["s1a", "s1b"]}}, 1: {"TFT": {2: ["s2a"]}}}
    rng = random.Random(0)
    series = lookup_series(identities, series_by_seed_opponent, "TFT", rng)
    assert series[0] in ("s1a", "s1b")
    assert series[1] == "s2a"


def test_lookup_series_missing_entry_returns_empty_string():
    identities = [(0, 99)]
    series_by_seed_opponent = {0: {"TFT": {}}}
    rng = random.Random(0)
    series = lookup_series(identities, series_by_seed_opponent, "TFT", rng)
    assert series == [""]
