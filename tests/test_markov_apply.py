import csv
import gzip
import random

from evopersonality_abm.markov_apply import (
    bootstrap_aggregate,
    bootstrap_aggregate_with_sampler,
    group_diversity_metrics,
    load_census,
    load_genotype_identities,
    load_human_weights,
    load_series_by_genotype_all_opponents,
    network_metrics,
    resample_cross_seed_group,
    resample_virtual_group,
    resample_within_seed_group_with_identity,
    series_to_word_set,
    similarity_to_human,
)
from evopersonality_abm.markov_recovery import build_edges, load_vocab, series_from_moves


# ------------------------------------------------------------------ load_census / load_series_by_genotype_all_opponents

def test_load_census(tmp_path):
    path = tmp_path / "census.csv.gz"
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        w.writerow([0, 1, 7, "0,1,0,1", "1,1"])
        w.writerow([1, 1, 3, "0,0,0,0", "0,0"])
    assert load_census(path) == {0: 7, 1: 3}


def _write_moves(path, rows):
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
        w.writerows(rows)


def test_load_series_by_genotype_all_opponents_matches_series_from_moves_directly(tmp_path):
    path = tmp_path / "moves.csv.gz"
    own_moves = ["C", "D", "C", "C", "D"]
    opp_moves = ["D", "D", "C", "C", "C"]
    rows = [(0, "TFT", 0, r + 1, own_moves[r], opp_moves[r]) for r in range(5)]
    _write_moves(path, rows)

    result = load_series_by_genotype_all_opponents(path, main_analysis_rounds=5)
    assert list(result.keys()) == ["TFT"]
    assert result["TFT"][0] == [series_from_moves(own_moves, opp_moves, n_rounds=5)]


def test_load_series_by_genotype_all_opponents_separates_opponents_and_repeats(tmp_path):
    path = tmp_path / "moves.csv.gz"
    rows = [
        (0, "TFT", 0, 1, "C", "C"), (0, "TFT", 0, 2, "C", "C"), (0, "TFT", 0, 3, "C", "C"),
        (0, "TFT", 1, 1, "D", "D"), (0, "TFT", 1, 2, "D", "D"), (0, "TFT", 1, 3, "D", "D"),
        (0, "AllD", 0, 1, "C", "D"), (0, "AllD", 0, 2, "C", "D"), (0, "AllD", 0, 3, "C", "D"),
        (1, "TFT", 0, 1, "C", "C"), (1, "TFT", 0, 2, "C", "C"), (1, "TFT", 0, 3, "C", "C"),
    ]
    _write_moves(path, rows)

    result = load_series_by_genotype_all_opponents(path, main_analysis_rounds=3)
    assert set(result.keys()) == {"TFT", "AllD"}
    assert len(result["TFT"][0]) == 2  # two repeats for genotype 0 vs TFT
    assert len(result["TFT"][1]) == 1
    assert len(result["AllD"][0]) == 1


# ------------------------------------------------------------------ resample_virtual_group

def test_resample_virtual_group_respects_zero_weight_exclusion():
    frequencies = {0: 10, 1: 0}
    series_by_genotype = {0: ["和和和,和和和。"], 1: ["底底底,底底底。"]}
    rng = random.Random(0)
    group = resample_virtual_group(frequencies, series_by_genotype, group_size=20, rng=rng)
    assert len(group) == 20
    assert all(s == "和和和,和和和。" for s in group)


def test_resample_virtual_group_returns_group_size_entries():
    frequencies = {0: 1, 1: 1, 2: 1}
    series_by_genotype = {0: ["a"], 1: ["b"], 2: ["c"]}
    group = resample_virtual_group(frequencies, series_by_genotype, group_size=15, rng=random.Random(1))
    assert len(group) == 15
    assert set(group) <= {"a", "b", "c"}


def test_resample_virtual_group_is_deterministic_given_same_rng_seed():
    frequencies = {0: 5, 1: 5}
    series_by_genotype = {0: ["x1", "x2"], 1: ["y1", "y2"]}
    g1 = resample_virtual_group(frequencies, series_by_genotype, 10, random.Random(42))
    g2 = resample_virtual_group(frequencies, series_by_genotype, 10, random.Random(42))
    assert g1 == g2


# ------------------------------------------------------------------ network_metrics

def test_network_metrics_hand_built_small_network():
    # 底底底 -(.8)-> 底底惨 -(.6)-> 底惨底 -(.9)-> 惨底底 -(.3)-> 底底底 (recovers to nothing but itself; no 和和和 at all)
    edges = {
        ("底底底", "底底惨"): 0.8,
        ("底底惨", "底惨底"): 0.6,
        ("底惨底", "惨底底"): 0.9,
        ("惨底底", "底底底"): 0.3,
    }
    m = network_metrics(edges, "AllD")
    assert m["has_bottom3"] is True
    assert m["has_top3"] is False
    assert m["recovery"] is None  # 和和和 absent entirely
    assert m["descent"] is None
    # AllD's circuit #2 is exactly this 4-cycle: 底底底->底底惨->底惨底->惨底底->底底底
    assert m["jstars"][2] == min(0.8, 0.6, 0.9, 0.3)


def test_network_metrics_recovery_defined_when_goal_reachable():
    edges = {
        ("底底底", "底底惨"): 0.9,
        ("底底惨", "惨惨惨"): 0.5,
        # a path all the way to 和和和 via made-up TFT-vocab nodes isn't needed --
        # just confirm a reachable goal gives a non-None threshold.
    }
    edges[("惨惨惨", "和和和")] = 0.4
    m = network_metrics(edges, "Random")
    # Random (vocab/circuit-list key "RAN") has exactly 1 named circuit
    # (the Bridge, 底惨底<->惨底惨); neither of its edges is present here.
    assert m["jstars"] == {0: 0.0}
    assert m["has_top3"] is True
    assert m["recovery"] == min(0.9, 0.5, 0.4)  # widest_path_threshold's bottleneck


def test_network_metrics_maps_probe_opponent_name_to_circuit_list_key():
    # regression test for the 2026-10-16 bug: NAMED_CIRCUITS is keyed by
    # the markov_recovery vocab label ("GRA"/"RAN"), not the probe.py
    # opponent name ("Gradual"/"Random") -- network_metrics must map
    # through OPPONENT_INFO, not look NAMED_CIRCUITS up by `opponent` directly.
    from evopersonality_abm.markov_recovery import NAMED_CIRCUITS

    edges = {("底惨底", "惨底惨"): 0.6, ("惨底惨", "底惨底"): 0.8}
    m_gradual = network_metrics(edges, "Gradual")
    assert len(m_gradual["jstars"]) == len(NAMED_CIRCUITS["GRA"])
    # GRA's circuit #2 is exactly this 2-node reciprocal pair
    assert m_gradual["jstars"][2] == min(0.6, 0.8)

    m_random = network_metrics(edges, "Random")
    assert len(m_random["jstars"]) == len(NAMED_CIRCUITS["RAN"]) == 1
    assert m_random["jstars"][0] == min(0.6, 0.8)


# ------------------------------------------------------------------ load_human_weights / similarity_to_human

def test_load_human_weights(tmp_path):
    path = tmp_path / "human.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "group", "n", "from", "to", "jaccard"])
        w.writerow(["TFT", "Secure", 56, "底底底", "底底惨", "0.5"])
        w.writerow(["TFT", "Mix", 46, "底底底", "底底惨", "0.2"])
    weights = load_human_weights(path)
    assert weights[("TFT", "Secure")] == {("底底底", "底底惨"): 0.5}
    assert weights[("TFT", "Mix")] == {("底底底", "底底惨"): 0.2}


def test_similarity_to_human_identical_networks():
    edges = {("a", "b"): 0.6, ("b", "c"): 0.4}
    corr, jacc = similarity_to_human(edges, dict(edges), threshold=0.30)
    assert abs(corr - 1.0) < 1e-9
    assert jacc == 1.0


def test_similarity_to_human_disjoint_networks_below_threshold():
    edges_a = {("a", "b"): 0.9}
    edges_b = {("c", "d"): 0.9}
    corr, jacc = similarity_to_human(edges_a, edges_b, threshold=0.30)
    # disjoint pairs: a's weight 0.9 vs b's 0; b's 0.9 vs a's 0 -- perfectly
    # anti-correlated structure collapses to a defined but non-trivial value
    assert corr is not None
    assert jacc == 0.0  # no edge in common between the two present-sets


def test_similarity_to_human_empty_networks_returns_none():
    assert similarity_to_human({}, {}) == (None, None)


# ------------------------------------------------------------------ bootstrap_aggregate

def test_bootstrap_aggregate_single_genotype_is_deterministic_across_boots():
    # Only one genotype, one repeat, group_size=5 -> every bootstrap draw
    # is identical (same series repeated), so rates must be exactly 0 or 1.
    focal = ["C"] * 20
    opp_always_d = ["D"] * 20
    series = series_from_moves(focal, opp_always_d, n_rounds=20)
    frequencies = {0: 1}
    series_by_genotype = {0: [series]}
    human_weights = {"Secure": {}, "Mix": {}, "Fearful": {}}

    result = bootstrap_aggregate(
        frequencies, series_by_genotype, "AllD", group_size=5, n_boot=10,
        rng=random.Random(0), human_weights_by_group=human_weights,
    )
    assert result["n_boot"] == 10
    assert result["rate_bottom3_present"] in (0.0, 1.0)
    assert result["rate_top3_present"] == 0.0  # AllC/AllD vocab never contains 和和和
    assert 0.0 <= result["rate_recovery_undefined"] <= 1.0


# ------------------------------------------------------------------ load_genotype_identities

def test_load_genotype_identities(tmp_path):
    path = tmp_path / "census.csv.gz"
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        w.writerow([0, 1, 7, "0,1,0,1", "1,1"])
        w.writerow([1, 1, 3, "0,0,0,0", "0,0"])
    idents = load_genotype_identities(path)
    assert idents == {0: (1, "0,1,0,1", "1,1"), 1: (1, "0,0,0,0", "0,0")}


# ------------------------------------------------------------------ resample_within_seed_group_with_identity

def test_resample_within_seed_group_with_identity_pairs_series_and_identity():
    frequencies = {0: 10, 1: 0}
    series_by_genotype = {0: ["s0"], 1: ["s1"]}
    identities = {0: (1, "t0", "p0"), 1: (1, "t1", "p1")}
    series, idents = resample_within_seed_group_with_identity(
        frequencies, series_by_genotype, identities, group_size=20, rng=random.Random(0)
    )
    assert len(series) == len(idents) == 20
    assert all(s == "s0" for s in series)  # genotype 1 has zero weight
    assert all(i == (1, "t0", "p0") for i in idents)


# ------------------------------------------------------------------ resample_cross_seed_group

def _seed_pool(genotype_id, series, identity, frequency=1):
    return ({genotype_id: frequency}, {genotype_id: [series]}, {genotype_id: identity})


def test_resample_cross_seed_group_draws_one_distinct_seed_each():
    seed_pools = {
        s: _seed_pool(0, f"series_{s}", (1, f"table_{s}", "p"))
        for s in range(5)
    }
    series, idents = resample_cross_seed_group(seed_pools, group_size=5, rng=random.Random(0))
    assert len(series) == 5
    # every seed contributes exactly one member (distinct seeds, group_size == n_seeds)
    assert sorted(series) == [f"series_{s}" for s in range(5)]
    assert len(set(idents)) == 5  # each seed's genotype table differs, so 5 distinct identities


def test_resample_cross_seed_group_raises_when_group_size_exceeds_available_seeds():
    seed_pools = {s: _seed_pool(0, f"series_{s}", (1, "t", "p")) for s in range(3)}
    try:
        resample_cross_seed_group(seed_pools, group_size=5, rng=random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resample_cross_seed_group_is_deterministic_given_same_rng_seed():
    seed_pools = {s: _seed_pool(0, f"series_{s}", (1, f"t{s}", "p")) for s in range(5)}
    out1 = resample_cross_seed_group(seed_pools, group_size=3, rng=random.Random(7))
    out2 = resample_cross_seed_group(seed_pools, group_size=3, rng=random.Random(7))
    assert out1 == out2


# ------------------------------------------------------------------ series_to_word_set / group_diversity_metrics

def test_series_to_word_set_matches_build_edges_per_participant_parsing():
    vocab = load_vocab("AllD")
    focal = ["D"] * 20
    opp = ["D"] * 20
    series = series_from_moves(focal, opp, n_rounds=20)
    word_set = series_to_word_set(series, vocab)
    assert word_set == {"底底底"}  # all-DD-throughout has only one 3-gram


def test_series_to_word_set_empty_series_is_empty_set():
    assert series_to_word_set("", load_vocab("TFT")) == frozenset()


def test_group_diversity_metrics_identical_members_have_zero_diversity():
    vocab = load_vocab("AllD")
    series = series_from_moves(["D"] * 20, ["D"] * 20, n_rounds=20)
    series_list = [series] * 4
    idents_list = [(1, "t", "p")] * 4
    result = group_diversity_metrics(series_list, idents_list, vocab)
    assert result["n_distinct_genotypes"] == 1
    assert result["mean_pairwise_jaccard_distance"] == 0.0


def test_group_diversity_metrics_all_distinct_members_have_higher_diversity():
    vocab = load_vocab("AllD")
    same = group_diversity_metrics(
        [series_from_moves(["D"] * 20, ["D"] * 20, n_rounds=20)] * 4,
        [(1, "t", "p")] * 4, vocab,
    )
    different_series = [
        series_from_moves(["D"] * 20, ["D"] * 20, n_rounds=20),          # all-D
        series_from_moves(["C"] * 20, ["D"] * 20, n_rounds=20),          # all-C vs D
        series_from_moves(["D", "C"] * 10, ["D"] * 20, n_rounds=20),
        series_from_moves(["C", "D"] * 10, ["D"] * 20, n_rounds=20),
    ]
    different = group_diversity_metrics(different_series, [(1, f"t{i}", "p") for i in range(4)], vocab)
    assert different["n_distinct_genotypes"] == 4
    assert different["mean_pairwise_jaccard_distance"] > same["mean_pairwise_jaccard_distance"]


# ------------------------------------------------------------------ bootstrap_aggregate_with_sampler

def test_bootstrap_aggregate_with_sampler_matches_bootstrap_aggregate_for_an_equivalent_sampler():
    # sample_fn wrapping resample_virtual_group (ignoring identities) must
    # reproduce bootstrap_aggregate's own numbers bit-for-bit, given the
    # same rng draw sequence.
    focal = ["C", "D"] * 10
    opp = ["D", "C"] * 10
    series = series_from_moves(focal, opp, n_rounds=20)
    frequencies = {0: 3, 1: 1}
    series_by_genotype = {0: [series], 1: [series_from_moves(["D"] * 20, ["D"] * 20, n_rounds=20)]}
    human_weights = {"Secure": {}, "Mix": {}, "Fearful": {}}
    vocab = load_vocab("TFT")

    direct = bootstrap_aggregate(
        frequencies, series_by_genotype, "TFT", group_size=4, n_boot=20,
        rng=random.Random(5), human_weights_by_group=human_weights,
    )

    def sample_fn(rng):
        return resample_virtual_group(frequencies, series_by_genotype, 4, rng), None

    via_sampler = bootstrap_aggregate_with_sampler(
        sample_fn, "TFT", n_boot=20, rng=random.Random(5),
        human_weights_by_group=human_weights, vocab=vocab, with_diversity=False,
    )
    assert direct == via_sampler


def test_bootstrap_aggregate_with_sampler_with_diversity_adds_diversity_keys():
    vocab = load_vocab("AllD")
    series = series_from_moves(["D"] * 20, ["D"] * 20, n_rounds=20)
    seed_pools = {s: _seed_pool(0, series, (1, f"t{s}", "p")) for s in range(5)}

    def sample_fn(rng):
        return resample_cross_seed_group(seed_pools, group_size=5, rng=rng)

    result = bootstrap_aggregate_with_sampler(
        sample_fn, "AllD", n_boot=5, rng=random.Random(1),
        human_weights_by_group={"Secure": {}, "Mix": {}, "Fearful": {}}, vocab=vocab,
        with_diversity=True,
    )
    assert "mean_n_distinct_genotypes" in result
    assert "mean_mean_pairwise_jaccard_distance" in result
    assert result["mean_n_distinct_genotypes"] == 5  # every seed_pool has a distinct identity
