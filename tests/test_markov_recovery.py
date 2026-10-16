import itertools
import random

import pytest

from evopersonality_abm.markov_recovery import (
    NAMED_CIRCUITS,
    build_edges,
    count_simple_cycles,
    cycle_jstar,
    load_vocab,
    series_from_moves,
    strongly_connected_components,
    widest_path_threshold,
)

FULL_VOCAB = frozenset("".join(p) for p in itertools.product("和惨搾底", repeat=3))


# ------------------------------------------------------------------ vocab files

def test_vocab_file_sizes():
    assert len(load_vocab("TFT")) == 16
    assert len(load_vocab("GRA")) == 64
    assert len(load_vocab("RAN")) == 64
    assert len(load_vocab("AllC")) == 8
    assert len(load_vocab("AllD")) == 8


def test_gra_and_ran_vocab_are_the_full_unrestricted_64():
    assert load_vocab("GRA") == FULL_VOCAB
    assert load_vocab("RAN") == FULL_VOCAB


def test_allc_vocab_is_only_states_with_opponent_cooperating():
    # AllC never defects, so the opponent's action (2nd letter) is always C:
    # only 和(CC) and 搾(DC) can appear
    for word in load_vocab("AllC"):
        assert all(ch in "和搾" for ch in word)


def test_alld_vocab_is_only_states_with_opponent_defecting():
    for word in load_vocab("AllD"):
        assert all(ch in "惨底" for ch in word)


# ------------------------------------------------------------------ series_from_moves

def test_series_from_moves_all_cooperate():
    focal = opp = ["C"] * 20
    s = series_from_moves(focal, opp, n_rounds=20)
    assert s == ",".join(["和和和"] * 18) + "。"


def test_series_from_moves_short_sequence_has_no_grams():
    # 2 rounds -> 0 sliding 3-grams
    assert series_from_moves(["C", "C"], ["C", "C"], n_rounds=20) == "。"


def test_series_from_moves_mixed():
    focal = ["C", "C", "D", "D"]
    opp = ["C", "D", "D", "C"]
    # states: CC, CD, DD, DC -> 和, 惨, 底, 搾
    s = series_from_moves(focal, opp, n_rounds=4)
    assert s == "和惨底,惨底搾。"


# ------------------------------------------------------------------ build_edges / widest_path_threshold
# (reference-implementation parity: these mirror markov3_widest_path_reference.py's
# own logic exactly, since build_edges/widest_path_threshold are a verbatim port)

def test_build_edges_requires_cooccurrence_within_a_participant_not_adjacency():
    # "AB" and "BC" never co-occur in the SAME participant's set -> no edge,
    # even though they are adjacent in a sequence string.
    vocab = frozenset({"ABA", "BAB"})
    series = ["ABA,ABA。"]  # only ABA ever appears
    edges = build_edges(series, vocab)
    assert edges == {}


def test_build_edges_jaccard_over_participants():
    vocab = frozenset({"AAB", "ABA"})
    # participant 0: both AAB and ABA appear -> contributes to both
    # participant 1: only AAB appears
    series = ["AAB,ABA。", "AAB。"]
    edges = build_edges(series, vocab)
    # AAB -> ABA: tail(AAB)="AB" == head(ABA)="AB" -> edge exists
    assert ("AAB", "ABA") in edges
    # lines(AAB) = {0,1}, lines(ABA) = {0} -> Jaccard = 1/2
    assert edges[("AAB", "ABA")] == pytest.approx(0.5)


def test_widest_path_threshold_hand_computed_example():
    # A -> B (j=0.8) -> C (j=0.2) -> GOAL ; A -> D (j=0.5) -> GOAL
    # path via B,C: bottleneck 0.2 ; path via D: bottleneck 0.5 -> widest = 0.5
    edges = {
        ("START", "B"): 0.8,
        ("B", "C"): 0.2,
        ("C", "GOAL"): 0.9,
        ("START", "D"): 0.5,
        ("D", "GOAL"): 0.5,
    }
    assert widest_path_threshold(edges, start="START", goal="GOAL") == pytest.approx(0.5)


def test_widest_path_threshold_none_when_start_or_goal_missing():
    edges = {("A", "B"): 0.5, ("B", "C"): 0.5}
    assert widest_path_threshold(edges, start="底底底", goal="和和和") is None
    assert widest_path_threshold(edges, start="A", goal="和和和") is None
    assert widest_path_threshold(edges, start="底底底", goal="C") is None


def test_widest_path_threshold_direct_edge():
    edges = {("底底底", "和和和"): 0.42}
    assert widest_path_threshold(edges) == pytest.approx(0.42)


# ------------------------------------------------------------------ cycle_jstar (7.1, ported from R)

def test_cycle_jstar_is_the_minimum_edge_weight_along_the_walk():
    # ALD-bridge-style 2-cycle: 底惨底 <-> 惨底惨
    edges = {
        ("底惨底", "惨底惨"): 0.7,
        ("惨底惨", "底惨底"): 0.3,
    }
    j = cycle_jstar(edges, ["底惨底", "惨底惨", "底惨底"])
    assert j == pytest.approx(0.3)


def test_cycle_jstar_missing_edge_counts_as_zero():
    edges = {("A", "B"): 0.9}  # B->C missing
    assert cycle_jstar(edges, ["A", "B", "C"]) == 0.0


def test_cycle_jstar_single_node_walk_is_empty():
    assert cycle_jstar({}, ["A"]) == 0.0


# ------------------------------------------------------------------ NAMED_CIRCUITS (Table 7's circuit list, 2026-10-16)

def test_named_circuits_covers_exactly_the_four_opponents_with_a_given_list():
    assert set(NAMED_CIRCUITS) == {"TFT", "GRA", "AllC", "AllD"}


def test_named_circuits_counts_match_what_was_given():
    assert len(NAMED_CIRCUITS["TFT"]) == 4
    assert len(NAMED_CIRCUITS["GRA"]) == 5
    assert len(NAMED_CIRCUITS["AllC"]) == 4
    assert len(NAMED_CIRCUITS["AllD"]) == 9


def test_named_circuits_are_closed_walks_except_the_allc_bridge():
    for opponent, circuits in NAMED_CIRCUITS.items():
        for seq in circuits:
            is_bridge = opponent == "AllC" and seq == ["和和和", "和和搾"]
            if is_bridge:
                assert seq[0] != seq[-1]
            else:
                assert seq[0] == seq[-1], f"{opponent} circuit not closed: {seq}"


def test_named_circuits_consecutive_nodes_form_valid_3gram_transitions():
    # every consecutive pair (a, b) in every circuit must satisfy the same
    # a[-2:] == b[:2] rule build_edges uses to decide an edge can exist.
    for opponent, circuits in NAMED_CIRCUITS.items():
        for seq in circuits:
            for a, b in zip(seq, seq[1:]):
                assert len(a) == 3 and len(b) == 3
                assert a[-2:] == b[:2], f"{opponent}: {a} -> {b} is not a valid 3-gram transition"


def test_named_circuits_nodes_respect_opponent_vocab_restriction():
    restricted = {"AllC": set("和搾"), "AllD": set("惨底")}
    for opponent, allowed in restricted.items():
        for seq in NAMED_CIRCUITS[opponent]:
            for node in seq:
                assert set(node) <= allowed, f"{opponent} node {node} uses chars outside {allowed}"


def test_named_circuits_cycle_jstar_runs_on_every_circuit_without_error():
    # sanity: each circuit is well-formed enough for cycle_jstar to run
    # (returns 0.0, the "missing edge" value, on an empty edge dict).
    for opponent, circuits in NAMED_CIRCUITS.items():
        for seq in circuits:
            assert cycle_jstar({}, seq) == 0.0


# ------------------------------------------------------------------ strongly_connected_components (Table 10)

def test_scc_hand_computed_two_components():
    # A<->B strongly connected; C is only reachable, not reaching back -> singleton
    edges = {
        ("A", "B"): 0.5,
        ("B", "A"): 0.5,
        ("B", "C"): 0.5,
    }
    comps = strongly_connected_components(edges, threshold=0.4)
    comp_sets = sorted([set(c) for c in comps], key=len)
    assert comp_sets == [{"C"}, {"A", "B"}]


def test_scc_threshold_splits_a_cycle_apart():
    # A->B (0.6) ->C (0.2) ->A (0.6): below threshold 0.4, the B->C edge
    # drops out, breaking the 3-cycle into singletons.
    edges = {("A", "B"): 0.6, ("B", "C"): 0.2, ("C", "A"): 0.6}
    comps_low = strongly_connected_components(edges, threshold=0.1)
    assert {"A", "B", "C"} in [set(c) for c in comps_low]
    comps_high = strongly_connected_components(edges, threshold=0.4)
    assert all(len(c) == 1 for c in comps_high)


def test_scc_isolated_nodes_are_singletons():
    edges = {("A", "B"): 0.9}
    comps = strongly_connected_components(edges, threshold=0.9)
    assert {frozenset(c) for c in comps} == {frozenset({"A"}), frozenset({"B"})}


# ------------------------------------------------------------------ count_simple_cycles (Table 12, new definition)

def test_count_simple_cycles_single_triangle():
    edges = {("A", "B"): 0.9, ("B", "C"): 0.9, ("C", "A"): 0.9}
    assert count_simple_cycles(edges, threshold=0.5, max_length=6) == 1


def test_count_simple_cycles_counts_rotations_once():
    # same triangle, but add the reverse-direction edges too: this creates
    # 3 two-cycles (A<->B, B<->C, A<->C) plus 2 three-cycles
    # (A->B->C->A and A->C->B->A) = 5 distinct simple cycles -- NOT 9
    # (3 rotations each of 3 three-cycle directions), which is what an
    # implementation that double-counts rotations would report.
    edges = {
        ("A", "B"): 0.9, ("B", "C"): 0.9, ("C", "A"): 0.9,
        ("B", "A"): 0.9, ("C", "B"): 0.9, ("A", "C"): 0.9,
    }
    assert count_simple_cycles(edges, threshold=0.5, max_length=6) == 5


def test_count_simple_cycles_respects_length_cap():
    # a single 4-cycle A->B->C->D->A
    edges = {("A", "B"): 0.9, ("B", "C"): 0.9, ("C", "D"): 0.9, ("D", "A"): 0.9}
    assert count_simple_cycles(edges, threshold=0.5, max_length=3) == 0
    assert count_simple_cycles(edges, threshold=0.5, max_length=4) == 1


def test_count_simple_cycles_no_cycles_in_a_dag():
    edges = {("A", "B"): 0.9, ("B", "C"): 0.9, ("A", "C"): 0.9}
    assert count_simple_cycles(edges, threshold=0.5, max_length=6) == 0


def test_count_simple_cycles_threshold_excludes_weak_edges():
    edges = {("A", "B"): 0.9, ("B", "C"): 0.9, ("C", "A"): 0.2}
    assert count_simple_cycles(edges, threshold=0.5, max_length=6) == 0
    assert count_simple_cycles(edges, threshold=0.1, max_length=6) == 1


def test_count_simple_cycles_handles_a_dense_random_graph_without_crashing():
    # sanity/perf check at roughly the real scale (64 nodes), not a value assertion
    rng = random.Random(0)
    nodes = [f"n{i}" for i in range(64)]
    edges = {}
    for a in nodes:
        for b in rng.sample(nodes, 5):
            if a != b:
                edges[(a, b)] = rng.random()
    n = count_simple_cycles(edges, threshold=0.3, max_length=6)
    assert isinstance(n, int) and n >= 0
