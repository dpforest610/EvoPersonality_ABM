import itertools

import pytest

from evopersonality_abm.agent import Agent
from evopersonality_abm.duel import run_duel
from evopersonality_abm.lindgren import (
    LindgrenTable,
    always_cooperate_table,
    always_defect_table,
    duplicate_memory,
    effective_memory_depth,
    pavlov_table,
    point_mutate,
    split_memory,
    tit_for_tat_table,
)
from evopersonality_abm.strategies import always_cooperate, always_defect, tit_for_tat


def test_rejects_wrong_table_length():
    with pytest.raises(ValueError):
        LindgrenTable(memory_depth=1, table=[0, 1, 0], prehistory=[1, 1])


def test_rejects_wrong_prehistory_length():
    with pytest.raises(ValueError):
        LindgrenTable(memory_depth=1, table=[0, 1, 0, 1], prehistory=[1])


def test_rejects_non_binary_entries():
    with pytest.raises(ValueError):
        LindgrenTable(memory_depth=1, table=[0, 1, 0, 2], prehistory=[1, 1])
    with pytest.raises(ValueError):
        LindgrenTable(memory_depth=1, table=[0, 1, 0, 1], prehistory=[1, 2])


def test_lookup_uses_own_and_opponent_last_move_directly():
    # anti-TFT: always play the opposite of the opponent's last move.
    table = LindgrenTable(memory_depth=1, table=[1, 0, 1, 0], prehistory=[0, 0])

    assert table.decide(own_history=[], opp_history=[]) == "C"  # prehistory opp=D -> C
    assert table.decide(own_history=["D"], opp_history=["C"]) == "D"
    assert table.decide(own_history=["C"], opp_history=["D"]) == "C"
    # own move is ignored by this table; only the opponent's last move matters.
    assert table.decide(own_history=["D"], opp_history=["D"]) == "C"


def test_depth1_table_matches_tit_for_tat_vs_always_defect():
    a = Agent("A", tit_for_tat_table().as_strategy())
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=5)

    ref_a = Agent("A", tit_for_tat)
    ref_b = Agent("B", always_defect)
    run_duel(ref_a, ref_b, rounds=5)

    assert a.history_self == ref_a.history_self == ["C", "D", "D", "D", "D"]


def test_depth1_table_matches_tit_for_tat_vs_always_cooperate():
    a = Agent("A", tit_for_tat_table().as_strategy())
    b = Agent("B", always_cooperate)
    run_duel(a, b, rounds=5)

    assert a.history_self == ["C"] * 5


def test_mutual_tit_for_tat_tables_cooperate_forever():
    a = Agent("A", tit_for_tat_table().as_strategy())
    b = Agent("B", tit_for_tat_table().as_strategy())
    run_duel(a, b, rounds=5)

    assert a.history_self == b.history_self == ["C"] * 5


def test_always_cooperate_and_always_defect_tables_match_hand_written_strategies():
    a = Agent("A", always_cooperate_table().as_strategy())
    b = Agent("B", always_defect_table().as_strategy())
    run_duel(a, b, rounds=5)

    ref_a = Agent("A", always_cooperate)
    ref_b = Agent("B", always_defect)
    run_duel(ref_a, ref_b, rounds=5)

    assert a.history_self == ref_a.history_self == ["C"] * 5
    assert b.history_self == ref_b.history_self == ["D"] * 5
    assert a.score == ref_a.score == 0
    assert b.score == ref_b.score == 25


def test_pavlov_table_cooperates_after_mutual_defection():
    a = Agent("A", pavlov_table().as_strategy())
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=4)

    # C (assumed CC), then D (mismatched with opp D -> shift), then C
    # (mutual DD -> shift back), then D again, alternating.
    assert a.history_self == ["C", "D", "C", "D"]


def test_mutual_pavlov_tables_cooperate_forever():
    a = Agent("A", pavlov_table().as_strategy())
    b = Agent("B", pavlov_table().as_strategy())
    run_duel(a, b, rounds=5)

    assert a.history_self == b.history_self == ["C"] * 5


# --- Duplication mutation: memory depth m -> m+1, must stay neutral ---


def test_duplicate_memory_grows_depth_by_one():
    depth2 = duplicate_memory(tit_for_tat_table())

    assert depth2.memory_depth == 2
    assert len(depth2.table) == 2 ** (2 * 2)
    assert len(depth2.prehistory) == 2 * 2


def test_duplicate_memory_from_depth_zero():
    # A depth-0 table has no history-dependence at all: one fixed move.
    always_defect_depth0 = LindgrenTable(memory_depth=0, table=[0], prehistory=[])
    depth1 = duplicate_memory(always_defect_depth0)

    assert depth1.memory_depth == 1
    for own_history, opp_history in [([], []), (["C"], ["C"]), (["D"], ["C"])]:
        assert (
            depth1.decide(own_history, opp_history)
            == always_defect_depth0.decide(own_history, opp_history)
            == "D"
        )


@pytest.mark.parametrize(
    "make_table",
    [tit_for_tat_table, always_cooperate_table, always_defect_table, pavlov_table],
)
def test_duplicate_memory_matches_original_on_every_possible_history(make_table):
    original = make_table()
    duplicated = duplicate_memory(original)

    # Exhaustively cover every own/opp history of length 0..4, not just the
    # ones a real duel would produce, since the table is a pure lookup and
    # must agree everywhere -- this is the direct, bit-level check that the
    # extra memory round is never actually consulted.
    for length in range(5):
        for own_bits in itertools.product("CD", repeat=length):
            for opp_bits in itertools.product("CD", repeat=length):
                own_history, opp_history = list(own_bits), list(opp_bits)
                assert duplicated.decide(own_history, opp_history) == original.decide(
                    own_history, opp_history
                )


@pytest.mark.parametrize(
    "make_table",
    [tit_for_tat_table, always_cooperate_table, always_defect_table, pavlov_table],
)
@pytest.mark.parametrize(
    "make_opponent_strategy",
    [
        lambda: always_defect,
        lambda: always_cooperate,
        lambda: tit_for_tat,
        lambda: pavlov_table().as_strategy(),
    ],
)
def test_duplicated_table_plays_identically_to_original_in_a_duel(
    make_table, make_opponent_strategy
):
    original = Agent("orig", make_table().as_strategy())
    duplicated = Agent("dup", duplicate_memory(make_table()).as_strategy())

    run_duel(original, Agent("opp", make_opponent_strategy()), rounds=30)
    run_duel(duplicated, Agent("opp", make_opponent_strategy()), rounds=30)

    assert duplicated.history_self == original.history_self
    assert duplicated.score == original.score


def test_repeated_duplication_stays_neutral_across_several_generations():
    original = tit_for_tat_table()
    grown = duplicate_memory(duplicate_memory(duplicate_memory(original)))
    assert grown.memory_depth == 4

    a = Agent("orig", original.as_strategy())
    b = Agent("grown", grown.as_strategy())
    run_duel(a, Agent("opp", always_defect), rounds=20)
    run_duel(b, Agent("opp", always_defect), rounds=20)

    assert b.history_self == a.history_self


# --- Split mutation: memory depth m -> m-1, exact inverse of duplication ---


def test_split_memory_rejects_depth_zero():
    depth0 = LindgrenTable(memory_depth=0, table=[0], prehistory=[])
    with pytest.raises(ValueError):
        split_memory(depth0)


def test_split_memory_shrinks_depth_by_one():
    depth0 = split_memory(tit_for_tat_table())
    assert depth0.memory_depth == 0
    assert len(depth0.table) == 1
    assert len(depth0.prehistory) == 0


@pytest.mark.parametrize(
    "make_table",
    [tit_for_tat_table, always_cooperate_table, always_defect_table, pavlov_table],
)
def test_split_undoes_duplicate_exactly(make_table):
    original = make_table()
    round_tripped = split_memory(duplicate_memory(original))

    assert round_tripped.memory_depth == original.memory_depth
    assert round_tripped.table == original.table
    assert round_tripped.prehistory == original.prehistory


@pytest.mark.parametrize(
    "make_table",
    [tit_for_tat_table, always_cooperate_table, always_defect_table, pavlov_table],
)
def test_split_then_duplicate_then_split_is_stable(make_table):
    # Once memory has been split back down, re-duplicating and splitting
    # again should keep reproducing the same (now depth-m) table -- the
    # round-trip identity holds at any depth, not just the original one.
    once = split_memory(duplicate_memory(make_table()))
    twice = split_memory(duplicate_memory(once))
    assert twice.memory_depth == once.memory_depth
    assert twice.table == once.table
    assert twice.prehistory == once.prehistory


def test_split_memory_keeps_the_forgotten_dd_quadrant_when_quadrants_diverge():
    # Build a depth-1 table by hand where the two "quadrants" a
    # subsequent duplicate_memory would create diverge only through a
    # later point mutation -- split_memory must recover exactly the
    # entries reachable by assuming the forgotten round was (D, D).
    depth2 = duplicate_memory(tit_for_tat_table())
    # Flip an entry that is NOT in the (own_extra=0, opp_extra=0)
    # quadrant, i.e. index >= len(depth2.table) // 4 with a leading 1 bit
    # in either half; pick one explicitly by construction instead of
    # searching, using the same indexing scheme as duplicate_memory/
    # split_memory (own bits then opp bits, oldest-to-newest, MSB first).
    m = 1  # original depth before duplication
    own_extra, opp_extra = 1, 0
    own_last_m, opp_last_m = 0, 1
    diverging_index = (
        (own_extra << m) | own_last_m
    ) << (m + 1) | ((opp_extra << m) | opp_last_m)
    mutated = point_mutate(depth2, diverging_index)

    recovered = split_memory(mutated)
    # The (0,0) quadrant was untouched by the mutation, so splitting
    # still reproduces the pre-duplication table exactly.
    assert recovered.table == tit_for_tat_table().table
    assert recovered.prehistory == tit_for_tat_table().prehistory


# --- Point mutation: flip one table entry, must stay local ---


@pytest.mark.parametrize(
    "make_table",
    [tit_for_tat_table, always_cooperate_table, always_defect_table, pavlov_table],
)
def test_point_mutate_flips_only_the_targeted_table_entry(make_table):
    original = make_table()
    for index in range(len(original.table)):
        mutated = point_mutate(original, index)

        assert mutated.memory_depth == original.memory_depth
        assert mutated.prehistory == original.prehistory
        for i, (old_bit, new_bit) in enumerate(zip(original.table, mutated.table)):
            if i == index:
                assert new_bit == 1 - old_bit
            else:
                assert new_bit == old_bit


def test_point_mutate_changes_decision_only_for_the_history_it_encodes():
    original = tit_for_tat_table()
    # index=1 -> own last move D (bit 0), opponent last move C (bit 1).
    mutated = point_mutate(original, index=1)

    # Exhaustively cover every own/opp history of length 0..4, as with the
    # duplication check: a table is a pure lookup, so "only that history
    # changes" must hold everywhere, not just on the histories a demo duel
    # happens to reach.
    changed_at_least_one = False
    for length in range(5):
        for own_bits in itertools.product("CD", repeat=length):
            for opp_bits in itertools.product("CD", repeat=length):
                own_history, opp_history = list(own_bits), list(opp_bits)
                targeted = original._table_index(own_history, opp_history) == 1
                mutated_move = mutated.decide(own_history, opp_history)
                original_move = original.decide(own_history, opp_history)
                if targeted:
                    assert mutated_move != original_move
                    changed_at_least_one = True
                else:
                    assert mutated_move == original_move
    assert changed_at_least_one


def test_point_mutated_tft_diverges_only_where_the_flipped_entry_applies():
    original = tit_for_tat_table()
    mutated_index = 1  # own last move D, opponent last move C
    mutated = point_mutate(original, mutated_index)

    pattern = ["C", "D", "C", "C", "D"]

    def scripted_opponent(own_history, opp_history):
        return pattern[len(own_history)]

    a = Agent("orig", original.as_strategy())
    b = Agent("mut", mutated.as_strategy())
    run_duel(a, Agent("opp", scripted_opponent), rounds=len(pattern))
    run_duel(b, Agent("opp", scripted_opponent), rounds=len(pattern))

    assert a.history_self != b.history_self  # the mutation does show up in play

    first_diff = next(
        i for i in range(len(pattern)) if a.history_self[i] != b.history_self[i]
    )
    assert a.history_self[:first_diff] == b.history_self[:first_diff]
    # confirm the divergence happens exactly where the flipped entry applies
    own_before = a.history_self[:first_diff]
    opp_before = pattern[:first_diff]
    assert original._table_index(own_before, opp_before) == mutated_index


# ------------------------------------------------------------------ effective_memory_depth

def test_effective_memory_depth_of_depth_zero_table_is_zero():
    t = LindgrenTable(0, [1], [])
    assert effective_memory_depth(t) == 0


def test_effective_memory_depth_of_tft_equals_its_nominal_depth():
    assert effective_memory_depth(tit_for_tat_table()) == 1


def test_effective_memory_depth_of_always_cooperate_is_zero():
    """always_cooperate_table has memory_depth=1 but never actually
    consults it -- every entry is 1 (cooperate)."""
    assert effective_memory_depth(always_cooperate_table()) == 0


def test_effective_memory_depth_of_always_defect_is_zero():
    assert effective_memory_depth(always_defect_table()) == 0


def test_duplicate_memory_never_increases_effective_depth():
    """duplicate_memory is neutral by construction, so folding must
    recover the original effective depth regardless of how many times
    the table was duplicated first."""
    tft = tit_for_tat_table()
    once = duplicate_memory(tft)
    twice = duplicate_memory(once)
    assert once.memory_depth == 2
    assert twice.memory_depth == 3
    assert effective_memory_depth(once) == 1
    assert effective_memory_depth(twice) == 1


def test_point_mutation_at_the_oldest_round_raises_effective_depth():
    """A point mutation that makes the decision depend on the oldest
    remembered round prevents folding back down -- effective depth must
    rise to (at least) the table's full nominal depth."""
    tft3 = duplicate_memory(duplicate_memory(tit_for_tat_table()))
    assert effective_memory_depth(tft3) == 1  # still foldable before mutation
    mutated = point_mutate(tft3, 0)
    assert effective_memory_depth(mutated) == 3


def test_effective_memory_depth_never_exceeds_nominal_depth():
    for table in (
        tit_for_tat_table(),
        always_cooperate_table(),
        always_defect_table(),
        pavlov_table(),
        duplicate_memory(tit_for_tat_table()),
    ):
        assert effective_memory_depth(table) <= table.memory_depth
