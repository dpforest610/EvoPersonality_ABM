import itertools

import pytest

from evopersonality_abm.agent import Agent
from evopersonality_abm.duel import run_duel
from evopersonality_abm.lindgren import (
    LindgrenTable,
    always_cooperate_table,
    always_defect_table,
    duplicate_memory,
    pavlov_table,
    point_mutate,
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
