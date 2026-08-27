import pytest

from evopersonality_abm.agent import Agent
from evopersonality_abm.duel import run_duel
from evopersonality_abm.lindgren import (
    LindgrenTable,
    always_cooperate_table,
    always_defect_table,
    pavlov_table,
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
