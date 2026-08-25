from evopersonality_abm.agent import Agent
from evopersonality_abm.duel import run_duel
from evopersonality_abm.strategies import always_cooperate, always_defect, tit_for_tat


def test_always_cooperate_vs_always_defect():
    a = Agent("A", always_cooperate)
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=5)

    assert a.history_self == ["C"] * 5
    assert b.history_self == ["D"] * 5
    assert a.score == 0
    assert b.score == 25


def test_tit_for_tat_mirrors_opponent_defect():
    a = Agent("A", tit_for_tat)
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=3)

    # TFT cooperates first, then mirrors the opponent's last move.
    assert a.history_self == ["C", "D", "D"]
