import random

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


def test_run_duel_default_noise_rate_is_fully_deterministic():
    # noise_rate=0.0 (the default) must behave exactly as before this was added.
    a = Agent("A", always_cooperate)
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=5)
    assert a.history_self == ["C"] * 5
    assert b.history_self == ["D"] * 5


def test_run_duel_noise_rate_one_always_flips():
    a = Agent("A", always_cooperate)
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=5, noise_rate=1.0, rng=random.Random(0))
    assert a.history_self == ["D"] * 5
    assert b.history_self == ["C"] * 5


def test_run_duel_noise_rate_zero_point_five_flips_some_but_not_all():
    a = Agent("A", always_cooperate)
    b = Agent("B", always_defect)
    run_duel(a, b, rounds=200, noise_rate=0.5, rng=random.Random(1))
    # with noise_rate=0.5 over 200 rounds, both flips and non-flips should occur
    assert "D" in a.history_self and "C" in a.history_self
    assert "C" in b.history_self and "D" in b.history_self


def test_run_duel_noise_only_affects_recorded_history_not_future_strategy_input():
    # tit_for_tat reads opp_history, which must reflect the REALIZED
    # (possibly noise-flipped) action, not the strategy's original intent.
    # noise_rate=1.0 flips BOTH players' every move (not just one side),
    # so B (always_cooperate, intends C every round) is realized as D every
    # round, and A (TFT) -- seeing that flipped D -- intends D from round 2
    # onward, which noise then flips again to C in A's own realized history.
    a = Agent("A", tit_for_tat)
    b = Agent("B", always_cooperate)
    run_duel(a, b, rounds=3, noise_rate=1.0, rng=random.Random(0))
    assert b.history_self == ["D", "D", "D"]
    # A's round-1 move (intended C, opp_history empty) is itself flipped to D;
    # from round 2 on, TFT intends D (mirroring B's flipped-to-D history),
    # which noise flips back to C.
    assert a.history_self == ["D", "C", "C"]
