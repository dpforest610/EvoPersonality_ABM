import random

from evopersonality_abm.agent import Agent
from evopersonality_abm.duel import run_duel
from evopersonality_abm.strategies import (
    always_defect,
    make_gradual_strategy,
    random_strategy_factory,
    tit_for_tat,
)


# ------------------------------------------------------------------ random_strategy_factory

def test_random_strategy_factory_respects_p_cooperate_extremes():
    always_c = random_strategy_factory(random.Random(0), p_cooperate=1.0)
    always_d = random_strategy_factory(random.Random(0), p_cooperate=0.0)
    assert [always_c([], []) for _ in range(10)] == ["C"] * 10
    assert [always_d([], []) for _ in range(10)] == ["D"] * 10


def test_random_strategy_factory_ignores_history():
    strat = random_strategy_factory(random.Random(0), p_cooperate=0.5)
    # same rng state each call -> same output regardless of (ignored) history args
    out1 = strat(["C", "D"], ["D", "D"])
    strat2 = random_strategy_factory(random.Random(0), p_cooperate=0.5)
    out2 = strat2([], [])
    assert out1 == out2


def test_random_strategy_factory_produces_both_moves_over_many_rounds():
    strat = random_strategy_factory(random.Random(0), p_cooperate=0.5)
    moves = [strat([], []) for _ in range(200)]
    assert "C" in moves and "D" in moves


# ------------------------------------------------------------------ make_gradual_strategy

def test_gradual_n1_matches_tit_for_tat_without_appeasement():
    """A single isolated defection, no appeasement: Gradual's response
    must match tit_for_tat exactly (requested explicitly in the design)."""
    gradual = make_gradual_strategy(appeasement=False)
    tft_agent = Agent("TFT", tit_for_tat)
    gradual_agent = Agent("GRA", gradual)

    # same fixed opponent sequence for both: defects once (round 1), then cooperates
    opp_moves = ["D", "C", "C", "C", "C"]
    own_history_tft, own_history_gra = [], []
    opp_history = []
    for m in opp_moves:
        own_history_tft.append(tit_for_tat(own_history_tft, opp_history))
        own_history_gra.append(gradual(own_history_gra, opp_history))
        opp_history.append(m)

    assert own_history_gra == own_history_tft


def test_gradual_without_appeasement_single_defection_retaliates_once():
    gradual = make_gradual_strategy(appeasement=False)
    own, opp = [], []
    for opp_move in ["D", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    # round1: C (no history yet); round2: D (retaliate for round1's D);
    # round3,4: back to C (no more defections)
    assert own == ["C", "D", "C", "C"]


def test_gradual_mid_punishment_defection_extends_the_current_punishment():
    """Explicit test requested in docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md
    section 2.3.1: a second defection arriving WHILE Gradual is already
    retaliating must lengthen the punishment in place (defection count is
    cumulative over the whole match, counted even mid-punishment)."""
    gradual_a = make_gradual_strategy(appeasement=False)
    gradual_b = make_gradual_strategy(appeasement=False)

    # scenario (a): opponent defects once (round 1) then always cooperates
    own_a, opp_a = [], []
    for opp_move in ["D", "C", "C", "C", "C"]:
        own_a.append(gradual_a(own_a, opp_a))
        opp_a.append(opp_move)
    assert own_a == ["C", "D", "C", "C", "C"]
    assert own_a.count("D") == 1

    # scenario (b): opponent defects round 1 AND round 2 (the latter falls
    # during Gradual's own retaliation), then always cooperates
    own_b, opp_b = [], []
    for opp_move in ["D", "D", "C", "C", "C"]:
        own_b.append(gradual_b(own_b, opp_b))
        opp_b.append(opp_move)
    # round1: C; round2: D (retaliate defection #1); round3: D (opponent's
    # round-2 defection is defection #2, extends punishment by +2, so still
    # punishing); round4: D (punishment continues); round5: C (done)
    assert own_b == ["C", "D", "D", "D", "C"]
    assert own_b.count("D") == 3
    assert own_b.count("D") > own_a.count("D")


def test_gradual_with_appeasement_cooperates_twice_after_punishment_ends():
    gradual = make_gradual_strategy(appeasement=True)
    own, opp = [], []
    for opp_move in ["D", "C", "C", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    # round1: C; round2: D (retaliate); round3,4: C,C (appeasement); round5,6: C (normal)
    assert own == ["C", "D", "C", "C", "C", "C"]


def test_gradual_with_appeasement_new_defection_interrupts_calming():
    gradual = make_gradual_strategy(appeasement=True)
    own, opp = [], []
    # round1: D triggers retaliation at round2; round3 would be calm #1,
    # but opponent defects again at round3 -> interrupts calm, extends punishment
    for opp_move in ["D", "C", "D", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    assert own[1] == "D"  # retaliating round1's defection
    assert own[2] == "C"  # calm #1 (punishment finished after round2)
    # opponent's round-3 defection (their 2nd ever) is seen going into round4:
    # interrupts the remaining calm, adds defection_count=2 more punishment
    assert own[3] == "D"
    assert own[4] == "D"


def test_gradual_full_duel_via_run_duel_against_always_defect():
    # sanity: Gradual should escalate harshly against a persistent defector
    # (every one of AllD's defections counts, even mid-punishment)
    gradual = make_gradual_strategy(appeasement=False)
    a = Agent("GRA", gradual)
    b = Agent("ALD", always_defect)
    run_duel(a, b, rounds=10)
    # after round 1 (opponent's 1st defection), Gradual should retaliate
    # and, since AllD keeps defecting throughout, never return to cooperation
    assert a.history_self[0] == "C"
    assert all(m == "D" for m in a.history_self[1:])
