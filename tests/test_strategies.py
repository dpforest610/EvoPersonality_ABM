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
    """A second defection arriving WHILE Gradual is already retaliating
    must lengthen the punishment (defection count is cumulative over the
    whole match, counted even mid-punishment) -- see docs/DEVIATION_gradual_rule.md."""
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
    # round-2 defection is defection #2 -- the recount rule sets the
    # remaining punishment to defection_count=2, not 1[leftover]+2=3, but
    # the leftover was already 0 by this point so recount and add happen
    # to coincide here); round4: D (punishment continues); round5: C (done)
    assert own_b == ["C", "D", "D", "D", "C"]
    assert own_b.count("D") == 3
    assert own_b.count("D") > own_a.count("D")


def test_gradual_recount_rule_sets_remaining_to_defection_count_not_a_sum():
    """docs/DEVIATION_gradual_rule.md: the human GRA condition (N=171)
    matches a rule where a new defection RESETS the remaining punishment to
    the current cumulative defection count, rather than adding to whatever
    punishment happens to be left -- a direct regression test of that
    recount formula (confirmed by human data to fit far better: 99.8%
    move match / 169 of 171 participants, vs. 99.2% / 155 of 171 for the
    "add" rule this project originally, incorrectly, shipped with)."""
    gradual = make_gradual_strategy(appeasement=False)
    own, opp = [], []
    for opp_move in ["D", "D", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    # after the 2nd defection (round 2), the recounted remaining punishment
    # is exactly defection_count=2 (rounds 3-4: two D's), not a longer run
    assert own == ["C", "D", "D", "D", "C"]
    assert own[2:4] == ["D", "D"]
    assert own[4] == "C"  # punishment is over after exactly 2 more D's


def test_gradual_recount_rule_prevents_overcounting_from_a_stale_leftover():
    """A chain of 3 defections in a row is where "add" and "recount"
    genuinely diverge (the 2nd defection's punishment hasn't fully
    decremented to 0 yet when the 3rd defection's effect is applied), and
    is the scenario that made the human-data mismatch visible. Under the
    old (add) rule this project originally shipped with, the punishment
    after the 3rd defection would have been set to 1[leftover]+3=4 rounds,
    for a total of 6 D's across this sequence with no cooperation yet by
    round 7. Under the fixed (recount) rule, it is set to exactly
    defection_count=3, for a total of 5 D's, with cooperation resuming at
    round 7."""
    gradual = make_gradual_strategy(appeasement=False)
    own, opp = [], []
    for opp_move in ["D", "D", "D", "C", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    assert own == ["C", "D", "D", "D", "D", "D", "C"]
    assert own.count("D") == 5


def test_gradual_with_appeasement_cooperates_twice_after_punishment_ends():
    gradual = make_gradual_strategy(appeasement=True)
    own, opp = [], []
    for opp_move in ["D", "C", "C", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    # round1: C; round2: D (retaliate); round3,4: C,C (appeasement); round5,6: C (normal)
    assert own == ["C", "D", "C", "C", "C", "C"]


def test_gradual_with_appeasement_defection_during_calming_does_not_interrupt_it():
    """docs/DEVIATION_gradual_rule.md: per Beaufils, Delahaye & Mathieu
    (1996)'s original, a defection seen during the 2 calming cooperations
    does NOT interrupt calming (both calming rounds are always played in
    full) -- it is merely counted toward the cumulative defection total,
    with punishment (using that recounted total) starting only once
    calming has finished. This project's original implementation
    incorrectly let a mid-calm defection cut calming short and retaliate
    immediately; that was its own bug, independent of the add-vs-recount
    deviation, fixed at the same time."""
    gradual = make_gradual_strategy(appeasement=True)
    own, opp = [], []
    # round1: D triggers retaliation at round2; round3 is calm #1; the
    # opponent defects again during round3 (their 2nd defection) -- round4
    # must still be calm #2 (uninterrupted), with punishment deferred to
    # round5 onward, recounted to defection_count=2
    for opp_move in ["D", "C", "D", "C", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    assert own[1] == "D"  # retaliating round1's defection
    assert own[2] == "C"  # calm #1
    assert own[3] == "C"  # calm #2 -- NOT interrupted by the opponent's round-3 defection
    assert own[4] == "D"  # punishment (recounted to 2) begins only after calming ends
    assert own[5] == "D"
    assert own[6] == "C"


def test_gradual_appeasement_always_cooperates_twice_even_if_opponent_defects_both_times():
    """Explicit test requested alongside docs/DEVIATION_gradual_rule.md:
    the 2 calming cooperations are unconditional -- they happen even if the
    opponent defects on both of those rounds."""
    gradual = make_gradual_strategy(appeasement=True)
    own, opp = [], []
    # opponent defects on round 1 (triggers punishment) AND on both calming
    # rounds (rounds 3 and 4) -- calming must still produce two C's in a row
    for opp_move in ["D", "C", "D", "D", "C", "C", "C"]:
        own.append(gradual(own, opp))
        opp.append(opp_move)
    assert own[1] == "D"  # retaliating round1's defection
    assert own[2] == "C"  # calm #1, despite the opponent's round-3 defection landing right after
    assert own[3] == "C"  # calm #2, despite the opponent's round-4 defection landing right after


def test_gradual_and_gradual_appeasement_can_diverge_for_the_same_opponent_sequence():
    """Gradual (no appeasement) and GradualAppeasement must actually behave
    differently for at least some opponent move sequence -- the opposite of
    the degenerate pre-fix bug reported in
    docs/MARKOV_STRUCTURE_EVOLUTION_RESULTS.md section 4, where every
    sequence produced identical output because calming and normal play
    both returned COOPERATE."""
    opp_moves = ["D", "C", "D", "C", "C", "C", "C"]

    no_appeasement = make_gradual_strategy(appeasement=False)
    own_no_app, opp_a = [], []
    for m in opp_moves:
        own_no_app.append(no_appeasement(own_no_app, opp_a))
        opp_a.append(m)

    with_appeasement = make_gradual_strategy(appeasement=True)
    own_with_app, opp_b = [], []
    for m in opp_moves:
        own_with_app.append(with_appeasement(own_with_app, opp_b))
        opp_b.append(m)

    assert own_no_app != own_with_app
    assert own_no_app == ["C", "D", "C", "D", "D", "C", "C"]
    assert own_with_app == ["C", "D", "C", "C", "D", "D", "C"]


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
