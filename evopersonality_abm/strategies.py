import random
from typing import Callable, List

from .game import COOPERATE, DEFECT

Action = str


def tit_for_tat(own_history, opp_history):
    if not opp_history:
        return COOPERATE
    return opp_history[-1]


def always_cooperate(own_history, opp_history):
    return COOPERATE


def always_defect(own_history, opp_history):
    return DEFECT


def random_strategy_factory(rng: random.Random, p_cooperate: float = 0.5) -> Callable[[List[Action], List[Action]], Action]:
    """A fresh independent-Bernoulli strategy per call, cooperating with
    probability p_cooperate each round regardless of history. Not a
    LindgrenTable (history-independent, so no fixed-memory-depth lookup
    table can represent it) -- same calling convention as the other
    functions in this module instead."""
    def strategy(own_history, opp_history):
        return COOPERATE if rng.random() < p_cooperate else DEFECT

    return strategy


def make_gradual_strategy(appeasement: bool) -> Callable[[List[Action], List[Action]], Action]:
    """Gradual (Beaufils, Delahaye & Mathieu 1996). Cooperates until the
    opponent defects; retaliates to the opponent's nth defection (counted
    cumulatively since the start of the match, INCLUDING defections seen
    while already retaliating) with n consecutive defections -- so a
    defection mid-punishment extends the current punishment, but by
    RECOUNTING the remaining punishment to the current cumulative
    defection count rather than adding to whatever punishment happens to
    be left (see docs/DEVIATION_gradual_rule.md: the human GRA condition's
    recorded moves, N=171, match a recount rule in 99.8% of moves / 169 of
    171 participants exactly, vs. 99.2% / 155 of 171 for the "add" rule
    this project originally shipped with -- the "add" rule was a deviation
    from the human study, not the human study's actual behavior). With
    appeasement=True, 2 unconditional cooperations ("calming") follow each
    completed punishment streak before normal play resumes -- per
    Beaufils et al.'s original, a defection seen during calming does NOT
    interrupt it (both calming cooperations are always played), but still
    counts toward the cumulative defection total, and punishment (using
    that recounted total) begins once calming ends. appeasement=False (the
    project's main-set "Gradual") omits the calming cooperations entirely,
    and degenerates to exactly `tit_for_tat`'s behavior when the opponent
    defects only once (n=1).

    Returns a fresh closure per call -- construct a new one per duel, the
    internal state is not meant to be shared/reset across duels."""
    state = {"punish_remaining": 0, "calm_remaining": 0, "defection_count": 0, "punish_pending": False}

    def strategy(own_history, opp_history):
        if opp_history and opp_history[-1] == DEFECT:
            state["defection_count"] += 1
            if state["calm_remaining"] > 0:
                # Beaufils et al.: a defection during calming counts toward
                # the cumulative total but does not interrupt calming;
                # punishment (recounted) starts once calming ends below.
                state["punish_pending"] = True
            else:
                state["punish_remaining"] = state["defection_count"]  # recount, don't add

        if state["punish_remaining"] > 0:
            state["punish_remaining"] -= 1
            if state["punish_remaining"] == 0 and appeasement:
                state["calm_remaining"] = 2
            return DEFECT
        if state["calm_remaining"] > 0:
            state["calm_remaining"] -= 1
            if state["calm_remaining"] == 0 and state["punish_pending"]:
                state["punish_remaining"] = state["defection_count"]
                state["punish_pending"] = False
            return COOPERATE
        return COOPERATE

    return strategy
