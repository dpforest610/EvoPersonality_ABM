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
    while already retaliating -- so a defection mid-punishment extends the
    current punishment in place, same as the original and the human-study
    spec, see docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 2.3.1)
    with n consecutive defections. With appeasement=True, 2 cooperations
    ("calming") follow each completed punishment streak before normal play
    resumes; appeasement=False (the project's main-set "Gradual") omits
    the calming cooperations, and degenerates to exactly `tit_for_tat`'s
    behavior when the opponent defects only once (n=1).

    Returns a fresh closure per call -- construct a new one per duel, the
    internal state is not meant to be shared/reset across duels."""
    state = {"punish_remaining": 0, "calm_remaining": 0, "defection_count": 0}

    def strategy(own_history, opp_history):
        if opp_history and opp_history[-1] == DEFECT:
            state["defection_count"] += 1
            state["punish_remaining"] += state["defection_count"]
            state["calm_remaining"] = 0  # a new defection interrupts any pending calm

        if state["punish_remaining"] > 0:
            state["punish_remaining"] -= 1
            if state["punish_remaining"] == 0 and appeasement:
                state["calm_remaining"] = 2
            return DEFECT
        if state["calm_remaining"] > 0:
            state["calm_remaining"] -= 1
            return COOPERATE
        return COOPERATE

    return strategy
