from .game import COOPERATE, DEFECT


def tit_for_tat(own_history, opp_history):
    if not opp_history:
        return COOPERATE
    return opp_history[-1]


def always_cooperate(own_history, opp_history):
    return COOPERATE


def always_defect(own_history, opp_history):
    return DEFECT
