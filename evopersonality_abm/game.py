COOPERATE = "C"
DEFECT = "D"

# (row player payoff, column player payoff)
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}


def payoff(action_a, action_b):
    return PAYOFFS[(action_a, action_b)]
