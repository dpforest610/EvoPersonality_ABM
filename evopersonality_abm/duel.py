import random
from typing import Optional

from .agent import Agent
from .game import COOPERATE, DEFECT, payoff
from .strategies import always_defect, tit_for_tat


def run_duel(
    agent_a: Agent,
    agent_b: Agent,
    rounds: int = 10,
    noise_rate: float = 0.0,
    rng: Optional[random.Random] = None,
):
    """noise_rate: implementation-error probability (independent per
    player per round) -- with probability noise_rate, the action actually
    played (and recorded into history) is the opposite of what the
    strategy decided, same as the "trembling hand" noise used in Lindgren
    (1991) and the broader noisy-IPD literature. 0.0 (the default) is the
    original fully deterministic behavior; `rng` is required whenever
    noise_rate > 0 (not defaulted to a fresh `random.Random()`, so callers
    control/seed the stream explicitly -- see
    docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 1.2)."""
    for _ in range(rounds):
        action_a = agent_a.decide()
        action_b = agent_b.decide()
        if noise_rate and rng.random() < noise_rate:
            action_a = DEFECT if action_a == COOPERATE else COOPERATE
        if noise_rate and rng.random() < noise_rate:
            action_b = DEFECT if action_b == COOPERATE else COOPERATE
        payoff_a, payoff_b = payoff(action_a, action_b)
        agent_a.record(action_a, action_b, payoff_a)
        agent_b.record(action_b, action_a, payoff_b)
    return agent_a, agent_b


def main():
    agent_a = Agent("A", tit_for_tat)
    agent_b = Agent("B", always_defect)
    run_duel(agent_a, agent_b, rounds=10)

    print(f"{'round':>5}  {agent_a.name:>3}  {agent_b.name:>3}")
    for i, (move_a, move_b) in enumerate(
        zip(agent_a.history_self, agent_b.history_self), start=1
    ):
        print(f"{i:>5}  {move_a:>3}  {move_b:>3}")

    for agent in (agent_a, agent_b):
        moves = "".join(agent.history_self)
        print(f"{agent.name}: score={agent.score}, moves={moves}")


if __name__ == "__main__":
    main()
