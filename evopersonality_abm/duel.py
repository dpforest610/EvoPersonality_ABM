from .agent import Agent
from .game import payoff
from .strategies import always_defect, tit_for_tat


def run_duel(agent_a: Agent, agent_b: Agent, rounds: int = 10):
    for _ in range(rounds):
        action_a = agent_a.decide()
        action_b = agent_b.decide()
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
