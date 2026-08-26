from dataclasses import dataclass, field
from typing import Callable, List

Action = str


@dataclass
class Agent:
    name: str
    strategy: Callable[[List[Action], List[Action]], Action]
    score: float = 0.0
    history_self: List[Action] = field(default_factory=list)
    history_opponent: List[Action] = field(default_factory=list)

    def decide(self) -> Action:
        return self.strategy(self.history_self, self.history_opponent)

    def record(self, own_action: Action, opp_action: Action, payoff: float) -> None:
        self.history_self.append(own_action)
        self.history_opponent.append(opp_action)
        self.score += payoff
