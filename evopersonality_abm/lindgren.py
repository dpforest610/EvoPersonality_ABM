"""Lindgren (1991)-style memory-m lookup-table strategy genome.

A strategy of memory depth m is encoded as:
  - `table`: 2**(2*m) bits. Indexed by the joint history of the last m
    rounds (own action bits, then opponent action bits), it gives the
    next action.
  - `prehistory`: 2*m bits describing the assumed (own, opponent) history
    before round 1, used to look up the first m decisions before any
    real history exists.

This module only fixes the lookup-table mechanics for a given, static
genome. Duplication/split mutation of the table (memory depth evolving)
is a later step.
"""

from typing import Callable, List, Sequence, Tuple

from .game import COOPERATE, DEFECT

Action = str
Bit = int

_ACTION_OF_BIT = {1: COOPERATE, 0: DEFECT}
_BIT_OF_ACTION = {COOPERATE: 1, DEFECT: 0}


class LindgrenTable:
    """A fixed-memory-depth lookup-table strategy genome."""

    def __init__(
        self,
        memory_depth: int,
        table: Sequence[Bit],
        prehistory: Sequence[Bit],
    ) -> None:
        if memory_depth < 0:
            raise ValueError("memory_depth must be >= 0")

        expected_table_len = 2 ** (2 * memory_depth)
        if len(table) != expected_table_len:
            raise ValueError(
                f"table must have 2**(2*{memory_depth})={expected_table_len} "
                f"entries, got {len(table)}"
            )
        expected_prehistory_len = 2 * memory_depth
        if len(prehistory) != expected_prehistory_len:
            raise ValueError(
                f"prehistory must have 2*{memory_depth}={expected_prehistory_len} "
                f"entries, got {len(prehistory)}"
            )
        if any(b not in (0, 1) for b in table):
            raise ValueError("table entries must be 0 or 1")
        if any(b not in (0, 1) for b in prehistory):
            raise ValueError("prehistory entries must be 0 or 1")

        self.memory_depth = memory_depth
        self.table: Tuple[Bit, ...] = tuple(table)
        self.prehistory: Tuple[Bit, ...] = tuple(prehistory)

    def _table_index(
        self, own_history: Sequence[Action], opp_history: Sequence[Action]
    ) -> int:
        m = self.memory_depth
        if m == 0:
            return 0

        pre_own, pre_opp = self.prehistory[:m], self.prehistory[m:]
        own_bits = (list(pre_own) + [_BIT_OF_ACTION[a] for a in own_history])[-m:]
        opp_bits = (list(pre_opp) + [_BIT_OF_ACTION[a] for a in opp_history])[-m:]

        index = 0
        for b in own_bits + opp_bits:
            index = (index << 1) | b
        return index

    def decide(
        self, own_history: Sequence[Action], opp_history: Sequence[Action]
    ) -> Action:
        return _ACTION_OF_BIT[self.table[self._table_index(own_history, opp_history)]]

    def as_strategy(self) -> Callable[[List[Action], List[Action]], Action]:
        return lambda own_history, opp_history: self.decide(own_history, opp_history)


def tit_for_tat_table() -> LindgrenTable:
    """Copy the opponent's last move; assume they cooperated before round 1."""
    return LindgrenTable(memory_depth=1, table=[0, 1, 0, 1], prehistory=[1, 1])


def always_cooperate_table() -> LindgrenTable:
    return LindgrenTable(memory_depth=1, table=[1, 1, 1, 1], prehistory=[1, 1])


def always_defect_table() -> LindgrenTable:
    return LindgrenTable(memory_depth=1, table=[0, 0, 0, 0], prehistory=[0, 0])


def pavlov_table() -> LindgrenTable:
    """Win-stay-lose-shift: cooperate iff both players matched last round."""
    return LindgrenTable(memory_depth=1, table=[1, 0, 0, 1], prehistory=[1, 1])
