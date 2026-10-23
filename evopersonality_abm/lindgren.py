"""Lindgren (1991)-style memory-m lookup-table strategy genome.

A strategy of memory depth m is encoded as:
  - `table`: 2**(2*m) bits. Indexed by the joint history of the last m
    rounds (own action bits, then opponent action bits), it gives the
    next action.
  - `prehistory`: 2*m bits describing the assumed (own, opponent) history
    before round 1, used to look up the first m decisions before any
    real history exists.

This module fixes the lookup-table mechanics for a given, static genome,
and provides two mutation operators on it:
  - `duplicate_memory` grows memory depth m by one round. It is
    behaviorally neutral: the resulting depth-(m+1) table makes exactly
    the same decisions as the depth-m table it was built from, for any
    actual play.
  - `point_mutate` flips a single table entry at a fixed memory depth. It
    is *not* neutral: it changes the action for exactly the one history
    condition that entry encodes, leaving every other entry (and hence
    every other history) untouched.
  - `split_memory` shrinks memory depth m by one round: the reverse of
    `duplicate_memory`. It keeps only the sub-table for the oldest
    remembered round being (own=D, opp=D) and discards the rest, so
    `split_memory(duplicate_memory(t)) == t` exactly for any table t, but
    applied to a table whose quadrants have since diverged via point
    mutation it is lossy -- shrinking forgets whatever the discarded
    quadrants encoded.
"""

import itertools
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


def duplicate_memory(table: LindgrenTable) -> LindgrenTable:
    """Gene-duplication mutation: memory_depth -> memory_depth + 1.

    Neutral by construction. The new table is indexed by one extra
    (own, opponent) round tacked onto the *old* end of the history window
    (own_bits/opp_bits are oldest-to-newest, so the new bits land at the
    most-significant end of each half of the index, per _table_index).
    Every entry that shares the same trailing m own-bits/opp-bits as an
    old entry copies that old entry's value, regardless of the two new
    leading bits -- so the extra round of memory is carried but never
    consulted, and decisions are unchanged for any real play. The two new
    prehistory bits are consulted only through that same ignored position,
    so their value cannot affect behavior either; they are set to 0.
    """
    m = table.memory_depth
    new_m = m + 1
    mask_m = (1 << m) - 1

    new_table: List[Bit] = [0] * (2 ** (2 * new_m))
    for idx in range(len(new_table)):
        idx2 = idx & ((1 << (2 * m + 1)) - 1)  # drop the leading own-extra bit
        own_last_m = idx2 >> (m + 1)
        opp_last_m = idx2 & mask_m  # drop the opp-extra bit in between
        old_index = (own_last_m << m) | opp_last_m
        new_table[idx] = table.table[old_index]

    new_prehistory = (
        [0] + list(table.prehistory[:m]) + [0] + list(table.prehistory[m:])
    )
    return LindgrenTable(memory_depth=new_m, table=new_table, prehistory=new_prehistory)


def split_memory(table: LindgrenTable) -> LindgrenTable:
    """Gene-deletion mutation: memory_depth -> memory_depth - 1.

    The reverse of `duplicate_memory`. For each entry of the new,
    depth-(m-1) table, look up the old depth-m table assuming the
    now-forgotten oldest round was (own=D, opp=D) -- i.e. keep the
    table.table quadrant where both new leading bits are 0, and drop the
    corresponding leading prehistory bit from each channel.

    Right after a `duplicate_memory` call (before any point mutation),
    all four quadrants of the grown table are identical copies of the
    original, so this exactly undoes it. On a table that has since
    diverged, the discarded quadrants' information is simply lost --
    unlike duplication (neutral) and point mutation (local), shrinking
    memory is a lossy, forgetful operation in general.
    """
    m = table.memory_depth
    if m < 1:
        raise ValueError("memory_depth must be >= 1 to split")
    new_m = m - 1
    mask_new_m = (1 << new_m) - 1

    new_table: List[Bit] = [0] * (2 ** (2 * new_m))
    for new_index in range(len(new_table)):
        own_last = new_index >> new_m
        opp_last = new_index & mask_new_m
        old_index = (own_last << m) | opp_last
        new_table[new_index] = table.table[old_index]

    new_prehistory = list(table.prehistory[1:m]) + list(table.prehistory[m + 1 :])
    return LindgrenTable(memory_depth=new_m, table=new_table, prehistory=new_prehistory)


def _table_ignores_oldest_round(table: LindgrenTable) -> bool:
    """True if `table`'s decisions never depend on the single oldest
    remembered round (own move and opponent move alike) -- i.e. the table
    is losslessly foldable to memory_depth - 1 via `split_memory`, the
    exact reverse of what `duplicate_memory` adds. Ported from
    `claude/markov-production-analysis` (added there for
    `effective_memory_depth` below).

    Checked via the public `decide` API with an explicit length-m history
    for both sides (not by inspecting table.table's bit layout directly),
    so it needs no assumption about _table_index's internals and is
    unaffected by prehistory (decide() never consults prehistory once the
    given history already has >= memory_depth entries -- see
    `_table_index`'s `[-m:]` slice)."""
    m = table.memory_depth
    if m == 0:
        return False  # nothing left to fold
    new_m = m - 1
    for own_suffix in itertools.product((DEFECT, COOPERATE), repeat=new_m):
        for opp_suffix in itertools.product((DEFECT, COOPERATE), repeat=new_m):
            decisions = {
                table.decide([own_oldest] + list(own_suffix), [opp_oldest] + list(opp_suffix))
                for own_oldest in (DEFECT, COOPERATE)
                for opp_oldest in (DEFECT, COOPERATE)
            }
            if len(decisions) > 1:
                return False
    return True


def effective_memory_depth(table: LindgrenTable) -> int:
    """The minimal memory depth `table` can be losslessly folded to by
    repeatedly checking `_table_ignores_oldest_round` and folding with
    `split_memory` -- i.e. how much of the genome's nominal memory_depth
    is actually used, independent of how it got there (duplication,
    mutation, or both). Ported from `claude/markov-production-analysis`
    (used there by the depth-stratified recovery analysis; needed here
    for docs/PREREGISTRATION_k_model_markov.md's H-C depth strata)."""
    current = table
    while current.memory_depth > 0 and _table_ignores_oldest_round(current):
        current = split_memory(current)
    return current.memory_depth


def point_mutate(table: LindgrenTable, index: int) -> LindgrenTable:
    """Point mutation: flip the table entry for one history condition.

    Unlike `duplicate_memory`, this is not neutral -- the entry at
    `index` (a joint own/opp history, encoded as in `_table_index`) picks
    the opposite action, while every other entry is untouched.
    """
    new_table = list(table.table)
    new_table[index] ^= 1
    return LindgrenTable(
        memory_depth=table.memory_depth, table=new_table, prehistory=table.prehistory
    )


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
