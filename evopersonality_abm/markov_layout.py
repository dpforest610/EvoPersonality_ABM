"""Fixed node layout for the Markov-network animation -- design doc
section 7.3, "option B" (confirmed 2026-10-16): a regular grid based on
the state symbols, needing no human data. The 1st and 2nd characters of
a 3-gram pick its "big cell" in a 4x4 grid; the 3rd character picks its
position along that cell's own diagonal, so 底底底 and 和和和 (whose
all-three-characters agree) land at the two opposite corners of the
whole layout.
"""

from typing import Tuple

# index 0 -> the corner nearest the origin, index 3 -> the corner nearest
# the opposite corner. 底底底 (all "底") must be diagonally opposite
# 和和和 (all "和"), so they anchor the two ends of this ordering.
SYMBOL_ORDER = ["底", "惨", "搾", "和"]


def node_position(word: str, cell_size: float = 1.0, margin: float = 0.1) -> Tuple[float, float]:
    """(x, y) in [0, 4*cell_size] x [0, 4*cell_size]. `word` is a 3-character
    3-gram (e.g. "底底惨"). margin keeps points off the gridlines between
    cells (0 < margin < 0.5)."""
    i1, i2, i3 = (SYMBOL_ORDER.index(c) for c in word)
    span = 1.0 - 2 * margin
    frac = i3 / (len(SYMBOL_ORDER) - 1)  # 0.0 .. 1.0, along the cell's own diagonal
    x = (i1 + margin + frac * span) * cell_size
    y = (i2 + margin + frac * span) * cell_size
    return x, y
