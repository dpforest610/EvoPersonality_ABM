import itertools

from evopersonality_abm.markov_layout import node_position

FULL_VOCAB = ["".join(p) for p in itertools.product("和惨搾底", repeat=3)]


def test_bottom3_and_top3_are_at_diagonally_opposite_corners():
    x_bottom, y_bottom = node_position("底底底")
    x_top, y_top = node_position("和和和")
    assert x_bottom < 1.0 and y_bottom < 1.0  # near the (0,0) corner
    assert x_top > 3.0 and y_top > 3.0  # near the opposite (4,4) corner


def test_all_64_words_get_distinct_positions():
    positions = {node_position(w) for w in FULL_VOCAB}
    assert len(positions) == len(FULL_VOCAB)


def test_positions_stay_within_the_4x4_grid_with_cell_size_one():
    for w in FULL_VOCAB:
        x, y = node_position(w, cell_size=1.0)
        assert 0.0 <= x <= 4.0
        assert 0.0 <= y <= 4.0


def test_first_two_characters_determine_the_big_cell():
    # same (1st, 2nd) -> same integer grid cell (floor of x, y), regardless of 3rd char
    for prefix in ["底底", "和搾", "惨搾"]:
        cells = {
            (int(node_position(prefix + c)[0]), int(node_position(prefix + c)[1]))
            for c in "和惨搾底"
        }
        assert len(cells) == 1


def test_cell_size_scales_positions_linearly():
    x1, y1 = node_position("底惨搾", cell_size=1.0)
    x2, y2 = node_position("底惨搾", cell_size=2.0)
    assert abs(x2 - 2 * x1) < 1e-9
    assert abs(y2 - 2 * y1) < 1e-9


def test_tft_vocab_subset_uses_the_same_64_cell_grid():
    from evopersonality_abm.markov_recovery import load_vocab
    tft_words = load_vocab("TFT")
    assert tft_words <= set(FULL_VOCAB)
    for w in tft_words:
        x, y = node_position(w)
        assert 0.0 <= x <= 4.0 and 0.0 <= y <= 4.0
