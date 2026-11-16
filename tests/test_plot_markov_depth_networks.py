import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "plot_markov_depth_networks", REPO_ROOT / "scripts" / "plot_markov_depth_networks.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_markov_depth_networks"] = mod
spec.loader.exec_module(mod)


def make_row(seed, age, gid=0, freq=1, effective_depth=0):
    return {"seed": seed, "age": age, "gid": gid, "freq": freq, "nominal_depth": effective_depth, "effective_depth": effective_depth}


# ------------------------------------------------------------------ select_depth_group

def test_select_depth_group_returns_all_when_fewer_than_max_size():
    pool = [make_row(0, 1000, effective_depth=3), make_row(1, 1000, effective_depth=3)]
    import random
    out = mod.select_depth_group(pool, max_size=10, depth=3, rng=random.Random(0))
    assert len(out) == 2


def test_select_depth_group_caps_at_max_size_when_plenty_available():
    pool = [make_row(seed, 1000, effective_depth=3) for seed in range(20)]
    import random
    out = mod.select_depth_group(pool, max_size=5, depth=3, rng=random.Random(0))
    assert len(out) == 5


def test_select_depth_group_only_considers_the_requested_depth():
    pool = [make_row(0, 1000, effective_depth=1), make_row(1, 1000, effective_depth=2)]
    import random
    out = mod.select_depth_group(pool, max_size=10, depth=1, rng=random.Random(0))
    assert len(out) == 1
    assert out[0]["effective_depth"] == 1


# ------------------------------------------------------------------ edges_with_self_loops

def test_edges_with_self_loops_merges_regular_and_self_loop_edges():
    vocab = frozenset({"AAB", "ABA", "和和和"})
    series = ["AAB,ABA,和和和。", "AAB,和和和。"]
    edges = mod.edges_with_self_loops(series, vocab)
    assert ("AAB", "ABA") in edges
    assert ("和和和", "和和和") in edges
    assert edges[("和和和", "和和和")] == 1.0


def test_edges_with_self_loops_applies_jaccard_threshold():
    # weak edge (jaccard < 0.30) must be filtered out
    vocab = frozenset({"AAB", "ABA"})
    series = ["AAB,ABA。", "AAB。", "AAB。", "AAB。"]  # jaccard(AAB,ABA) = 1/4 = 0.25
    edges = mod.edges_with_self_loops(series, vocab)
    assert ("AAB", "ABA") not in edges


# ------------------------------------------------------------------ circular_layout

def test_circular_layout_places_every_node_on_the_unit_circle():
    import numpy as np
    pos = mod.circular_layout(["A", "B", "C"])
    assert set(pos) == {"A", "B", "C"}
    for x, y in pos.values():
        assert abs(np.hypot(x, y) - 1.0) < 1e-9


def test_circular_layout_is_deterministic_regardless_of_input_order():
    pos1 = mod.circular_layout(["B", "A", "C"])
    pos2 = mod.circular_layout(["A", "C", "B"])
    assert pos1 == pos2
