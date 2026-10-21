import csv
import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "export_markov_edge_weights_for_viewer", REPO_ROOT / "scripts" / "export_markov_edge_weights_for_viewer.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["export_markov_edge_weights_for_viewer"] = mod
spec.loader.exec_module(mod)


# ------------------------------------------------------------------ effective_depth_of_identity

def test_effective_depth_of_identity_matches_effective_memory_depth():
    # TFT-like table: memory_depth=1, genuinely uses it
    identity = (1, "0,1,0,1", "1,1")
    assert mod.effective_depth_of_identity(identity) == 1


def test_effective_depth_of_identity_handles_depth_zero_empty_prehistory():
    identity = (0, "1", "")
    assert mod.effective_depth_of_identity(identity) == 0


# ------------------------------------------------------------------ depth_ranges

def test_depth_ranges_are_contiguous_and_cover_0_to_max_depth():
    ranges = mod.depth_ranges(d1=1, d2=3, max_depth=6)
    assert list(ranges["depthLow"]) == [0, 1]
    assert list(ranges["depthMid"]) == [2, 3]
    assert list(ranges["depthHigh"]) == [4, 5, 6]


# ------------------------------------------------------------------ filter_pool_by_depth

def test_filter_pool_by_depth_keeps_only_matching_genotypes():
    pool = {
        0: ({1: 5, 2: 3}, {1: ["s1"], 2: ["s2"]}, {1: (1, "0,1,0,1", "1,1"), 2: (0, "1", "")}),
    }
    # genotype 1 has effective depth 1, genotype 2 has effective depth 0
    restricted = mod.filter_pool_by_depth(pool, range(1, 2))
    assert set(restricted[0][0]) == {1}  # only genotype 1 survives


def test_filter_pool_by_depth_drops_seeds_with_no_matches():
    pool = {0: ({1: 5}, {1: ["s1"]}, {1: (0, "1", "")})}  # effective depth 0
    restricted = mod.filter_pool_by_depth(pool, range(3, 5))
    assert restricted == {}


# ------------------------------------------------------------------ average_edges_over_draws

def test_average_edges_over_draws_returns_none_when_pool_too_small():
    pool = {0: ({1: 1}, {1: ["AAB,ABA。"]}, {1: (0, "", "")})}
    vocab = frozenset({"AAB", "ABA"})
    rng = random.Random(0)
    assert mod.average_edges_over_draws(pool, vocab, group_size=5, n_draws=3, rng=rng) is None


def test_average_edges_over_draws_averages_across_draws():
    # 2 seeds, each with ONE genotype whose series always gives the same edge
    pool = {
        0: ({1: 1}, {1: ["AAB,ABA。"]}, {1: (0, "", "")}),
        1: ({1: 1}, {1: ["AAB,ABA。"]}, {1: (0, "", "")}),
    }
    vocab = frozenset({"AAB", "ABA"})
    rng = random.Random(0)
    edges = mod.average_edges_over_draws(pool, vocab, group_size=2, n_draws=5, rng=rng)
    assert edges is not None
    # every draw uses both seeds -> same 2-person group each time -> jaccard(AAB,ABA)=1.0 every draw
    assert edges[("AAB", "ABA")] == 1.0


# ------------------------------------------------------------------ edges_to_rows

def test_edges_to_rows_sorts_and_rounds():
    edges = {("B", "A"): 0.123456, ("A", "B"): 0.5}
    rows = mod.edges_to_rows(edges)
    assert rows == [["A", "B", 0.5], ["B", "A", 0.1235]]


# ------------------------------------------------------------------ build_human_section

def test_build_human_section_reformats_csv_into_nested_schema(tmp_path):
    path = tmp_path / "human_weights.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "group", "n", "from", "to", "jaccard"])
        w.writerow(["TFT", "Secure", 56, "和和和", "和和搾", 0.5])
        w.writerow(["TFT", "Mix", 46, "和和和", "和和搾", 0.3])

    section = mod.build_human_section(path)
    assert section["TFT"]["Secure"] == [["和和和", "和和搾", 0.5]]
    assert section["TFT"]["Mix"] == [["和和和", "和和搾", 0.3]]
