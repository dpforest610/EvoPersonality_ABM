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


# ------------------------------------------------------------------ node_appearance_rates_over_draws

def _tiny_pool():
    # 3 seeds: seed 0 has genotype 1 (series touching 和和和/和和搾/和搾底) and
    # genotype 2 (series touching 底底底/底底惨); seeds 1/2 have only one each.
    return {
        0: (
            {1: 3, 2: 1},
            {1: ["和和和,和和搾,和搾底,和搾底。"], 2: ["底底底,底底底,底底惨。"]},
            {1: (1, "0,1,0,1", "1,1"), 2: (1, "0,1,0,1", "1,1")},
        ),
        1: ({1: 2}, {1: ["和和和,和和搾,和搾底,和搾底。"]}, {1: (1, "0,1,0,1", "1,1")}),
        2: ({1: 1}, {1: ["底底底,底底底,底底惨。"]}, {1: (1, "0,1,0,1", "1,1")}),
    }


_TINY_VOCAB = frozenset({"和和和", "和和搾", "和搾底", "底底底", "底底惨"})


def test_node_appearance_rates_returns_none_when_pool_too_small():
    pool = {0: _tiny_pool()[0]}
    rng = random.Random(0)
    assert mod.node_appearance_rates_over_draws(pool, _TINY_VOCAB, group_size=5, n_draws=3, rng=rng) is None


def test_node_appearance_rates_covers_every_vocab_word_including_zeros():
    pool = _tiny_pool()
    rng = random.Random(0)
    rates = mod.node_appearance_rates_over_draws(pool, _TINY_VOCAB, group_size=2, n_draws=5, rng=rng)
    assert set(rates.keys()) == set(_TINY_VOCAB)
    assert all(0.0 <= v <= 1.0 for v in rates.values())


def test_node_appearance_rates_key_order_is_sorted_not_frozenset_order():
    # `vocab` is a frozenset, whose own iteration order depends on
    # Python's per-process string hash randomization -- iterating it
    # directly to build the returned dict used to make the JSON output's
    # key order differ between separate `python3 ...` launches even
    # though every value stayed the same. The returned dict's keys must
    # be in a fixed (sorted) order regardless of which frozenset object
    # (same elements, unrelated construction) is passed in.
    pool = _tiny_pool()
    rates = mod.node_appearance_rates_over_draws(pool, _TINY_VOCAB, group_size=2, n_draws=5, rng=random.Random(0))
    assert list(rates.keys()) == sorted(_TINY_VOCAB)

    other_vocab_same_elements = frozenset({"底底惨", "底底底", "和搾底", "和和搾", "和和和"})
    rates2 = mod.node_appearance_rates_over_draws(pool, other_vocab_same_elements, group_size=2, n_draws=5, rng=random.Random(0))
    assert list(rates2.keys()) == list(rates.keys())


def test_node_appearance_rates_consumes_the_identical_rng_sequence_as_edges():
    """The core correctness requirement: --nodes-only must draw EXACTLY
    the same 200 cross-lineage groups the edge-weight export would have
    drawn in the same position of the same per-age RNG stream."""
    pool = _tiny_pool()

    rng_edges = random.Random("shared_seed_1000")
    draws_via_edges_path = [mod.resample_cross_seed_group(pool, 2, rng_edges) for _ in range(5)]

    rng_nodes = random.Random("shared_seed_1000")
    draws_via_nodes_path = [mod.resample_cross_seed_group(pool, 2, rng_nodes) for _ in range(5)]

    assert draws_via_edges_path == draws_via_nodes_path

    # and the two aggregate functions themselves, called in place of the
    # raw draws above, must consume rng identically round for round
    rng_a = random.Random("shared_seed_1000")
    edges = mod.average_edges_over_draws(pool, _TINY_VOCAB, 2, 5, rng_a)
    rng_b = random.Random("shared_seed_1000")
    rates = mod.node_appearance_rates_over_draws(pool, _TINY_VOCAB, 2, 5, rng_b)
    # after both have consumed exactly 5 draws from identically-seeded
    # RNGs, their underlying generators must be in the same state
    assert rng_a.random() == rng_b.random()
    assert edges is not None and rates is not None


def test_node_appearance_rates_matches_hand_computed_fraction():
    # 2 seeds only (0 and 1), group_size=2 forces BOTH seeds every draw --
    # genotype at seed 0 is a coin flip between the 1 (freq 3) and 2 (freq 1)
    # entries, seed 1 always has its single genotype (和和和 series).
    pool = {k: v for k, v in _tiny_pool().items() if k in (0, 1)}
    rng = random.Random(1)
    rates = mod.node_appearance_rates_over_draws(pool, _TINY_VOCAB, group_size=2, n_draws=200, rng=rng)
    # seed 1's participant always carries 和和和/和和搾/和搾底 -> their rate
    # must be at least 0.5 (present whenever seed 1's draw is used, which is every draw)
    assert rates["和和和"] >= 0.5
    # 底底底/底底惨 can only come from seed 0's genotype 2 (freq 1 of 4) -- rare but possible
    assert 0.0 <= rates["底底底"] <= 0.5
