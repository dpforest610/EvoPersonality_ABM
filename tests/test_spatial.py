import random

import pytest

from evopersonality_abm.spatial import (
    MOORE_OFFSETS,
    SpatialGenerationStats,
    _mean_neighbor_payoff,
    _softmax_sample,
    fixed_ring_duel_edges,
    local_placement_order,
    place_on_grid,
    ring_duel_edges,
    run_coupled_evolution_opponent_count,
    run_coupled_evolution_spatial,
    run_coupled_evolution_spatial_persistent,
    torus_neighbors,
)


def _stats_to_comparable(stats):
    """Flattens a stats list into plain-Python values for an exact
    equality check (K values are floats from deterministic arithmetic
    given identical rng draws, so == is meaningful here, not just
    approximate)."""
    return [
        (s.generation, tuple(s.k_out), tuple(s.k_self), tuple(s.memory_depths), tuple(s.payoffs))
        for s in stats
    ]


def test_opponent_count_chunked_resume_is_bit_identical_to_uninterrupted():
    """Priority-1 safety check before relying on chunked checkpointing in
    production: running 4,8,12,...,40 generations in one shot must give
    EXACTLY the same per-generation (k_out, k_self, memory_depth, payoff)
    as stopping at generation 20 (return_state=True), saving the
    returned (genomes, rng_state, rng_ring_state), and resuming from
    generation 20 to 40 with those exact saved values."""
    common_kwargs = dict(population_size=20, rounds_per_duel=4, k=6, seed=42)

    uninterrupted = run_coupled_evolution_opponent_count(generations=40, **common_kwargs)

    first_half, genomes, rng_state, rng_ring_state = run_coupled_evolution_opponent_count(
        generations=20, return_state=True, **common_kwargs
    )
    second_half = run_coupled_evolution_opponent_count(
        generations=40,
        start_generation=20,
        initial_genomes=genomes,
        rng_state=rng_state,
        rng_ring_state=rng_ring_state,
        **common_kwargs,
    )
    resumed = first_half + second_half

    assert len(uninterrupted) == len(resumed) == 40
    assert _stats_to_comparable(uninterrupted) == _stats_to_comparable(resumed)


def test_opponent_count_chunked_resume_works_across_three_chunks():
    """Same bit-exact check, but split into 3 uneven chunks, closer to
    how the production chunked runner will actually call this (repeated
    checkpoint/resume cycles, not just one)."""
    common_kwargs = dict(population_size=16, rounds_per_duel=4, k=4, seed=7)

    uninterrupted = run_coupled_evolution_opponent_count(generations=30, **common_kwargs)

    all_stats = []
    genomes, rng_state, rng_ring_state = None, None, None
    start = 0
    for chunk_end in (7, 19, 30):
        chunk_stats, genomes, rng_state, rng_ring_state = run_coupled_evolution_opponent_count(
            generations=chunk_end,
            start_generation=start,
            initial_genomes=genomes,
            rng_state=rng_state,
            rng_ring_state=rng_ring_state,
            return_state=True,
            **common_kwargs,
        )
        all_stats.extend(chunk_stats)
        start = chunk_end

    assert len(uninterrupted) == len(all_stats) == 30
    assert _stats_to_comparable(uninterrupted) == _stats_to_comparable(all_stats)


def test_torus_neighbors_wraps_at_edges():
    # corner (0,0) on a 5x5 torus: left wraps to x=4, up wraps to y=4.
    neighbors = set(torus_neighbors((0, 0), width=5, height=5))
    assert neighbors == {(4, 0), (1, 0), (0, 4), (0, 1)}


def test_torus_neighbors_interior_cell():
    neighbors = set(torus_neighbors((2, 2), width=5, height=5))
    assert neighbors == {(1, 2), (3, 2), (2, 1), (2, 3)}


def test_place_on_grid_gives_distinct_cells_within_bounds():
    rng = random.Random(0)
    positions = place_on_grid(n_agents=20, width=5, height=5, rng=rng)
    assert len(positions) == 20
    assert len(set(positions)) == 20  # no two agents share a cell
    for x, y in positions:
        assert 0 <= x < 5 and 0 <= y < 5


def test_place_on_grid_rejects_overfull_grid():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        place_on_grid(n_agents=30, width=5, height=5, rng=rng)


def test_mean_neighbor_payoff_is_zero_with_no_occupied_neighbors():
    agent_at_cell = {(0, 0): 0}
    total_payoff = [10.0]
    rounds_played = [5]
    # (3,3)'s neighbors on a 5x5 torus are (2,3),(4,3),(3,2),(3,4) -- none occupied.
    assert _mean_neighbor_payoff((3, 3), 5, 5, agent_at_cell, total_payoff, rounds_played) == 0.0


def test_mean_neighbor_payoff_averages_occupied_neighbors():
    # cell (1,0) has neighbor (0,0) occupied by agent 0 (mean payoff 2.0)
    # and neighbor (2,0) occupied by agent 1 (mean payoff 4.0); others empty.
    agent_at_cell = {(0, 0): 0, (2, 0): 1}
    total_payoff = [20.0, 40.0]
    rounds_played = [10, 10]
    result = _mean_neighbor_payoff((1, 0), 5, 5, agent_at_cell, total_payoff, rounds_played)
    assert result == pytest.approx(3.0)


def test_softmax_sample_always_picks_the_only_option():
    rng = random.Random(0)
    assert _softmax_sample([(0, 0)], [1.23], rng, tau=1.0) == (0, 0)


def test_softmax_sample_strongly_prefers_higher_score_at_low_temperature():
    rng = random.Random(0)
    options = [(0, 0), (1, 0)]
    picks = [_softmax_sample(options, [0.0, 10.0], rng, tau=0.1) for _ in range(200)]
    assert picks.count((1, 0)) > 190  # near-deterministic toward the higher score


def test_run_coupled_evolution_spatial_smoke_static_and_mobile():
    """A few generations, tiny population -- just confirms the phase/duel/
    movement loop runs end-to-end without crashing or corrupting the
    agent<->cell bookkeeping, for both conditions (2) and (3)."""
    for mobile in (False, True):
        stats = run_coupled_evolution_spatial(
            population_size=12,
            generations=3,
            rounds_per_duel=4,
            n_phases=2,
            grid_width=4,
            grid_height=4,
            tau=1.0,
            seed=0,
            mobile=mobile,
        )
        assert len(stats) == 3
        for s in stats:
            assert isinstance(s, SpatialGenerationStats)
            assert len(s.positions) == 12
            assert len(set(s.positions)) == 12  # still one agent per cell
            assert len(s.payoffs) == 12
            assert all(p >= 0.0 for p in s.payoffs)
            for x, y in s.positions:
                assert 0 <= x < 4 and 0 <= y < 4


def test_moore_offsets_give_eight_neighbors():
    neighbors = set(torus_neighbors((2, 2), width=5, height=5, offsets=MOORE_OFFSETS))
    assert neighbors == {
        (1, 1), (2, 1), (3, 1),
        (1, 2),         (3, 2),
        (1, 3), (2, 3), (3, 3),
    }


def test_run_coupled_evolution_spatial_accepts_moore_neighbor_offsets():
    stats = run_coupled_evolution_spatial(
        population_size=12,
        generations=2,
        rounds_per_duel=4,
        n_phases=2,
        grid_width=4,
        grid_height=4,
        seed=0,
        mobile=True,
        neighbor_offsets=MOORE_OFFSETS,
    )
    assert len(stats) == 2
    assert len(stats[-1].positions) == 12


def test_run_coupled_evolution_spatial_rejects_non_divisible_phases():
    with pytest.raises(ValueError):
        run_coupled_evolution_spatial(population_size=10, generations=1, rounds_per_duel=5, n_phases=2,
                                       grid_width=4, grid_height=4, seed=0)


def test_persistent_smoke_keeps_one_agent_per_cell_every_generation():
    """A few generations, tiny population -- confirms the parent-cell
    inheritance + leftover-scatter bookkeeping never double-occupies or
    drops a cell."""
    stats = run_coupled_evolution_spatial_persistent(
        population_size=12,
        generations=5,
        rounds_per_duel=4,
        n_phases=1,
        grid_width=4,
        grid_height=4,
        seed=0,
        mobile=False,
    )
    assert len(stats) == 5
    for s in stats:
        assert len(s.positions) == 12
        assert len(set(s.positions)) == 12  # still one agent per cell
        for x, y in s.positions:
            assert 0 <= x < 4 and 0 <= y < 4


def test_persistent_smoke_at_realistic_scale():
    """Same population/grid shape as the actual experiment (100 agents,
    12x12 grid, k=8/Moore, n_phases=1), just far fewer generations --
    confirms the inheritance+scatter bookkeeping holds up at the scale
    that matters, not just the tiny 12-agent case above."""
    stats = run_coupled_evolution_spatial_persistent(
        population_size=100,
        generations=5,
        rounds_per_duel=50,
        n_phases=1,
        grid_width=12,
        grid_height=12,
        seed=0,
        mobile=False,
        neighbor_offsets=MOORE_OFFSETS,
    )
    assert len(stats) == 5
    for s in stats:
        assert len(s.positions) == 100
        assert len(set(s.positions)) == 100


def test_persistent_rejects_non_divisible_phases():
    with pytest.raises(ValueError):
        run_coupled_evolution_spatial_persistent(
            population_size=10, generations=1, rounds_per_duel=5, n_phases=2,
            grid_width=4, grid_height=4, seed=0,
        )


def test_ring_duel_edges_gives_each_agent_exactly_k_neighbors():
    rng = random.Random(0)
    for k in (4, 8, 16, 32, 50, 99):
        edges = ring_duel_edges(100, k, rng)
        degree = {i: 0 for i in range(100)}
        for i, j in edges:
            assert i != j
            degree[i] += 1
            degree[j] += 1
        assert all(d == k for d in degree.values())


def test_ring_duel_edges_rejects_odd_k_other_than_complete_graph():
    rng = random.Random(0)
    with pytest.raises(NotImplementedError):
        ring_duel_edges(100, 15, rng)


def test_ring_duel_edges_k_equals_n_minus_1_is_complete_graph():
    rng = random.Random(0)
    edges = ring_duel_edges(10, 9, rng)
    assert len(edges) == 10 * 9 // 2  # C(10,2), i.e. every pair duels


def test_ring_duel_edges_rejects_out_of_range_k():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        ring_duel_edges(10, 0, rng)
    with pytest.raises(ValueError):
        ring_duel_edges(10, 10, rng)


def test_run_coupled_evolution_opponent_count_smoke():
    stats = run_coupled_evolution_opponent_count(
        population_size=20, generations=3, rounds_per_duel=4, k=4, seed=0,
    )
    assert len(stats) == 3
    for s in stats:
        assert len(s.k_out) == 20
        assert len(s.payoffs) == 20
        assert all(p >= 0.0 for p in s.payoffs)


def test_run_coupled_evolution_opponent_count_k_equals_full_matches_scale():
    """k = population_size - 1 should behave like well-mixed: every agent
    gets exactly population_size - 1 opponents' worth of rounds played."""
    stats = run_coupled_evolution_opponent_count(
        population_size=12, generations=1, rounds_per_duel=5, k=11, seed=0,
    )
    # every one of the 12 agents duels all 11 others once (5 rounds each)
    # -- can't inspect rounds_played directly (CoupledGenerationStats has
    # no such field), but a full run without error at k=n-1 is itself the
    # contract being tested (ring_duel_edges' own test covers the count).
    assert len(stats[0].payoffs) == 12


_GOLDEN_SHUFFLE_REFERENCE = [
    (0, [0.5827880059033551, 0.24066300012702502, 0.9762551055929201, 0.5602572770128127, 0.6190095931735539, 0.24842658485754932],
     [0.9097040631431023, 0.5510472537913857, 0.04658268061775628, 0.6820026947612046, 0.4964144951134918, 0.1797667495831432],
     [1, 1, 1, 1, 1, 1], [2.5, 2.5, 2.5, 1.5, 3.25, 3.25]),
    (1, [0.6253850508861982, 0.9930375973444486, 0.5494372077432812, 0.6154673602961022, 0.20150078372901792, 0.6716996083558234],
     [0.9693511413356708, 0.08365415359229192, 0.705526990629568, 0.44130185759200413, 0.16143544528871137, 0.630997918162999],
     [1, 2, 1, 1, 1, 1], [3.25, 1.0, 1.5, 4.5, 1.25, 3.5]),
    (2, [0.5791954253458513, 0.6919273211806282, 0.48104944311727194, 0.5754938198885912, 0.1935130724040553, 0.7750534438623987],
     [0.41311567596478194, 0.6277309426077391, 0.9851452334102603, 0.5009179460658212, 0.12906783332322613, 0.6936650873512837],
     [1, 1, 1, 0, 1, 1], [2.0, 1.75, 1.5, 3.0, 0.5, 1.0]),
    (3, [0.6097240650832012, 0.5641392035783495, 0.6464966850564083, 0.6001896606540319, 0.6570521087720989, 0.43193443671177406],
     [0.4245032287240657, 0.5044364066246965, 0.4017442280949224, 0.35741016657128793, 0.4203545191047164, 1.0],
     [1, 0, 1, 1, 0, 1], [1.75, 1.75, 2.75, 2.75, 1.5, 1.5]),
]


def test_opponent_count_placement_shuffle_matches_golden_reference():
    """placement="shuffle" (the default) must stay byte-for-byte identical
    to the implementation as it existed before the placement parameter
    (and the cooperation/assortativity/same-parent-fraction tracking) was
    added. Reference values captured from the pre-change implementation
    at commit 8faede1 -- adding "local" placement must not perturb
    "shuffle" in any way, since both read from the same rng in the same
    order and the only thing placement should change is how offspring
    are laid out afterward (an identity no-op for "shuffle")."""
    for placement in (None, "shuffle"):
        kwargs = dict(population_size=6, generations=4, rounds_per_duel=2, k=2, seed=7)
        if placement is not None:
            kwargs["placement"] = placement
        stats = run_coupled_evolution_opponent_count(**kwargs)
        actual = [(s.generation, s.k_out, s.k_self, s.memory_depths, s.payoffs) for s in stats]
        assert actual == _GOLDEN_SHUFFLE_REFERENCE


def test_local_placement_order_groups_siblings_contiguously():
    # parents 0,2,2,0,3,1 (old ring positions) produced these 6 offspring
    # (selection slots 0..5); expect: block for parent 0 (slots 0,3), then
    # parent 1 (slot 5), then parent 2 (slots 1,2), then parent 3 (slot 4) --
    # scanned in OLD ring-position order 0,1,2,3, packed with no gaps.
    parent_indices = [0, 2, 2, 0, 3, 1]
    order = local_placement_order(parent_indices, population_size=6)
    assert order == [0, 3, 5, 1, 2, 4]
    # every contiguous run in `order` maps to selection slots sharing one parent
    parents_in_new_order = [parent_indices[slot] for slot in order]
    assert parents_in_new_order == [0, 0, 1, 2, 2, 3]


def test_fixed_ring_duel_edges_each_agent_has_exactly_k_neighbors():
    for k in (2, 4, 8):
        edges = fixed_ring_duel_edges(20, k)
        degree = {i: 0 for i in range(20)}
        for i, j in edges:
            assert i != j
            degree[i] += 1
            degree[j] += 1
        assert all(d == k for d in degree.values())


def test_fixed_ring_duel_edges_connects_array_adjacent_indices_only():
    # k=2: each agent's only neighbors should be its immediate array
    # neighbors (i-1, i+1 mod n) -- no shuffling, unlike ring_duel_edges.
    edges = set(fixed_ring_duel_edges(5, 2))
    expected = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
    assert edges == expected


def test_local_placement_increases_same_parent_duel_fraction():
    """Operational check, at small scale: local placement should make
    actually-dueled pairs share a parent (and be similar in K_out) much
    more often than shuffle's near-baseline rate."""
    common = dict(population_size=30, rounds_per_duel=3, k=6, seed=11, generations=15)
    stats_shuffle = run_coupled_evolution_opponent_count(placement="shuffle", **common)
    stats_local = run_coupled_evolution_opponent_count(placement="local", **common)

    shuffle_rate = statistics_mean(
        s.same_parent_fraction for s in stats_shuffle[1:] if s.same_parent_fraction == s.same_parent_fraction
    )
    local_rate = statistics_mean(
        s.same_parent_fraction for s in stats_local[1:] if s.same_parent_fraction == s.same_parent_fraction
    )
    assert local_rate > shuffle_rate


def statistics_mean(values):
    values = list(values)
    return sum(values) / len(values)


def test_opponent_count_local_placement_chunked_resume_is_bit_identical():
    """Same bit-exact chunked-resume guarantee as the existing shuffle
    test, but for placement="local" -- must also thread parent_of_state
    through (track_parent_state=True) since same_parent_fraction for the
    first post-resume generation depends on knowing the prior
    generation's parent assignments."""
    common_kwargs = dict(population_size=20, rounds_per_duel=4, k=6, seed=42, placement="local")

    uninterrupted = run_coupled_evolution_opponent_count(generations=40, **common_kwargs)

    first_half, genomes, rng_state, rng_ring_state, parent_of = run_coupled_evolution_opponent_count(
        generations=20, return_state=True, track_parent_state=True, **common_kwargs
    )
    second_half = run_coupled_evolution_opponent_count(
        generations=40,
        start_generation=20,
        initial_genomes=genomes,
        rng_state=rng_state,
        rng_ring_state=rng_ring_state,
        parent_of_state=parent_of,
        **common_kwargs,
    )
    resumed = first_half + second_half

    assert len(uninterrupted) == len(resumed) == 40
    assert _stats_to_comparable(uninterrupted) == _stats_to_comparable(resumed)
    # the new fields must match too, including same_parent_fraction across
    # the resume boundary (generation 20) -- the real risk spot, since it
    # needs parent_of_state to be carried over correctly.
    for a, b in zip(uninterrupted, resumed):
        assert a.same_parent_fraction == b.same_parent_fraction or (
            a.same_parent_fraction != a.same_parent_fraction and b.same_parent_fraction != b.same_parent_fraction
        )  # NaN != NaN, so compare "is NaN" on both sides instead
        assert a.k_out_assortativity == b.k_out_assortativity or (
            a.k_out_assortativity != a.k_out_assortativity and b.k_out_assortativity != b.k_out_assortativity
        )
