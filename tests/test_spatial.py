import random

import pytest

from evopersonality_abm.spatial import (
    MOORE_OFFSETS,
    SpatialGenerationStats,
    _mean_neighbor_payoff,
    _softmax_sample,
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
