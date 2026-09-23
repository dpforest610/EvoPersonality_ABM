import random

import pytest

from evopersonality_abm.spatial import (
    SpatialGenerationStats,
    _mean_neighbor_payoff,
    _softmax_sample,
    place_on_grid,
    run_coupled_evolution_spatial,
    torus_neighbors,
)


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


def test_run_coupled_evolution_spatial_rejects_non_divisible_phases():
    with pytest.raises(ValueError):
        run_coupled_evolution_spatial(population_size=10, generations=1, rounds_per_duel=5, n_phases=2,
                                       grid_width=4, grid_height=4, seed=0)
