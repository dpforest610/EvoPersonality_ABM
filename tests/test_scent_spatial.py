import random
import statistics

import pytest

from evopersonality_abm.scent_spatial import (
    assign_offspring_cells,
    moore_offsets,
    nearest_empty_cell,
    run_coupled_evolution_scent,
)


def test_moore_offsets_radius_1_has_8_cells():
    offsets = moore_offsets(1)
    assert len(offsets) == 8
    assert (0, 0) not in offsets


def test_moore_offsets_radius_2_has_24_cells():
    offsets = moore_offsets(2)
    assert len(offsets) == 24


def test_nearest_empty_cell_prefers_closest_ring():
    width = height = 10
    occupied = {(5, 5)}  # target itself is occupied
    rng = random.Random(0)
    chosen = nearest_empty_cell((5, 5), width, height, occupied, rng)
    # must be a distance-1 Chebyshev neighbor of (5, 5)
    dx = min(abs(chosen[0] - 5), width - abs(chosen[0] - 5))
    dy = min(abs(chosen[1] - 5), height - abs(chosen[1] - 5))
    assert max(dx, dy) == 1


def test_assign_offspring_cells_first_child_takes_parents_old_cell():
    # agent 0's only cell is (1, 1); if its only offspring is slot 0,
    # that offspring must land on (1, 1).
    old_cells = [(1, 1), (5, 5)]
    parent_indices = [0, 1]  # offspring 0 <- parent 0, offspring 1 <- parent 1
    rng = random.Random(0)
    new_cells = assign_offspring_cells(parent_indices, old_cells, 10, 10, rng)
    assert new_cells[0] == (1, 1)
    assert new_cells[1] == (5, 5)


def test_assign_offspring_cells_extra_children_cluster_near_parent():
    old_cells = [(0, 0), (5, 5)]
    # parent 0 has 5 offspring, parent 1 has none
    parent_indices = [0, 0, 0, 0, 0]
    rng = random.Random(0)
    new_cells = assign_offspring_cells(parent_indices, old_cells, 20, 20, rng)
    assert new_cells[0] == (0, 0)
    for cell in new_cells[1:]:
        dx = min(abs(cell[0] - 0), 20 - abs(cell[0] - 0))
        dy = min(abs(cell[1] - 0), 20 - abs(cell[1] - 0))
        # a 20x20 grid has ample room, so even the 5th sibling should
        # still land within a couple of rings of the parent's old cell
        assert max(dx, dy) <= 3


def test_m0_no_movement_within_a_generation():
    stats = run_coupled_evolution_scent(
        population_size=20, generations=5, rounds_per_duel=10, n_phases=2,
        grid_width=6, grid_height=6, movement="none", seed=3,
    )
    for s in stats:
        assert s.move_rate == 0.0


def test_m1_move_rate_matches_p_move():
    p_move = 0.4
    stats = run_coupled_evolution_scent(
        population_size=60, generations=20, rounds_per_duel=12, n_phases=4,
        grid_width=14, grid_height=14, movement="random", p_move=p_move, seed=7,
    )
    observed = statistics.fmean(s.move_rate for s in stats)
    assert abs(observed - p_move) < 0.05


def test_scent_movement_produces_nonzero_move_rate():
    stats = run_coupled_evolution_scent(
        population_size=30, generations=5, rounds_per_duel=12, n_phases=3,
        grid_width=8, grid_height=8, movement="scent", tau=1.0, seed=5,
    )
    assert all(s.move_rate > 0.0 for s in stats)


def test_chunked_checkpoint_resume_is_bit_identical():
    kwargs = dict(
        population_size=24, rounds_per_duel=12, n_phases=3,
        grid_width=8, grid_height=8, movement="scent", tau=1.0, seed=11,
    )
    uninterrupted = run_coupled_evolution_scent(generations=6, **kwargs)

    stats1, genomes1, cells1, rng_state1, rng_spatial_state1, parent_of1 = run_coupled_evolution_scent(
        generations=3, track_parent_state=True, return_state=True, **kwargs
    )
    stats2, genomes2, cells2, rng_state2, rng_spatial_state2 = run_coupled_evolution_scent(
        generations=6, start_generation=3,
        initial_genomes=genomes1, initial_cells=cells1,
        rng_state=rng_state1, rng_spatial_state=rng_spatial_state1,
        parent_of_state=parent_of1, return_state=True,
        **kwargs,
    )
    resumed = stats1 + stats2

    assert len(uninterrupted) == len(resumed) == 6
    for a, b in zip(uninterrupted, resumed):
        assert a.generation == b.generation
        assert a.k_out == b.k_out
        assert a.k_self == b.k_self
        assert a.payoffs == b.payoffs
        assert a.positions == b.positions
        assert a.move_rate == b.move_rate
        assert a.cooperation_rate == b.cooperation_rate or (
            a.cooperation_rate != a.cooperation_rate and b.cooperation_rate != b.cooperation_rate
        )


def test_resume_requires_divisible_rounds_per_duel():
    with pytest.raises(ValueError):
        run_coupled_evolution_scent(
            population_size=10, generations=1, rounds_per_duel=10, n_phases=3,
            grid_width=6, grid_height=6, movement="none",
        )


def test_population_must_fit_on_grid():
    with pytest.raises(ValueError):
        run_coupled_evolution_scent(
            population_size=50, generations=1, rounds_per_duel=10, n_phases=1,
            grid_width=5, grid_height=5, movement="none",
        )


def test_random_movement_requires_p_move():
    with pytest.raises(ValueError):
        run_coupled_evolution_scent(
            population_size=10, generations=1, rounds_per_duel=10, n_phases=1,
            grid_width=6, grid_height=6, movement="random",
        )
