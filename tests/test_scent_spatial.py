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


def test_scent_index_is_nan_for_non_scent_movement():
    stats_none = run_coupled_evolution_scent(
        population_size=20, generations=3, rounds_per_duel=10, n_phases=2,
        grid_width=6, grid_height=6, movement="none", seed=2,
    )
    for s in stats_none:
        assert s.scent_index_mean != s.scent_index_mean  # NaN

    stats_random = run_coupled_evolution_scent(
        population_size=20, generations=3, rounds_per_duel=10, n_phases=2,
        grid_width=6, grid_height=6, movement="random", p_move=0.3, seed=2,
    )
    for s in stats_random:
        assert s.scent_index_mean != s.scent_index_mean  # NaN


def test_scent_index_decreases_toward_zero_as_tau_grows():
    kwargs = dict(
        population_size=40, generations=10, rounds_per_duel=10, n_phases=2,
        grid_width=10, grid_height=10, movement="scent", seed=4,
    )
    low_tau_stats = run_coupled_evolution_scent(tau=0.01, **kwargs)
    high_tau_stats = run_coupled_evolution_scent(tau=100.0, **kwargs)

    low_tau_index = statistics.fmean(
        s.scent_index_mean for s in low_tau_stats if s.scent_index_mean == s.scent_index_mean
    )
    high_tau_index = statistics.fmean(
        s.scent_index_mean for s in high_tau_stats if s.scent_index_mean == s.scent_index_mean
    )
    # near-greedy (low tau) should follow the scent score far more
    # closely than near-uniform (high tau) random choice among candidates
    assert low_tau_index > 0.7
    assert abs(high_tau_index) < 0.2
    assert low_tau_index > high_tau_index


def test_phase_recording_captures_one_entry_per_phase_for_requested_generations():
    recorder = {}
    run_coupled_evolution_scent(
        population_size=20, generations=5, rounds_per_duel=12, n_phases=3,
        grid_width=6, grid_height=6, movement="scent", tau=1.0, seed=2,
        record_phases_for_generations={1, 3}, phase_recorder_out=recorder,
    )
    assert sorted(recorder.keys()) == [1, 3]
    for gen in (1, 3):
        assert len(recorder[gen]) == 3  # n_phases
        for phase_idx, positions, moved, payoff_so_far in recorder[gen]:
            assert len(positions) == 20
            assert len(moved) == 20
            assert len(payoff_so_far) == 20
            assert all(isinstance(p, tuple) and len(p) == 2 for p in positions)
            # non-isolated agents by the last phase should have a real
            # (non-NaN) running payoff-so-far
            if phase_idx == 2:
                assert sum(1 for v in payoff_so_far if v == v) > 0


def test_phase_recording_moved_flags_are_false_for_m0():
    recorder = {}
    run_coupled_evolution_scent(
        population_size=20, generations=3, rounds_per_duel=12, n_phases=3,
        grid_width=6, grid_height=6, movement="none", seed=2,
        record_phases_for_generations={1}, phase_recorder_out=recorder,
    )
    for _, _, moved, _ in recorder[1]:
        assert not any(moved)


def test_phase_recording_last_phase_payoff_matches_generation_payoffs():
    recorder = {}
    stats = run_coupled_evolution_scent(
        population_size=20, generations=5, rounds_per_duel=12, n_phases=3,
        grid_width=6, grid_height=6, movement="scent", tau=1.0, seed=2,
        record_phases_for_generations={1, 3}, phase_recorder_out=recorder,
    )
    for gen in (1, 3):
        last_phase_idx, _, _, payoff_so_far = recorder[gen][-1]
        assert last_phase_idx == 2  # n_phases - 1
        gen_payoffs = stats[gen].payoffs
        for v, p in zip(payoff_so_far, gen_payoffs):
            if v == v:  # not NaN -> agent had at least one duel
                assert v == pytest.approx(p)


def test_phase_recording_requires_recorder_out():
    with pytest.raises(ValueError):
        run_coupled_evolution_scent(
            population_size=20, generations=3, rounds_per_duel=12, n_phases=3,
            grid_width=6, grid_height=6, movement="none", seed=2,
            record_phases_for_generations={1},
        )


def test_phase_recording_does_not_change_simulation_results():
    kwargs = dict(
        population_size=20, generations=5, rounds_per_duel=12, n_phases=3,
        grid_width=6, grid_height=6, movement="scent", tau=1.0, seed=2,
    )
    plain = run_coupled_evolution_scent(**kwargs)
    recorder = {}
    recorded = run_coupled_evolution_scent(
        record_phases_for_generations={1, 3}, phase_recorder_out=recorder, **kwargs
    )
    for a, b in zip(plain, recorded):
        assert a.k_out == b.k_out
        assert a.k_self == b.k_self
        assert a.payoffs == b.payoffs
        assert a.positions == b.positions
        assert a.move_rate == b.move_rate


def test_consider_move_prob_scales_down_move_rate_scent():
    kwargs = dict(
        population_size=60, generations=10, rounds_per_duel=10, n_phases=5,
        grid_width=12, grid_height=12, movement="scent", tau=1.0, seed=6,
    )
    full = run_coupled_evolution_scent(**kwargs)
    gated = run_coupled_evolution_scent(consider_move_prob=0.1, **kwargs)
    full_rate = statistics.fmean(s.move_rate for s in full)
    gated_rate = statistics.fmean(s.move_rate for s in gated)
    assert gated_rate < full_rate * 0.3  # roughly a 10x reduction, generous tolerance


def test_consider_move_prob_scales_down_move_rate_random():
    kwargs = dict(
        population_size=60, generations=10, rounds_per_duel=10, n_phases=5,
        grid_width=12, grid_height=12, movement="random", p_move=0.8, seed=6,
    )
    full = run_coupled_evolution_scent(**kwargs)
    gated = run_coupled_evolution_scent(consider_move_prob=0.1, **kwargs)
    full_rate = statistics.fmean(s.move_rate for s in full)
    gated_rate = statistics.fmean(s.move_rate for s in gated)
    assert gated_rate < full_rate * 0.3


def test_consider_move_prob_none_is_unchanged_from_default():
    kwargs = dict(
        population_size=30, generations=8, rounds_per_duel=10, n_phases=5,
        grid_width=10, grid_height=10, movement="scent", tau=1.0, seed=9,
    )
    plain = run_coupled_evolution_scent(**kwargs)
    explicit_none = run_coupled_evolution_scent(consider_move_prob=None, **kwargs)
    for a, b in zip(plain, explicit_none):
        assert a.positions == b.positions
        assert a.move_rate == b.move_rate
        assert a.k_out == b.k_out


def test_consider_move_prob_zero_means_nobody_moves():
    stats = run_coupled_evolution_scent(
        population_size=30, generations=5, rounds_per_duel=10, n_phases=5,
        grid_width=10, grid_height=10, movement="scent", tau=1.0, seed=4,
        consider_move_prob=0.0,
    )
    for s in stats:
        assert s.move_rate == 0.0
