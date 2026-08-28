import random

import pytest

from evopersonality_abm.coupling import (
    Genome,
    mutate_genome,
    random_genome,
    round_robin_payoffs_coupled,
    run_coupled_duel,
    run_coupled_evolution,
    update_internal_state,
)
from evopersonality_abm.lindgren import always_defect_table, tit_for_tat_table


def test_update_internal_state_full_deliberate_no_noise_snaps_to_r():
    # k_self=1, k_out=0 -> s(t) tracks r_i exactly, regardless of s(t-1).
    assert update_internal_state(s_prev=0.5, k_self=1.0, r=1, k_out=0.0, a=0.0, noise=0.0) == 1.0
    assert update_internal_state(s_prev=0.5, k_self=1.0, r=0, k_out=0.0, a=1.0, noise=0.0) == 0.0


def test_update_internal_state_full_automatic_no_noise_snaps_to_a():
    assert update_internal_state(s_prev=0.5, k_self=0.0, r=0.0, k_out=1.0, a=1, noise=0.0) == 1.0


def test_update_internal_state_no_coupling_holds_still_apart_from_noise():
    assert update_internal_state(s_prev=0.3, k_self=0.0, r=1, k_out=0.0, a=0, noise=0.0) == 0.3


def test_update_internal_state_clips_to_unit_interval():
    # k_self=k_out=1, both pulling toward 1 from s_prev=0 overshoots to 2.
    assert update_internal_state(s_prev=0.0, k_self=1.0, r=1, k_out=1.0, a=1, noise=0.0) == 1.0
    # symmetric overshoot below 0.
    assert update_internal_state(s_prev=1.0, k_self=1.0, r=0, k_out=1.0, a=0, noise=0.0) == 0.0


def test_random_genome_shape_and_range():
    rng = random.Random(0)
    genome = random_genome(memory_depth=2, rng=rng)
    assert genome.table.memory_depth == 2
    assert 0.0 <= genome.k_out <= 1.0
    assert 0.0 <= genome.k_self <= 1.0


def test_run_coupled_duel_matches_pure_lindgren_when_fully_deliberate_and_noiseless():
    # k_self=1, k_out=0, noise=0: r_i always wins outright, so the coupled
    # duel must reproduce the pure Lindgren table-vs-table outcome exactly.
    tft = Genome(table=tit_for_tat_table(), k_out=0.0, k_self=1.0)
    defector = Genome(table=always_defect_table(), k_out=0.0, k_self=1.0)
    rng = random.Random(1)

    payoff_tft, payoff_defector = run_coupled_duel(tft, defector, rounds=10, rng=rng, noise_std=0.0)

    # Known result (see test_lindgren.py): TFT starts C then mirrors D
    # forever against an always-defector -> TFT gets 0 the first round,
    # 1 each round after; defector gets 5 the first round, 1 after.
    assert payoff_tft == 0 + 1 * 9
    assert payoff_defector == 5 + 1 * 9


def test_round_robin_payoffs_coupled_empty_for_single_agent():
    genome = Genome(table=always_defect_table(), k_out=0.5, k_self=0.5)
    rng = random.Random(2)
    assert round_robin_payoffs_coupled([genome], rounds_per_duel=10, rng=rng, noise_std=0.05) == [0.0]


def test_mutate_genome_keeps_k_within_unit_interval_even_at_the_boundary():
    rng = random.Random(3)
    genome = Genome(table=tit_for_tat_table(), k_out=1.0, k_self=0.0)
    for _ in range(200):
        genome = mutate_genome(
            genome,
            rng,
            point_mutation_rate=0.0,
            duplication_rate=0.0,
            split_rate=0.0,
            max_memory_depth=5,
            min_memory_depth=0,
            k_mutation_std=0.3,
        )
        assert 0.0 <= genome.k_out <= 1.0
        assert 0.0 <= genome.k_self <= 1.0


def test_run_coupled_evolution_produces_expected_stats_shape():
    stats = run_coupled_evolution(
        population_size=10, generations=5, rounds_per_duel=8, seed=42
    )
    assert [s.generation for s in stats] == list(range(5))
    for s in stats:
        assert len(s.k_out) == 10
        assert len(s.k_self) == 10
        assert len(s.memory_depths) == 10
        assert len(s.payoffs) == 10
        assert all(0.0 <= k <= 1.0 for k in s.k_out)
        assert all(0.0 <= k <= 1.0 for k in s.k_self)


def test_run_coupled_evolution_is_deterministic_given_a_seed():
    stats_a = run_coupled_evolution(population_size=8, generations=4, rounds_per_duel=6, seed=123)
    stats_b = run_coupled_evolution(population_size=8, generations=4, rounds_per_duel=6, seed=123)
    for a, b in zip(stats_a, stats_b):
        assert a.k_out == b.k_out
        assert a.k_self == b.k_self
        assert a.memory_depths == b.memory_depths
        assert a.payoffs == b.payoffs
