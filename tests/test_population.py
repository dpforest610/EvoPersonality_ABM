import random

from evopersonality_abm.lindgren import (
    always_cooperate_table,
    always_defect_table,
    duplicate_memory,
    tit_for_tat_table,
)
from evopersonality_abm.population import (
    mutate_individual,
    random_table,
    round_robin_payoffs,
    run_evolution,
    select_and_reproduce,
)


def test_random_table_has_correct_shape_for_depth():
    rng = random.Random(0)
    table = random_table(memory_depth=2, rng=rng)
    assert table.memory_depth == 2
    assert len(table.table) == 2 ** (2 * 2)
    assert len(table.prehistory) == 2 * 2


def test_round_robin_payoffs_matches_known_pd_outcomes():
    # 3 agents: always-cooperate, always-defect, tit-for-tat.
    # Payoffs below are hand-computable from the PD payoff matrix.
    tables = [always_cooperate_table(), always_defect_table(), tit_for_tat_table()]
    payoffs = round_robin_payoffs(tables, rounds_per_duel=10)

    # always-defect exploits always-cooperate every round (CC->AD payoff 5,
    # AC payoff 0) but mutually defects against TFT after round 1.
    # always-cooperate should earn strictly less than tit-for-tat, which in
    # turn earns strictly less than always-defect.
    assert payoffs[1] > payoffs[2] > payoffs[0]


def test_round_robin_payoffs_empty_for_single_agent():
    assert round_robin_payoffs([always_defect_table()], rounds_per_duel=10) == [0.0]


def test_select_and_reproduce_keeps_population_size_and_only_existing_genomes():
    tables = [always_cooperate_table(), always_defect_table()]
    rng = random.Random(1)
    # Zero-weight the cooperator so only the defector can be selected.
    next_gen = select_and_reproduce(tables, payoffs=[0.0, 1.0], rng=rng)

    assert len(next_gen) == 2
    assert all(t is tables[1] for t in next_gen)


def test_mutate_individual_respects_max_memory_depth_cap():
    rng = random.Random(2)
    table = tit_for_tat_table()  # depth 1
    mutated = mutate_individual(
        table,
        rng,
        point_mutation_rate=0.0,
        duplication_rate=1.0,  # always try to duplicate
        max_memory_depth=1,  # but already at the cap
    )
    assert mutated.memory_depth == 1


def test_mutate_individual_duplication_only_grows_depth_by_one():
    rng = random.Random(3)
    table = tit_for_tat_table()  # depth 1
    mutated = mutate_individual(
        table,
        rng,
        point_mutation_rate=0.0,
        duplication_rate=1.0,
        max_memory_depth=5,
    )
    assert mutated.memory_depth == 2


def test_mutate_individual_no_mutation_when_rates_are_zero():
    rng = random.Random(4)
    table = tit_for_tat_table()
    mutated = mutate_individual(
        table, rng, point_mutation_rate=0.0, duplication_rate=0.0, max_memory_depth=5
    )
    assert mutated.memory_depth == table.memory_depth
    assert mutated.table == table.table
    assert mutated.prehistory == table.prehistory


def test_run_evolution_produces_one_stats_entry_per_generation_with_full_population():
    stats = run_evolution(population_size=10, generations=5, rounds_per_duel=8, seed=42)

    assert [s.generation for s in stats] == list(range(5))
    for s in stats:
        assert len(s.memory_depths) == 10
        assert len(s.payoffs) == 10
        assert s.min_depth <= s.mean_depth <= s.max_depth


def test_run_evolution_depth_never_shrinks_when_split_rate_is_zero():
    # split_rate defaults to 0.0, so with it left off no individual can
    # ever have a smaller memory depth than the initial population.
    stats = run_evolution(
        population_size=20,
        generations=15,
        rounds_per_duel=10,
        initial_memory_depth=1,
        duplication_rate=0.1,
        seed=7,
    )
    assert all(min(s.memory_depths) >= 1 for s in stats)


def test_mutate_individual_split_shrinks_depth_by_one():
    rng = random.Random(5)
    table = duplicate_memory(tit_for_tat_table())  # depth 2
    mutated = mutate_individual(
        table,
        rng,
        point_mutation_rate=0.0,
        duplication_rate=0.0,
        split_rate=1.0,
        max_memory_depth=5,
        min_memory_depth=0,
    )
    assert mutated.memory_depth == 1


def test_mutate_individual_respects_min_memory_depth_floor():
    rng = random.Random(6)
    table = tit_for_tat_table()  # depth 1
    mutated = mutate_individual(
        table,
        rng,
        point_mutation_rate=0.0,
        duplication_rate=0.0,
        split_rate=1.0,
        max_memory_depth=5,
        min_memory_depth=1,  # already at the floor
    )
    assert mutated.memory_depth == 1


def test_mutate_individual_duplication_and_split_are_mutually_exclusive():
    # With both rates at 1.0, duplication (tried first) always wins --
    # depth grows, it never nets out to a no-op or a shrink.
    rng = random.Random(8)
    table = tit_for_tat_table()  # depth 1
    mutated = mutate_individual(
        table,
        rng,
        point_mutation_rate=0.0,
        duplication_rate=1.0,
        split_rate=1.0,
        max_memory_depth=5,
        min_memory_depth=0,
    )
    assert mutated.memory_depth == 2


def test_run_evolution_with_split_can_shrink_below_initial_depth():
    stats = run_evolution(
        population_size=20,
        generations=25,
        rounds_per_duel=10,
        initial_memory_depth=3,
        point_mutation_rate=0.05,
        duplication_rate=0.0,
        split_rate=0.3,
        max_memory_depth=5,
        min_memory_depth=0,
        seed=9,
    )
    assert any(min(s.memory_depths) < 3 for s in stats)


def test_run_evolution_is_deterministic_given_a_seed():
    stats_a = run_evolution(population_size=8, generations=4, rounds_per_duel=6, seed=123)
    stats_b = run_evolution(population_size=8, generations=4, rounds_per_duel=6, seed=123)

    for a, b in zip(stats_a, stats_b):
        assert a.memory_depths == b.memory_depths
        assert a.payoffs == b.payoffs


# ------------------------------------------------------------------ noise_rate (see
# docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 1.2)

def test_round_robin_payoffs_default_noise_rate_matches_pre_noise_behavior():
    # noise_rate=0.0 (the default) must reproduce the exact original payoffs,
    # with no rng required at all.
    tables = [always_cooperate_table(), always_defect_table(), tit_for_tat_table()]
    assert round_robin_payoffs(tables, rounds_per_duel=10) == round_robin_payoffs(
        tables, rounds_per_duel=10, noise_rate=0.0, rng=random.Random(0)
    )


def test_run_evolution_default_noise_rate_matches_pre_noise_behavior():
    stats_no_arg = run_evolution(population_size=8, generations=4, rounds_per_duel=6, seed=123)
    stats_explicit_zero = run_evolution(
        population_size=8, generations=4, rounds_per_duel=6, seed=123, noise_rate=0.0
    )
    for a, b in zip(stats_no_arg, stats_explicit_zero):
        assert a.memory_depths == b.memory_depths
        assert a.payoffs == b.payoffs


def test_run_evolution_noise_rate_changes_dynamics():
    # Same seed, only noise_rate differs: full determinism (noise_rate=0.0) must
    # diverge from noisy play (noise_rate=1.0, every realized action flipped).
    stats_clean = run_evolution(
        population_size=10, generations=5, rounds_per_duel=8, seed=42, noise_rate=0.0
    )
    stats_noisy = run_evolution(
        population_size=10, generations=5, rounds_per_duel=8, seed=42, noise_rate=1.0
    )
    assert any(a.payoffs != b.payoffs for a, b in zip(stats_clean, stats_noisy))


def test_run_evolution_noise_rate_is_still_deterministic_given_a_seed():
    stats_a = run_evolution(
        population_size=8, generations=4, rounds_per_duel=6, seed=321, noise_rate=0.01
    )
    stats_b = run_evolution(
        population_size=8, generations=4, rounds_per_duel=6, seed=321, noise_rate=0.01
    )
    for a, b in zip(stats_a, stats_b):
        assert a.memory_depths == b.memory_depths
        assert a.payoffs == b.payoffs
