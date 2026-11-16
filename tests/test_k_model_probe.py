import random

from evopersonality_abm.coupling import Genome, random_genome
from evopersonality_abm.lindgren import (
    LindgrenTable,
    always_defect_table,
    pavlov_table,
    tit_for_tat_table,
)
from evopersonality_abm.k_model_probe import (
    FIXED_OPPONENTS,
    OPPONENTS,
    SELFPLAY_OPPONENTS,
    census_genomes,
    draw_partner_genome,
    genotype_key,
    k_recovery_time_vs_fixed,
    k_recovery_time_vs_genome,
    run_k_selfplay_duel,
    run_k_vs_fixed_duel,
)


# ------------------------------------------------------------------ OPPONENTS registry

def test_fixed_opponents_are_the_6_annen_named_not_gradual_appeasement():
    assert FIXED_OPPONENTS == ["TFT", "AllC", "AllD", "Random", "Gradual", "WSLS"]
    assert "GradualAppeasement" not in OPPONENTS


def test_every_fixed_opponent_has_exactly_one_of_table_or_strategy_factory():
    for name, spec in OPPONENTS.items():
        has_table = "table" in spec
        has_factory = "strategy_factory" in spec
        assert has_table != has_factory, name


# ------------------------------------------------------------------ genotype_key / census_genomes

def test_genotype_key_includes_k_out_k_self_exactly_unrounded():
    table = tit_for_tat_table()
    a = Genome(table=table, k_out=0.3, k_self=0.7)
    b = Genome(table=table, k_out=0.300001, k_self=0.7)
    assert genotype_key(a) != genotype_key(b)


def test_genotype_key_identical_for_genomes_with_identical_fields():
    table = tit_for_tat_table()
    a = Genome(table=table, k_out=0.3, k_self=0.7)
    b = Genome(table=table, k_out=0.3, k_self=0.7)
    assert a is not b
    assert genotype_key(a) == genotype_key(b)


def test_census_genomes_dedupes_exact_duplicates_and_counts_frequency():
    table = tit_for_tat_table()
    g1 = Genome(table=table, k_out=0.3, k_self=0.7)
    g2 = Genome(table=table, k_out=0.3, k_self=0.7)  # same fields, distinct object
    g3 = Genome(table=table, k_out=0.31, k_self=0.7)  # differs by K_out alone
    census = census_genomes([g1, g2, g3])
    assert len(census) == 2
    (_key0, freq0, _rep0), (_key1, freq1, _rep1) = census
    assert freq0 == 2 and freq1 == 1


def test_census_genomes_random_genomes_are_almost_all_distinct():
    # docs/K_MODEL_MARKOV_DESIGN.md section 8-3: continuous K means
    # near-total per-individual uniqueness is EXPECTED, not a bug.
    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(50)]
    census = census_genomes(genomes)
    assert len(census) == 50  # vanishingly unlikely to collide by chance
    assert all(freq == 1 for _key, freq, _rep in census)


# ------------------------------------------------------------------ run_k_vs_fixed_duel

def test_run_k_vs_fixed_duel_returns_equal_length_move_lists():
    genome = random_genome(1, random.Random(0))
    own, opp = run_k_vs_fixed_duel(genome, "AllD", random.Random(1), rounds=15)
    assert len(own) == 15 and len(opp) == 15
    assert set(own) <= {"C", "D"} and set(opp) <= {"C", "D"}


def test_run_k_vs_fixed_duel_is_deterministic_given_the_same_rng_seed():
    genome = random_genome(1, random.Random(0))
    a = run_k_vs_fixed_duel(genome, "Random", random.Random(7), rounds=10)
    b = run_k_vs_fixed_duel(genome, "Random", random.Random(7), rounds=10)
    assert a == b


def test_run_k_vs_fixed_duel_noise_changes_realized_moves():
    genome = Genome(table=always_defect_table(), k_out=0.0, k_self=1.0)
    clean = run_k_vs_fixed_duel(genome, "AllC", random.Random(0), rounds=10, noise_std=0.0)
    noisy = run_k_vs_fixed_duel(genome, "AllC", random.Random(0), rounds=10, noise_std=5.0)
    assert clean != noisy


def test_run_k_vs_fixed_duel_degenerate_k_self1_k_out0_noise0_matches_pure_lindgren_tft_vs_alld():
    """k_self=1, k_out=0, noise_std=0 makes s equal the table's own
    recommendation exactly every round (s = s_prev + 1*(r-s_prev) + 0 + 0
    = r), reducing the K layer to a deterministic pass-through -- this
    must reproduce the SAME moves as plain LindgrenTable.decide() would
    (the pure-Lindgren TFT-vs-AllD precedent: tests/test_probe.py's
    test_run_probe_duel_against_alld_matches_the_genome_table_behavior_directly)."""
    genome = Genome(table=tit_for_tat_table(), k_out=0.0, k_self=1.0)
    own, opp = run_k_vs_fixed_duel(genome, "AllD", random.Random(0), rounds=5, noise_std=0.0)
    assert own == ["C", "D", "D", "D", "D"]
    assert opp == ["D"] * 5


# ------------------------------------------------------------------ run_k_selfplay_duel

def test_run_k_selfplay_duel_returns_equal_length_move_lists():
    g1 = random_genome(1, random.Random(0))
    g2 = random_genome(1, random.Random(1))
    own, opp = run_k_selfplay_duel(g1, g2, random.Random(2), rounds=12)
    assert len(own) == 12 and len(opp) == 12


def test_run_k_selfplay_duel_self_uses_two_independent_local_states_not_shared():
    # same frozen Genome object on both sides -- must not share mutable
    # state (there is none stored on Genome, so this should just work).
    genome = Genome(table=always_defect_table(), k_out=0.0, k_self=1.0)
    own, opp = run_k_selfplay_duel(genome, genome, random.Random(0), rounds=5, noise_std=0.0)
    assert own == ["D"] * 5
    assert opp == ["D"] * 5


def test_run_k_selfplay_duel_is_deterministic_given_the_same_rng_seed():
    g1 = random_genome(1, random.Random(0))
    g2 = random_genome(1, random.Random(1))
    a = run_k_selfplay_duel(g1, g2, random.Random(5), rounds=10)
    b = run_k_selfplay_duel(g1, g2, random.Random(5), rounds=10)
    assert a == b


# ------------------------------------------------------------------ draw_partner_genome

def test_draw_partner_genome_only_returns_genomes_from_the_given_gids():
    g0 = random_genome(1, random.Random(0))
    g1 = random_genome(1, random.Random(1))
    genomes = {0: g0, 1: g1}
    rng = random.Random(0)
    for _ in range(20):
        drawn = draw_partner_genome([0, 1], [1, 1], genomes, rng)
        assert drawn is g0 or drawn is g1


# ------------------------------------------------------------------ k_recovery_time_vs_fixed

def test_k_recovery_time_vs_fixed_degenerate_tft_vs_tft_never_recovers():
    """Degenerate K genome (k_self=1,k_out=0,noise=0) with TFT's table,
    against the fixed TFT opponent, reduces to pure-Lindgren TFT-vs-TFT
    from a forced-bottom start -- established earlier (this session) to
    never recover (neither side ever initiates cooperation)."""
    genome = Genome(table=tit_for_tat_table(), k_out=0.0, k_self=1.0)
    rt = k_recovery_time_vs_fixed(genome, "TFT", random.Random(0), free_rounds=20, noise_std=0.0)
    assert rt is None


def test_k_recovery_time_vs_fixed_degenerate_wsls_vs_tft_never_recovers():
    genome = Genome(table=pavlov_table(), k_out=0.0, k_self=1.0)
    rt = k_recovery_time_vs_fixed(genome, "TFT", random.Random(0), free_rounds=20, noise_std=0.0)
    assert rt is None


def test_k_recovery_time_vs_fixed_returns_none_or_a_positive_int_within_free_rounds():
    genome = random_genome(1, random.Random(0))
    rt = k_recovery_time_vs_fixed(genome, "WSLS", random.Random(1), free_rounds=20)
    assert rt is None or (isinstance(rt, int) and 1 <= rt <= 20)


# ------------------------------------------------------------------ k_recovery_time_vs_genome

def test_k_recovery_time_vs_genome_degenerate_wsls_self_recovers_in_2_rounds():
    """Hand-verified (and cross-checked by direct execution): a
    degenerate WSLS-table K genome, forced to 3 rounds of mutual
    defection against an identical copy of itself, has both sides'
    tables recommend COOPERATE from round 1 of free play (WSLS's
    win-stay-lose-shift: index(own=D,opp=D) -> table[0]=1=>C), but the
    2-consecutive-CC condition isn't met until the round after that
    (round 1 follows a D,D round, so it doesn't count as "2 consecutive")
    -- recovers at free-round 2."""
    genome = Genome(table=pavlov_table(), k_out=0.0, k_self=1.0)
    rt = k_recovery_time_vs_genome(genome, genome, random.Random(0), free_rounds=20, noise_std=0.0)
    assert rt == 2


def test_k_recovery_time_vs_genome_is_deterministic_given_the_same_rng_seed():
    g1 = random_genome(1, random.Random(0))
    g2 = random_genome(1, random.Random(1))
    a = k_recovery_time_vs_genome(g1, g2, random.Random(3), free_rounds=20)
    b = k_recovery_time_vs_genome(g1, g2, random.Random(3), free_rounds=20)
    assert a == b
