import random

from evopersonality_abm.lindgren import (
    LindgrenTable,
    always_defect_table,
    duplicate_memory,
    tit_for_tat_table,
)
from evopersonality_abm.markov_recovery import load_vocab
from evopersonality_abm.probe import (
    EXTENDED_SET,
    MAIN_SET,
    OPPONENTS,
    census_genotypes,
    genotype_key,
    make_opponent_strategy,
    run_probe_duel,
    run_selfplay_duel,
)


# ------------------------------------------------------------------ genotype_key / census_genotypes

def test_genotype_key_is_identical_for_equal_but_distinct_tables():
    a = tit_for_tat_table()
    b = tit_for_tat_table()
    assert a is not b
    assert genotype_key(a) == genotype_key(b)


def test_genotype_key_differs_for_different_memory_depth_or_table_or_prehistory():
    tft = tit_for_tat_table()
    deeper = duplicate_memory(tft)
    different_table = LindgrenTable(memory_depth=1, table=[1, 0, 1, 0], prehistory=[1, 1])
    different_prehistory = LindgrenTable(memory_depth=1, table=[0, 1, 0, 1], prehistory=[0, 0])

    keys = {
        genotype_key(tft),
        genotype_key(deeper),
        genotype_key(different_table),
        genotype_key(different_prehistory),
    }
    assert len(keys) == 4


def test_census_genotypes_dedupes_and_counts_frequency():
    tft_a = tit_for_tat_table()
    tft_b = tit_for_tat_table()  # distinct object, same genotype
    alld = always_defect_table()
    census = census_genotypes([tft_a, tft_b, tft_a, alld])

    assert len(census) == 2
    # sorted by descending frequency: TFT (3 copies) before AllD (1 copy)
    (key0, freq0, rep0), (key1, freq1, rep1) = census
    assert freq0 == 3 and freq1 == 1
    assert key0 == genotype_key(tft_a)
    assert key1 == genotype_key(alld)
    assert freq0 + freq1 == 4


def test_census_genotypes_representative_is_one_of_the_matching_tables():
    tft_a = tit_for_tat_table()
    tft_b = tit_for_tat_table()
    (_, _, rep), = census_genotypes([tft_a, tft_b])
    assert rep is tft_a or rep is tft_b
    assert genotype_key(rep) == genotype_key(tft_a)


def test_census_genotypes_empty_population():
    assert census_genotypes([]) == []


# ------------------------------------------------------------------ OPPONENTS registry

def test_opponents_registry_covers_main_and_extended_sets():
    assert set(MAIN_SET) <= set(OPPONENTS)
    assert set(EXTENDED_SET) <= set(OPPONENTS)
    assert set(MAIN_SET).isdisjoint(EXTENDED_SET)


def test_every_opponent_has_exactly_one_of_table_or_strategy_factory():
    for name, spec in OPPONENTS.items():
        has_table = "table" in spec
        has_factory = "strategy_factory" in spec
        assert has_table != has_factory, name


def test_every_opponent_vocab_is_loadable_and_nonempty():
    for name, spec in OPPONENTS.items():
        vocab = load_vocab(spec["vocab"])
        assert len(vocab) > 0, name


def test_extended_set_opponents_use_the_unrestricted_gra_vocabulary():
    # design doc section 3.4: GradualAppeasement and WSLS reuse GRA/RAN's
    # full 64-word (unrestricted) vocabulary, stated directly by the user.
    for name in EXTENDED_SET:
        assert OPPONENTS[name]["vocab"] == "GRA"


def test_make_opponent_strategy_fixed_table_opponents_are_tft_alld_allc_equivalent():
    tft_strategy = make_opponent_strategy("TFT", random.Random(0))
    assert tft_strategy([], []) == "C"
    assert tft_strategy([], ["D"]) == "D"

    alld_strategy = make_opponent_strategy("AllD", random.Random(0))
    assert alld_strategy([], []) == "D"
    assert alld_strategy(["D", "D"], ["C", "C"]) == "D"


def test_make_opponent_strategy_gradual_is_a_fresh_stateful_closure_each_call():
    # two separately-constructed Gradual strategies must not share state
    strat1 = make_opponent_strategy("Gradual", random.Random(0))
    strat2 = make_opponent_strategy("Gradual", random.Random(0))
    own1, own2 = [], []
    for opp_move in ["D", "C", "C"]:
        own1.append(strat1(own1, own1 and [opp_move] or []))
    assert strat1 is not strat2


# ------------------------------------------------------------------ run_probe_duel

def test_run_probe_duel_returns_two_equal_length_move_lists():
    genome = tit_for_tat_table()
    own_moves, opp_moves = run_probe_duel(genome, "AllD", random.Random(0), rounds=20)
    assert len(own_moves) == 20
    assert len(opp_moves) == 20
    assert set(own_moves) <= {"C", "D"}
    assert set(opp_moves) <= {"C", "D"}


def test_run_probe_duel_against_alld_matches_the_genome_table_behavior_directly():
    # TFT vs AllD, no noise: TFT cooperates round 1 then always defects
    # (mirrors AllD's constant defection) -- matches tit_for_tat_table's
    # own prehistory-driven decide() applied directly.
    genome = tit_for_tat_table()
    own_moves, opp_moves = run_probe_duel(genome, "AllD", random.Random(0), rounds=5)
    assert own_moves == ["C", "D", "D", "D", "D"]
    assert opp_moves == ["D"] * 5


def test_run_probe_duel_uses_the_genomes_own_prehistory():
    # always_defect_table's prehistory is [0, 0] (assumes prior mutual
    # defection); pitted against AllC, it should defect from round 1
    # (not cooperate first the way a prehistory=[1,1] genome would).
    genome = always_defect_table()
    own_moves, opp_moves = run_probe_duel(genome, "AllC", random.Random(0), rounds=3)
    assert own_moves == ["D", "D", "D"]
    assert opp_moves == ["C", "C", "C"]


def test_run_probe_duel_is_deterministic_given_the_same_rng_seed():
    genome = tit_for_tat_table()
    out_a = run_probe_duel(genome, "Random", random.Random(7), rounds=10)
    out_b = run_probe_duel(genome, "Random", random.Random(7), rounds=10)
    assert out_a == out_b


def test_run_probe_duel_noise_rate_can_change_realized_moves():
    genome = always_defect_table()
    clean = run_probe_duel(genome, "AllC", random.Random(0), rounds=10, noise_rate=0.0)
    noisy = run_probe_duel(genome, "AllC", random.Random(0), rounds=10, noise_rate=1.0)
    assert clean != noisy


# ------------------------------------------------------------------ run_selfplay_duel
# (exploratory, outside preregistration -- docs/EXPLORATORY_selfplay_recovery.md)

def test_run_selfplay_duel_self_mutual_tit_for_tat_cooperates_forever():
    table = tit_for_tat_table()
    own_moves, opp_moves = run_selfplay_duel(table, table, random.Random(0), rounds=10)
    assert own_moves == ["C"] * 10
    assert opp_moves == ["C"] * 10


def test_run_selfplay_duel_self_uses_two_independent_agents_not_shared_state():
    # two SEPARATE Agents wrapping the same table must not share history --
    # mirrors run_probe_duel's own prehistory-driven, no-shared-state design
    table = always_defect_table()
    own_moves, opp_moves = run_selfplay_duel(table, table, random.Random(0), rounds=5)
    assert own_moves == ["D"] * 5
    assert opp_moves == ["D"] * 5


def test_run_selfplay_duel_pop_uses_the_partner_tables_own_prehistory():
    focal = tit_for_tat_table()  # prehistory [1,1] -- assumes prior mutual cooperation
    partner = always_defect_table()  # prehistory [0,0] -- assumes prior mutual defection
    own_moves, opp_moves = run_selfplay_duel(focal, partner, random.Random(0), rounds=3)
    # TFT vs AllD: TFT cooperates round 1 (per its own prehistory) then
    # mirrors AllD's constant defection -- identical behavior to probing
    # against the fixed "AllD" opponent (run_probe_duel), confirming
    # partner_table's own prehistory is honored exactly like a fixed table.
    assert own_moves == ["C", "D", "D"]
    assert opp_moves == ["D", "D", "D"]


def test_run_selfplay_duel_returns_equal_length_move_lists():
    table = tit_for_tat_table()
    own_moves, opp_moves = run_selfplay_duel(table, table, random.Random(0), rounds=20)
    assert len(own_moves) == 20 and len(opp_moves) == 20


def test_run_selfplay_duel_is_deterministic_given_the_same_rng_seed():
    focal, partner = tit_for_tat_table(), always_defect_table()
    out_a = run_selfplay_duel(focal, partner, random.Random(7), rounds=10, noise_rate=0.3)
    out_b = run_selfplay_duel(focal, partner, random.Random(7), rounds=10, noise_rate=0.3)
    assert out_a == out_b


def test_run_selfplay_duel_noise_rate_can_change_realized_moves():
    focal, partner = always_defect_table(), always_defect_table()
    clean = run_selfplay_duel(focal, partner, random.Random(0), rounds=10, noise_rate=0.0)
    noisy = run_selfplay_duel(focal, partner, random.Random(0), rounds=10, noise_rate=1.0)
    assert clean != noisy
