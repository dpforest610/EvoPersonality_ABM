import csv
import gzip
import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "export_markov_selfplay_recovery_by_depth", REPO_ROOT / "scripts" / "export_markov_selfplay_recovery_by_depth.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["export_markov_selfplay_recovery_by_depth"] = mod
spec.loader.exec_module(mod)

from evopersonality_abm.lindgren import LindgrenTable, always_defect_table, duplicate_memory, pavlov_table, tit_for_tat_table  # noqa: E402


def write_census(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        for row in rows:
            w.writerow(row)


# ------------------------------------------------------------------ depth_group_lookup

def test_depth_group_lookup_matches_edge_mods_depth_ranges():
    lookup = mod.depth_group_lookup(2, 3, 6)
    assert lookup[0] == "depthLow" and lookup[2] == "depthLow"
    assert lookup[3] == "depthMid"
    assert lookup[4] == "depthHigh" and lookup[6] == "depthHigh"
    assert 7 not in lookup


# ------------------------------------------------------------------ is_depth1_wsls_rule

def test_is_depth1_wsls_rule_true_for_pavlov_table():
    assert mod.is_depth1_wsls_rule(pavlov_table())


def test_is_depth1_wsls_rule_false_for_other_depth1_tables():
    assert not mod.is_depth1_wsls_rule(tit_for_tat_table())
    assert not mod.is_depth1_wsls_rule(always_defect_table())


def test_is_depth1_wsls_rule_false_for_non_depth1_tables_even_if_behaviorally_equivalent():
    # duplicate_memory grows pavlov_table to memory_depth=2 without
    # changing its behavior, but its raw table is now 16 entries, not the
    # literal 4-entry [1,0,0,1] match -- out of scope for this count by design.
    deeper = duplicate_memory(pavlov_table())
    assert deeper.memory_depth == 2
    assert not mod.is_depth1_wsls_rule(deeper)


def test_is_depth1_wsls_rule_ignores_prehistory():
    same_rule_different_prehistory = LindgrenTable(memory_depth=1, table=[1, 0, 0, 1], prehistory=[0, 0])
    assert mod.is_depth1_wsls_rule(same_rule_different_prehistory)


# ------------------------------------------------------------------ wsls_rule_counts_for_seed_age

def test_wsls_rule_counts_for_seed_age_counts_only_depth1_frequency_weighted():
    census = [
        (0, 5, pavlov_table()),  # depth1, matches WSLS rule
        (1, 3, tit_for_tat_table()),  # depth1, does not match
        (2, 10, duplicate_memory(pavlov_table())),  # depth2, excluded from n_depth1 entirely
    ]
    n_depth1, n_wsls_rule = mod.wsls_rule_counts_for_seed_age(census)
    assert n_depth1 == 8  # 5 + 3, depth2 genotype excluded
    assert n_wsls_rule == 5


def test_wsls_rule_counts_for_seed_age_zero_when_no_depth1_genotypes():
    census = [(0, 10, duplicate_memory(pavlov_table()))]
    assert mod.wsls_rule_counts_for_seed_age(census) == (0, 0)


# ------------------------------------------------------------------ recovery_times_by_depth_for_seed_age

def test_recovery_times_by_depth_tags_entries_with_the_focal_genotypes_depth_group():
    shallow = tit_for_tat_table()  # effective depth 1 -> depthLow under (2,3,6)
    deep = duplicate_memory(duplicate_memory(duplicate_memory(tit_for_tat_table())))  # nominal depth 4, effective depth still 1 (duplicate_memory is behaviorally neutral)
    census = [(0, 1, shallow), (1, 1, deep)]
    depth_group_of_depth = mod.depth_group_lookup(2, 3, 6)
    rng = random.Random(0)
    out = mod.recovery_times_by_depth_for_seed_age(census, probe_repeats=2, free_rounds=5, noise_rate=0.0, rng=rng, depth_group_of_depth=depth_group_of_depth)
    # both genotypes have effective_memory_depth==1 -> both land in depthLow
    assert set(k[1] for k in out.keys()) == {"depthLow"}
    assert set(k[0] for k in out.keys()) == {"TFT", "SELF", "POP", "WSLS"}
    for opponent in ("TFT", "SELF", "POP", "WSLS"):
        assert len(out[(opponent, "depthLow")]) == 4  # 2 genotypes x probe_repeats=2


def test_recovery_times_by_depth_skips_genotypes_outside_every_range():
    # depth_group_lookup(2, 3, 6) only covers effective depths 0-6; a
    # genotype effectively deeper than max_depth is skipped entirely.
    table = tit_for_tat_table()
    census = [(0, 1, table)]
    narrow_lookup = {}  # no depth covered at all
    rng = random.Random(0)
    out = mod.recovery_times_by_depth_for_seed_age(census, probe_repeats=2, free_rounds=5, noise_rate=0.0, rng=rng, depth_group_of_depth=narrow_lookup)
    assert out == {}


def test_recovery_times_by_depth_pop_partner_drawn_from_full_population_regardless_of_depth_group():
    # POP partners come from the WHOLE census, not filtered by the focal
    # genotype's own depth group -- only the key under which results are
    # recorded depends on the focal's depth.
    shallow = tit_for_tat_table()  # effective depth 1 (decision depends on opponent's last bit)
    constant = always_defect_table()  # effective depth 0 (table is constant, folds to depth 0)
    census = [(0, 1, shallow), (1, 1, constant)]
    depth_group_of_depth = {0: "depthLow", 1: "depthLow"}  # both effective depths land in the same group
    rng = random.Random(0)
    out = mod.recovery_times_by_depth_for_seed_age(census, probe_repeats=3, free_rounds=5, noise_rate=0.0, rng=rng, depth_group_of_depth=depth_group_of_depth)
    assert len(out[("POP", "depthLow")]) == 6  # both genotypes qualify, 2 x probe_repeats=3


def test_recovery_times_by_depth_is_deterministic_given_the_same_rng_seed():
    census = [(0, 2, tit_for_tat_table()), (1, 1, always_defect_table())]
    lookup = mod.depth_group_lookup(2, 3, 6)
    out_a = mod.recovery_times_by_depth_for_seed_age(census, 2, 5, 0.2, random.Random(7), lookup)
    out_b = mod.recovery_times_by_depth_for_seed_age(census, 2, 5, 0.2, random.Random(7), lookup)
    assert out_a == out_b


# ------------------------------------------------------------------ summarize_recovery_times_by_depth

def test_summarize_recovery_times_by_depth_computes_median_and_rate_per_key():
    pooled = {
        ("TFT", "depthLow"): [2, 4, None, 2],
        ("WSLS", "depthHigh"): [None, None],
    }
    rows = mod.summarize_recovery_times_by_depth(1000, pooled)
    by_key = {(r["opponent"], r["depth_group"]): r for r in rows}
    assert by_key[("TFT", "depthLow")]["n"] == 4
    assert by_key[("TFT", "depthLow")]["median_recovery_round"] == 2
    assert by_key[("TFT", "depthLow")]["never_recovered_rate"] == round(1 / 4, 4)
    assert by_key[("WSLS", "depthHigh")]["n_recovered"] == 0
    assert by_key[("WSLS", "depthHigh")]["never_recovered_rate"] == 1.0


def test_summarize_recovery_times_by_depth_skips_empty_keys():
    rows = mod.summarize_recovery_times_by_depth(1000, {("TFT", "depthLow"): [1], ("SELF", "depthMid"): []})
    assert {(r["opponent"], r["depth_group"]) for r in rows} == {("TFT", "depthLow")}


# ------------------------------------------------------------------ process_one_age (integration)

def test_process_one_age_combines_census_and_recovery_computation(tmp_path):
    data_dir = tmp_path / "data"
    write_census(mod.reprobe_mod.census_path(data_dir, 0, 1000), [
        (0, 1, 5, "1,0,0,1", "1,1"),  # pavlov/WSLS rule, depth1, frequency=5
        (1, 1, 3, "0,1,0,1", "1,1"),  # TFT, depth1, frequency=3
    ])
    result = mod.process_one_age(str(data_dir), 1000, [0], probe_repeats=2, free_rounds=5, noise_rate=0.0, d1=2, d2=3, max_depth=6, master_seed="t")
    assert result["age"] == 1000
    assert result["wsls_rule_row"] == {"age": 1000, "n_depth1": 8, "n_wsls_rule": 5, "fraction": round(5 / 8, 4)}
    assert len(result["recovery_rows"]) > 0
    assert all(r["depth_group"] == "depthLow" for r in result["recovery_rows"])  # both genotypes are depth1 -> effective depth 1 -> depthLow


def test_process_one_age_skips_seed_with_no_census(tmp_path):
    data_dir = tmp_path / "data"
    result = mod.process_one_age(str(data_dir), 1000, [0], probe_repeats=1, free_rounds=5, noise_rate=0.0, d1=2, d2=3, max_depth=6, master_seed="t")
    assert result["recovery_rows"] == []
    assert result["wsls_rule_row"] == {"age": 1000, "n_depth1": 0, "n_wsls_rule": 0, "fraction": ""}
