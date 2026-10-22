import csv
import gzip
import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "export_markov_selfplay_recovery", REPO_ROOT / "scripts" / "export_markov_selfplay_recovery.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["export_markov_selfplay_recovery"] = mod
spec.loader.exec_module(mod)

from evopersonality_abm.lindgren import always_defect_table, tit_for_tat_table  # noqa: E402


# ------------------------------------------------------------------ build_selfplay_series_for_seed_age

def test_build_selfplay_series_covers_every_genotype_with_probe_repeats_entries_each():
    census = [(0, 5, tit_for_tat_table()), (1, 2, always_defect_table())]
    rng = random.Random(0)
    self_series, pop_series = mod.build_selfplay_series_for_seed_age(
        census, probe_repeats=4, probe_rounds=10, main_analysis_rounds=5, noise_rate=0.0, rng=rng,
    )
    assert set(self_series.keys()) == {0, 1}
    assert set(pop_series.keys()) == {0, 1}
    assert all(len(v) == 4 for v in self_series.values())
    assert all(len(v) == 4 for v in pop_series.values())


def test_build_selfplay_series_self_mutual_tft_is_all_cooperate_series():
    census = [(0, 1, tit_for_tat_table())]
    rng = random.Random(0)
    self_series, _pop = mod.build_selfplay_series_for_seed_age(
        census, probe_repeats=1, probe_rounds=10, main_analysis_rounds=5, noise_rate=0.0, rng=rng,
    )
    # mutual TFT with no noise cooperates forever -> every 3-gram is 和和和
    # (main_analysis_rounds=5 -> sliding 3-grams at indices 2,3,4 -> 3 grams)
    assert self_series[0] == ["和和和,和和和,和和和。"]


def test_build_selfplay_series_is_deterministic_given_the_same_rng_seed():
    census = [(0, 3, tit_for_tat_table()), (1, 1, always_defect_table())]
    out_a = mod.build_selfplay_series_for_seed_age(census, 3, 10, 5, 0.2, random.Random(42))
    out_b = mod.build_selfplay_series_for_seed_age(census, 3, 10, 5, 0.2, random.Random(42))
    assert out_a == out_b


# ------------------------------------------------------------------ build_pools_for_age

def write_census(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        for row in rows:
            w.writerow(row)


def write_moves(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
        for row in rows:
            w.writerow(row)


def test_build_pools_for_age_includes_self_pop_and_wsls_when_data_present(tmp_path):
    data_dir = tmp_path / "data"
    write_census(mod.reprobe_mod.census_path(data_dir, 0, 1000), [(0, 1, 5, "0,1,0,1", "1,1")])
    moves_rows = [(0, "WSLS", 0, r, "C", "C") for r in range(1, 6)]
    write_moves(data_dir / "seed_0_probe_age_1000_noise0p01_moves.csv.gz", moves_rows)

    pools, recovery_times = mod.build_pools_for_age(data_dir, "noise0p01", 1000, [0], probe_repeats=2, probe_rounds=5, main_analysis_rounds=3, noise_rate=0.0, master_seed="t")
    assert set(pools.keys()) == {"SELF", "POP", "WSLS"}
    assert 0 in pools["SELF"] and 0 in pools["POP"] and 0 in pools["WSLS"]
    frequencies, series_by_genotype, identities = pools["SELF"][0]
    assert frequencies == {0: 5}
    assert 0 in series_by_genotype
    assert identities[0][0] == 1  # memory_depth
    assert set(recovery_times.keys()) == {"TFT", "SELF", "POP", "WSLS"}
    assert len(recovery_times["TFT"]) == 2  # probe_repeats, one genotype
    assert len(recovery_times["POP"]) == 2


def test_build_pools_for_age_skips_wsls_when_moves_file_missing(tmp_path):
    data_dir = tmp_path / "data"
    write_census(mod.reprobe_mod.census_path(data_dir, 0, 1000), [(0, 1, 5, "0,1,0,1", "1,1")])
    # no moves file written

    pools, recovery_times = mod.build_pools_for_age(data_dir, "noise0p01", 1000, [0], probe_repeats=1, probe_rounds=3, main_analysis_rounds=2, noise_rate=0.0, master_seed="t")
    assert 0 not in pools["WSLS"]
    assert 0 in pools["SELF"]  # SELF/POP don't need the moves file
    # recovery_time_from_forced_bottom's own WSLS opponent is the fixed
    # pavlov_table(), not the (possibly missing) production moves data
    assert len(recovery_times["WSLS"]) == 1


def test_build_pools_for_age_skips_seed_with_no_census(tmp_path):
    data_dir = tmp_path / "data"
    pools, recovery_times = mod.build_pools_for_age(data_dir, "noise0p01", 1000, [0], probe_repeats=1, probe_rounds=3, main_analysis_rounds=2, noise_rate=0.0, master_seed="t")
    assert pools == {"SELF": {}, "POP": {}, "WSLS": {}}
    assert recovery_times == {"TFT": [], "SELF": [], "POP": [], "WSLS": []}


# ------------------------------------------------------------------ draw_partner_table

def test_draw_partner_table_only_returns_tables_from_the_given_gids():
    gids = [0, 1]
    tables = {0: tit_for_tat_table(), 1: always_defect_table()}
    rng = random.Random(0)
    for _ in range(20):
        drawn = mod.draw_partner_table(gids, [1, 1], tables, rng)
        assert drawn is tables[0] or drawn is tables[1]


def test_draw_partner_table_is_deterministic_given_the_same_rng_seed():
    gids = [0, 1]
    tables = {0: tit_for_tat_table(), 1: always_defect_table()}
    a = mod.draw_partner_table(gids, [3, 1], tables, random.Random(5))
    b = mod.draw_partner_table(gids, [3, 1], tables, random.Random(5))
    assert a is b


# ------------------------------------------------------------------ recovery_times_for_seed_age

def test_recovery_times_for_seed_age_covers_all_four_opponents_with_probe_repeats_each():
    census = [(0, 1, tit_for_tat_table()), (1, 1, always_defect_table())]
    rng = random.Random(0)
    out = mod.recovery_times_for_seed_age(census, probe_repeats=3, free_rounds=10, noise_rate=0.0, rng=rng)
    assert set(out.keys()) == {"TFT", "SELF", "POP", "WSLS"}
    # 2 genotypes x 3 repeats each for TFT/SELF/WSLS/POP
    for opponent in ("TFT", "SELF", "WSLS", "POP"):
        assert len(out[opponent]) == 6


def test_recovery_times_for_seed_age_self_mutual_tft_never_recovers():
    # TFT only ever reciprocates, never initiates cooperation -- from a
    # forced-DD history both copies keep mirroring the other's last (D)
    # move forever, so mutual TFT is locked at the bottom (same structural
    # non-recovery as TFT-vs-focal in recovery_time_from_forced_bottom's
    # own docstring example, just with TFT on both sides this time).
    census = [(0, 1, tit_for_tat_table())]
    rng = random.Random(0)
    out = mod.recovery_times_for_seed_age(census, probe_repeats=1, free_rounds=10, noise_rate=0.0, rng=rng)
    assert out["SELF"][0] is None


# ------------------------------------------------------------------ summarize_recovery_times

def test_summarize_recovery_times_computes_median_and_never_recovered_rate():
    recovery_times = {"TFT": [2, 4, None, 2], "SELF": [None, None]}
    rows = mod.summarize_recovery_times(1000, recovery_times)
    by_opponent = {r["opponent"]: r for r in rows}
    assert by_opponent["TFT"]["n"] == 4
    assert by_opponent["TFT"]["n_recovered"] == 3
    assert by_opponent["TFT"]["median_recovery_round"] == 2
    assert by_opponent["TFT"]["never_recovered_rate"] == round(1 / 4, 4)
    assert by_opponent["SELF"]["n_recovered"] == 0
    assert by_opponent["SELF"]["median_recovery_round"] == ""
    assert by_opponent["SELF"]["never_recovered_rate"] == 1.0


def test_summarize_recovery_times_skips_opponents_with_zero_samples():
    rows = mod.summarize_recovery_times(1000, {"TFT": [1], "SELF": []})
    assert {r["opponent"] for r in rows} == {"TFT"}


# ------------------------------------------------------------------ write_widest_path_csv

def test_write_widest_path_csv_round_trips(tmp_path):
    rows = [
        {"opponent": "SELF", "age": 1000, "mean_recovery": 0.5},
        {"opponent": "POP", "age": 1000, "mean_recovery": 0.3, "rate_bottom3_present": 1.0},
    ]
    path = tmp_path / "out.csv"
    mod.write_widest_path_csv(path, rows)
    with open(path, newline="") as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 2
    assert {r["opponent"] for r in read_rows} == {"SELF", "POP"}


def test_write_widest_path_csv_handles_empty_rows(tmp_path):
    path = tmp_path / "out.csv"
    mod.write_widest_path_csv(path, [])
    assert path.exists()
