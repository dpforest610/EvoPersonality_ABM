import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "export_k_model_markov_viewer", REPO_ROOT / "scripts" / "export_k_model_markov_viewer.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["export_k_model_markov_viewer"] = mod
spec.loader.exec_module(mod)


# ------------------------------------------------------------------ classify_type

def test_classify_type_all_4_quadrants():
    # threshold_kout=0.5, threshold_kself=0.5
    assert mod.classify_type(k_out=0.7, k_self=0.7, threshold_kout=0.5, threshold_kself=0.5) == "White"
    assert mod.classify_type(k_out=0.3, k_self=0.7, threshold_kout=0.5, threshold_kself=0.5) == "Blue"
    assert mod.classify_type(k_out=0.7, k_self=0.3, threshold_kout=0.5, threshold_kself=0.5) == "Green"
    assert mod.classify_type(k_out=0.3, k_self=0.3, threshold_kout=0.5, threshold_kself=0.5) == "Pink"


def test_classify_type_exact_threshold_goes_to_high_side():
    assert mod.classify_type(k_out=0.5, k_self=0.5, threshold_kout=0.5, threshold_kself=0.5) == "White"


# ------------------------------------------------------------------ load_census_with_k / load_recovery

def write_census(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory", "k_out", "k_self"])
        for row in rows:
            w.writerow(row)


def write_moves(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
        w.writerows(rows)


def write_recovery(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "recovery_round"])
        w.writerows(rows)


def test_load_census_with_k_round_trips(tmp_path):
    path = tmp_path / "census.csv.gz"
    write_census(path, [(0, 1, 3, "0,1,0,1", "1,1", 0.3, 0.7)])
    census = mod.load_census_with_k(path)
    assert census == [(0, 3, 1, "0,1,0,1", "1,1", 0.3, 0.7)]


def test_load_recovery_parses_empty_string_as_none(tmp_path):
    path = tmp_path / "recovery.csv.gz"
    write_recovery(path, [(0, "TFT", 0, ""), (0, "TFT", 1, 3)])
    rows = mod.load_recovery(path)
    assert rows == [(0, "TFT", 0, None), (0, "TFT", 1, 3)]


# ------------------------------------------------------------------ build_seed_pool_for_type / merge_pools_across_seeds

def test_build_seed_pool_for_type_filters_by_classification():
    census = [
        (0, 2, 1, "0,1,0,1", "1,1", 0.7, 0.7),  # White
        (1, 1, 1, "1,0,1,0", "0,0", 0.3, 0.7),  # Blue
    ]
    series_by_genotype = {0: ["和和和,和和和。"], 1: ["底底底,底底底。"]}
    pool = mod.build_seed_pool_for_type(census, series_by_genotype, "White", threshold_kout=0.5, threshold_kself=0.5)
    frequencies, filtered_series, identities = pool
    assert frequencies == {0: 2}
    assert 1 not in filtered_series
    assert identities[0] == (1, "0,1,0,1", "1,1")


def test_build_seed_pool_for_type_skips_genotype_with_no_series():
    census = [(0, 5, 1, "0,1,0,1", "1,1", 0.7, 0.7)]
    pool = mod.build_seed_pool_for_type(census, {}, "White", threshold_kout=0.5, threshold_kself=0.5)
    assert pool == ({}, {}, {})


def test_merge_pools_across_seeds_remaps_to_globally_unique_ids():
    pool_a = ({0: 1}, {0: ["s1"]}, {0: (1, "t", "p")})
    pool_b = ({0: 1}, {0: ["s2"]}, {0: (1, "t", "p")})  # same local id 0, different seed
    merged = mod.merge_pools_across_seeds([pool_a, pool_b])
    frequencies, series_by_genotype, identities = merged
    assert len(frequencies) == 2  # not collapsed despite colliding local ids
    assert sorted(v[0] for v in series_by_genotype.values()) == ["s1", "s2"]


def test_pool_size_counts_distinct_genotypes():
    pool = ({0: 5, 1: 3}, {0: ["s"], 1: ["s"]}, {0: (1, "t", "p"), 1: (1, "t", "p")})
    assert mod.pool_size(pool) == 2


# ------------------------------------------------------------------ summarize_recovery

def test_summarize_recovery_computes_rate_and_median():
    row = mod.summarize_recovery(1000, "TFT", "White", [2, 4, None, 2])
    assert row["n"] == 4
    assert row["n_recovered"] == 3
    assert row["median_recovery_round"] == 2
    assert row["never_recovered_rate"] == round(1 / 4, 4)


def test_summarize_recovery_returns_none_for_empty_list():
    assert mod.summarize_recovery(1000, "TFT", "White", []) is None


# ------------------------------------------------------------------ discover_ages

def test_discover_ages_across_seeds(tmp_path):
    write_census(mod.pilot_mod.census_path(tmp_path, 0, 1000), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    write_census(mod.pilot_mod.census_path(tmp_path, 1, 1050), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    assert mod.discover_ages(tmp_path) == [1000, 1050]


# ------------------------------------------------------------------ end-to-end smoke (process_one_age)

def test_process_one_age_end_to_end_tiny(tmp_path):
    import random

    from evopersonality_abm.coupling import random_genome
    from evopersonality_abm.k_model_probe import census_genomes

    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(10)]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import importlib

    pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py")
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    pilot_spec.loader.exec_module(pilot_module)

    for seed in (0, 1):
        pilot_module.run_probes_for_age(genomes, age=1000, seed=seed, out_dir=data_dir, probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}")

    k_values = [g.k_out for g in genomes] + [g.k_self for g in genomes]
    threshold_kout = sorted(g.k_out for g in genomes)[len(genomes) // 2]
    threshold_kself = sorted(g.k_self for g in genomes)[len(genomes) // 2]

    result = mod.process_one_age(data_dir, 1000, [0, 1], threshold_kout, threshold_kself, group_size=3, n_draws=3, master_seed="test")
    assert result["age"] == 1000
    assert len(result["type_count_rows"]) == 4  # White/Blue/Green/Pink
    assert sum(r["n_individuals"] for r in result["type_count_rows"]) == 20  # 10 genomes x 2 seeds
    assert set(result["edges_by_grouping"].keys()) <= set(mod.ALL_OPPONENTS)


def test_main_writes_all_5_output_files(tmp_path):
    import random

    from evopersonality_abm.coupling import random_genome

    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(10)]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    import importlib
    pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py")
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    pilot_spec.loader.exec_module(pilot_module)
    pilot_module.run_probes_for_age(genomes, age=1000, seed=0, out_dir=data_dir, probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed="t")

    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"threshold_kout": 0.5, "threshold_kself": 0.5}, f)

    out_dir = tmp_path / "out"
    sys_argv_backup = sys.argv
    sys.argv = [
        "export_k_model_markov_viewer.py",
        "--data-dir", str(data_dir), "--seeds", "1", "--seed-offset", "0",
        "--thresholds-json", str(thresholds_path), "--group-size", "3", "--n-draws", "3",
        "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    for name in ["k_model_type_counts.csv", "k_model_widest_path.csv", "k_model_recovery_time.csv", "k_model_edges.json", "k_model_nodes.json"]:
        assert (out_dir / name).exists(), name

    with open(out_dir / "k_model_edges.json") as f:
        edges = json.load(f)
    assert edges["caveat"] == mod.CAVEAT_TEXT
