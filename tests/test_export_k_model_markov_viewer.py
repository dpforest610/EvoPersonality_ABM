import csv
import gzip
import importlib.util
import json
import random
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
    assert mod.classify_type(k_out=0.7, k_self=0.7, threshold_kout=0.5, threshold_kself=0.5) == "K_Secure"
    assert mod.classify_type(k_out=0.3, k_self=0.7, threshold_kout=0.5, threshold_kself=0.5) == "K_Dismissing"
    assert mod.classify_type(k_out=0.7, k_self=0.3, threshold_kout=0.5, threshold_kself=0.5) == "K_Preoccupied"
    assert mod.classify_type(k_out=0.3, k_self=0.3, threshold_kout=0.5, threshold_kself=0.5) == "K_Fearful"


def test_classify_type_exact_threshold_goes_to_high_side():
    assert mod.classify_type(k_out=0.5, k_self=0.5, threshold_kout=0.5, threshold_kself=0.5) == "K_Secure"


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


def test_load_series_first_rounds_matches_naive_loader(tmp_path):
    path = tmp_path / "moves.csv.gz"
    rows = []
    for rnd in range(1, 101):
        rows.append((0, "TFT", 0, rnd, "C", "D"))
    write_moves(path, rows)
    from evopersonality_abm.k_model_cross_lineage import load_series_first_rounds
    assert load_series_first_rounds(path, 20) == mod.load_series_by_genotype_all_opponents(path, 20)


# ------------------------------------------------------------------ census_true_types / discover_ages

def test_census_true_types_classifies_every_genotype():
    census = [
        (0, 2, 1, "0,1,0,1", "1,1", 0.7, 0.7),  # K_Secure
        (1, 1, 1, "1,0,1,0", "0,0", 0.3, 0.7),  # K_Dismissing
    ]
    types = mod.census_true_types(census, threshold_kout=0.5, threshold_kself=0.5)
    assert types == {0: "K_Secure", 1: "K_Dismissing"}


def test_discover_ages_across_seeds(tmp_path):
    write_census(mod.pilot_mod.census_path(tmp_path, 0, 1000), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    write_census(mod.pilot_mod.census_path(tmp_path, 1, 1050), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    assert mod.discover_ages(tmp_path) == [1000, 1050]


# ------------------------------------------------------------------ summarize_recovery

def test_summarize_recovery_computes_rate_and_median():
    row = mod.summarize_recovery(1000, "TFT", "K_Secure", [2, 4, None, 2])
    assert row["n"] == 4
    assert row["n_recovered"] == 3
    assert row["median_recovery_round"] == 2
    assert row["never_recovered_rate"] == round(1 / 4, 4)


def test_summarize_recovery_returns_none_for_empty_list():
    assert mod.summarize_recovery(1000, "TFT", "K_Secure", []) is None


# ------------------------------------------------------------------ build_human_section

def test_build_human_section_reads_real_human_weights_csv():
    section = mod.build_human_section(REPO_ROOT / "docs" / "human_network_weights.csv")
    assert "TFT" in section
    assert "Secure" in section["TFT"]
    assert "Mix" in section["TFT"]
    assert "Fearful" in section["TFT"]
    assert all(len(row) == 3 for row in section["TFT"]["Secure"])


# ------------------------------------------------------------------ cross-lineage sampling helpers (opponent-independent)

def test_average_edges_over_draws_returns_none_for_zero_group_size():
    assert mod.average_edges_over_draws({}, frozenset(), 0, 10, "TFT", {}, random.Random(0)) is None


def test_node_appearance_rates_over_draws_returns_none_for_zero_group_size():
    assert mod.node_appearance_rates_over_draws({}, frozenset(), 0, 10, "TFT", {}, random.Random(0)) is None


# ------------------------------------------------------------------ end-to-end smoke (process_one_age)

def test_process_one_age_end_to_end_tiny(tmp_path):
    from evopersonality_abm.coupling import random_genome

    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(10)]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import importlib

    pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py")
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    pilot_spec.loader.exec_module(pilot_module)

    for seed in (0, 1, 2, 3, 4, 5):
        g = [random_genome(1, random.Random(seed)) for _ in range(10)]
        pilot_module.run_probes_for_age(g, age=1000, seed=seed, out_dir=data_dir, probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}")

    k_outs, k_selfs = [], []
    for seed in range(6):
        c = mod.load_census_with_k(mod.pilot_mod.census_path(data_dir, seed, 1000))
        k_outs.extend(r[5] for r in c)
        k_selfs.extend(r[6] for r in c)
    k_outs.sort()
    k_selfs.sort()
    threshold_kout = k_outs[len(k_outs) // 2]
    threshold_kself = k_selfs[len(k_selfs) // 2]

    result = mod.process_one_age(data_dir, 1000, list(range(6)), threshold_kout, threshold_kself, n_draws=3, master_seed="test")
    assert result["age"] == 1000
    assert len(result["type_count_rows"]) == 4  # K_Secure/K_Dismissing/K_Preoccupied/K_Fearful
    assert sum(r["n_individuals"] for r in result["type_count_rows"]) == 60  # 10 genomes x 6 seeds
    assert set(result["edges_by_opponent"].keys()) <= set(mod.ALL_OPPONENTS)
    assert set(result["effective_group_size"].keys()) == set(mod.GROUPINGS)
    for grouping, gs in result["effective_group_size"].items():
        assert gs <= mod.TARGET_GROUP_SIZE[grouping]


def test_main_writes_all_output_files_with_human_section(tmp_path):
    from evopersonality_abm.coupling import random_genome

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    import importlib
    pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py")
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    pilot_spec.loader.exec_module(pilot_module)
    for seed in (0, 1, 2):
        g = [random_genome(1, random.Random(seed)) for _ in range(10)]
        pilot_module.run_probes_for_age(g, age=1000, seed=seed, out_dir=data_dir, probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}")

    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"threshold_kout": 0.5, "threshold_kself": 0.5}, f)

    out_dir = tmp_path / "out"
    # ProcessPoolExecutor pickles process_one_age by (module name, qualname)
    # and re-resolves it through sys.modules in the worker -- another test
    # file loading a script under this SAME ad-hoc module name (several do)
    # can overwrite the registry entry between collection and this test's
    # own execution, breaking that identity check. Re-assert it right
    # before the call that actually triggers multiprocessing.
    sys.modules["export_k_model_markov_viewer"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "export_k_model_markov_viewer.py",
        "--data-dir", str(data_dir), "--seeds", "3", "--seed-offset", "0",
        "--thresholds-json", str(thresholds_path), "--n-draws", "3", "--workers", "1",
        "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    for name in [
        "k_model_type_counts.csv", "k_model_widest_path.csv", "k_model_recovery_time.csv",
        "k_model_effective_group_size.csv", "k_model_edges.json", "k_model_nodes.json",
    ]:
        assert (out_dir / name).exists(), name

    with open(out_dir / "k_model_edges.json") as f:
        edges = json.load(f)
    assert edges["caveat"] == mod.CAVEAT_TEXT
    assert "human" in edges
    assert "TFT" in edges["human"]

    with open(out_dir / "k_model_nodes.json") as f:
        nodes = json.load(f)
    assert nodes["caveat"] == mod.CAVEAT_TEXT
    assert "human" not in nodes  # same limitation as the Lindgren-side export


def test_main_resumes_from_age_cache(tmp_path):
    """A second run with the same --out-dir should skip the already-cached
    age (no recomputation needed) -- same resumability contract as
    scripts/run_k_model_confirmatory_analysis.py."""
    from evopersonality_abm.coupling import random_genome

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import importlib
    pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py")
    pilot_module = importlib.util.module_from_spec(pilot_spec)
    pilot_spec.loader.exec_module(pilot_module)
    g = [random_genome(1, random.Random(0)) for _ in range(10)]
    pilot_module.run_probes_for_age(g, age=1000, seed=0, out_dir=data_dir, probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed="t0")

    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"threshold_kout": 0.5, "threshold_kself": 0.5}, f)
    out_dir = tmp_path / "out"

    def run():
        sys.modules["export_k_model_markov_viewer"] = mod  # see comment in the test above
        sys_argv_backup = sys.argv
        sys.argv = [
            "export_k_model_markov_viewer.py",
            "--data-dir", str(data_dir), "--seeds", "1", "--seed-offset", "0",
            "--thresholds-json", str(thresholds_path), "--n-draws", "2", "--workers", "1",
            "--out-dir", str(out_dir),
        ]
        try:
            mod.main()
        finally:
            sys.argv = sys_argv_backup

    run()
    cache_file = out_dir / "_age_cache" / "age_1000.pkl"
    assert cache_file.exists()
    mtime_before = cache_file.stat().st_mtime_ns
    run()
    assert cache_file.stat().st_mtime_ns == mtime_before  # untouched -- skipped, not recomputed
