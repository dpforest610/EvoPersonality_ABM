import importlib.util
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", SCRIPTS_DIR / "run_coupling_markov_pilot.py")
pilot_mod = importlib.util.module_from_spec(_pilot_spec)
sys.modules["run_coupling_markov_pilot"] = pilot_mod
_pilot_spec.loader.exec_module(pilot_mod)

_confirmatory_spec = importlib.util.spec_from_file_location("run_k_model_confirmatory_analysis", SCRIPTS_DIR / "run_k_model_confirmatory_analysis.py")
confirmatory_mod = importlib.util.module_from_spec(_confirmatory_spec)
sys.modules["run_k_model_confirmatory_analysis"] = confirmatory_mod
_confirmatory_spec.loader.exec_module(confirmatory_mod)

_mod_spec = importlib.util.spec_from_file_location("analyze_k_model_permutation_null_center", SCRIPTS_DIR / "analyze_k_model_permutation_null_center.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["analyze_k_model_permutation_null_center"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_cache(tmp_path, n_seeds=6, n_genomes=12, age=1000):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        pilot_mod.run_probes_for_age(
            genomes, age=age, seed=seed, out_dir=data_dir,
            probe_repeats=3, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}",
        )

    import csv
    import gzip
    k_outs, k_selfs = [], []
    for seed in range(n_seeds):
        with gzip.open(pilot_mod.census_path(data_dir, seed, age), "rt", newline="") as f:
            for row in csv.DictReader(f):
                k_outs.append(float(row["k_out"]))
                k_selfs.append(float(row["k_self"]))
    k_outs.sort()
    k_selfs.sort()
    threshold_kout = k_outs[len(k_outs) // 2]
    threshold_kself = k_selfs[len(k_selfs) // 2]

    seeds = list(range(n_seeds))
    seedboot_resamples = confirmatory_mod.build_seedboot_resamples(seeds, 3, "test")
    tiny_group_size = {"K_Secure": 2, "Mix": 2, "K_Fearful": 2}
    result = confirmatory_mod.process_one_age(
        str(data_dir), age, seeds, threshold_kout, threshold_kself,
        str(REPO_ROOT / "docs" / "human_network_weights.csv"),
        n_draws_main=2, seedboot_resamples=seedboot_resamples,
        n_permutations=3, permutation_draws_ha=2, master_seed="test",
        ha_group_size=tiny_group_size,
    )

    results_dir = tmp_path / "results"
    cache_dir = results_dir / "_age_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"age_{age}.pkl", "wb") as f:
        pickle.dump(result, f)
    return results_dir


def test_load_cached_ages_filters_by_window(tmp_path):
    results_dir = make_tiny_cache(tmp_path, age=1000)
    cache_dir = results_dir / "_age_cache"
    in_window = mod.load_cached_ages(cache_dir, 1000, 1000)
    out_of_window = mod.load_cached_ages(cache_dir, 1050, 1100)
    assert len(in_window) == 1
    assert len(out_of_window) == 0


def test_build_rows_contains_true_null_and_effect_for_every_comparison_with_data(tmp_path):
    results_dir = make_tiny_cache(tmp_path, age=1000)
    cache_dir = results_dir / "_age_cache"
    results = mod.load_cached_ages(cache_dir, 1000, 1000)
    rows = mod.build_rows(results)
    assert rows  # at least some comparisons produced data at this tiny scale
    for row in rows:
        assert row["effect_type_only"] == row["true_diff_window_avg"] - row["null_center"]
        assert row["n_permutations_used"] >= 1


def test_main_writes_csv(tmp_path):
    results_dir = make_tiny_cache(tmp_path, age=1000)
    out_csv = tmp_path / "out" / "permutation_null_center.csv"
    sys_argv_backup = sys.argv
    sys.argv = [
        "analyze_k_model_permutation_null_center.py",
        "--results-dir", str(results_dir),
        "--window-start", "1000", "--window-end", "1000",
        "--out-csv", str(out_csv),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup
    assert out_csv.exists()
    import csv
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert set(rows[0].keys()) == {"hypothesis", "comparison", "true_diff_window_avg", "null_center", "effect_type_only", "n_permutations_used"}
