import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", SCRIPTS_DIR / "run_coupling_markov_pilot.py")
pilot_mod = importlib.util.module_from_spec(_pilot_spec)
sys.modules["run_coupling_markov_pilot"] = pilot_mod
_pilot_spec.loader.exec_module(pilot_mod)

_mod_spec = importlib.util.spec_from_file_location("plot_k_model_recovery_figures", SCRIPTS_DIR / "plot_k_model_recovery_figures.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["plot_k_model_recovery_figures"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, n_seeds=6, n_genomes=14, ages=(1000, 1050)):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        for age in ages:
            pilot_mod.run_probes_for_age(
                genomes, age=age, seed=seed, out_dir=data_dir,
                probe_repeats=3, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}_{age}",
            )
    return data_dir


def median_thresholds(data_dir, n_seeds, age):
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
    return k_outs[len(k_outs) // 2], k_selfs[len(k_selfs) // 2]


def test_load_age_data_skips_moves_and_returns_counts(tmp_path):
    n_seeds = 6
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, ages=ages)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, ages[0])
    seeds = list(range(n_seeds))

    true_type_by_seed, counts_by_gid = mod.load_age_data(data_dir, seeds, ages[0], threshold_kout, threshold_kself)
    assert set(true_type_by_seed.keys()) == set(seeds)
    assert set(counts_by_gid.keys()) == set(seeds)
    for seed_counts in counts_by_gid.values():
        for gid_counts in seed_counts.values():
            for key, (n, n_rec) in gid_counts.items():
                assert n > 0
                assert 0 <= n_rec <= n


def test_rate_for_weights_matches_manual_fraction(tmp_path):
    n_seeds = 6
    ages = (1000,)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, ages=ages)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, ages[0])
    seeds = list(range(n_seeds))

    true_type_by_seed, counts_by_gid = mod.load_age_data(data_dir, seeds, ages[0], threshold_kout, threshold_kself)
    for type_name in mod.TYPES:
        r = mod.rate_for_weights(counts_by_gid, true_type_by_seed, ("opp", "SELF"), type_name)
        if r is not None:
            assert 0.0 <= r <= 1.0


def test_build_figure_a_data_produces_points_for_every_age(tmp_path):
    n_seeds = 6
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, ages=ages)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, ages[0])
    seeds = list(range(n_seeds))

    points, bands = mod.build_figure_a_data(data_dir, seeds, list(ages), threshold_kout, threshold_kself, n_bootstrap_seed=5, master_seed="test")
    for type_name in mod.TYPES:
        assert len(points[type_name]) == len(ages)
        assert len(bands[type_name]) == len(ages)


def test_build_figure_b_data_produces_bars_for_available_strata(tmp_path):
    n_seeds = 6
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, ages=ages)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, ages[0])
    seeds = list(range(n_seeds))

    bars, bands = mod.build_figure_b_data(data_dir, seeds, list(ages), threshold_kout, threshold_kself, n_bootstrap_seed=5, master_seed="test")
    for key, val in bars.items():
        assert 0.0 <= val <= 1.0
        if key in bands:
            lo, hi = bands[key]
            assert lo <= hi


def test_main_writes_both_figures(tmp_path):
    import json
    n_seeds = 6
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, ages=ages)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, ages[0])
    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"threshold_kout": threshold_kout, "threshold_kself": threshold_kself}, f)

    out_dir = tmp_path / "figures"
    sys_argv_backup = sys.argv
    sys.argv = [
        "plot_k_model_recovery_figures.py",
        "--data-dir", str(data_dir), "--seeds", str(n_seeds), "--seed-offset", "0",
        "--thresholds-json", str(thresholds_path),
        "--window-start", str(ages[0]), "--window-end", str(ages[-1]),
        "--n-bootstrap-seed", "5",
        "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    assert (out_dir / "k_model_recovery_rate_by_generation_SELF.png").exists()
    assert (out_dir / "k_model_recovery_rate_by_depth_stratum_SELF.png").exists()
