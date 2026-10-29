import csv
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

_mod_spec = importlib.util.spec_from_file_location("export_k_model_equal_group_size_circuits", SCRIPTS_DIR / "export_k_model_equal_group_size_circuits.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["export_k_model_equal_group_size_circuits"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, n_seeds=8, n_genomes=14, age=1000):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        pilot_mod.run_probes_for_age(
            genomes, age=age, seed=seed, out_dir=data_dir,
            probe_repeats=3, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}",
        )
    return data_dir


def median_thresholds(data_dir, n_seeds, age):
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


def test_compute_metrics_handles_missing_opponent():
    result = mod.compute_metrics({"TFT": None})
    assert result["TFT_widest_底底底-和和和"] is None


def test_compute_metrics_computes_widest_and_jstar():
    edges = {
        "TFT": {("底底底", "底底底"): 1.0, ("底底底", "和和和"): 0.5, ("和和和", "和和和"): 1.0},
    }
    result = mod.compute_metrics({"TFT": edges["TFT"], "Gradual": None, "AllC": None, "AllD": None})
    assert result["TFT_widest_底底底-和和和"] == 0.5


def test_process_one_age_smoke(tmp_path):
    n_seeds = 8
    age = 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, age=age)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, age)
    seeds = list(range(n_seeds))

    result = mod.process_one_age(str(data_dir), age, seeds, threshold_kout, threshold_kself, n_draws=3, master_seed="test")
    assert result["age"] == age
    rows = result["rows"]
    assert len(rows) == len(mod.GROUPS) * len(mod.METRICS)
    for row in rows:
        assert row["group"] in mod.GROUPS
        assert row["shared46_group_size"] <= mod.SHARED_TARGET
        assert row["human_matched_group_size"] <= mod.HUMAN_MATCHED_TARGET[row["group"]]


def test_main_writes_csv(tmp_path):
    import json
    n_seeds = 8
    age = 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, age=age)
    threshold_kout, threshold_kself = median_thresholds(data_dir, n_seeds, age)
    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"threshold_kout": threshold_kout, "threshold_kself": threshold_kself}, f)

    out_csv = tmp_path / "out" / "k_model_equal_group_size_circuits.csv"
    sys.modules["export_k_model_equal_group_size_circuits"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "export_k_model_equal_group_size_circuits.py",
        "--data-dir", str(data_dir), "--seeds", str(n_seeds), "--seed-offset", "0",
        "--thresholds-json", str(thresholds_path),
        "--window-start", str(age), "--window-end", str(age),
        "--n-draws", "3", "--workers", "1",
        "--out-csv", str(out_csv),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(mod.GROUPS) * len(mod.METRICS)
    assert set(rows[0].keys()) == {"age", "group", "metric", "shared46_group_size", "shared46_value", "human_matched_group_size", "human_matched_value"}
