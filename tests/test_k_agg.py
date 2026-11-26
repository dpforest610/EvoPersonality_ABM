import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", SCRIPTS_DIR / "run_coupling_markov_pilot.py")
pilot_mod = importlib.util.module_from_spec(_pilot_spec)
sys.modules["run_coupling_markov_pilot"] = pilot_mod
_pilot_spec.loader.exec_module(pilot_mod)

_mod_spec = importlib.util.spec_from_file_location("k_agg", SCRIPTS_DIR / "exploratory_hbes2026" / "k_agg.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["k_agg"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, n_seeds=6, n_genomes=16, age=1000):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        pilot_mod.run_probes_for_age(
            genomes, age=age, seed=seed, out_dir=data_dir,
            probe_repeats=2, probe_rounds=20, free_rounds=3, probe_noise_std=0.05, master_seed=f"t{seed}",
        )
    return data_dir, list(range(n_seeds))


def write_thresholds(tmp_path, seeds, threshold_kout=0.5, threshold_kself=0.5):
    path = tmp_path / "type_thresholds.json"
    with open(path, "w") as f:
        json.dump({"threshold_kout": threshold_kout, "threshold_kself": threshold_kself, "seeds": seeds}, f)
    return path


def test_classify_and_metrics_basic():
    assert mod.classify(0.9, 0.9, 0.5, 0.5) == "K_Secure"
    assert mod.classify(0.1, 0.9, 0.5, 0.5) == "K_Dismissing"
    assert mod.classify(0.9, 0.1, 0.5, 0.5) == "K_Preoccupied"
    assert mod.classify(0.1, 0.1, 0.5, 0.5) == "K_Fearful"

    coop, entry, reach = mod.metrics(["和"] * 20)
    assert coop == 1.0
    assert reach is None  # no "底" ever appears


def test_main_writes_well_formed_csv(tmp_path):
    age = 1000
    data_dir, seeds = make_tiny_production_data(tmp_path, age=age)
    thresholds_path = write_thresholds(tmp_path, seeds)
    out_path = tmp_path / "k_by_type_opponent.csv"

    sys.modules["k_agg"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "k_agg.py",
        "--data-dir", str(data_dir), "--thresholds-json", str(thresholds_path),
        "--age-start", str(age), "--age-end", str(age), "--age-step", "50",
        "--workers", "1", "--n-boot", "20",
        "--out", str(out_path),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    assert out_path.exists()
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "expected at least one aggregated row"
    assert set(rows[0].keys()) == {"opponent", "metric", "group", "mean", "n_seeds", "ci_lo", "ci_hi"}
    for row in rows:
        assert row["opponent"] in set(mod.OPP.values())
        assert row["metric"] in mod.METRICS
        assert row["group"] in mod.TYPES
        float(row["mean"])  # parses as a number
        float(row["ci_lo"])
        float(row["ci_hi"])
