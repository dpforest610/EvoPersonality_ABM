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

_mod_spec = importlib.util.spec_from_file_location("k_alt", SCRIPTS_DIR / "exploratory_hbes2026" / "k_alt.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["k_alt"] = mod
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


def test_per_series_basic():
    alt_count, alt3_any, retaliation, longest = mod.per_series(["搾", "惨"] * 10)
    assert alt_count == 19.0
    assert alt3_any == 1.0
    assert retaliation == 1.0  # every "惨" is followed by "搾" here
    assert longest == 19.0


def test_main_writes_both_csvs(tmp_path):
    age = 1000
    data_dir, seeds = make_tiny_production_data(tmp_path, age=age)
    thresholds_path = write_thresholds(tmp_path, seeds)
    out_prefix = str(tmp_path / "out" / "k_alt")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    sys.modules["k_alt"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "k_alt.py",
        "--data-dir", str(data_dir), "--thresholds-json", str(thresholds_path),
        "--age-start", str(age), "--age-end", str(age), "--age-step", "50",
        "--workers", "1", "--n-boot", "20",
        "--out-prefix", out_prefix,
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    by_type_path = Path(f"{out_prefix}_by_type.csv")
    regression_path = Path(f"{out_prefix}_regression.csv")
    assert by_type_path.exists()
    assert regression_path.exists()

    with open(by_type_path, newline="") as f:
        by_type_rows = list(csv.DictReader(f))
    assert by_type_rows, "expected at least one by-type row"
    assert set(by_type_rows[0].keys()) == {"opponent", "metric", "group", "mean", "n_seeds", "ci_lo", "ci_hi"}
    for row in by_type_rows:
        assert row["opponent"] in set(mod.OPP.values())
        assert row["metric"] in mod.METRICS
        float(row["mean"])
        float(row["ci_lo"])
        float(row["ci_hi"])

    with open(regression_path, newline="") as f:
        reg_rows = list(csv.DictReader(f))
    assert set(reg_rows[0].keys()) == {"opponent", "metric", "term", "b", "n_seeds", "ci_lo", "ci_hi"}
    terms = {"切片", "不安相当(-K_self)", "回避相当(-K_out)", "積"}
    for row in reg_rows:
        assert row["term"] in terms
        float(row["b"])
