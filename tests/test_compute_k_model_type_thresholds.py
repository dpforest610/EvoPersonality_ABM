import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "compute_k_model_type_thresholds", REPO_ROOT / "scripts" / "compute_k_model_type_thresholds.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["compute_k_model_type_thresholds"] = mod
spec.loader.exec_module(mod)


def write_census(path, rows):
    """rows: [(genotype_id, memory_depth, frequency, table, prehistory, k_out, k_self), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory", "k_out", "k_self"])
        for row in rows:
            w.writerow(row)


# ------------------------------------------------------------------ load_k_values

def test_load_k_values_repeats_by_frequency(tmp_path):
    path = tmp_path / "census.csv.gz"
    write_census(path, [
        (0, 1, 3, "0,1,0,1", "1,1", 0.3, 0.7),
        (1, 1, 1, "1,0,1,0", "0,0", 0.5, 0.5),
    ])
    values = mod.load_k_values(path)
    assert values.count((0.3, 0.7)) == 3
    assert values.count((0.5, 0.5)) == 1
    assert len(values) == 4


# ------------------------------------------------------------------ discover_ages

def test_discover_ages_finds_ages_across_seeds(tmp_path):
    write_census(mod.census_path(tmp_path, 100, 1000), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    write_census(mod.census_path(tmp_path, 101, 1050), [(0, 1, 1, "0,1,0,1", "1,1", 0.3, 0.7)])
    ages = mod.discover_ages(tmp_path, [100, 101])
    assert ages == [1000, 1050]


# ------------------------------------------------------------------ main (integration)

def test_main_computes_pooled_median_and_per_generation_csv(tmp_path, capsys):
    data_dir = tmp_path / "data"
    # seed 100: age 1000 has k_self values 0.1,0.2 ; age 1050 has 0.9
    write_census(mod.census_path(data_dir, 100, 1000), [
        (0, 1, 1, "0,1,0,1", "1,1", 0.1, 0.1),
        (1, 1, 1, "0,1,0,1", "1,1", 0.2, 0.2),
    ])
    write_census(mod.census_path(data_dir, 100, 1050), [
        (0, 1, 1, "0,1,0,1", "1,1", 0.9, 0.9),
    ])
    # seed 101: age 1000 has 0.8
    write_census(mod.census_path(data_dir, 101, 1000), [
        (0, 1, 1, "0,1,0,1", "1,1", 0.8, 0.8),
    ])

    out_dir = tmp_path / "out"
    sys_argv_backup = sys.argv
    sys.argv = [
        "compute_k_model_type_thresholds.py",
        "--data-dir", str(data_dir), "--seeds", "2", "--seed-offset", "100",
        "--min-generation", "1000", "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    with open(out_dir / "type_thresholds.json") as f:
        thresholds = json.load(f)
    # pooled k_self values across both ages/seeds: 0.1, 0.2, 0.9, 0.8 -> median 0.5
    assert thresholds["threshold_kself"] == 0.5
    assert thresholds["n_total"] == 4

    with open(out_dir / "type_thresholds_by_generation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    by_age = {int(r["age"]): r for r in rows}
    assert by_age[1000]["n"] == "3"  # 2 from seed100 + 1 from seed101
    assert float(by_age[1000]["median_kself"]) == 0.2  # median of [0.1, 0.2, 0.8]
    assert by_age[1050]["n"] == "1"
    assert float(by_age[1050]["median_kself"]) == 0.9


def test_main_min_generation_excludes_earlier_ages_from_pooled_json(tmp_path):
    data_dir = tmp_path / "data"
    write_census(mod.census_path(data_dir, 100, 500), [(0, 1, 1, "0,1,0,1", "1,1", 0.1, 0.1)])
    write_census(mod.census_path(data_dir, 100, 1000), [(0, 1, 1, "0,1,0,1", "1,1", 0.9, 0.9)])

    out_dir = tmp_path / "out"
    sys_argv_backup = sys.argv
    sys.argv = [
        "compute_k_model_type_thresholds.py",
        "--data-dir", str(data_dir), "--seeds", "1", "--seed-offset", "100",
        "--min-generation", "1000", "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    with open(out_dir / "type_thresholds.json") as f:
        thresholds = json.load(f)
    assert thresholds["n_total"] == 1  # only age 1000, not age 500
    assert thresholds["ages_pooled"] == [1000]

    # but the per-generation CSV still covers age 500 too (sensitivity variant, no min_generation filter)
    with open(out_dir / "type_thresholds_by_generation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {int(r["age"]) for r in rows} == {500, 1000}
