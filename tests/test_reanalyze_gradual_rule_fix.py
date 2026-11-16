import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "reanalyze_gradual_rule_fix", REPO_ROOT / "scripts" / "reanalyze_gradual_rule_fix.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["reanalyze_gradual_rule_fix"] = mod
spec.loader.exec_module(mod)

reprobe_spec = importlib.util.spec_from_file_location(
    "reprobe_gradual_rule_fix", REPO_ROOT / "scripts" / "reprobe_gradual_rule_fix.py"
)
reprobe_mod = importlib.util.module_from_spec(reprobe_spec)
sys.modules["reprobe_gradual_rule_fix"] = reprobe_mod
reprobe_spec.loader.exec_module(reprobe_mod)


# ------------------------------------------------------------------ window_average_by_key

def test_window_average_by_key_filters_by_window_and_normalizes_group_size_to_str():
    rows = [
        {"opponent": "Gradual", "scheme": "B", "group_size": "56", "age": "1000", "mean_recovery": "0.2"},
        {"opponent": "Gradual", "scheme": "B", "group_size": "56", "age": "1050", "mean_recovery": "0.4"},
        {"opponent": "Gradual", "scheme": "B", "group_size": 56, "age": "900", "mean_recovery": "99.0"},  # outside window, int group_size
    ]
    out = mod.window_average_by_key(rows, (1000, 5000))
    key = ("Gradual", "B", "56", "mean_recovery")
    assert key in out
    assert abs(out[key] - 0.3) < 1e-9


def test_window_average_by_key_skips_nan_values():
    rows = [
        {"opponent": "AllC", "scheme": "B", "group_size": "56", "age": "1000", "mean_recovery": ""},
        {"opponent": "AllC", "scheme": "B", "group_size": "56", "age": "1050", "mean_recovery": "0.5"},
    ]
    out = mod.window_average_by_key(rows, (1000, 5000))
    assert out[("AllC", "B", "56", "mean_recovery")] == 0.5


# ------------------------------------------------------------------ write_merged_csv / load_csv_rows round trip

def test_write_merged_csv_combines_kept_and_new_rows(tmp_path):
    kept = [{"opponent": "TFT", "scheme": "B", "age": "1000", "group_size": "56", "mean_corr_Secure": "0.2"}]
    new = [{"opponent": "Gradual", "scheme": "B", "age": "1000", "group_size": 56, "mean_recovery": 0.1}]
    out_path = tmp_path / "merged.csv"
    mod.write_merged_csv(out_path, kept, new)

    rows = mod.load_csv_rows(out_path)
    assert len(rows) == 2
    opponents = {r["opponent"] for r in rows}
    assert opponents == {"TFT", "Gradual"}
    gradual_row = next(r for r in rows if r["opponent"] == "Gradual")
    assert gradual_row["group_size"] == "56"  # round-trips through CSV as a string
    assert gradual_row["mean_recovery"] == "0.1"


# ------------------------------------------------------------------ load_seed_pools_for_age_reprobe

def write_census(path, rows):
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        for row in rows:
            w.writerow(row)


def test_load_seed_pools_for_age_reprobe_reads_census_from_data_dir_and_moves_from_reprobe_dir(tmp_path):
    data_dir = tmp_path / "data"
    reprobe_dir = tmp_path / "reprobe"
    write_census(data_dir / "seed_0_probe_age_1000_census.csv.gz", [(0, 1, 5, "0,1,1,0", "1,0")])

    written = reprobe_mod.reprobe_one_seed_age(
        data_dir, reprobe_dir, seed=0, age=1000, probe_noise_rate=0.01,
        probe_repeats=2, probe_rounds=5, main_analysis_rounds=2, master_seed="test",
    )
    assert written is True

    pools = mod.load_seed_pools_for_age_reprobe(data_dir, reprobe_dir, 0.01, 1000, [0])
    assert set(pools.keys()) == {0}
    assert set(pools[0].keys()) == {"Gradual", "GradualAppeasement"}
    frequencies, series_by_genotype, identities = pools[0]["Gradual"]
    assert frequencies == {0: 5}
    assert 0 in series_by_genotype


def test_load_seed_pools_for_age_reprobe_skips_seed_missing_reprobe_data(tmp_path):
    data_dir = tmp_path / "data"
    reprobe_dir = tmp_path / "reprobe"
    write_census(data_dir / "seed_0_probe_age_1000_census.csv.gz", [(0, 1, 5, "0,1,1,0", "1,0")])
    # no reprobe written for seed 0 at this age
    pools = mod.load_seed_pools_for_age_reprobe(data_dir, reprobe_dir, 0.01, 1000, [0])
    assert pools == {}
