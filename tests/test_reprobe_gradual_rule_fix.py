import csv
import gzip
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "reprobe_gradual_rule_fix", REPO_ROOT / "scripts" / "reprobe_gradual_rule_fix.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["reprobe_gradual_rule_fix"] = mod
spec.loader.exec_module(mod)


def write_census(path, rows):
    """rows: [(genotype_id, memory_depth, frequency, table_str, prehistory_str), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        for row in rows:
            w.writerow(row)


# ------------------------------------------------------------------ load_census_genotypes

def test_load_census_genotypes_round_trips_a_normal_table(tmp_path):
    path = tmp_path / "seed_0_probe_age_0_census.csv.gz"
    write_census(path, [(0, 1, 5, "0,1,1,0", "1,0")])
    out = mod.load_census_genotypes(path)
    assert len(out) == 1
    gid, freq, rep = out[0]
    assert gid == 0
    assert freq == 5
    assert rep.memory_depth == 1
    assert rep.table == (0, 1, 1, 0)
    assert rep.prehistory == (1, 0)


def test_load_census_genotypes_handles_memory_depth_zero_empty_prehistory(tmp_path):
    """memory_depth=0 has an empty prehistory column (2*0=0 entries) --
    "".split(",") would otherwise produce [''] and crash on int('')."""
    path = tmp_path / "seed_0_probe_age_0_census.csv.gz"
    write_census(path, [(0, 0, 3, "1", "")])
    out = mod.load_census_genotypes(path)
    gid, freq, rep = out[0]
    assert rep.memory_depth == 0
    assert rep.table == (1,)
    assert rep.prehistory == ()


# ------------------------------------------------------------------ reprobe_one_seed_age

def test_reprobe_one_seed_age_writes_only_gradual_opponents(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_census(mod.census_path(data_dir, 0, 0), [
        (0, 1, 5, "0,1,1,0", "1,0"),
        (1, 0, 3, "1", ""),
    ])

    written = mod.reprobe_one_seed_age(
        data_dir, out_dir, seed=0, age=0, probe_noise_rate=0.0,
        probe_repeats=2, probe_rounds=5, main_analysis_rounds=2, master_seed="test",
    )
    assert written is True

    mp = mod.reprobe_moves_path(out_dir, 0, 0, 0.0)
    sp = mod.reprobe_summary_path(out_dir, 0, 0, 0.0)
    assert mp.exists() and sp.exists()

    with gzip.open(mp, "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    opponents = {row["opponent"] for row in rows}
    assert opponents == {"Gradual", "GradualAppeasement"}
    # 2 genotypes x 2 opponents x 2 repeats x 5 rounds
    assert len(rows) == 2 * 2 * 2 * 5

    with open(sp, newline="") as f:
        summary_rows = list(csv.DictReader(f))
    assert {row["opponent"] for row in summary_rows} == {"Gradual", "GradualAppeasement"}
    assert len(summary_rows) == 2 * 2  # 2 genotypes x 2 opponents


def test_reprobe_one_seed_age_is_resumable(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_census(mod.census_path(data_dir, 0, 0), [(0, 1, 5, "0,1,1,0", "1,0")])

    first = mod.reprobe_one_seed_age(
        data_dir, out_dir, seed=0, age=0, probe_noise_rate=0.0,
        probe_repeats=2, probe_rounds=5, main_analysis_rounds=2, master_seed="test",
    )
    assert first is True

    second = mod.reprobe_one_seed_age(
        data_dir, out_dir, seed=0, age=0, probe_noise_rate=0.0,
        probe_repeats=2, probe_rounds=5, main_analysis_rounds=2, master_seed="test",
    )
    assert second is False  # already done -- skipped, not recomputed


def test_reprobe_one_seed_age_skips_when_census_missing(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    # no census written for (seed=0, age=0)
    written = mod.reprobe_one_seed_age(
        data_dir, out_dir, seed=0, age=0, probe_noise_rate=0.0,
        probe_repeats=1, probe_rounds=2, main_analysis_rounds=1, master_seed="test",
    )
    assert written is False
    assert not mod.reprobe_moves_path(out_dir, 0, 0, 0.0).exists()


# ------------------------------------------------------------------ reprobe_one_age

def test_reprobe_one_age_counts_written_skipped_and_missing(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    write_census(mod.census_path(data_dir, 0, 0), [(0, 1, 5, "0,1,1,0", "1,0")])
    # seed 1 has no census at this age -> should be counted as missing

    age, written, skipped_done, skipped_missing = mod.reprobe_one_age(
        str(data_dir), str(out_dir), [0, 1], age=0, probe_noise_rate=0.0,
        probe_repeats=1, probe_rounds=2, main_analysis_rounds=1, master_seed="test",
    )
    assert age == 0
    assert written == 1
    assert skipped_done == 0
    assert skipped_missing == 1

    # second pass: seed 0 is now already done
    _age, written2, skipped_done2, skipped_missing2 = mod.reprobe_one_age(
        str(data_dir), str(out_dir), [0, 1], age=0, probe_noise_rate=0.0,
        probe_repeats=1, probe_rounds=2, main_analysis_rounds=1, master_seed="test",
    )
    assert written2 == 0
    assert skipped_done2 == 1
    assert skipped_missing2 == 1
