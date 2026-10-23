import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_coupling_markov_pilot", REPO_ROOT / "scripts" / "run_coupling_markov_pilot.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_coupling_markov_pilot"] = mod
spec.loader.exec_module(mod)

TINY_KWARGS = dict(
    population_size=8, generations=10, rounds_per_duel=6, initial_memory_depth=1,
    point_mutation_rate=0.1, duplication_rate=0.05, split_rate=0.02,
    max_memory_depth=3, min_memory_depth=0, k_mutation_std=0.05,
    evolution_noise_std=0.05, probe_start=0, probe_interval=5,
    probe_repeats=2, probe_rounds=10, free_rounds=5, probe_noise_std=0.05,
    master_seed="test_v1",
)


def test_atomic_pickle_dump_replaces_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "seed_0_checkpoint.pkl"
    mod._atomic_pickle_dump(path, {"gen": 5})
    mod._atomic_pickle_dump(path, {"gen": 10})
    with open(path, "rb") as f:
        assert pickle.load(f)["gen"] == 10
    assert [p.name for p in tmp_path.iterdir()] == ["seed_0_checkpoint.pkl"]


# ------------------------------------------------------------------ write_genotype_census

def test_write_genotype_census_round_trips(tmp_path):
    from evopersonality_abm.coupling import Genome
    from evopersonality_abm.lindgren import tit_for_tat_table

    g = Genome(table=tit_for_tat_table(), k_out=0.3, k_self=0.7)
    census = [((1, (0, 1, 0, 1), (1, 1), 0.3, 0.7), 2, g)]
    path = tmp_path / "census.csv.gz"
    mod.write_genotype_census(path, census)

    import csv
    import gzip
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["frequency"] == "2"
    assert rows[0]["table"] == "0,1,0,1"
    assert float(rows[0]["k_out"]) == 0.3
    assert float(rows[0]["k_self"]) == 0.7


# ------------------------------------------------------------------ run_probes_for_age

def test_run_probes_for_age_covers_all_8_opponents(tmp_path):
    import random

    from evopersonality_abm.coupling import random_genome

    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(5)]
    mod.run_probes_for_age(genomes, age=0, seed=0, out_dir=tmp_path, probe_repeats=1, probe_rounds=5, free_rounds=3, probe_noise_std=0.05, master_seed="t")

    import csv
    import gzip
    with gzip.open(mod.moves_path(tmp_path, 0, 0), "rt") as f:
        moves = list(csv.DictReader(f))
    with gzip.open(mod.recovery_path(tmp_path, 0, 0), "rt") as f:
        recovery = list(csv.DictReader(f))

    assert {r["opponent"] for r in moves} == set(mod.ALL_PROBE_OPPONENTS)
    assert {r["opponent"] for r in recovery} == set(mod.ALL_PROBE_OPPONENTS)
    assert mod.ALL_PROBE_OPPONENTS == ["TFT", "AllC", "AllD", "Random", "Gradual", "WSLS", "SELF", "POP"]


def test_run_probes_for_age_moves_have_5_rounds_each():
    import random

    from evopersonality_abm.coupling import random_genome
    import tempfile

    rng = random.Random(0)
    genomes = [random_genome(1, rng) for _ in range(3)]
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        mod.run_probes_for_age(genomes, age=0, seed=0, out_dir=out_dir, probe_repeats=1, probe_rounds=5, free_rounds=3, probe_noise_std=0.05, master_seed="t")
        import csv
        import gzip
        with gzip.open(mod.moves_path(out_dir, 0, 0), "rt") as f:
            moves = list(csv.DictReader(f))
        tft_rows = [r for r in moves if r["opponent"] == "TFT" and r["genotype_id"] == "0" and r["repeat"] == "0"]
        assert len(tft_rows) == 5
        assert [r["round"] for r in tft_rows] == ["1", "2", "3", "4", "5"]


# ------------------------------------------------------------------ run_one_seed resumability

def test_run_one_seed_interrupted_and_resumed_matches_uninterrupted_run(tmp_path, monkeypatch):
    """Same resumability guarantee as run_lindgren_markov_sweep.py's own
    test: a seed run straight through vs. one genuinely interrupted
    partway (simulated by making round_robin_payoffs_coupled raise after
    a few generations, past at least one probe_interval=5 checkpoint)
    and then resumed, must produce a bit-identical final
    generation_stats.csv."""
    straight_dir = tmp_path / "straight"
    resumed_dir = tmp_path / "resumed"
    straight_dir.mkdir()
    resumed_dir.mkdir()

    mod.run_one_seed(seed=0, out_dir=straight_dir, **TINY_KWARGS)

    real_round_robin = mod.round_robin_payoffs_coupled
    call_count = {"n": 0}

    def crashing_round_robin(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > 6:  # past the probe_interval=5 checkpoint, before generations=10 finishes
            raise RuntimeError("simulated crash")
        return real_round_robin(*args, **kwargs)

    monkeypatch.setattr(mod, "round_robin_payoffs_coupled", crashing_round_robin)
    with pytest.raises(RuntimeError, match="simulated crash"):
        mod.run_one_seed(seed=0, out_dir=resumed_dir, **TINY_KWARGS)
    assert mod.checkpoint_path_for(resumed_dir, 0).exists()

    monkeypatch.setattr(mod, "round_robin_payoffs_coupled", real_round_robin)
    mod.run_one_seed(seed=0, out_dir=resumed_dir, **TINY_KWARGS)
    assert not mod.checkpoint_path_for(resumed_dir, 0).exists()  # deleted on completion

    straight_csv = (straight_dir / "seed_0_generation_stats.csv").read_text()
    resumed_csv = (resumed_dir / "seed_0_generation_stats.csv").read_text()
    assert straight_csv == resumed_csv


def test_run_one_seed_writes_census_at_every_probe_age_including_final():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        mod.run_one_seed(seed=0, out_dir=out_dir, **TINY_KWARGS)
        # probe_start=0, probe_interval=5, generations=10 -> ages 0, 5, 10
        for age in (0, 5, 10):
            assert mod.census_path(out_dir, 0, age).exists(), age
