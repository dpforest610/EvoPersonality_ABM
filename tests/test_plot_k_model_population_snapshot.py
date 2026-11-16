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

_mod_spec = importlib.util.spec_from_file_location("plot_k_model_population_snapshot", SCRIPTS_DIR / "plot_k_model_population_snapshot.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["plot_k_model_population_snapshot"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, seed=0, n_genomes=20, ages=(1000, 3000, 5000)):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    genomes = [random_genome(1, rng) for _ in range(n_genomes)]
    for age in ages:
        pilot_mod.run_probes_for_age(
            genomes, age=age, seed=seed, out_dir=data_dir,
            probe_repeats=1, probe_rounds=5, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}_{age}",
        )
    return data_dir


def test_load_snapshot_returns_all_individuals_at_one_seed_age(tmp_path):
    n_genomes = 20
    data_dir = make_tiny_production_data(tmp_path, seed=0, n_genomes=n_genomes, ages=(1000,))
    values = mod.load_snapshot(data_dir, seed=0, age=1000)
    assert len(values) == n_genomes
    for k_out, k_self in values:
        assert 0.0 <= k_out <= 1.0
        assert 0.0 <= k_self <= 1.0


def test_load_snapshot_missing_file_raises():
    import pytest

    with pytest.raises(SystemExit):
        mod.load_snapshot(Path("/nonexistent"), seed=0, age=1000)


def test_plot_snapshots_writes_png(tmp_path):
    snapshots = [
        (1000, [(0.1, 0.2), (0.8, 0.9), (0.5, 0.5)]),
        (3000, [(0.3, 0.4), (0.6, 0.6)]),
        (5000, [(0.4, 0.5)]),
    ]
    out_path = tmp_path / "snapshot.png"
    mod.plot_snapshots(snapshots, seed=200, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_writes_png(tmp_path):
    data_dir = make_tiny_production_data(tmp_path, seed=0, n_genomes=20, ages=(1000, 3000, 5000))
    out_path = tmp_path / "out" / "snapshot.png"
    sys.modules["plot_k_model_population_snapshot"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "plot_k_model_population_snapshot.py",
        "--data-dir", str(data_dir), "--seed", "0",
        "--ages", "1000", "3000", "5000",
        "--out-path", str(out_path),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup
    assert out_path.exists()
