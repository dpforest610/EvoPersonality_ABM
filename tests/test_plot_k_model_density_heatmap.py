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

_mod_spec = importlib.util.spec_from_file_location("plot_k_model_density_heatmap", SCRIPTS_DIR / "plot_k_model_density_heatmap.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["plot_k_model_density_heatmap"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, n_seeds=4, n_genomes=20, ages=(1000, 1050)):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        for age in ages:
            pilot_mod.run_probes_for_age(
                genomes, age=age, seed=seed, out_dir=data_dir,
                probe_repeats=1, probe_rounds=5, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}_{age}",
            )
    return data_dir


def test_load_pooled_k_values_pools_across_seeds_and_ages(tmp_path):
    n_seeds, n_genomes = 4, 20
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=n_genomes, ages=ages)
    seeds = list(range(n_seeds))
    values = mod.load_pooled_k_values(data_dir, seeds, min_generation=1000, max_generation=10**9)
    assert len(values) == n_seeds * n_genomes * len(ages)
    for k_out, k_self in values:
        assert 0.0 <= k_out <= 1.0
        assert 0.0 <= k_self <= 1.0


def test_load_pooled_k_values_respects_min_generation(tmp_path):
    n_seeds, n_genomes = 4, 20
    ages = (1000, 1050)
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=n_genomes, ages=ages)
    seeds = list(range(n_seeds))
    values = mod.load_pooled_k_values(data_dir, seeds, min_generation=1050, max_generation=10**9)
    assert len(values) == n_seeds * n_genomes  # only age 1050


def test_plot_density_heatmap_writes_png(tmp_path):
    values = [(0.1, 0.2), (0.8, 0.9), (0.5, 0.5), (0.3, 0.7)] * 10
    out_path = tmp_path / "heatmap.png"
    mod.plot_density_heatmap(values, bins=10, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_split_interior_and_boundary_excludes_exact_0_and_1():
    values = [(0.1, 0.2), (0.0, 0.5), (0.5, 1.0), (1.0, 1.0), (0.3, 0.7)]
    interior, boundary_pct = mod.split_interior_and_boundary(values)
    assert interior == [(0.1, 0.2), (0.3, 0.7)]
    assert boundary_pct == 60.0


def test_split_interior_and_boundary_all_interior_is_zero_pct():
    values = [(0.1, 0.2), (0.3, 0.7)]
    interior, boundary_pct = mod.split_interior_and_boundary(values)
    assert interior == values
    assert boundary_pct == 0.0


def test_find_mode_bin_locates_peak_bin_center():
    import numpy as np

    counts = np.zeros((4, 4))
    counts[1, 2] = 100  # x-bin 1, y-bin 2 is the densest
    xedges = np.linspace(0, 1, 5)
    yedges = np.linspace(0, 1, 5)
    mode_x, mode_y = mod.find_mode_bin(counts, xedges, yedges)
    assert mode_x == (xedges[1] + xedges[2]) / 2
    assert mode_y == (yedges[2] + yedges[3]) / 2


def test_plot_density_heatmap_handles_boundary_pileup(tmp_path):
    # Half the mass pinned at the K_out==0 boundary; the heatmap/mode/
    # color-scale must be computed from the remaining interior-only data,
    # and must not crash when a sizable boundary fraction is excluded.
    interior = [(0.4, 0.5), (0.5, 0.5), (0.6, 0.5)] * 10
    boundary = [(0.0, 0.5)] * 10
    out_path = tmp_path / "heatmap_boundary.png"
    mod.plot_density_heatmap(interior + boundary, bins=10, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_main_writes_png(tmp_path):
    data_dir = make_tiny_production_data(tmp_path, n_seeds=4, n_genomes=20, ages=(1000,))
    out_path = tmp_path / "out" / "heatmap.png"
    sys.modules["plot_k_model_density_heatmap"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "plot_k_model_density_heatmap.py",
        "--data-dir", str(data_dir), "--seeds", "4", "--seed-offset", "0",
        "--min-generation", "1000", "--bins", "10",
        "--out-path", str(out_path),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup
    assert out_path.exists()
