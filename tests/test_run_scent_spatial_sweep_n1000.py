import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_scent_spatial_sweep_n1000", REPO_ROOT / "scripts" / "run_scent_spatial_sweep_n1000.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_scent_spatial_sweep_n1000"] = mod
spec.loader.exec_module(mod)


def test_parse_gen_range():
    assert mod.parse_gen_range(None) is None
    assert mod.parse_gen_range("0-19") == set(range(0, 20))
    assert mod.parse_gen_range("5-5") == {5}


def test_parse_gen_range_multiple_comma_separated():
    assert mod.parse_gen_range("0-19,4980-4999") == set(range(0, 20)) | set(range(4980, 5000))
    assert mod.parse_gen_range("1-1,3-3,5-5") == {1, 3, 5}


def _fake_snapshot(n, rng):
    k_out = rng.random(n)
    k_self = rng.random(n)
    payoff = 1.0 + rng.random(n)
    opp_k_self = rng.random(n)
    opp_k_out = rng.random(n)
    return dict(k_out=k_out.tolist(), k_self=k_self.tolist(), payoff=payoff.tolist(),
                opp_k_self=opp_k_self.tolist(), opp_k_out=opp_k_out.tolist())


def test_compute_seed_summary_basic_shape_and_values():
    rng = np.random.default_rng(0)
    snapshots = {gen: _fake_snapshot(50, rng) for gen in range(2000, 2500, 100)}
    summary = mod.compute_seed_summary(seed=3, snapshots_by_gen=snapshots)

    assert summary["seed"] == 3
    assert summary["n_qualifying_gens"] == len(snapshots)
    for field in mod.SUMMARY_FIELDS:
        assert field in summary
    # with real variance in random data, these should all be finite numbers
    for field in ["beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout",
                   "beta1", "beta2", "gamma11", "gamma22", "gamma12",
                   "mean_beta1", "mean_beta2", "mean_gamma11", "mean_gamma22", "mean_gamma12",
                   "var_kout", "var_kself", "bc_kout", "bc_kself"]:
        assert math.isfinite(summary[field]), f"{field} should be finite, got {summary[field]}"


def test_compute_seed_summary_skips_degenerate_generations():
    # a generation with constant k_out should be excluded (no variance to regress on)
    snapshots = {
        2000: dict(k_out=[0.5] * 10, k_self=[0.1, 0.9] * 5, payoff=[1.0, 2.0] * 5,
                   opp_k_self=[0.2] * 10, opp_k_out=[0.3] * 10),
    }
    summary = mod.compute_seed_summary(seed=0, snapshots_by_gen=snapshots)
    assert summary["n_qualifying_gens"] == 0
    assert summary["beta1"] != summary["beta1"]  # NaN


def test_write_and_read_summary_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    snapshots = {gen: _fake_snapshot(30, rng) for gen in range(2000, 2300, 100)}
    summary = mod.compute_seed_summary(seed=7, snapshots_by_gen=snapshots)
    path = tmp_path / "seed_7_summary.csv"
    mod.write_summary_csv(path, summary)

    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["seed"]) == 7
    assert int(rows[0]["n_qualifying_gens"]) == summary["n_qualifying_gens"]
    assert abs(float(rows[0]["beta1"]) - summary["beta1"]) < 1e-6


def test_atomic_pickle_dump_replaces_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "seed_0_checkpoint.pkl"
    mod._atomic_pickle_dump(path, {"next_generation": 250})
    mod._atomic_pickle_dump(path, {"next_generation": 500})
    import pickle
    with open(path, "rb") as f:
        assert pickle.load(f)["next_generation"] == 500
    assert [p.name for p in tmp_path.iterdir()] == ["seed_0_checkpoint.pkl"]


def test_out_root_for_respects_experiment_root_override():
    custom = Path("/tmp/not-production/scent_spatial_sweep_n1000_phase_recording")
    out = mod.out_root_for("none", experiment_root=custom)
    assert str(out).startswith(str(custom))
    assert mod.EXPERIMENT_ROOT not in out.parents and out != mod.EXPERIMENT_ROOT


def test_resolve_experiment_root_defaults_to_production():
    assert mod.resolve_experiment_root(None) == mod.EXPERIMENT_ROOT
    assert mod.resolve_experiment_root("") == mod.EXPERIMENT_ROOT


def test_resolve_experiment_root_accepts_a_different_directory():
    custom = "scripts/output/scent_spatial_sweep_n1000_phase_recording"
    out = mod.resolve_experiment_root(custom)
    assert out == Path(custom)
    assert out.resolve() != mod.EXPERIMENT_ROOT.resolve()


def test_resolve_experiment_root_rejects_the_production_root_itself():
    with pytest.raises(ValueError):
        mod.resolve_experiment_root(str(mod.EXPERIMENT_ROOT))
    with pytest.raises(ValueError):
        mod.resolve_experiment_root(str(mod.EXPERIMENT_ROOT) + "/")


def _small_model_params(**overrides):
    params = dict(
        population_size=20, rounds_per_duel=10, point_mutation_rate=0.05,
        duplication_rate=0.02, split_rate=0.02, max_memory_depth=8, min_memory_depth=0,
        k_mutation_std=0.05, noise_std=0.05, grid_width=6, grid_height=6, n_phases=2,
        tau=1.0, radius=1,
    )
    params.update(overrides)
    return params


def test_run_one_seed_chunked_default_interval_matches_log_interval(tmp_path):
    seed, agg_rows, individual_rows, summary, phase_frames, elapsed, n_chunks, gens = mod.run_one_seed_chunked(
        seed=0, generations=5, chunk_size=5, out_root_str=str(tmp_path),
        movement="none", p_move=None, model_params=_small_model_params(),
        record_phases_seed=None, record_phases_gens=None,
    )
    # default individual_log_interval == LOG_INTERVAL (100): with only 5
    # generations, only gen 0 is ever a multiple of it
    gens_logged = sorted({r[0] for r in individual_rows})
    assert gens_logged == [0]


def test_run_one_seed_chunked_dense_individual_log_interval(tmp_path):
    seed, agg_rows, individual_rows, summary, phase_frames, elapsed, n_chunks, gens = mod.run_one_seed_chunked(
        seed=0, generations=5, chunk_size=5, out_root_str=str(tmp_path),
        movement="none", p_move=None, model_params=_small_model_params(),
        record_phases_seed=None, record_phases_gens=None, individual_log_interval=1,
    )
    gens_logged = sorted({r[0] for r in individual_rows})
    assert gens_logged == [0, 1, 2, 3, 4]
    # every generation's full population is present
    for g in gens_logged:
        assert len({r[1] for r in individual_rows if r[0] == g}) == 20  # population_size


def test_individual_log_interval_never_changes_seed_summary():
    """A dense individual log must not change seed_summary.csv -- that
    always follows the fixed LOG_INTERVAL/STEADY_STATE_MIN_GEN cadence,
    independent of --individual-log-interval."""
    kwargs = dict(
        seed=1, generations=5, chunk_size=5,
        movement="none", p_move=None, model_params=_small_model_params(),
        record_phases_seed=None, record_phases_gens=None,
    )
    import tempfile
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        r_sparse = mod.run_one_seed_chunked(out_root_str=d1, individual_log_interval=100, **kwargs)
        r_dense = mod.run_one_seed_chunked(out_root_str=d2, individual_log_interval=1, **kwargs)
        summary_sparse, summary_dense = r_sparse[3], r_dense[3]
        assert summary_sparse.keys() == summary_dense.keys()
        for k in summary_sparse:
            a, b = summary_sparse[k], summary_dense[k]
            assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))


def test_scratch_branch_is_per_condition():
    a = mod.scratch_branch_for(Path("scripts/output/scent_spatial_sweep_n1000/scent/host"))
    b = mod.scratch_branch_for(Path("scripts/output/scent_spatial_sweep_n1000/scent_low/host"))
    assert a != b and a.endswith("-scent") and b.endswith("-scent_low")


def test_warn_if_oversubscribed(capsys, monkeypatch):
    monkeypatch.setattr(mod, "total_cores", lambda: 8)
    mod.warn_if_oversubscribed(8)
    assert "WARNING" not in capsys.readouterr().err
    mod.warn_if_oversubscribed(9)
    assert "WARNING" in capsys.readouterr().err
