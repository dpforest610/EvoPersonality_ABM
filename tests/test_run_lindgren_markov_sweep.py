import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_lindgren_markov_sweep", REPO_ROOT / "scripts" / "run_lindgren_markov_sweep.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_lindgren_markov_sweep"] = mod
spec.loader.exec_module(mod)

TINY_KWARGS = dict(
    noise_rate=0.0, probe_noise_rates=(0.0,), population_size=8, generations=20,
    rounds_per_duel=6, initial_memory_depth=1, point_mutation_rate=0.1,
    duplication_rate=0.05, split_rate=0.02, max_memory_depth=3, min_memory_depth=0,
    probe_interval=5, probe_repeats=2, probe_rounds=10, main_analysis_rounds=5,
    push_partial_progress=False,
)


def test_atomic_pickle_dump_replaces_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "seed_0_checkpoint.pkl"
    mod._atomic_pickle_dump(path, {"age": 5})
    mod._atomic_pickle_dump(path, {"age": 10})
    import pickle
    with open(path, "rb") as f:
        assert pickle.load(f)["age"] == 10
    assert [p.name for p in tmp_path.iterdir()] == ["seed_0_checkpoint.pkl"]


def test_seed_is_complete_false_when_file_missing(tmp_path):
    assert mod.seed_is_complete(tmp_path, seed=0, generations=20) is False


def test_seed_is_complete_false_when_partial_row_count(tmp_path):
    mod.write_generation_stats(mod.generation_stats_path(tmp_path, 0), [(g, 1.0, 0.0, 1, 1, 2.0) for g in range(5)])
    assert mod.seed_is_complete(tmp_path, seed=0, generations=20) is False


def test_seed_is_complete_true_when_full_row_count(tmp_path):
    mod.write_generation_stats(mod.generation_stats_path(tmp_path, 0), [(g, 1.0, 0.0, 1, 1, 2.0) for g in range(20)])
    assert mod.seed_is_complete(tmp_path, seed=0, generations=20) is True


def test_run_one_seed_interrupted_and_resumed_matches_uninterrupted_run(tmp_path, monkeypatch):
    """The resumability guarantee claimed in the Mac runbook: a seed run
    straight through, versus one that's genuinely interrupted partway
    (simulated by making round_robin_payoffs raise partway through, so
    run_one_seed crashes mid-run leaving its last checkpoint on disk --
    lowering --generations would NOT simulate this, since run_one_seed
    would just treat that as a complete, shorter run and delete its
    checkpoint) and then resumed with an unpatched call, must produce a
    bit-identical final generation_stats.csv."""
    straight_dir = tmp_path / "straight"
    resumed_dir = tmp_path / "resumed"
    straight_dir.mkdir()
    resumed_dir.mkdir()

    mod.run_one_seed(seed=0, out_dir=straight_dir, **TINY_KWARGS)

    real_round_robin_payoffs = mod.round_robin_payoffs
    call_count = {"n": 0}

    def crashing_round_robin_payoffs(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] > 8:  # past at least one probe_interval=5 checkpoint, before generations=20 finishes
            raise RuntimeError("simulated crash")
        return real_round_robin_payoffs(*args, **kwargs)

    monkeypatch.setattr(mod, "round_robin_payoffs", crashing_round_robin_payoffs)
    with pytest.raises(RuntimeError, match="simulated crash"):
        mod.run_one_seed(seed=0, out_dir=resumed_dir, **TINY_KWARGS)
    assert mod.checkpoint_path_for(resumed_dir, 0).exists()
    assert not mod.seed_is_complete(resumed_dir, 0, TINY_KWARGS["generations"])

    monkeypatch.setattr(mod, "round_robin_payoffs", real_round_robin_payoffs)
    mod.run_one_seed(seed=0, out_dir=resumed_dir, **TINY_KWARGS)
    assert not mod.checkpoint_path_for(resumed_dir, 0).exists()  # deleted on completion

    straight_csv = (straight_dir / "seed_0_generation_stats.csv").read_text()
    resumed_csv = (resumed_dir / "seed_0_generation_stats.csv").read_text()
    assert straight_csv == resumed_csv


def test_run_one_seed_push_partial_progress_false_reports_once_at_the_end():
    import queue as queue_module
    from multiprocessing import Manager

    manager = Manager()
    q = manager.Queue()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        kwargs = dict(TINY_KWARGS)
        kwargs["push_partial_progress"] = False
        mod.run_one_seed(seed=0, out_dir=Path(d), progress_queue=q, **kwargs)

        reports = []
        while True:
            try:
                reports.append(q.get_nowait())
            except queue_module.Empty:
                break
        assert len(reports) == 1  # exactly one batch, at completion
        seed, paths = reports[0]
        assert seed == 0
        assert len(paths) > 1  # the whole seed's committed-tier files, not just one


def test_run_one_seed_push_partial_progress_true_reports_multiple_times():
    import queue as queue_module
    from multiprocessing import Manager

    manager = Manager()
    q = manager.Queue()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        kwargs = dict(TINY_KWARGS)
        kwargs["push_partial_progress"] = True
        mod.run_one_seed(seed=0, out_dir=Path(d), progress_queue=q, **kwargs)

        n_reports = 0
        while True:
            try:
                q.get_nowait()
                n_reports += 1
            except queue_module.Empty:
                break
        # generations=20, probe_interval=5 -> probes at age 0,5,10,15,20 = 5
        # reports (one per probe timepoint); the final generations%probe_interval==0
        # case contributes no extra duplicate report beyond that last probe.
        assert n_reports == 5
