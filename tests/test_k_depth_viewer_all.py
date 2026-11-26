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

_mod_spec = importlib.util.spec_from_file_location("k_depth_viewer_all", SCRIPTS_DIR / "exploratory_hbes2026" / "k_depth_viewer_all.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["k_depth_viewer_all"] = mod
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


def test_classify_matches_module_convention():
    assert mod.classify(0.9, 0.9, 0.5, 0.5) == "K_Secure"
    assert mod.classify(0.1, 0.1, 0.5, 0.5) == "K_Fearful"


def test_main_writes_edges_nodes_and_group_size(tmp_path):
    age = 1000
    data_dir, seeds = make_tiny_production_data(tmp_path, age=age)
    thresholds_path = write_thresholds(tmp_path, seeds)
    out_dir = tmp_path / "out"

    sys.modules["k_depth_viewer_all"] = mod
    sys_argv_backup = sys.argv
    sys.argv = [
        "k_depth_viewer_all.py",
        "--data-dir", str(data_dir), "--thresholds-json", str(thresholds_path),
        "--age-start", str(age), "--age-end", str(age), "--age-step", "100",
        "--workers", "1", "--reservoir", "10", "--n-draws", "3", "--min-group", "1",
        "--out-dir", str(out_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    edges_path = out_dir / "k_depth_edges.json"
    nodes_path = out_dir / "k_depth_nodes.json"
    gs_path = out_dir / "k_depth_group_size.csv"
    assert edges_path.exists()
    assert nodes_path.exists()
    assert gs_path.exists()

    edges = json.loads(edges_path.read_text())
    assert "abm" in edges and "meta" in edges
    nodes = json.loads(nodes_path.read_text())
    assert "abm_nodes" in nodes and "meta" in nodes

    # at least one (opponent, grouping, depth) has non-empty edge data
    found_edges = any(
        depth_edges
        for grouping_map in edges["abm"].values()
        for depth_edges in grouping_map.values()
    )
    assert found_edges, "expected at least one grouping/depth with edge data"

    import csv
    with open(gs_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "expected at least one group-size row"
    assert set(rows[0].keys()) == {"grouping", "depth", "seeds_available", "group_size", "individuals"}
    for row in rows:
        assert row["grouping"] in mod.GROUPINGS
        int(row["depth"])
        int(row["seeds_available"])
        int(row["group_size"])
        int(row["individuals"])
