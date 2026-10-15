import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "animate_scent_n1000_heatbugs", REPO_ROOT / "scripts" / "animate_scent_n1000_heatbugs.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["animate_scent_n1000_heatbugs"] = mod
spec.loader.exec_module(mod)


def test_tcode_four_quadrants():
    # columns: pos_x, pos_y, k_self, k_out, payoff (field() needs 5 cols;
    # tcode only reads columns 2 and 3)
    a = np.array([
        [0, 0, 0.9, 0.9, 3.0],   # high/high -> 0
        [0, 0, 0.9, 0.1, 3.0],   # high/low -> 1
        [0, 0, 0.1, 0.9, 3.0],   # low/high -> 2
        [0, 0, 0.1, 0.1, 3.0],   # low/low -> 3
    ])
    t = mod.tcode(a, ts=0.5, to=0.5)
    assert list(t) == [0, 1, 2, 3]


def test_tcode_boundary_is_inclusive_high():
    a = np.array([[0, 0, 0.5, 0.5, 3.0]])
    assert mod.tcode(a, ts=0.5, to=0.5)[0] == 0  # >= counts as "high"


def test_field_mean_of_four_orthogonal_occupied_neighbors():
    # agent at (2,2) with payoff 2.0, agent at (3,2) [east neighbor] with payoff 4.0;
    # cell (2,2) itself should be NaN (occupied); a cell neighboring only one of them
    # sees that one payoff value
    a = np.array([
        [2, 2, 0.5, 0.5, 2.0],
        [3, 2, 0.5, 0.5, 4.0],
    ])
    f = mod.field(a, width=10, height=10)
    assert f[2, 2] != f[2, 2]  # NaN: occupied cell
    # cell (2,1) [x=2,y=1] is an orthogonal neighbor of (2,2) only
    assert f[1, 2] == 2.0
    # cell (3,1) [x=3,y=1] is an orthogonal neighbor of (3,2) only
    assert f[1, 3] == 4.0
    # cell (2,3) [x=2,y=3], north of (2,2): also neighbors only (2,2)
    assert f[3, 2] == 2.0


def test_field_averages_multiple_occupied_neighbors():
    a = np.array([
        [1, 1, 0.5, 0.5, 2.0],
        [3, 1, 0.5, 0.5, 6.0],
    ])
    f = mod.field(a, width=10, height=10)
    # cell (2,1) is an orthogonal neighbor of both (1,1) and (3,1)
    assert f[1, 2] == 4.0


def test_compute_trajectory_share_and_cumulative_deviation():
    gens = [0, 1]
    by_gen = {
        0: np.array([
            [0, 0, 0.9, 0.9, 4.0],  # type 0
            [1, 0, 0.9, 0.9, 4.0],  # type 0
            [2, 0, 0.1, 0.1, 2.0],  # type 3
            [3, 0, 0.1, 0.1, 2.0],  # type 3
        ]),
        1: np.array([
            [0, 0, 0.9, 0.9, 5.0],
            [1, 0, 0.9, 0.9, 5.0],
            [2, 0, 0.1, 0.1, 1.0],
            [3, 0, 0.1, 0.1, 1.0],
        ]),
    }
    share, cum_dev = mod.compute_trajectory(gens, by_gen, ts=0.5, to=0.5)
    # gen 0: types 0 and 3 each 50%, types 1 and 2 absent (0%)
    assert list(share[0]) == [50.0, 0.0, 0.0, 50.0]
    # gen 0: type 0 mean payoff 4.0, population mean 3.0 -> deviation +1.0;
    # type 3 mean payoff 2.0 -> deviation -1.0; absent types -> 0
    assert np.allclose(cum_dev[0], [1.0, 0.0, 0.0, -1.0])
    # gen 1: type 0 mean 5.0, population mean 3.0 -> deviation +2.0, cumulative +3.0
    # type 3 mean 1.0 -> deviation -2.0, cumulative -3.0
    assert np.allclose(cum_dev[1], [3.0, 0.0, 0.0, -3.0])


def test_load_thresholds_shape():
    # thresholds.json may not exist in a fresh checkout before the compute
    # script has run -- only check the loader's parsing logic using a
    # temporary file with the same schema
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"threshold_kself": 0.777, "threshold_kout": 0.5025}, f)
        path = f.name
    orig = mod.THRESHOLDS_JSON
    try:
        mod.THRESHOLDS_JSON = Path(path)
        ts, to = mod.load_thresholds()
        assert ts == 0.777
        assert to == 0.5025
    finally:
        mod.THRESHOLDS_JSON = orig
