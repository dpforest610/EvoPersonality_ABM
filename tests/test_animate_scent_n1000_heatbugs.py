import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_field_uses_8_cell_moore_neighborhood_not_4_cell():
    # the model's own scent field (_mean_neighbor_payoff, called with
    # offsets=moore_offsets(radius=1) from _run_generation) uses all 8
    # surrounding cells, not just the 4 orthogonal ones -- the bug in the
    # first prototype (heatbugs_style_reference.py). A diagonal neighbor
    # must be picked up.
    a = np.array([[5, 5, 0.5, 0.5, 7.0]])
    f = mod.field(a, width=12, height=12)
    assert f[5, 5] != f[5, 5]  # NaN: occupied cell itself
    assert f[6, 6] == 7.0      # diagonal neighbor (Chebyshev distance 1)
    assert f[6, 5] == 7.0      # orthogonal neighbor, still included
    assert f[7, 7] != f[7, 7]  # Chebyshev distance 2 -> outside radius 1, NaN


def test_field_averages_multiple_occupied_moore_neighbors():
    a = np.array([
        [1, 1, 0.5, 0.5, 2.0],
        [3, 1, 0.5, 0.5, 6.0],
    ])
    f = mod.field(a, width=10, height=10)
    # cell (2,1) is a Moore neighbor of both (1,1) and (3,1)
    assert f[1, 2] == 4.0


def test_field_isolated_agent_only_affects_its_own_moore_neighborhood():
    a = np.array([
        [1, 1, 0.5, 0.5, 2.0],
        [8, 8, 0.5, 0.5, 4.0],
    ])
    f = mod.field(a, width=12, height=12)
    assert f[2, 2] == 2.0       # Moore neighbor of (1,1) only
    assert f[7, 7] == 4.0       # Moore neighbor of (8,8) only
    assert f[5, 5] != f[5, 5]   # far from both -> NaN


def test_field_matches_model_mean_neighbor_payoff():
    """field() must reproduce the model's own scent score exactly: the
    same payoff_so_far values, through the same moore_offsets(radius=1)
    neighborhood, computed by the same function _run_generation actually
    scores scent candidates with
    (evopersonality_abm.spatial._mean_neighbor_payoff)."""
    from evopersonality_abm.spatial import _mean_neighbor_payoff

    rng = np.random.default_rng(0)
    width = height = 12
    n_agents = 25
    positions = []
    seen = set()
    while len(positions) < n_agents:
        pos = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        if pos not in seen:
            seen.add(pos)
            positions.append(pos)
    payoffs = rng.uniform(1.0, 5.0, size=n_agents)  # strictly positive: 0.0 never a real average
    a = np.array([[x, y, 0.5, 0.5, p] for (x, y), p in zip(positions, payoffs)])

    f = mod.field(a, width=width, height=height)

    agent_at_cell = {pos: i for i, pos in enumerate(positions)}
    total_payoff = list(payoffs)
    rounds_played = [1] * n_agents  # total_payoff[j]/rounds_played[j] == payoffs[j]

    checked_a_real_neighbor_average = False
    for x in range(width):
        for y in range(height):
            expected = _mean_neighbor_payoff(
                (x, y), width, height, agent_at_cell, total_payoff, rounds_played, mod.NEIGHBOR_OFFSETS
            )
            got = f[y, x]
            if (x, y) in agent_at_cell:
                assert got != got  # field() always blanks occupied cells to NaN
            elif expected == 0.0:
                # _mean_neighbor_payoff's "no occupied neighbor" floor;
                # field() uses NaN for that same case instead
                assert got != got
            else:
                assert got == pytest.approx(expected)
                checked_a_real_neighbor_average = True
    assert checked_a_real_neighbor_average


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
