import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "animate_scent_n1000_phase_recording", REPO_ROOT / "scripts" / "animate_scent_n1000_phase_recording.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["animate_scent_n1000_phase_recording"] = mod
spec.loader.exec_module(mod)


def test_torus_delta_no_wrap():
    assert mod.torus_delta(5, 8, 41) == 3
    assert mod.torus_delta(8, 5, 41) == -3


def test_torus_delta_takes_the_short_way_around():
    # on a size-10 torus, 1 -> 9 the short way is -2 (wrap), not +8
    assert mod.torus_delta(1, 9, 10) == -2
    assert mod.torus_delta(9, 1, 10) == 2


def test_interpolate_positions_linear_no_wrap():
    out = mod.interpolate_positions([(5, 5)], [(8, 5)], grid_width=41, grid_height=41, frac=0.5)
    assert out[0, 0] == 6.5
    assert out[0, 1] == 5.0


def test_interpolate_positions_wraps_the_short_way():
    # size-10 torus, 1 -> 9 at frac 0.5 should land at 0 (halfway along -2), not at 5
    out = mod.interpolate_positions([(1, 0)], [(9, 0)], grid_width=10, grid_height=10, frac=0.5)
    assert out[0, 0] == 0.0


def _fake_phase_frames(gens, n_phases=3, n_agents=4):
    """Build a minimal phase_frames dict: positions drift by (1, 0) each
    phase within a generation, reset to (0, 0) at each new generation
    (simulating offspring replacing parents at a fixed cell)."""
    out = {}
    for gi, gen in enumerate(gens):
        frames = []
        for p in range(n_phases):
            positions = [((p) % 41, 0) for _ in range(n_agents)]
            moved = [p > 0 for _ in range(n_agents)]
            payoff_so_far = [1.0 + p for _ in range(n_agents)]
            frames.append((p, positions, moved, payoff_so_far))
        out[gen] = dict(frames=frames, k_out=[0.5] * n_agents, k_self=[0.5] * n_agents)
    return out


def test_build_render_frames_one_entry_per_phase():
    pf = _fake_phase_frames([0, 1], n_phases=3)
    frames = mod.build_render_frames(pf, n_phases=3, subframes=4)
    assert len(frames) == 6  # 2 gens x 3 phases


def test_build_render_frames_hard_cuts_at_generation_boundary_only():
    pf = _fake_phase_frames([0, 1, 2], n_phases=3)
    frames = mod.build_render_frames(pf, n_phases=3, subframes=4)
    # phase 0 of each generation after the first is a cut; nothing else is
    cuts = [(f["gen"], f["phase"]) for f in frames if f["is_cut"]]
    assert cuts == [(1, 0), (2, 0)]
    # within-generation transitions (phase 1, phase 2) carry interpolation data
    for f in frames:
        if f["phase"] > 0:
            assert "_interp_from" in f and "_interp_to" in f
        else:
            assert "_interp_from" not in f


def test_build_render_frames_respects_gen_filter():
    pf = _fake_phase_frames([0, 1, 2, 3], n_phases=2)
    frames = mod.build_render_frames(pf, n_phases=2, subframes=2, gen_filter=lambda g: g in (1, 3))
    gens_seen = sorted({f["gen"] for f in frames})
    assert gens_seen == [1, 3]
    # first frame of this filtered sequence is still treated as "no prior
    # frame to interpolate from", not a cut (out is empty at that point)
    assert frames[0]["is_cut"] is False


def test_expand_interpolated_frames_count_and_arrival_flag():
    pf = _fake_phase_frames([0], n_phases=3, n_agents=2)
    key_frames = mod.build_render_frames(pf, n_phases=3, subframes=4)
    expanded = mod.expand_interpolated_frames(key_frames, grid_width=41, grid_height=41)
    # phase 0: 1 held frame; phases 1,2: 4 subframes each -> 1 + 4 + 4 = 9
    assert len(expanded) == 9
    arrivals = [f.get("is_arrival") for f in expanded]
    # the held phase-0 frame has no is_arrival key; each transition's last
    # subframe is marked as arrival, the rest are not
    assert arrivals.count(True) == 2
    for f in expanded:
        assert "positions" in f and f["positions"].shape == (2, 2)


def test_expand_interpolated_frames_never_jumps_across_a_generation_cut():
    pf = _fake_phase_frames([0, 1], n_phases=2, n_agents=1)
    key_frames = mod.build_render_frames(pf, n_phases=2, subframes=3)
    expanded = mod.expand_interpolated_frames(key_frames, grid_width=41, grid_height=41)
    # the frame right after the generation-1 cut must equal gen 1's own
    # phase-0 recorded position exactly (no blend with gen 0's last phase)
    cut_frame = next(f for f in expanded if f.get("is_cut"))
    assert cut_frame["gen"] == 1 and cut_frame["phase"] == 0
    assert tuple(cut_frame["positions"][0]) == (0.0, 0.0)


def test_mean_neighbor_payoff_grid_basic():
    # 5x5 grid, one occupied cell at (2,2) with payoff 3.0; its 8 Moore
    # neighbors should show 3.0, everything else (no occupied neighbor) NaN
    positions = [(2, 2)]
    payoff = np.array([3.0])
    grid, has_neighbor = mod.mean_neighbor_payoff_grid(positions, payoff, grid_width=5, grid_height=5)
    assert grid[2, 2] != grid[2, 2]  # the occupied cell itself has no occupied neighbor of its own
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            assert grid[2 + dy, 2 + dx] == 3.0
            assert has_neighbor[2 + dy, 2 + dx]
    assert grid[0, 0] != grid[0, 0]  # far corner: NaN (no occupied neighbor)


def test_mean_neighbor_payoff_grid_averages_multiple_neighbors():
    positions = [(1, 1), (3, 1)]
    payoff = np.array([2.0, 6.0])
    grid, _ = mod.mean_neighbor_payoff_grid(positions, payoff, grid_width=5, grid_height=5)
    # cell (2,1) is a Moore-neighbor of both occupied cells
    assert grid[1, 2] == 4.0


def test_mean_neighbor_payoff_grid_ignores_nan_payoff():
    # agent at (1,1) has NaN payoff (no duels yet this generation); agent
    # at (1,2) has a real payoff. Cell (1,3) is a Moore-neighbor of (1,2)
    # only (too far from (1,1)), so it should see exactly 5.0, not NaN
    # and not an average with a dropped NaN term.
    positions = [(1, 1), (1, 2)]
    payoff = np.array([float("nan"), 5.0])
    grid, has_neighbor = mod.mean_neighbor_payoff_grid(positions, payoff, grid_width=5, grid_height=5)
    assert grid[3, 1] == 5.0
    assert has_neighbor[3, 1]
    # cell (0,0) is (toroidally) a Moore-neighbor of the NaN-payoff agent
    # at (1,1) only, not of (1,2) -- its only candidate is filtered out
    assert grid[0, 0] != grid[0, 0]  # NaN
    assert not has_neighbor[0, 0]
