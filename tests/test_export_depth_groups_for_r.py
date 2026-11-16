import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "export_depth_groups_for_r", REPO_ROOT / "scripts" / "export_depth_groups_for_r.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["export_depth_groups_for_r"] = mod
spec.loader.exec_module(mod)


def make_row(seed, age, gid=0, freq=1, effective_depth=0, nominal_depth=None):
    return {
        "seed": seed, "age": age, "gid": gid, "freq": freq,
        "nominal_depth": nominal_depth if nominal_depth is not None else effective_depth,
        "effective_depth": effective_depth,
    }


# ------------------------------------------------------------------ propose_cutpoints

def test_propose_cutpoints_finds_a_feasible_balanced_split():
    pool = []
    # depth 0: seeds {0,1}; depth 1 adds seed {2} -- cutting low at depth 0
    # alone is infeasible (only 2 seeds), so d1=1 is the unique choice
    for seed in (0, 1):
        pool.append(make_row(seed, 1000, effective_depth=0))
    pool.append(make_row(2, 1000, effective_depth=1))
    # depth 2: seeds {3,4}; depth 3 adds seed {5} -- same logic for mid
    for seed in (3, 4):
        pool.append(make_row(seed, 1000, effective_depth=2))
    pool.append(make_row(5, 1000, effective_depth=3))
    # depth 4-5: seeds {6,7,8} (either alone would be < 3, need both)
    pool.append(make_row(6, 1000, effective_depth=4))
    for seed in (7, 8):
        pool.append(make_row(seed, 1000, effective_depth=5))

    result = mod.propose_cutpoints(pool, group_size=3)
    assert result is not None
    d1, d2, counts = result
    assert d1 == 1
    assert d2 == 3
    assert counts == (3, 3, 3)


def test_propose_cutpoints_returns_none_when_infeasible():
    pool = [make_row(0, 1000, effective_depth=0)]  # 1 distinct seed total -- can't make 3 groups of 56
    assert mod.propose_cutpoints(pool, group_size=56) is None


# ------------------------------------------------------------------ depth_ranges_from_cutpoints

def test_depth_ranges_from_cutpoints_are_contiguous_and_cover_full_range():
    pool = [make_row(0, 1000, effective_depth=d) for d in range(7)]  # 0..6
    ranges = mod.depth_ranges_from_cutpoints(pool, d1=1, d2=3)
    assert list(ranges["Low"]) == [0, 1]
    assert list(ranges["Mid"]) == [2, 3]
    assert list(ranges["High"]) == [4, 5, 6]


# ------------------------------------------------------------------ select_group

def test_select_group_prefers_one_per_distinct_seed():
    pool = [make_row(seed, 1000, effective_depth=0) for seed in range(5)]
    rng = random.Random(0)
    selected = mod.select_group(pool, group_size=5, depth_range=range(0, 1), rng=rng)
    assert len(selected) == 5
    assert len({p["seed"] for p in selected}) == 5  # all distinct


def test_select_group_reuses_a_seed_at_a_different_age_when_seeds_run_out():
    pool = [
        make_row(0, 1000, effective_depth=0),
        make_row(0, 1050, effective_depth=0),  # same seed, different age
        make_row(1, 1000, effective_depth=0),
    ]
    rng = random.Random(0)
    selected = mod.select_group(pool, group_size=3, depth_range=range(0, 1), rng=rng)
    assert len(selected) == 3
    # seed 0 must appear twice, but at two DIFFERENT ages (never the same (seed,age) twice)
    seed0_ages = sorted(p["age"] for p in selected if p["seed"] == 0)
    assert seed0_ages == [1000, 1050]


def test_select_group_raises_when_truly_not_enough_candidates():
    pool = [make_row(0, 1000, effective_depth=0)]  # only 1 candidate ever
    rng = random.Random(0)
    try:
        mod.select_group(pool, group_size=2, depth_range=range(0, 1), rng=rng)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_select_group_only_considers_rows_within_the_depth_range():
    pool = [
        make_row(0, 1000, effective_depth=0),
        make_row(1, 1000, effective_depth=5),  # out of range
    ]
    rng = random.Random(0)
    selected = mod.select_group(pool, group_size=1, depth_range=range(0, 1), rng=rng)
    assert len(selected) == 1
    assert selected[0]["effective_depth"] == 0


# ------------------------------------------------------------------ write_depth_distribution

def test_write_depth_distribution_aggregates_frequency_by_age_kind_depth(tmp_path):
    pool = [
        make_row(0, 1000, freq=3, nominal_depth=2, effective_depth=1),
        make_row(1, 1000, freq=2, nominal_depth=2, effective_depth=1),
        make_row(0, 1050, freq=5, nominal_depth=1, effective_depth=1),
    ]
    out_path = tmp_path / "dist.csv"
    mod.write_depth_distribution(pool, out_path)

    import csv
    rows = list(csv.DictReader(open(out_path)))
    row_1000_eff_1 = next(r for r in rows if r["age"] == "1000" and r["depth_kind"] == "effective" and r["depth"] == "1")
    assert int(row_1000_eff_1["frequency_weighted_count"]) == 5  # 3+2
    row_1000_nom_2 = next(r for r in rows if r["age"] == "1000" and r["depth_kind"] == "nominal" and r["depth"] == "2")
    assert int(row_1000_nom_2["frequency_weighted_count"]) == 5
