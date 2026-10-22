import csv
import importlib.util
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "analyze_selfplay_recovery_by_depth", REPO_ROOT / "scripts" / "analyze_selfplay_recovery_by_depth.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_selfplay_recovery_by_depth"] = mod
spec.loader.exec_module(mod)


def write_by_depth_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["age", "never_recovered_rate", "median_recovery_round", "opponent", "depth_group", "n_recovered", "n"])
        w.writeheader()
        w.writerows(rows)


def write_pooled_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["opponent", "age", "n_recovered", "n", "never_recovered_rate", "median_recovery_round"])
        w.writeheader()
        w.writerows(rows)


def write_per_seed_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "age", "opponent", "depth_group", "n", "n_recovered"])
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ load_by_depth / load_pooled

def test_load_by_depth_nests_by_opponent_age_group(tmp_path):
    path = tmp_path / "by_depth.csv"
    write_by_depth_csv(path, [
        {"age": 1000, "never_recovered_rate": 0.5, "median_recovery_round": 2, "opponent": "TFT", "depth_group": "depthLow", "n_recovered": 5, "n": 10},
    ])
    out = mod.load_by_depth(path)
    assert out["TFT"][1000]["depthLow"]["never_recovered_rate"] == "0.5"


def test_load_pooled_nests_by_opponent_age(tmp_path):
    path = tmp_path / "pooled.csv"
    write_pooled_csv(path, [{"opponent": "TFT", "age": 1000, "n_recovered": 5, "n": 10, "never_recovered_rate": 0.5, "median_recovery_round": 2}])
    out = mod.load_pooled(path)
    assert out["TFT"][1000]["never_recovered_rate"] == "0.5"


# ------------------------------------------------------------------ weights_and_rates

def test_weights_and_rates_normalizes_n_to_shares_and_flips_to_recovery_rate():
    by_age = {
        1000: {
            "depthLow": {"n": "10", "never_recovered_rate": "0.8"},
            "depthMid": {"n": "10", "never_recovered_rate": "0.5"},
            "depthHigh": {"n": "20", "never_recovered_rate": "0.2"},
        }
    }
    weights, rates = mod.weights_and_rates(by_age, 1000)
    assert weights == {"depthLow": 0.25, "depthMid": 0.25, "depthHigh": 0.5}
    assert rates == pytest.approx({"depthLow": 0.2, "depthMid": 0.5, "depthHigh": 0.8})


# ------------------------------------------------------------------ decompose

def _uniform_by_age(age_to_group_n_rate):
    """{age: {group: (n, never_recovered_rate)}} -> the by_age dict decompose() expects."""
    return {
        age: {g: {"n": str(n), "never_recovered_rate": str(nrr)} for g, (n, nrr) in groups.items()}
        for age, groups in age_to_group_n_rate.items()
    }


def test_decompose_within_only_when_weights_constant_across_ages():
    # equal, UNCHANGING weights across 2 ages; only the recovery rates move
    by_age = _uniform_by_age({
        1000: {"depthLow": (10, 0.8), "depthMid": (10, 0.8), "depthHigh": (10, 0.8)},
        2000: {"depthLow": (10, 0.6), "depthMid": (10, 0.6), "depthHigh": (10, 0.6)},
    })
    d = mod.decompose(by_age)
    assert abs(d["composition_total"]) < 1e-12
    assert abs(d["within_total"] - d["total_change"]) < 1e-12
    assert d["total_change"] == (1 - 0.6) - (1 - 0.8)  # recovery_rate = 1 - never_recovered_rate


def test_decompose_composition_only_when_rates_constant_across_ages():
    # recovery rates (never_recovered_rate) identical across ages; only the
    # group SHARES shift (depthHigh gains weight, and has a higher recovery
    # rate than depthLow -- so composition alone should move the total up).
    by_age = _uniform_by_age({
        1000: {"depthLow": (20, 0.8), "depthMid": (10, 0.5), "depthHigh": (10, 0.2)},
        2000: {"depthLow": (10, 0.8), "depthMid": (10, 0.5), "depthHigh": (20, 0.2)},
    })
    d = mod.decompose(by_age)
    assert abs(d["within_total"]) < 1e-12
    assert abs(d["composition_total"] - d["total_change"]) < 1e-12
    assert d["total_change"] > 0  # shifted weight towards the higher-recovery-rate group


def test_decompose_residual_is_exactly_zero_up_to_float_rounding():
    by_age = _uniform_by_age({
        1000: {"depthLow": (7, 0.9), "depthMid": (13, 0.4), "depthHigh": (31, 0.1)},
        1500: {"depthLow": (11, 0.7), "depthMid": (9, 0.6), "depthHigh": (25, 0.2)},
        2000: {"depthLow": (5, 0.5), "depthMid": (20, 0.5), "depthHigh": (40, 0.3)},
    })
    d = mod.decompose(by_age)
    assert abs(d["residual_check"]) < 1e-9
    assert abs((d["within_total"] + d["composition_total"]) - d["total_change"]) < 1e-9


def test_decompose_chained_over_many_steps_still_exact_even_if_non_monotonic():
    # non-monotonic path (goes up then down then up) -- the chain sum must
    # still equal EXACTLY the endpoint-to-endpoint total change.
    by_age = _uniform_by_age({
        1000: {"depthLow": (10, 0.9), "depthMid": (10, 0.5), "depthHigh": (10, 0.1)},
        1200: {"depthLow": (15, 0.3), "depthMid": (10, 0.5), "depthHigh": (5, 0.9)},
        1400: {"depthLow": (5, 0.9), "depthMid": (10, 0.5), "depthHigh": (15, 0.1)},
        1600: {"depthLow": (8, 0.4), "depthMid": (12, 0.4), "depthHigh": (10, 0.6)},
    })
    d = mod.decompose(by_age)
    assert abs(d["residual_check"]) < 1e-9


# ------------------------------------------------------------------ mean_rate_by_group

def test_mean_rate_by_group_averages_across_all_probed_ages():
    by_age = _uniform_by_age({
        1000: {"depthLow": (10, 0.8), "depthMid": (10, 0.5), "depthHigh": (10, 0.2)},
        2000: {"depthLow": (10, 0.6), "depthMid": (10, 0.5), "depthHigh": (10, 0.4)},
    })
    m = mod.mean_rate_by_group(by_age)
    assert m == pytest.approx({"depthLow": 0.7, "depthMid": 0.5, "depthHigh": 0.3})


# ------------------------------------------------------------------ select_japanese_font

def test_select_japanese_font_prefers_hiragino_sans_when_present():
    font, is_hiragino = mod.select_japanese_font({"Hiragino Sans", "Arial", "IPAGothic"})
    assert font == "Hiragino Sans"
    assert is_hiragino is True


def test_select_japanese_font_falls_back_to_ipagothic_when_no_hiragino():
    font, is_hiragino = mod.select_japanese_font({"Arial", "DejaVu Sans"})
    assert font == "IPAGothic"
    assert is_hiragino is False


def test_select_japanese_font_tries_other_hiragino_variants_in_order():
    font, is_hiragino = mod.select_japanese_font({"Hiragino Kaku Gothic Pro"})
    assert font == "Hiragino Kaku Gothic Pro"
    assert is_hiragino is True


# ------------------------------------------------------------------ load_per_seed / tft_pooled_per_seed

def test_load_per_seed_nests_by_opponent_age_group_seed(tmp_path):
    path = tmp_path / "per_seed.csv"
    write_per_seed_csv(path, [
        {"seed": 100, "age": 1000, "opponent": "TFT", "depth_group": "depthLow", "n": 60, "n_recovered": 6},
    ])
    out = mod.load_per_seed(path)
    assert out["TFT"][1000]["depthLow"][100] == (60, 6)


def test_tft_pooled_per_seed_sums_across_the_3_depth_groups_within_a_seed(tmp_path):
    path = tmp_path / "per_seed.csv"
    write_per_seed_csv(path, [
        {"seed": 100, "age": 1000, "opponent": "TFT", "depth_group": "depthLow", "n": 60, "n_recovered": 6},
        {"seed": 100, "age": 1000, "opponent": "TFT", "depth_group": "depthMid", "n": 30, "n_recovered": 3},
        {"seed": 100, "age": 1000, "opponent": "TFT", "depth_group": "depthHigh", "n": 90, "n_recovered": 9},
        {"seed": 101, "age": 1000, "opponent": "TFT", "depth_group": "depthLow", "n": 10, "n_recovered": 1},
        # a different opponent in the same file must NOT leak into TFT's pooled sums
        {"seed": 100, "age": 1000, "opponent": "SELF", "depth_group": "depthLow", "n": 1000, "n_recovered": 999},
    ])
    per_seed = mod.load_per_seed(path)
    pooled = mod.tft_pooled_per_seed(per_seed)
    assert pooled[1000][100] == (180, 18)
    assert pooled[1000][101] == (10, 1)


# ------------------------------------------------------------------ seed_bootstrap_recovered_rate

def test_seed_bootstrap_recovered_rate_none_with_fewer_than_2_seeds():
    assert mod.seed_bootstrap_recovered_rate({100: (10, 5)}, n_boot=100, rng=random.Random(0)) is None
    assert mod.seed_bootstrap_recovered_rate({}, n_boot=100, rng=random.Random(0)) is None


def test_seed_bootstrap_recovered_rate_band_straddles_the_point_estimate():
    # 4 seeds with recovery rates 0.0, 0.33, 0.67, 1.0 -- pooled point
    # estimate is 0.5; a 95% band over enough draws should bracket it.
    seed_counts = {0: (10, 0), 1: (9, 3), 2: (9, 6), 3: (10, 10)}
    lo, hi = mod.seed_bootstrap_recovered_rate(seed_counts, n_boot=2000, rng=random.Random(0))
    assert lo <= 0.5 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_seed_bootstrap_recovered_rate_degenerate_all_seeds_identical_gives_a_point_band():
    # every seed has the exact same rate -> every bootstrap draw is
    # identical -> lo == hi == that rate.
    seed_counts = {0: (10, 5), 1: (20, 10), 2: (30, 15)}
    lo, hi = mod.seed_bootstrap_recovered_rate(seed_counts, n_boot=200, rng=random.Random(0))
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_seed_bootstrap_recovered_rate_is_deterministic_given_the_same_rng_seed():
    seed_counts = {0: (10, 2), 1: (15, 9), 2: (8, 1), 3: (12, 6)}
    a = mod.seed_bootstrap_recovered_rate(seed_counts, n_boot=500, rng=random.Random(42))
    b = mod.seed_bootstrap_recovered_rate(seed_counts, n_boot=500, rng=random.Random(42))
    assert a == b


# ------------------------------------------------------------------ plot_academic_figure (smoke)

def _make_depth_fixture():
    by_depth = {
        opp: {
            age: {g: {"never_recovered_rate": str(round(0.5 + 0.01 * i, 4))} for i, g in enumerate(mod.DEPTH_GROUPS)}
            for age in (1000, 1050)
        }
        for opp in mod.OPPONENTS
    }
    per_seed = {
        opp: {
            age: {g: {100 + s: (20, 10 + s) for s in range(5)} for g in mod.DEPTH_GROUPS}
            for age in (1000, 1050)
        }
        for opp in mod.OPPONENTS
    }
    pooled = {"TFT": {age: {"never_recovered_rate": "0.85"} for age in (1000, 1050)}}
    return by_depth, per_seed, pooled


def test_plot_academic_figure_writes_a_png_without_crashing(tmp_path):
    by_depth, per_seed, pooled = _make_depth_fixture()
    tft_per_seed_pooled = mod.tft_pooled_per_seed(per_seed)
    out_path = tmp_path / "academic_SELF.png"
    mod.plot_academic_figure("SELF", by_depth, per_seed, tft_per_seed_pooled, pooled, n_boot=50, rng=random.Random(0), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
