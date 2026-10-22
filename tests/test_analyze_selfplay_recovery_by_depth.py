import csv
import importlib.util
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
