import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "plot_markov_confirmatory_trends", REPO_ROOT / "scripts" / "plot_markov_confirmatory_trends.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_markov_confirmatory_trends"] = mod
spec.loader.exec_module(mod)


def test_cycle_threshold_columns_extracts_and_sorts_thresholds():
    rows = [
        {"mean_cycle_count_0.3": "5", "other": "x"},
        {"mean_cycle_count_0.1": "2", "mean_cycle_count_0.5": "1"},
    ]
    assert mod.cycle_threshold_columns(rows) == [0.1, 0.3, 0.5]


def test_cycle_threshold_columns_empty_when_absent():
    rows = [{"mean_recovery": "0.5"}]
    assert mod.cycle_threshold_columns(rows) == []


def test_get_returns_nan_for_missing_or_empty():
    import math
    assert math.isnan(mod.get({}, "missing"))
    assert math.isnan(mod.get({"x": ""}, "x"))
    assert mod.get({"x": "0.5"}, "x") == 0.5


def test_mean_ignoring_nan_skips_nan_values():
    import math
    values = [1.0, float("nan"), 3.0]
    assert mod.mean_ignoring_nan(values) == 2.0


def test_mean_ignoring_nan_all_nan_returns_nan():
    import math
    assert math.isnan(mod.mean_ignoring_nan([float("nan"), float("nan")]))


def test_aggregate_filters_by_scheme_opponent_and_group_size():
    rows = [
        {"scheme": "A", "opponent": "TFT", "age": "1000", "group_size": "56", "mean_recovery": "0.9"},  # wrong scheme
        {"scheme": "B", "opponent": "AllD", "age": "1000", "group_size": "56", "mean_recovery": "0.9"},  # wrong opponent
        {"scheme": "B", "opponent": "TFT", "age": "1000", "group_size": "46", "mean_recovery": "0.9"},  # wrong group_size
        {"scheme": "B", "opponent": "TFT", "age": "1000", "group_size": "56", "mean_recovery": "0.3"},
    ]
    by_age = mod.aggregate(rows, "TFT", 56)
    assert list(by_age.keys()) == [1000]
    assert by_age[1000]["mean_recovery"] == [0.3]


def test_human_label_of_covers_only_opponents_with_recovery_reference():
    assert mod.HUMAN_LABEL_OF == {"TFT": "TFT", "Random": "RAN", "Gradual": "GRA"}
    # AllC/AllD/GradualAppeasement/WSLS have no human recovery/descent reference
    for opp in ("AllC", "AllD", "GradualAppeasement", "WSLS"):
        assert opp not in mod.HUMAN_LABEL_OF
