import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "plot_markov_confirmatory_sampling_comparison",
    REPO_ROOT / "scripts" / "plot_markov_confirmatory_sampling_comparison.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_markov_confirmatory_sampling_comparison"] = mod
spec.loader.exec_module(mod)


def test_series_for_filters_by_opponent_scheme_and_group_size():
    rows = [
        {"opponent": "TFT", "scheme": "A", "age": "1000", "group_size": "56", "mean_n_distinct_genotypes": "3.0"},
        {"opponent": "TFT", "scheme": "B", "age": "1000", "group_size": "56", "mean_n_distinct_genotypes": "9.0"},
        {"opponent": "AllD", "scheme": "A", "age": "1000", "group_size": "56", "mean_n_distinct_genotypes": "99.0"},
        {"opponent": "TFT", "scheme": "A", "age": "1000", "group_size": "46", "mean_n_distinct_genotypes": "99.0"},
    ]
    ages, vals = mod.series_for(rows, "TFT", "A", 56, "mean_n_distinct_genotypes")
    assert ages == [1000]
    assert vals == [3.0]


def test_series_for_averages_multiple_rows_at_the_same_age():
    rows = [
        {"opponent": "TFT", "scheme": "B", "age": "1000", "group_size": "56", "mean_recovery": "0.2"},
        {"opponent": "TFT", "scheme": "B", "age": "1000", "group_size": "56", "mean_recovery": "0.4"},
    ]
    ages, vals = mod.series_for(rows, "TFT", "B", 56, "mean_recovery")
    assert ages == [1000]
    assert abs(vals[0] - 0.3) < 1e-9


def test_series_for_returns_empty_when_no_matching_rows():
    ages, vals = mod.series_for([], "TFT", "B", 56, "mean_recovery")
    assert ages == [] and vals == []
