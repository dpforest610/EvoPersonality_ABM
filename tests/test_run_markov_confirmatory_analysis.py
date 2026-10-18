import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "run_markov_confirmatory_analysis", REPO_ROOT / "scripts" / "run_markov_confirmatory_analysis.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_markov_confirmatory_analysis"] = mod
spec.loader.exec_module(mod)

GROUP_FOR_HUMAN = {"Secure": 56, "Mix": 46, "Fearful": 69}


# ------------------------------------------------------------------ compute_point_estimates

def test_compute_point_estimates_averages_within_window_only():
    rows = [
        {"scheme": "B", "opponent": "TFT", "age": 900, "group_size": 56, "mean_corr_Secure": 0.9},  # outside window
        {"scheme": "B", "opponent": "TFT", "age": 1000, "group_size": 56, "mean_corr_Secure": 0.1},
        {"scheme": "B", "opponent": "TFT", "age": 1050, "group_size": 56, "mean_corr_Secure": 0.3},
        {"scheme": "B", "opponent": "TFT", "age": 5100, "group_size": 56, "mean_corr_Secure": 0.9},  # outside window
    ]
    result = mod.compute_point_estimates(rows, (1000, 5000), GROUP_FOR_HUMAN)
    assert abs(result[("TFT", "S")] - 0.2) < 1e-9  # mean(0.1, 0.3)


def test_compute_point_estimates_ignores_scheme_a_and_non_h_opponents():
    rows = [
        {"scheme": "A", "opponent": "TFT", "age": 1000, "group_size": 56, "mean_corr_Secure": 0.9},
        {"scheme": "B", "opponent": "Random", "age": 1000, "group_size": 56, "mean_corr_Secure": 0.9},
        {"scheme": "B", "opponent": "TFT", "age": 1000, "group_size": 56, "mean_corr_Secure": 0.5},
    ]
    result = mod.compute_point_estimates(rows, (1000, 5000), GROUP_FOR_HUMAN)
    assert result == {("TFT", "S"): 0.5}


def test_compute_point_estimates_matches_group_size_to_the_right_human_letter():
    rows = [
        {"scheme": "B", "opponent": "AllD", "age": 1000, "group_size": 56, "mean_corr_Secure": 0.2, "mean_corr_Mix": 0.9, "mean_corr_Fearful": 0.9},
        {"scheme": "B", "opponent": "AllD", "age": 1000, "group_size": 46, "mean_corr_Secure": 0.9, "mean_corr_Mix": 0.4, "mean_corr_Fearful": 0.9},
        {"scheme": "B", "opponent": "AllD", "age": 1000, "group_size": 69, "mean_corr_Secure": 0.9, "mean_corr_Mix": 0.9, "mean_corr_Fearful": 0.6},
    ]
    result = mod.compute_point_estimates(rows, (1000, 5000), GROUP_FOR_HUMAN)
    # row with group_size=56 contributes only to "S" (via mean_corr_Secure), etc.
    assert abs(result[("AllD", "S")] - 0.2) < 1e-9
    assert abs(result[("AllD", "M")] - 0.4) < 1e-9
    assert abs(result[("AllD", "F")] - 0.6) < 1e-9


def test_compute_point_estimates_skips_nan():
    rows = [
        {"scheme": "B", "opponent": "TFT", "age": 1000, "group_size": 56, "mean_corr_Secure": float("nan")},
        {"scheme": "B", "opponent": "TFT", "age": 1050, "group_size": 56, "mean_corr_Secure": 0.4},
    ]
    result = mod.compute_point_estimates(rows, (1000, 5000), GROUP_FOR_HUMAN)
    assert abs(result[("TFT", "S")] - 0.4) < 1e-9


# ------------------------------------------------------------------ build_bootstrap_diff_distributions

def test_build_bootstrap_diff_distributions_averages_across_ages_per_iteration():
    # iteration 0: age1000 gives TFT S=0.1,M=0.2,F=0.3; age1050 gives S=0.3,M=0.4,F=0.5
    #   -> time-window avg: S=0.2, M=0.3, F=0.4
    # need all 3 opponents present for iteration 0 to be included
    sums = {}
    counts = {}
    for opponent in ("TFT", "AllD", "AllC"):
        for age, (s, m, f) in [(1000, (0.1, 0.2, 0.3)), (1050, (0.3, 0.4, 0.5))]:
            for group, val in zip(("Secure", "Mix", "Fearful"), (s, m, f)):
                key = (0, opponent, group)
                sums[key] = sums.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1

    diffs = mod.build_bootstrap_diff_distributions(sums, counts, n_bootstrap_seed=1)
    assert len(diffs[("TFT", "M-S")]) == 1
    assert abs(diffs[("TFT", "M-S")][0] - (0.3 - 0.2)) < 1e-9
    assert abs(diffs[("TFT", "F-S")][0] - (0.4 - 0.2)) < 1e-9


def test_build_bootstrap_diff_distributions_drops_iterations_with_missing_data():
    # iteration 0 has data for all 3 opponents; iteration 1 is missing AllC entirely
    sums, counts = {}, {}
    for opponent in ("TFT", "AllD", "AllC"):
        for group in ("Secure", "Mix", "Fearful"):
            sums[(0, opponent, group)] = 0.3
            counts[(0, opponent, group)] = 1
    for opponent in ("TFT", "AllD"):  # AllC missing for iteration 1
        for group in ("Secure", "Mix", "Fearful"):
            sums[(1, opponent, group)] = 0.5
            counts[(1, opponent, group)] = 1

    diffs = mod.build_bootstrap_diff_distributions(sums, counts, n_bootstrap_seed=2)
    # only iteration 0 is complete -> every comparison has exactly 1 value
    for key, values in diffs.items():
        assert len(values) == 1


def test_build_bootstrap_diff_distributions_covers_all_six_comparisons():
    sums, counts = {}, {}
    for i in range(3):
        for opponent in ("TFT", "AllD", "AllC"):
            for group in ("Secure", "Mix", "Fearful"):
                sums[(i, opponent, group)] = 0.1 * i
                counts[(i, opponent, group)] = 1
    diffs = mod.build_bootstrap_diff_distributions(sums, counts, n_bootstrap_seed=3)
    expected_keys = {(o, g) for _h, _l, o, g in mod.COMPARISONS}
    assert set(diffs.keys()) == expected_keys
    for values in diffs.values():
        assert len(values) == 3


# ------------------------------------------------------------------ age_cache_path

def test_age_cache_path_is_unique_per_age(tmp_path):
    p1 = mod.age_cache_path(tmp_path, 1000)
    p2 = mod.age_cache_path(tmp_path, 1050)
    assert p1 != p2
    assert p1.parent == tmp_path
