from evopersonality_abm.k_model_confirmatory import (
    HA_COMPARISONS,
    HB_COMPARISONS,
    depth_stratum,
    ha_comparison_diff,
    hb_comparison_diff,
    hc_comparison_diff,
    holm_correction,
    one_sided_bootstrap_pvalue,
    permutation_pvalue,
    verdict,
)


def test_ha_comparisons_cover_all_6_and_both_directions_per_type():
    assert len(HA_COMPARISONS) == 6
    hypotheses = {c[0] for c in HA_COMPARISONS}
    assert hypotheses == {"HA1_Secure", "HA2_Mix", "HA3_Fearful"}
    for hyp in hypotheses:
        assert sum(1 for c in HA_COMPARISONS if c[0] == hyp) == 2


def test_hb_comparisons_cover_self_and_pop():
    assert len(HB_COMPARISONS) == 2
    assert {c[2] for c in HB_COMPARISONS} == {"SELF", "POP"}


def test_ha_comparison_diff():
    r = {("K_Secure", "Secure"): 0.8, ("K_Secure", "Mix"): 0.3, ("K_Secure", "Fearful"): 0.1}
    assert ha_comparison_diff(r, "K_Secure", "Secure", "Mix") == 0.5


def test_hb_comparison_diff_fearful_minus_secure():
    rate = {("K_Fearful", "SELF"): 0.6, ("K_Secure", "SELF"): 0.2}
    assert abs(hb_comparison_diff(rate, "SELF") - 0.4) < 1e-12


def test_hc_comparison_diff():
    rate = {("K_Fearful", "depthLow"): 0.5, ("K_Secure", "depthLow"): 0.2}
    assert hc_comparison_diff(rate, "depthLow") == 0.3


def test_depth_stratum_cutpoints():
    assert depth_stratum(0) == "depthLow"
    assert depth_stratum(2) == "depthLow"
    assert depth_stratum(3) == "depthMid"
    assert depth_stratum(4) == "depthHigh"
    assert depth_stratum(8) == "depthHigh"


def test_one_sided_bootstrap_pvalue_all_positive_is_small():
    p = one_sided_bootstrap_pvalue([0.1] * 100)
    assert p == 1 / 101


def test_one_sided_bootstrap_pvalue_all_negative_is_large():
    p = one_sided_bootstrap_pvalue([-0.1] * 100)
    assert p == 101 / 101


def test_permutation_pvalue_true_diff_beats_all_null_is_small():
    p = permutation_pvalue(1.0, [0.1] * 100)
    assert p == 1 / 101


def test_permutation_pvalue_true_diff_matches_null_distribution():
    null = [0.0, 0.5, 1.0, 1.5]
    # true_diff=1.0: null values >=1.0 are {1.0, 1.5} -> b=2
    assert permutation_pvalue(1.0, null) == 3 / 5


def test_holm_correction_single_comparison():
    assert holm_correction([0.01]) == [True]
    assert holm_correction([0.5]) == [False]


def test_holm_correction_stops_at_first_failure():
    # sorted order: 0.01, 0.02, 0.5 -- thresholds .05/3, .05/2, .05/1
    p_values = [0.5, 0.01, 0.02]
    rejects = holm_correction(p_values)
    assert rejects == [False, True, True]


def test_verdict_all_significant_is_supported():
    assert verdict([True, True]) == "支持"


def test_verdict_some_significant_is_partial():
    assert verdict([True, False]) == "部分的に支持"


def test_verdict_none_significant_is_not_supported():
    assert verdict([False, False]) == "支持されない"


def test_verdict_empty_is_untestable():
    assert verdict([]) == "検定不能"
