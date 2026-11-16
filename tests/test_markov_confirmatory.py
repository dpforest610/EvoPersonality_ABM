from evopersonality_abm.markov_confirmatory import (
    COMPARISONS,
    comparison_diff,
    evaluate_hypotheses,
    holm_correction,
    one_sided_bootstrap_pvalue,
)


# ------------------------------------------------------------------ comparison_diff

def test_comparison_diff_subtracts_in_the_given_order():
    r = {"S": 0.1, "M": 0.4, "F": 0.3}
    assert comparison_diff(r, "M-S") == 0.4 - 0.1
    assert comparison_diff(r, "F-S") == 0.3 - 0.1
    assert comparison_diff(r, "F-M") == 0.3 - 0.4


# ------------------------------------------------------------------ one_sided_bootstrap_pvalue

def test_one_sided_bootstrap_pvalue_all_positive_gives_small_p():
    diffs = [0.1] * 999 + [-0.01]  # 1/1000 at or below 0
    p = one_sided_bootstrap_pvalue(diffs)
    assert abs(p - (1 + 1) / (1000 + 1)) < 1e-12


def test_one_sided_bootstrap_pvalue_all_nonpositive_gives_p_near_one():
    diffs = [-0.1] * 500
    p = one_sided_bootstrap_pvalue(diffs)
    assert abs(p - (500 + 1) / (500 + 1)) < 1e-12
    assert p == 1.0


def test_one_sided_bootstrap_pvalue_never_exactly_zero():
    diffs = [0.5] * 2000  # every single bootstrap draw supports the hypothesis
    p = one_sided_bootstrap_pvalue(diffs)
    assert p > 0.0
    assert abs(p - 1 / 2001) < 1e-12


def test_one_sided_bootstrap_pvalue_zero_counts_as_not_supporting():
    diffs = [0.0, 0.1, 0.1]  # diff<=0 includes exactly 0
    p = one_sided_bootstrap_pvalue(diffs)
    assert abs(p - (1 + 1) / (3 + 1)) < 1e-12


# ------------------------------------------------------------------ holm_correction

def test_holm_correction_hand_computed_example():
    # classic textbook-style example, alpha=.05, 4 comparisons:
    # sorted p-values: .01, .02, .03, .20
    # thresholds (rank 0..3): .05/4=.0125, .05/3=.01667, .05/2=.025, .05/1=.05
    # step-down: p=.01 <= .0125 -> reject; p=.02 <= .01667? NO -> stop (Holm halts at first non-reject)
    p_values = [0.03, 0.01, 0.20, 0.02]  # unsorted order on purpose
    reject = holm_correction(p_values, alpha=0.05)
    # only the smallest p-value (.01, index 1) should be rejected
    assert reject == [False, True, False, False]


def test_holm_correction_all_significant_when_all_pvalues_tiny():
    p_values = [0.0001] * 6
    assert holm_correction(p_values, alpha=0.05) == [True] * 6


def test_holm_correction_none_significant_when_all_pvalues_large():
    p_values = [0.9] * 6
    assert holm_correction(p_values, alpha=0.05) == [False] * 6


def test_holm_correction_is_at_least_as_conservative_as_uncorrected():
    # Holm-rejected set must be a subset of what plain p<alpha would reject
    p_values = [0.01, 0.03, 0.04, 0.2, 0.001, 0.049]
    reject = holm_correction(p_values, alpha=0.05)
    uncorrected = [p < 0.05 for p in p_values]
    for r, u in zip(reject, uncorrected):
        assert (not r) or u  # r implies u


# ------------------------------------------------------------------ evaluate_hypotheses

def _make_diffs(value, n=100, noise=0.0):
    import random
    rng = random.Random(0)
    return [value + (rng.random() - 0.5) * noise for _ in range(n)]


def test_evaluate_hypotheses_full_support_when_both_comparisons_strongly_positive():
    point_estimates = {}
    bootstrap_diffs = {}
    for hyp, label, opponent, groups in COMPARISONS:
        point_estimates[(opponent, groups)] = 0.3
        bootstrap_diffs[(opponent, groups)] = _make_diffs(0.3, n=500, noise=0.05)

    results = evaluate_hypotheses(point_estimates, bootstrap_diffs, alpha=0.05)
    assert len(results) == 3
    for r in results:
        assert r.verdict == "support"
        assert len(r.comparisons) == 2


def test_evaluate_hypotheses_not_supported_when_diffs_are_negative():
    point_estimates = {}
    bootstrap_diffs = {}
    for hyp, label, opponent, groups in COMPARISONS:
        point_estimates[(opponent, groups)] = -0.2
        bootstrap_diffs[(opponent, groups)] = _make_diffs(-0.2, n=500, noise=0.05)

    results = evaluate_hypotheses(point_estimates, bootstrap_diffs, alpha=0.05)
    for r in results:
        assert r.verdict == "not supported"


def test_evaluate_hypotheses_partial_support_when_only_one_comparison_significant():
    point_estimates = {}
    bootstrap_diffs = {}
    for hyp, label, opponent, groups in COMPARISONS:
        if opponent == "TFT":
            # H1's two comparisons: one strongly positive, one centered on zero/negative
            if groups == "M-S":
                point_estimates[(opponent, groups)] = 0.4
                bootstrap_diffs[(opponent, groups)] = _make_diffs(0.4, n=500, noise=0.02)
            else:
                point_estimates[(opponent, groups)] = -0.3
                bootstrap_diffs[(opponent, groups)] = _make_diffs(-0.3, n=500, noise=0.02)
        else:
            point_estimates[(opponent, groups)] = 0.0
            bootstrap_diffs[(opponent, groups)] = _make_diffs(-0.5, n=500, noise=0.02)

    results = evaluate_hypotheses(point_estimates, bootstrap_diffs, alpha=0.05)
    h1 = next(r for r in results if r.hypothesis == "H1_TFT")
    assert h1.verdict == "partially supported"
    sig_labels = {c.label for c in h1.comparisons if c.significant}
    assert sig_labels == {"r(M)-r(S)"}


def test_evaluate_hypotheses_returns_all_three_opponents():
    point_estimates = {}
    bootstrap_diffs = {}
    for hyp, label, opponent, groups in COMPARISONS:
        point_estimates[(opponent, groups)] = 0.1
        bootstrap_diffs[(opponent, groups)] = _make_diffs(0.1, n=200, noise=0.03)
    results = evaluate_hypotheses(point_estimates, bootstrap_diffs)
    assert {r.opponent for r in results} == {"TFT", "AllD", "AllC"}
