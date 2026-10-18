"""Confirmatory-hypothesis statistics for the Markov-structure-evolution
production study -- docs/PREREGISTRATION_markov_structure_evolution.md
sections 4-5 (confirmed commit ccc0b1b7a76f22e150e7c097b1757b1fb8e54e0e).

Pure statistics only (no IO, no simulation) -- the orchestrating script
(scripts/run_markov_confirmatory_analysis.py) produces the raw bootstrap
difference distributions this module turns into p-values and verdicts.

The 6 directional comparisons (section 4), each predicted positive under
its hypothesis:
    H1 (TFT):  r(M) - r(S) > 0   and   r(F) - r(S) > 0
    H2 (AllD): r(S) - r(M) > 0   and   r(F) - r(M) > 0
    H3 (AllC): r(F) - r(S) > 0   and   r(F) - r(M) > 0
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

COMPARISONS: List[Tuple[str, str, str, str]] = [
    # (hypothesis, comparison_label, opponent, "minuend-subtrahend" groups)
    ("H1_TFT", "r(M)-r(S)", "TFT", "M-S"),
    ("H1_TFT", "r(F)-r(S)", "TFT", "F-S"),
    ("H2_AllD", "r(S)-r(M)", "AllD", "S-M"),
    ("H2_AllD", "r(F)-r(M)", "AllD", "F-M"),
    ("H3_AllC", "r(F)-r(S)", "AllC", "F-S"),
    ("H3_AllC", "r(F)-r(M)", "AllC", "F-M"),
]


def comparison_diff(r_by_group: Dict[str, float], groups: str) -> float:
    """groups like "M-S": r_by_group["M"] - r_by_group["S"]."""
    a, b = groups.split("-")
    return r_by_group[a] - r_by_group[b]


def one_sided_bootstrap_pvalue(bootstrap_diffs: Sequence[float]) -> float:
    """One-sided p for H0: diff <= 0 vs H1: diff > 0 (the hypothesis's
    predicted direction -- every comparison in COMPARISONS is written so
    the predicted sign is positive). Add-one continuity correction
    ((b+1)/(B+1)) so p is never reported as exactly 0 from a finite
    bootstrap -- standard practice for a bootstrap/permutation p-value."""
    b = sum(1 for d in bootstrap_diffs if d <= 0)
    return (b + 1) / (len(bootstrap_diffs) + 1)


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction. Returns a list parallel to
    `p_values` (same order, NOT sorted) of whether each is significant at
    `alpha` after correction."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, i in enumerate(order):  # rank 0 = smallest p
        threshold = alpha / (n - rank)
        if p_values[i] <= threshold:
            reject[i] = True
        else:
            break  # Holm stops rejecting at the first failure to reject
    return reject


@dataclass
class ComparisonResult:
    hypothesis: str
    label: str
    opponent: str
    point_estimate_diff: float
    p_value: float
    significant: bool


@dataclass
class HypothesisResult:
    hypothesis: str
    opponent: str
    comparisons: List[ComparisonResult]

    @property
    def verdict(self) -> str:
        n_sig = sum(1 for c in self.comparisons if c.significant)
        if n_sig == len(self.comparisons):
            return "support"
        if n_sig == 0:
            return "not supported"
        return "partially supported"


def evaluate_hypotheses(
    point_estimate_diffs: Dict[Tuple[str, str], float],
    bootstrap_diffs: Dict[Tuple[str, str], Sequence[float]],
    alpha: float = 0.05,
) -> List[HypothesisResult]:
    """point_estimate_diffs/bootstrap_diffs are keyed by (opponent, groups)
    e.g. ("TFT", "M-S") -> ... , matching COMPARISONS' (opponent, groups)
    pairs. Runs Holm correction across all 6 comparisons jointly (section
    5's "6つの比較にHolm法を適用"), then groups the per-comparison
    results back into the 3 hypotheses (section 5's verdict rule:
    support/partial/not-supported by how many of a hypothesis's 2
    comparisons are significant)."""
    p_values = [
        one_sided_bootstrap_pvalue(bootstrap_diffs[(opponent, groups)])
        for _, _, opponent, groups in COMPARISONS
    ]
    significances = holm_correction(p_values, alpha=alpha)

    by_hypothesis: Dict[str, List[ComparisonResult]] = {}
    for (hypothesis, label, opponent, groups), p, sig in zip(COMPARISONS, p_values, significances):
        result = ComparisonResult(
            hypothesis=hypothesis, label=label, opponent=opponent,
            point_estimate_diff=point_estimate_diffs[(opponent, groups)],
            p_value=p, significant=sig,
        )
        by_hypothesis.setdefault(hypothesis, []).append(result)

    opponent_of = {"H1_TFT": "TFT", "H2_AllD": "AllD", "H3_AllC": "AllC"}
    return [
        HypothesisResult(hypothesis=h, opponent=opponent_of[h], comparisons=comps)
        for h, comps in by_hypothesis.items()
    ]
