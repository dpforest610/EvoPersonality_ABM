"""Confirmatory-hypothesis statistics for the K-model Markov-structure
production study -- docs/PREREGISTRATION_k_model_markov.md sections 4-5
(confirmed commit: see that document's header).

Pure statistics only (no IO, no simulation) -- the orchestrating script
(scripts/run_k_model_confirmatory_analysis.py) produces the raw point
estimates, seed-bootstrap difference distributions, and permutation null
distributions this module turns into comparisons, p-values, Holm
correction, and the one_sided_bootstrap_pvalue/holm_correction functions
are ported unchanged from evopersonality_abm.markov_confirmatory (the
Lindgren-side equivalent, see docs/PREREGISTRATION_markov_structure_evolution.md
sections 4-5) -- same statistics, different comparison lists.

H-A's 6 directional comparisons (section 4), built from a 3x3 grid of
r(type, human_group) -- each predicted positive:
    HA1_Secure:  r(K_Secure,Secure)  - r(K_Secure,Mix)      > 0
                 r(K_Secure,Secure)  - r(K_Secure,Fearful)  > 0
    HA2_Mix:     r(Mix,Mix)          - r(Mix,Secure)        > 0
                 r(Mix,Mix)          - r(Mix,Fearful)        > 0
    HA3_Fearful: r(K_Fearful,Fearful)- r(K_Fearful,Secure)  > 0
                 r(K_Fearful,Fearful)- r(K_Fearful,Mix)      > 0

H-B's 2 directional comparisons -- never_recovered_rate(K_Fearful, opp)
is predicted higher than never_recovered_rate(K_Secure, opp):
    HB1_SELF: never_recovered_rate(K_Fearful,SELF) - never_recovered_rate(K_Secure,SELF) > 0
    HB2_POP:  never_recovered_rate(K_Fearful,POP)  - never_recovered_rate(K_Secure,POP)  > 0

H-C's comparisons are built at runtime (one per depth stratum that passes
the minimum-count rule, section 4) -- see HC_comparison_label/HC_STRATA
below rather than a fixed list, since which strata are testable depends
on the production data.
"""

from typing import Dict, List, Sequence, Tuple

HA_COMPARISONS: List[Tuple[str, str, str, str, str]] = [
    # (hypothesis, comparison_label, type, group_same, group_other)
    ("HA1_Secure", "r(K_Secure,Secure)-r(K_Secure,Mix)", "K_Secure", "Secure", "Mix"),
    ("HA1_Secure", "r(K_Secure,Secure)-r(K_Secure,Fearful)", "K_Secure", "Secure", "Fearful"),
    ("HA2_Mix", "r(Mix,Mix)-r(Mix,Secure)", "Mix", "Mix", "Secure"),
    ("HA2_Mix", "r(Mix,Mix)-r(Mix,Fearful)", "Mix", "Mix", "Fearful"),
    ("HA3_Fearful", "r(K_Fearful,Fearful)-r(K_Fearful,Secure)", "K_Fearful", "Fearful", "Secure"),
    ("HA3_Fearful", "r(K_Fearful,Fearful)-r(K_Fearful,Mix)", "K_Fearful", "Fearful", "Mix"),
]

HB_COMPARISONS: List[Tuple[str, str, str]] = [
    # (hypothesis, comparison_label, opponent)
    ("HB1_SELF", "never_recovered_rate(K_Fearful,SELF)-never_recovered_rate(K_Secure,SELF)", "SELF"),
    ("HB2_POP", "never_recovered_rate(K_Fearful,POP)-never_recovered_rate(K_Secure,POP)", "POP"),
]

# H-C's depth strata (section 4): cutpoints 2,3 on effective_memory_depth.
HC_STRATA = ["depthLow", "depthMid", "depthHigh"]
HC_MIN_STRATUM_N = 20


def depth_stratum(effective_depth: int) -> str:
    """depthLow=0-2, depthMid=3, depthHigh=4+ (preregistration section 4)."""
    if effective_depth <= 2:
        return "depthLow"
    if effective_depth == 3:
        return "depthMid"
    return "depthHigh"


def ha_comparison_diff(r_by_type_group: Dict[Tuple[str, str], float], type_: str, group_same: str, group_other: str) -> float:
    return r_by_type_group[(type_, group_same)] - r_by_type_group[(type_, group_other)]


def hb_comparison_diff(rate_by_type_opponent: Dict[Tuple[str, str], float], opponent: str) -> float:
    return rate_by_type_opponent[("K_Fearful", opponent)] - rate_by_type_opponent[("K_Secure", opponent)]


def hc_comparison_diff(rate_by_type_stratum: Dict[Tuple[str, str], float], stratum: str) -> float:
    """Same direction as HB1 (SELF only, section 4 H-C1), restricted to one depth stratum."""
    return rate_by_type_stratum[("K_Fearful", stratum)] - rate_by_type_stratum[("K_Secure", stratum)]


def one_sided_bootstrap_pvalue(bootstrap_diffs: Sequence[float]) -> float:
    """One-sided p for H0: diff <= 0 vs H1: diff > 0 (every comparison
    above is written so the predicted sign is positive). Add-one
    continuity correction ((b+1)/(B+1)) so p is never reported as
    exactly 0 from a finite bootstrap. Ported unchanged from
    evopersonality_abm.markov_confirmatory."""
    b = sum(1 for d in bootstrap_diffs if d <= 0)
    return (b + 1) / (len(bootstrap_diffs) + 1)


def permutation_pvalue(true_diff: float, null_diffs: Sequence[float]) -> float:
    """One-sided empirical p for the label-permutation test (preregistration
    section 5-4): P(null >= true_diff), continuity-corrected the same way
    as one_sided_bootstrap_pvalue. `true_diff` must be computed with the
    SAME sampling procedure (draw count) as `null_diffs` -- see
    docs/PREREGISTRATION_k_model_markov.md section 5-4 step 3."""
    b = sum(1 for d in null_diffs if d >= true_diff)
    return (b + 1) / (len(null_diffs) + 1)


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down correction. Returns a list parallel to
    `p_values` (same order, NOT sorted) of whether each is significant at
    `alpha` after correction. Ported unchanged from
    evopersonality_abm.markov_confirmatory."""
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


def verdict(rejects: Sequence[bool]) -> str:
    """"支持" if every comparison in the group is significant, "部分的に
    支持" if only some are, "支持されない" if none are (preregistration
    section 5-3's rule, same as the Lindgren precedent)."""
    if not rejects:
        return "検定不能"
    if all(rejects):
        return "支持"
    if any(rejects):
        return "部分的に支持"
    return "支持されない"
