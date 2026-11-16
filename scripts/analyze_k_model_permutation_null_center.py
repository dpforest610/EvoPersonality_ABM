"""Exploratory (NOT preregistered): the label-permutation test's null
distribution center, for each H-A/H-B/H-C comparison -- see
docs/K_MODEL_MARKOV_RESULTS.md section 3.

scripts/run_k_model_confirmatory_analysis.py's permutation test (preregistration
section 5-4) only reports WHERE the true statistic falls in the null
distribution (the p-value) -- it never reports the null distribution's own
mean. Because labels are shuffled WITHIN each (seed, age) population (not
across seeds), the null distribution can be centered away from 0: it still
reflects whatever between-lineage differences existed before shuffling,
just with the type assignment randomized. This script computes, for each
comparison:
    true_diff_window_avg  -- the real-label statistic, window-averaged
                              across ages (same value already in
                              k_model_confirmatory_summary.json's
                              "permutation" rows)
    null_center            -- the permutation null distribution's mean
                              (window-averaged per iteration across ages,
                              same procedure as the true statistic and as
                              run_k_model_confirmatory_analysis.build_results'
                              own seed-bootstrap/permutation pooling)
    effect_type_only        -- true_diff_window_avg - null_center: the part
                              of the real difference that is NOT explained
                              by shuffling labels within seeds (i.e. not
                              explained by between-lineage differences
                              alone)

Reads the per-age cache (scripts/run_k_model_confirmatory_analysis.py's
--results-dir/_age_cache/age_<N>.pkl) -- these are Mac-local only
(gitignored, never pushed), so THIS SCRIPT MUST RUN ON THE MAC where the
confirmatory analysis was run. Writes one small CSV, which IS meant to be
committed and pushed.

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 9):
    venv/bin/python scripts/analyze_k_model_permutation_null_center.py \\
        --results-dir scripts/output/k_model_confirmatory_production \\
        --window-start 1000 --window-end 5000 \\
        --out-csv scripts/output/k_model_confirmatory_production/permutation_null_center.csv
"""

import argparse
import csv
import importlib.util
import pickle
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("run_k_model_confirmatory_analysis", SCRIPTS_DIR / "run_k_model_confirmatory_analysis.py")
confirmatory_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_k_model_confirmatory_analysis"] = confirmatory_mod
_spec.loader.exec_module(confirmatory_mod)

from evopersonality_abm.k_model_confirmatory import HA_COMPARISONS, HB_COMPARISONS, HC_STRATA  # noqa: E402

import statistics  # noqa: E402


def load_cached_ages(cache_dir: Path, window_start: int, window_end: int):
    results = []
    for p in sorted(cache_dir.glob("age_*.pkl")):
        age = int(p.stem.split("_")[1])
        if not (window_start <= age <= window_end):
            continue
        with open(p, "rb") as f:
            results.append(confirmatory_mod.normalize_cached_result(pickle.load(f)))
    return results


def build_rows(results):
    # window_average_per_iteration is a nested function inside
    # run_k_model_confirmatory_analysis.build_results, not a top-level
    # name -- reimplemented here identically (same inputs/outputs) rather
    # than refactored out, to avoid touching the already-tested,
    # already-pushed confirmatory script for a non-preregistered,
    # exploratory follow-up.
    from collections import defaultdict

    def _window_average_per_iteration(per_age_dicts):
        by_label_iter = defaultdict(lambda: defaultdict(list))
        for d in per_age_dicts:
            for label, by_iter in d.items():
                for i, v in by_iter.items():
                    by_label_iter[label][i].append(v)
        return {label: [statistics.fmean(vs) for vs in by_iter.values()] for label, by_iter in by_label_iter.items()}

    ha_null = _window_average_per_iteration([r["ha_perm_null"] for r in results])
    hb_null = _window_average_per_iteration([r["hb_perm_null"] for r in results])
    hc_null = _window_average_per_iteration([r["hc_perm_null"] for r in results])

    rows = []
    for hyp, label, type_, g_same, g_other in HA_COMPARISONS:
        trues = [r["ha_perm_true"].get(label) for r in results if label in r["ha_perm_true"]]
        nulls = ha_null.get(label, [])
        if trues and nulls:
            true_avg = statistics.fmean(trues)
            null_center = statistics.fmean(nulls)
            rows.append({"hypothesis": hyp, "comparison": label, "true_diff_window_avg": true_avg, "null_center": null_center, "effect_type_only": true_avg - null_center, "n_permutations_used": len(nulls)})
    for hyp, label, opponent in HB_COMPARISONS:
        trues = [r["hb_perm_true"].get(label) for r in results if label in r["hb_perm_true"]]
        nulls = hb_null.get(label, [])
        if trues and nulls:
            true_avg = statistics.fmean(trues)
            null_center = statistics.fmean(nulls)
            rows.append({"hypothesis": hyp, "comparison": label, "true_diff_window_avg": true_avg, "null_center": null_center, "effect_type_only": true_avg - null_center, "n_permutations_used": len(nulls)})
    for stratum in HC_STRATA:
        trues = [r["hc_perm_true"].get(stratum) for r in results if stratum in r["hc_perm_true"]]
        nulls = hc_null.get(stratum, [])
        if trues and nulls:
            true_avg = statistics.fmean(trues)
            null_center = statistics.fmean(nulls)
            rows.append({"hypothesis": "HC1", "comparison": stratum, "true_diff_window_avg": true_avg, "null_center": null_center, "effect_type_only": true_avg - null_center, "n_permutations_used": len(nulls)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    cache_dir = Path(args.results_dir) / "_age_cache"
    results = load_cached_ages(cache_dir, args.window_start, args.window_end)
    print(f"loaded {len(results)} cached ages from {cache_dir}", flush=True)

    rows = build_rows(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hypothesis", "comparison", "true_diff_window_avg", "null_center", "effect_type_only", "n_permutations_used"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"wrote {out_path} ({len(rows)} rows)")
    for row in rows:
        print(f"  {row['hypothesis']:12s} {row['comparison']:55s} true={row['true_diff_window_avg']:+.4f} null_center={row['null_center']:+.4f} effect_type_only={row['effect_type_only']:+.4f}")


if __name__ == "__main__":
    main()
