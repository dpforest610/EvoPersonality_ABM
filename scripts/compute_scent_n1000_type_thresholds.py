"""Fixed median-split thresholds for the heatbugs-style 4-type figures
(docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md additions B and C): pooled
median of K_self and K_out across ALL 30 seeds x all 3 conditions
(M0/M1/M2), gen>=2000, from the committed N=1000 production individual
logs. Computed once here and used as a fixed constant everywhere else
(scripts/plot_scent_n1000_type_summary.py, scripts/animate_scent_n1000_heatbugs.py)
-- replacing the seed-0-only prototype thresholds (0.7835265 / 0.4566165).

Usage:
    python3 scripts/compute_scent_n1000_type_thresholds.py
"""

import gzip
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("scripts/output/scent_spatial_sweep_n1000")
MACHINE = "doeRyanoMacBook-Neo.local"
CONDITIONS = ["none", "random", "scent"]
SEEDS = list(range(30))
STEADY_STATE_MIN_GEN = 2000

OUT_JSON = ROOT / "analysis" / "type_thresholds.json"


def main():
    kself_all, kout_all = [], []
    n_rows = 0
    per_condition_n = {}
    for cond in CONDITIONS:
        cond_n = 0
        for seed in SEEDS:
            path = ROOT / cond / MACHINE / f"seed_{seed}_individual_log.csv.gz"
            with gzip.open(path, "rt") as f:
                for row in csv.DictReader(f):
                    if int(row["generation"]) >= STEADY_STATE_MIN_GEN:
                        kself_all.append(float(row["k_self"]))
                        kout_all.append(float(row["k_out"]))
                        cond_n += 1
        per_condition_n[cond] = cond_n
        n_rows += cond_n
        print(f"{cond}: {cond_n} rows (30 seeds x gen>=2000)")

    kself_all = np.array(kself_all)
    kout_all = np.array(kout_all)
    threshold_kself = float(np.median(kself_all))
    threshold_kout = float(np.median(kout_all))

    print(f"\npooled n = {n_rows} (= 3 conditions x 30 seeds x 30 snapshots x 1000 agents)")
    print(f"median K_self (threshold_kself) = {threshold_kself:.7f}")
    print(f"median K_out  (threshold_kout)  = {threshold_kout:.7f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            dict(
                threshold_kself=threshold_kself, threshold_kout=threshold_kout,
                n_rows=n_rows, per_condition_n=per_condition_n,
                conditions=CONDITIONS, seeds=SEEDS, steady_state_min_gen=STEADY_STATE_MIN_GEN,
                note="pooled across all 3 conditions x 30 seeds x gen>=2000, replacing the "
                     "seed-0-only prototype thresholds (0.7835265 / 0.4566165)",
            ),
            f, indent=2,
        )
        f.write("\n")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
