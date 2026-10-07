"""Verifies the shuffle-replay production run (scripts/output/
local_placement_sweep_shuffle_replay/) numerically reproduces the
original precision k-sweep (scripts/output/opponent_count_precision_
sweep/) for k=4,8,16,32, all 50 seeds -- same seed, same PARAMS,
placement="shuffle" is untouched code, so this should be an exact
replay (see docs/PREREGISTRATION_local_placement.md step 2: if it does
NOT match, that means the replay's own code has a bug, and the run
must be investigated before any further analysis).

Checks, per (k, seed):
  - individual_log.csv's shared columns (generation, agent_index,
    k_out, k_self, memory_depth, payoff) -- exact string match (both
    files are written with the same ":.6f" formatting), numeric
    tolerance as a fallback diagnostic only.
  - timeseries.csv's shared columns (generation, mean_k_out, std_k_out,
    mean_k_self, std_k_self, mean_depth, mean_payoff).

Usage:
    python3 scripts/verify_shuffle_replay_matches_existing.py
"""

import csv
from pathlib import Path

REPLAY_ROOT = Path("scripts/output/local_placement_sweep_shuffle_replay")
EXISTING_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 8, 16, 32]
SEEDS = list(range(50))

SHARED_INDIVIDUAL_COLS = ["generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff"]
SHARED_TIMESERIES_COLS = ["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
                          "mean_depth", "mean_payoff"]
NUMERIC_TOL = 1e-9


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def compare_rows(replay_rows, existing_rows, cols, label):
    if len(replay_rows) != len(existing_rows):
        return [f"{label}: row count mismatch ({len(replay_rows)} vs {len(existing_rows)})"]
    mismatches = []
    for i, (r, e) in enumerate(zip(replay_rows, existing_rows)):
        for col in cols:
            rv, ev = r[col], e[col]
            if rv == ev:
                continue
            try:
                if abs(float(rv) - float(ev)) <= NUMERIC_TOL:
                    continue
            except ValueError:
                pass
            mismatches.append(f"{label} row {i} col {col}: replay={rv!r} existing={ev!r}")
            if len(mismatches) >= 5:
                return mismatches
    return mismatches


def main():
    print(f"Verifying shuffle replay against existing precision-sweep data "
          f"(k={K_VALUES}, {len(SEEDS)} seeds each)...\n")

    missing = []
    for k in K_VALUES:
        replay_dir = REPLAY_ROOT / f"k{k}"
        for s in SEEDS:
            if not (replay_dir / f"seed_{s}_timeseries.csv").exists():
                missing.append((k, s))
    if missing:
        print(f"Replay not yet complete: {len(missing)} (k,seed) units missing, "
              f"e.g. {missing[:5]}. Stopping -- re-run once shuffle production finishes.")
        return

    all_mismatches = []
    n_checked = 0
    for k in K_VALUES:
        replay_dir = REPLAY_ROOT / f"k{k}"
        existing_dir = EXISTING_ROOT / f"k{k}"
        for s in SEEDS:
            n_checked += 1
            replay_ind = load_csv(replay_dir / f"seed_{s}_individual_log.csv")
            existing_ind = load_csv(existing_dir / f"seed_{s}_individual_log.csv")
            all_mismatches.extend(
                compare_rows(replay_ind, existing_ind, SHARED_INDIVIDUAL_COLS, f"k={k} seed={s} individual_log")
            )

            replay_ts = load_csv(replay_dir / f"seed_{s}_timeseries.csv")
            existing_ts = load_csv(existing_dir / f"seed_{s}_timeseries.csv")
            all_mismatches.extend(
                compare_rows(replay_ts, existing_ts, SHARED_TIMESERIES_COLS, f"k={k} seed={s} timeseries")
            )

            if all_mismatches:
                break
        if all_mismatches:
            break

    print(f"Checked {n_checked}/{len(K_VALUES) * len(SEEDS)} (k,seed) units before "
          f"{'stopping at first mismatch' if all_mismatches else 'finishing'}.\n")

    if all_mismatches:
        print("MISMATCH FOUND -- shuffle replay does NOT reproduce the existing precision-sweep data:\n")
        for m in all_mismatches:
            print(f"  {m}")
        print("\nStopping. Investigate the replay code before proceeding to any further analysis.")
    else:
        print(f"OK: all {n_checked} (k,seed) units match exactly "
              f"(individual_log.csv + timeseries.csv shared columns).")
        print("Proceed to the full analysis.")


if __name__ == "__main__":
    main()
