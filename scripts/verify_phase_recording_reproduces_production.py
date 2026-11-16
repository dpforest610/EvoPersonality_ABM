"""Step 3 of the N=1000 presentation-animation procedure (see
docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md): verify that a phase-recording
run of scripts/run_scent_spatial_sweep_n1000.py (same seed, same
parameters, a non-production --out-root) reproduces the real committed
production seed-0 trajectory bit-for-bit, at every generation the
production individual log actually has a snapshot for (every
LOG_INTERVAL=100th generation: 0, 100, ..., 4900). Recording is
opt-in/side-effect-only in evopersonality_abm/scent_spatial.py (it does
not consume any extra RNG draws -- see
tests/test_scent_spatial.py::test_phase_recording_does_not_change_simulation_results),
so an exact match here means the recording run IS production, not a
re-simulation that merely resembles it.

Compares position, K_out, K_self, payoff per agent at each overlapping
generation. Prints every mismatch and exits non-zero if there is one.

Usage (run from the same machine/hostname that produced the committed
production files, so machine_tag() resolves to the same directory for
both -- pass --production-machine to override if not):
    python3 scripts/verify_phase_recording_reproduces_production.py \\
        --movement scent --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording
"""

import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scent_spatial_sweep_n1000 import (  # noqa: E402
    EXPERIMENT_ROOT,
    individual_gz_path_for,
    machine_tag,
    out_root_for,
)

FIELDS = ["pos_x", "pos_y", "k_out", "k_self", "payoff"]


def load_by_gen(path: Path):
    by_gen = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            by_gen[int(row["generation"])][int(row["agent_index"])] = row
    return by_gen


def compare(prod_path: Path, rec_path: Path, label: str) -> int:
    """Returns the number of mismatched (generation, agent, field) triples."""
    for name, path in [("production", prod_path), ("recording", rec_path)]:
        if not path.exists():
            print(f"ERROR [{label}]: {name} file not found: {path}", file=sys.stderr)
            return -1

    prod = load_by_gen(prod_path)
    rec = load_by_gen(rec_path)
    shared_gens = sorted(set(prod) & set(rec))
    if not shared_gens:
        print(f"ERROR [{label}]: no overlapping generations between production and recording "
              "individual logs -- production logs every 100th generation (0, 100, ..., 4900); "
              "make sure the recording run covered the full generations=5000 (not a short run)",
              file=sys.stderr)
        return -1

    print(f"[{label}] comparing {len(shared_gens)} overlapping generation(s): {shared_gens}")
    total_fields = 0
    mismatches = 0
    for gen in shared_gens:
        p_rows, r_rows = prod[gen], rec[gen]
        if set(p_rows) != set(r_rows):
            print(f"  [{label}] gen {gen}: agent_index sets differ! "
                  f"production has {len(p_rows)}, recording has {len(r_rows)}")
            mismatches += 1
            continue
        for idx in sorted(p_rows):
            a, b = p_rows[idx], r_rows[idx]
            for field in FIELDS:
                total_fields += 1
                if a[field] != b[field]:
                    mismatches += 1
                    print(f"  [{label}] gen {gen} agent {idx} {field}: "
                          f"production={a[field]!r} recording={b[field]!r}")

    print(f"[{label}] {mismatches} mismatch(es) out of {total_fields} "
          f"(generation, agent, field) comparisons across {len(shared_gens)} generation(s).")
    return mismatches


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--movement", choices=["none", "random", "scent"], action="append",
                   help="repeatable; omit to check all three (none, random, scent)")
    p.add_argument("--recording-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--production-machine", type=str, default=None,
                   help="override the production machine-tag subdirectory name if this "
                        "verification is not being run on the same machine that produced "
                        "scripts/output/scent_spatial_sweep_n1000/<movement>/<machine>/ "
                        "(default: this machine's own tag, i.e. assumes same machine)")
    args = p.parse_args()

    movements = args.movement or ["none", "random", "scent"]
    prod_machine = args.production_machine or machine_tag()

    any_fail = False
    for movement in movements:
        rec_root = out_root_for(movement, experiment_root=args.recording_root)
        prod_root = EXPERIMENT_ROOT / movement / prod_machine
        m = compare(
            individual_gz_path_for(prod_root, args.seed),
            individual_gz_path_for(rec_root, args.seed),
            label=movement,
        )
        if m != 0:
            any_fail = True

    if any_fail:
        print("\nNOT reproducible for at least one condition -- see mismatches above.", file=sys.stderr)
        sys.exit(1)
    print("\nReproducible: every condition's recording run matches production exactly "
          "at every compared generation.")


if __name__ == "__main__":
    main()
