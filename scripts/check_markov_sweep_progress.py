"""Progress check for the Markov-structure-evolution production run
(scripts/run_lindgren_markov_sweep.py), read directly from the output
directories on local disk -- see docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md
and the Mac runbook (docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md).

Usage:
    python3 scripts/check_markov_sweep_progress.py
    python3 scripts/check_markov_sweep_progress.py --n-seeds 70 --seed-offset 100 --generations 3000 \\
        --out-dirs scripts/output/markov_sweep_production_noise01,scripts/output/markov_sweep_production_noise00
"""

import argparse
import pickle
import sys
from pathlib import Path

# Unpickling a checkpoint needs evopersonality_abm (the tables inside are
# LindgrenTable instances) -- without this, pickle.load() fails the moment
# any in-progress checkpoint is read.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUT_DIRS = [
    "scripts/output/markov_sweep_production_noise01",
    "scripts/output/markov_sweep_production_noise00",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dirs", type=str, default=",".join(DEFAULT_OUT_DIRS))
    p.add_argument("--n-seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--generations", type=int, default=3000)
    args = p.parse_args()

    out_dirs = [Path(d) for d in args.out_dirs.split(",") if d.strip()]
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    print(f"{'condition':>40}  {'done':>8}  in-progress (seed:age/total)")
    total_done, total_units = 0, 0
    for out_dir in out_dirs:
        total_units += len(seeds)
        if not out_dir.exists():
            print(f"{out_dir!s:>40}  {'(no dir)':>8}")
            continue

        done = []
        in_progress = []
        for s in seeds:
            gs_path = out_dir / f"seed_{s}_generation_stats.csv"
            if gs_path.exists():
                with open(gs_path, newline="") as f:
                    n_rows = sum(1 for _ in f) - 1
                if n_rows == args.generations:
                    done.append(s)
                    continue

            ckpt_path = out_dir / f"seed_{s}_checkpoint.pkl"
            if ckpt_path.exists():
                try:
                    with open(ckpt_path, "rb") as f:
                        ckpt = pickle.load(f)
                    in_progress.append(f"{s}:{ckpt['age']}/{args.generations}")
                except Exception as e:
                    in_progress.append(f"{s}:?(unreadable: {e})")

        total_done += len(done)
        ip_str = ", ".join(in_progress) if in_progress else "-"
        print(f"{out_dir!s:>40}  {len(done):>3}/{len(seeds):<4}  {ip_str}")

    print(f"\nTotal: {total_done}/{total_units} (condition, seed) units complete.")


if __name__ == "__main__":
    main()
