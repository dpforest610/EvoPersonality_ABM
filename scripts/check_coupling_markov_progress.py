"""Progress check for the K-model Markov-structure production run
(scripts/run_coupling_markov_pilot.py run at production scale), read
directly from the output directory on local disk -- see
docs/K_MODEL_MARKOV_MAC_RUNBOOK.md.

Usage:
    python3 scripts/check_coupling_markov_progress.py \\
        --out-dir scripts/output/coupling_markov_production --n-seeds 70 --seed-offset 200 --generations 5000
"""

import argparse
import pickle
import sys
from pathlib import Path

# Unpickling a checkpoint needs evopersonality_abm (the genomes inside
# are instances of its classes) -- without this, pickle.load() fails the
# moment any in-progress checkpoint is read.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="scripts/output/coupling_markov_production")
    p.add_argument("--n-seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=200)
    p.add_argument("--generations", type=int, default=5000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    if not out_dir.exists():
        print(f"{out_dir} does not exist yet")
        return

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
                in_progress.append(f"{s}:{ckpt['gen']}/{args.generations}")
            except Exception as e:
                in_progress.append(f"{s}:?(unreadable: {e})")

    ip_str = ", ".join(in_progress) if in_progress else "-"
    print(f"{out_dir!s}  done: {len(done)}/{len(seeds)}  in-progress (seed:gen/total): {ip_str}")


if __name__ == "__main__":
    main()
