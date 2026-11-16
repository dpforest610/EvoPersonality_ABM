"""精密kスイープ(既存9点+延長k=64,80,99)の進捗を、ローカルディスク上の
出力ディレクトリから直接読んで一覧表示する。

Usage:
    python3 scripts/check_progress.py
    python3 scripts/check_progress.py --out-root scripts/output/opponent_count_precision_sweep
    python3 scripts/check_progress.py --k-values 64,80,99
"""

import argparse
import pickle
import sys
from pathlib import Path

# Unpickling a checkpoint needs to import evopersonality_abm (the genomes
# inside are instances of its classes) -- without this, pickle.load() fails
# with "No module named 'evopersonality_abm'" the moment any in-progress
# checkpoint is read, since running `python3 scripts/check_progress.py`
# only puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUT_ROOT = Path("scripts/output/opponent_count_precision_sweep")
DEFAULT_K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50, 64, 80, 99]
DEFAULT_N_SEEDS = 50
DEFAULT_GENERATIONS = 5000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES))
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    args = p.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    seeds = list(range(args.n_seeds))

    print(f"{'k':>6}  {'done':>8}  in-progress (seed:next_gen/total)")
    total_done, total_units = 0, 0
    for k in k_values:
        k_dir = args.out_root / f"k{k}"
        total_units += len(seeds)
        if not k_dir.exists():
            print(f"{k:>6}  {'(no dir)':>8}")
            continue

        done = sum(
            1 for s in seeds
            if (k_dir / f"seed_{s}_timeseries.csv").exists()
            and (k_dir / f"seed_{s}_individual_log.csv").exists()
        )
        total_done += done

        in_progress = []
        for s in seeds:
            ckpt_path = k_dir / f"seed_{s}_checkpoint.pkl"
            if not ckpt_path.exists():
                continue
            try:
                with open(ckpt_path, "rb") as f:
                    ckpt = pickle.load(f)
                in_progress.append(f"{s}:{ckpt['next_generation']}/{args.generations}")
            except Exception as e:
                in_progress.append(f"{s}:?(unreadable: {e})")

        ip_str = ", ".join(in_progress) if in_progress else "-"
        print(f"{k:>6}  {done:>3}/{len(seeds):<4}  {ip_str}")

    print(f"\nTotal: {total_done}/{total_units} (k,seed) units complete.")


if __name__ == "__main__":
    main()
