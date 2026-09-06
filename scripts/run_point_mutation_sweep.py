"""Run phase for the point-mutation-rate sweep.

Question: does point_mutation_rate set the ~77-generation K_out/K_self
quasi-period found by the 50-seed coupling sweep? Prediction: raising
point_mutation_rate shrinks the period.

Sweeps point_mutation_rate over PMR_LEVELS = [0.02, 0.05, 0.10], 50
seeds each (range(0, total_seeds), same convention as
run_coupling_seed_sweep_50.py), 5000 generations, every other parameter
fixed at the same values as the existing 50-seed sweep
(population_size=100, duplication_rate=0.02, split_rate=0.02,
max/min_memory_depth=8/0, k_mutation_std=0.05, noise_std=0.05,
rounds_per_duel=50).

This script ONLY runs simulations and saves the raw per-generation
series (mean_k_out, mean_k_self, mean_depth, one column each, length =
generations) -- one file per (level, seed) cell. It does not measure
anything; see analyze_point_mutation_sweep.py, which reads only these
files and can be re-run with different windows/nperseg without
recomputing anything here. Simulation is the expensive part; measurement
conditions are the part you want to revisit -- keeping them separate
means a re-measurement is free.

Resumable: skips any (level, seed) cell whose CSV already exists.
Re-invoking after an interruption (or letting it die and restarting)
only computes what's missing.

Usage:
    python3 scripts/run_point_mutation_sweep.py                # full: 3 levels x 50 seeds x 5000 gen
    python3 scripts/run_point_mutation_sweep.py --smoke         # wiring check: 3 levels x 3 seeds x 1000 gen
    python3 scripts/run_point_mutation_sweep.py --levels 0.02,0.10 --total-seeds 10
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.coupling import run_coupled_evolution

DEFAULT_LEVELS = [0.02, 0.05, 0.10]
Row = Tuple[int, float, float, float]  # generation, mean_k_out, mean_k_self, mean_depth


def level_dirname(rate: float) -> str:
    return f"pmr_{rate:.3g}"


def run_one_cell(
    seed: int,
    point_mutation_rate: float,
    population_size: int,
    generations: int,
    rounds_per_duel: int,
    duplication_rate: float,
    split_rate: float,
    max_memory_depth: int,
    min_memory_depth: int,
    k_mutation_std: float,
    noise_std: float,
) -> Tuple[int, float, List[Row]]:
    stats = run_coupled_evolution(
        population_size=population_size,
        generations=generations,
        rounds_per_duel=rounds_per_duel,
        point_mutation_rate=point_mutation_rate,
        duplication_rate=duplication_rate,
        split_rate=split_rate,
        max_memory_depth=max_memory_depth,
        min_memory_depth=min_memory_depth,
        k_mutation_std=k_mutation_std,
        noise_std=noise_std,
        seed=seed,
    )
    rows: List[Row] = [
        (s.generation, s.mean_k_out, s.mean_k_self, statistics.fmean(s.memory_depths)) for s in stats
    ]
    return seed, point_mutation_rate, rows


def write_csv(path: Path, rows: List[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "mean_k_self", "mean_depth"])
        for gen, mko, mks, md in rows:
            w.writerow([gen, f"{mko:.6f}", f"{mks:.6f}", f"{md:.4f}"])


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: List[Path], message: str) -> None:
    add = _git("add", *[str(p) for p in paths])
    if add.returncode != 0:
        print(f"git add failed for {message}: {add.stderr.strip()}", flush=True)
        return
    commit = _git("commit", "-m", message)
    if commit.returncode != 0:
        if "nothing to commit" in (commit.stdout + commit.stderr):
            return
        print(f"git commit failed for {message}: {commit.stderr.strip()}", flush=True)
        return
    delay = 2
    for attempt in range(4):
        push = _git("push")
        if push.returncode == 0:
            print(f"pushed: {message}", flush=True)
            return
        print(f"git push failed (attempt {attempt + 1}) for {message}: {push.stderr.strip()}", flush=True)
        time.sleep(delay)
        delay *= 2
    print(f"git push ultimately failed for: {message}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="scripts/output/point_mutation_sweep")
    p.add_argument("--levels", type=str, default=",".join(str(x) for x in DEFAULT_LEVELS))
    p.add_argument("--total-seeds", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=10, help="cumulative cells per commit+push")
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=5000)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=8)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--k-mutation-std", type=float, default=0.05)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument(
        "--smoke", action="store_true",
        help="wiring check only: 3 seeds x 1000 gen, separate out-dir, no git push",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        total_seeds = 3
        generations = 1000
        out_dir = Path(args.out_dir + "_smoke")
        do_push = False
    else:
        total_seeds = args.total_seeds
        generations = args.generations
        out_dir = Path(args.out_dir)
        do_push = True

    levels = [float(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(total_seeds))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "point_mutation_rate_levels": levels,
                "seeds": seeds,
                "generations": generations,
                "population_size": args.population_size,
                "rounds_per_duel": args.rounds_per_duel,
                "duplication_rate": args.duplication_rate,
                "split_rate": args.split_rate,
                "max_memory_depth": args.max_memory_depth,
                "min_memory_depth": args.min_memory_depth,
                "k_mutation_std": args.k_mutation_std,
                "noise_std": args.noise_std,
                "smoke": args.smoke,
            },
            f,
            indent=2,
        )

    def csv_path_for(rate: float, seed: int) -> Path:
        d = out_dir / level_dirname(rate)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"seed_{seed}.csv"

    all_cells = [(rate, seed) for rate in levels for seed in seeds]
    already_done = [(r, s) for r, s in all_cells if csv_path_for(r, s).exists()]
    pending = [(r, s) for r, s in all_cells if not csv_path_for(r, s).exists()]

    print(f"{len(already_done)}/{len(all_cells)} cells already done", flush=True)
    print(f"{len(pending)} cells pending", flush=True)

    if not pending:
        print("All cells already complete -- nothing to run.", flush=True)
        return

    completed_count = len(already_done)
    last_checkpoint = (completed_count // args.checkpoint_every) * args.checkpoint_every
    paths_since_checkpoint: List[Path] = [manifest_path]
    batch_since_checkpoint: List[Tuple[float, int]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one_cell,
                seed,
                rate,
                args.population_size,
                generations,
                args.rounds_per_duel,
                args.duplication_rate,
                args.split_rate,
                args.max_memory_depth,
                args.min_memory_depth,
                args.k_mutation_std,
                args.noise_std,
            ): (rate, seed)
            for rate, seed in pending
        }
        for fut in as_completed(futures):
            seed, rate, rows = fut.result()
            path = csv_path_for(rate, seed)
            write_csv(path, rows)

            completed_count += 1
            batch_since_checkpoint.append((rate, seed))
            paths_since_checkpoint.append(path)
            print(f"pmr={rate} seed={seed} done ({completed_count}/{len(all_cells)})", flush=True)

            if do_push:
                threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                if threshold > last_checkpoint:
                    message = (
                        f"Point mutation sweep: {completed_count}/{len(all_cells)} cells complete "
                        f"(batch: {batch_since_checkpoint})"
                    )
                    commit_and_push(paths_since_checkpoint, message)
                    last_checkpoint = threshold
                    batch_since_checkpoint = []
                    paths_since_checkpoint = [manifest_path]

    if do_push and batch_since_checkpoint:
        message = (
            f"Point mutation sweep: {completed_count}/{len(all_cells)} cells complete "
            f"(final partial batch: {batch_since_checkpoint})"
        )
        commit_and_push(paths_since_checkpoint, message)

    print(f"\nAll {completed_count}/{len(all_cells)} cells complete.", flush=True)


if __name__ == "__main__":
    main()
