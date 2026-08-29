"""Extend the 5-seed, 5000-generation K-coupled sweep to 50 seeds, same
model parameters throughout (only the seed changes), to check whether
the phase-structure findings from the 5-seed sweep -- CCF period
~70-90gen with a negative trough, PSD broad hump around ~80gen with
Q~1, and a detrended mean radius from the origin ~0.07 common across
seeds -- hold up with 10x more seeds. That re-analysis (CCF, PSD,
phase-plane) is deliberately NOT redone here; this script's only job is
piling up scripts/output/coupling_seed_sweep_5000gen/seed_<seed>_timeseries.csv
for all 50 seeds.

Seeds: range(0, 50) -- 50 consecutive integers, which already contains
the original five (42, 1, 2, 3, 4), so no seed is silently added or
dropped, and the list is trivially reproducible. Recorded again in
seeds.json next to the per-seed CSVs.

Resumable: writes into the SAME directory the original 5-seed sweep
used, and skips any seed whose CSV already exists (both the original 5
and, on a re-invocation after an interruption, whatever this script
already finished) -- no seed is ever recomputed.

Checkpointing: every time the cumulative completed-seed count crosses a
new multiple of --checkpoint-every (default 10: 10, 20, 30, 40, 50),
commits and pushes every CSV written since the last checkpoint, with a
commit message stating the cumulative count, the seeds in that batch,
and the running mean/std (across ALL completed seeds so far) of each
seed's detrended mean radius from the origin -- a cheap amplitude sanity
check, not the full re-analysis.

Usage:
    python3 scripts/run_coupling_seed_sweep_50.py
    python3 scripts/run_coupling_seed_sweep_50.py --workers 5 --checkpoint-every 10
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.coupling import run_coupled_evolution

# gen, mean_k_out, std_k_out, mean_k_self, std_k_self, mean_depth, mean_payoff
Row = Tuple[int, float, float, float, float, float, float]


def run_one_seed(
    seed: int,
    population_size: int,
    generations: int,
    rounds_per_duel: int,
    point_mutation_rate: float,
    duplication_rate: float,
    split_rate: float,
    max_memory_depth: int,
    min_memory_depth: int,
    k_mutation_std: float,
    noise_std: float,
) -> Tuple[int, List[Row]]:
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
        (
            s.generation,
            s.mean_k_out,
            statistics.pstdev(s.k_out),
            s.mean_k_self,
            statistics.pstdev(s.k_self),
            statistics.fmean(s.memory_depths),
            s.mean_payoff,
        )
        for s in stats
    ]
    return seed, rows


def write_csv(path: Path, rows: List[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self", "mean_depth", "mean_payoff"])
        for gen, mko, sko, mks, sks, md, mp in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}", f"{md:.4f}", f"{mp:.6f}"])


def read_csv_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    int(r["generation"]),
                    float(r["mean_k_out"]),
                    float(r["std_k_out"]),
                    float(r["mean_k_self"]),
                    float(r["std_k_self"]),
                    float(r["mean_depth"]),
                    float(r["mean_payoff"]),
                )
            )
    return rows


def detrended_mean_radius(rows: List[Row], window: int) -> float:
    """Same statistic as the phase-plane analysis: mean distance from
    the origin of the (K_self, K_out) trajectory after subtracting a
    centered moving average from each channel."""
    kout = np.array([r[1] for r in rows])
    kself = np.array([r[3] for r in rows])
    kernel = np.ones(window) / window
    trend_out = np.convolve(kout, kernel, mode="valid")
    trend_self = np.convolve(kself, kernel, mode="valid")
    offset = (window - 1) // 2
    kout_dt = kout[offset : offset + len(trend_out)] - trend_out
    kself_dt = kself[offset : offset + len(trend_self)] - trend_self
    radius = np.sqrt(kout_dt**2 + kself_dt**2)
    return float(radius.mean())


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
    p.add_argument("--out-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen")
    p.add_argument("--total-seeds", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=5000)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=8)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--k-mutation-std", type=float, default=0.05)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--detrend-window", type=int, default=75)
    p.add_argument("--workers", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.total_seeds))
    seeds_path = out_dir / "seeds.json"
    with open(seeds_path, "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "generation_rule": "range(0, total_seeds)",
                "note": "already contains the original 5-seed sweep's 42, 1, 2, 3, 4",
            },
            f,
            indent=2,
        )

    def csv_path_for(seed: int) -> Path:
        return out_dir / f"seed_{seed}_timeseries.csv"

    radius_by_seed: Dict[int, float] = {}
    already_done: List[int] = []
    pending: List[int] = []
    for s in seeds:
        path = csv_path_for(s)
        if path.exists():
            radius_by_seed[s] = detrended_mean_radius(read_csv_rows(path), args.detrend_window)
            already_done.append(s)
        else:
            pending.append(s)

    print(f"{len(already_done)} seeds already done: {already_done}", flush=True)
    print(f"{len(pending)} seeds pending: {pending}", flush=True)

    if not pending:
        print("All seeds already complete -- nothing to run.", flush=True)
        return

    completed_count = len(already_done)
    last_checkpoint = (completed_count // args.checkpoint_every) * args.checkpoint_every
    batch_since_checkpoint: List[int] = []
    paths_since_checkpoint: List[Path] = [seeds_path]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one_seed,
                seed,
                args.population_size,
                args.generations,
                args.rounds_per_duel,
                args.point_mutation_rate,
                args.duplication_rate,
                args.split_rate,
                args.max_memory_depth,
                args.min_memory_depth,
                args.k_mutation_std,
                args.noise_std,
            ): seed
            for seed in pending
        }
        for fut in as_completed(futures):
            seed, rows = fut.result()
            path = csv_path_for(seed)
            write_csv(path, rows)
            radius_by_seed[seed] = detrended_mean_radius(rows, args.detrend_window)

            completed_count += 1
            batch_since_checkpoint.append(seed)
            paths_since_checkpoint.append(path)
            print(f"seed={seed} done ({completed_count}/{args.total_seeds})", flush=True)

            threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
            if threshold > last_checkpoint:
                radii = list(radius_by_seed.values())
                mean_r = statistics.fmean(radii)
                std_r = statistics.pstdev(radii) if len(radii) > 1 else 0.0
                message = (
                    f"Coupling 50-seed sweep: {completed_count}/{args.total_seeds} seeds complete "
                    f"(batch: seeds {sorted(batch_since_checkpoint)}); "
                    f"cumulative detrended-radius mean={mean_r:.4f} std={std_r:.4f} (n={len(radii)})"
                )
                commit_and_push(paths_since_checkpoint, message)
                last_checkpoint = threshold
                batch_since_checkpoint = []
                paths_since_checkpoint = [seeds_path]

    if batch_since_checkpoint:
        radii = list(radius_by_seed.values())
        mean_r = statistics.fmean(radii)
        std_r = statistics.pstdev(radii) if len(radii) > 1 else 0.0
        message = (
            f"Coupling 50-seed sweep: {completed_count}/{args.total_seeds} seeds complete "
            f"(final partial batch: seeds {sorted(batch_since_checkpoint)}); "
            f"cumulative detrended-radius mean={mean_r:.4f} std={std_r:.4f} (n={len(radii)})"
        )
        commit_and_push(paths_since_checkpoint, message)

    print(f"\nAll {completed_count}/{args.total_seeds} seeds complete.", flush=True)


if __name__ == "__main__":
    main()
