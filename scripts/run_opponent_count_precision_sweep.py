"""第二章: PRECISION opponent-count sweep -- is the inverted-U real, and
where is its peak?

The coarse scan (run_opponent_count_scan.py, n=10 seeds x 1500 gen) was
not monotonic and not individually significant at any k -- consistent
with either a real non-monotonic (inverted-U) shape, or noise at n=10.
This sweep re-runs at FULL chapter-1 precision (n=50 seeds x 5000
generations, matching chapter 1 exactly so the two are directly
comparable) across 9 k values, finely spaced to locate a peak if one
exists: k = 4, 6, 8, 12, 16, 24, 32, 40, 50. All even (ring_duel_edges
requires even k for exact k-regularity, except k=population_size-1).
k=99 is NOT re-simulated -- chapter 1's own well-mixed data (ring
k=population_size-1 is mathematically the complete graph) is the
reference point for that end of the curve.

Same model as the coarse scan: 1D ring (fresh random circular order
every generation), each agent duels its k nearest ring-neighbors for
one continuous rounds_per_duel-round duel (n_phases=1, no movement --
both already ruled out as drivers in earlier experiments). No new
mechanisms are introduced here -- this sweep exists ONLY to pin down
whether the k=8-region effect is real and where it peaks, per explicit
instruction to stop after reporting results and not chase the
inverted-U's "meaning" until its existence is confirmed.

Resumable (skips any (k, seed) whose both output files already exist)
and checkpointed: commits + pushes every --checkpoint-every completed
(k, seed) units (default 5), so a container restart loses at most a
few seeds of work. Designed to fit a ~72 hour budget across 9 k x 50
seeds x 5000 generations with 4 parallel workers (estimated ~63.5
CPU-hours / 4 cores =~ 63.5 wall hours, ordered cheapest-k-first so the
most diagnostic points -- the small-k end, including whether k=4 stays
a trough at full power -- finish earliest).

Usage:
    python3 scripts/run_opponent_count_precision_sweep.py
    python3 scripts/run_opponent_count_precision_sweep.py --smoke   # 2 seeds x 200 gen, one k, no push
"""

import argparse
import csv
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_opponent_count

OUT_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [4, 6, 8, 12, 16, 24, 32, 40, 50]  # ascending: cheapest (most diagnostic) first
SEEDS = list(range(50))
GENERATIONS = 5000
LOG_INTERVAL = 100  # matches chapter 1's convention exactly

PARAMS = dict(
    population_size=100,
    rounds_per_duel=50,
    point_mutation_rate=0.05,
    duplication_rate=0.02,
    split_rate=0.02,
    max_memory_depth=8,
    min_memory_depth=0,
    k_mutation_std=0.05,
    noise_std=0.05,
)

AggRow = Tuple[int, float, float, float, float, float, float]
IndividualRow = Tuple[int, int, float, float, int, float]


def run_one_seed(k: int, seed: int, generations: int) -> Tuple[int, int, List[AggRow], List[IndividualRow], float]:
    t0 = time.time()
    stats = run_coupled_evolution_opponent_count(seed=seed, generations=generations, k=k, **PARAMS)
    elapsed = time.time() - t0

    agg_rows: List[AggRow] = []
    individual_rows: List[IndividualRow] = []
    for s in stats:
        agg_rows.append(
            (
                s.generation,
                s.mean_k_out,
                statistics.pstdev(s.k_out),
                s.mean_k_self,
                statistics.pstdev(s.k_self),
                statistics.fmean(s.memory_depths),
                s.mean_payoff,
            )
        )
        if s.generation % LOG_INTERVAL == 0:
            for i in range(len(s.k_out)):
                individual_rows.append(
                    (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i])
                )

    return k, seed, agg_rows, individual_rows, elapsed


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
                    "mean_depth", "mean_payoff"])
        for gen, mko, sko, mks, sks, md, mp in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                        f"{md:.4f}", f"{mp:.6f}"])


def write_individual_csv(path: Path, rows: List[IndividualRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff"])
        for gen, idx, k_out, k_self, depth, payoff in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}"])


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="git commit+push every N newly-completed (k,seed) units")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true",
                   help="wiring check: k=8 only, 2 seeds x 200 gen, separate out-dir, no push")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        k_values = [8]
        seeds = [0, 1]
        generations = 200
        out_root = Path(str(OUT_ROOT) + "_smoke")
        do_push = False
    else:
        k_values = K_VALUES
        seeds = SEEDS
        generations = GENERATIONS
        out_root = OUT_ROOT
        do_push = True

    units = [(k, seed) for k in k_values for seed in seeds]

    def agg_path_for(k, seed):
        return out_root / f"k{k}" / f"seed_{seed}_timeseries.csv"

    def individual_path_for(k, seed):
        return out_root / f"k{k}" / f"seed_{seed}_individual_log.csv"

    for k in k_values:
        (out_root / f"k{k}").mkdir(parents=True, exist_ok=True)

    pending = [(k, s) for k, s in units if not (agg_path_for(k, s).exists() and individual_path_for(k, s).exists())]
    already_done = len(units) - len(pending)
    print(f"{already_done}/{len(units)} (k,seed) units already done", flush=True)
    print(f"{len(pending)} pending", flush=True)

    if not pending:
        print("All units already complete -- nothing to run.", flush=True)
        return

    completed_count = already_done
    last_checkpoint = (completed_count // args.checkpoint_every) * args.checkpoint_every
    batch_since_checkpoint: List[Tuple[int, int]] = []
    paths_since_checkpoint: List[Path] = []

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_seed, k, seed, generations): (k, seed)
            for k, seed in pending
        }
        for future in as_completed(futures):
            k, seed, agg_rows, individual_rows, elapsed = future.result()
            agg_path = agg_path_for(k, seed)
            ind_path = individual_path_for(k, seed)
            write_agg_csv(agg_path, agg_rows)
            write_individual_csv(ind_path, individual_rows)

            completed_count += 1
            batch_since_checkpoint.append((k, seed))
            paths_since_checkpoint.extend([agg_path, ind_path])
            per_gen_ms = elapsed / generations * 1000
            print(f"k={k} seed={seed} done ({completed_count}/{len(units)}): "
                  f"{elapsed:.1f}s, {per_gen_ms:.1f} ms/gen", flush=True)

            if do_push:
                threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                if threshold > last_checkpoint:
                    message = (
                        f"Opponent-count precision sweep: {completed_count}/{len(units)} units complete "
                        f"(batch: {sorted(batch_since_checkpoint)})"
                    )
                    commit_and_push(paths_since_checkpoint, message)
                    last_checkpoint = threshold
                    batch_since_checkpoint = []
                    paths_since_checkpoint = []

    if do_push and batch_since_checkpoint:
        message = (
            f"Opponent-count precision sweep: {completed_count}/{len(units)} units complete "
            f"(final partial batch: {sorted(batch_since_checkpoint)})"
        )
        commit_and_push(paths_since_checkpoint, message)

    print(f"\nAll {completed_count}/{len(units)} units complete in {time.time() - t_start:.1f}s wall-clock.",
          flush=True)


if __name__ == "__main__":
    main()
