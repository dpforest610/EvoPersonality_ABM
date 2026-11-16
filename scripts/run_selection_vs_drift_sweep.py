"""Run phase: does selection act on K_out/K_self, or are they drifting
like a (nearly) neutral trait?

The OU/AR(1) fit (analyze_ou_fit_diagnostic.py) named a bounded,
strongly-persistent drift but couldn't say whether it's driven by
selection or by finite-population genetic drift, because every prior
sweep only ever saved population-level aggregates (mean/std per
generation) -- never the per-individual (K_out, K_self, payoff) triples
needed to ask "do individuals with higher K also get higher payoff
within their own generation?" This script adds that per-individual log
on top of the existing coupled-evolution simulation (no changes to
evopersonality_abm/coupling.py -- CoupledGenerationStats already carries
per-individual k_out/k_self/payoffs lists in memory; this script is the
first to write them to disk).

Same model and defaults as the original 50-seed confirmatory sweep
(run_coupling_seed_sweep_50.py): population_size=100, generations=5000,
rounds_per_duel=50, point_mutation_rate=0.05, duplication_rate=0.02,
split_rate=0.02, max/min_memory_depth=8/0, k_mutation_std=0.05,
noise_std=0.05, seeds=range(0,50) -- only the logging changes. Two
files per seed:

  seed_<n>_timeseries.csv      -- same 7-column aggregate format as the
                                   original confirmatory sweep (a free
                                   sanity check: same seed + same
                                   params should reproduce it exactly).
  seed_<n>_individual_log.csv  -- generation, agent_index, k_out,
                                   k_self, memory_depth, payoff -- one
                                   row per agent, logged every
                                   --log-interval generations (default
                                   100) across the whole run (so both
                                   the transient and the gen>=2000
                                   steady state are covered; the steady
                                   state gets the most snapshots since
                                   it's most of the run). memory_depth
                                   is included so a later analysis can
                                   check whether a K-payoff correlation
                                   survives controlling for the
                                   co-evolving Lindgren table's depth
                                   (a possible confound).

Resumable: skips any seed whose BOTH files already exist. Checkpointed
git commit+push every --checkpoint-every (default 10) cumulative
completed seeds, same pattern as run_coupling_seed_sweep_50.py.

Usage:
    python3 scripts/run_selection_vs_drift_sweep.py --smoke
    python3 scripts/run_selection_vs_drift_sweep.py
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
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.coupling import run_coupled_evolution

# gen, mean_k_out, std_k_out, mean_k_self, std_k_self, mean_depth, mean_payoff
AggRow = Tuple[int, float, float, float, float, float, float]
# gen, agent_index, k_out, k_self, memory_depth, payoff
IndividualRow = Tuple[int, int, float, float, int, float]


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
    log_interval: int,
) -> Tuple[int, List[AggRow], List[IndividualRow]]:
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
    agg_rows: List[AggRow] = [
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
    individual_rows: List[IndividualRow] = []
    for s in stats:
        if s.generation % log_interval == 0:
            for i in range(len(s.k_out)):
                individual_rows.append(
                    (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i])
                )
    return seed, agg_rows, individual_rows


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self", "mean_depth", "mean_payoff"])
        for gen, mko, sko, mks, sks, md, mp in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}", f"{md:.4f}", f"{mp:.6f}"])


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="scripts/output/selection_vs_drift_sweep_with_depth")
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
    p.add_argument("--log-interval", type=int, default=100, help="log per-individual data every N generations")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument(
        "--smoke", action="store_true",
        help="wiring check only: 3 seeds x 1000 gen, log-interval 100, separate out-dir, no git push",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(total_seeds))
    seeds_path = out_dir / "seeds.json"
    with open(seeds_path, "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "generation_rule": "range(0, total_seeds)",
                "log_interval": args.log_interval,
                "generations": generations,
                "smoke": args.smoke,
            },
            f,
            indent=2,
        )

    def agg_path_for(seed: int) -> Path:
        return out_dir / f"seed_{seed}_timeseries.csv"

    def individual_path_for(seed: int) -> Path:
        return out_dir / f"seed_{seed}_individual_log.csv"

    already_done = [s for s in seeds if agg_path_for(s).exists() and individual_path_for(s).exists()]
    pending = [s for s in seeds if s not in already_done]

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
                generations,
                args.rounds_per_duel,
                args.point_mutation_rate,
                args.duplication_rate,
                args.split_rate,
                args.max_memory_depth,
                args.min_memory_depth,
                args.k_mutation_std,
                args.noise_std,
                args.log_interval,
            ): seed
            for seed in pending
        }
        for fut in as_completed(futures):
            seed, agg_rows, individual_rows = fut.result()
            agg_path = agg_path_for(seed)
            ind_path = individual_path_for(seed)
            write_agg_csv(agg_path, agg_rows)
            write_individual_csv(ind_path, individual_rows)

            completed_count += 1
            batch_since_checkpoint.append(seed)
            paths_since_checkpoint.extend([agg_path, ind_path])
            print(f"seed={seed} done ({completed_count}/{len(seeds)}), "
                  f"{len(individual_rows)} individual-log rows", flush=True)

            if do_push:
                threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                if threshold > last_checkpoint:
                    message = (
                        f"Selection-vs-drift sweep: {completed_count}/{len(seeds)} seeds complete "
                        f"(batch: seeds {sorted(batch_since_checkpoint)})"
                    )
                    commit_and_push(paths_since_checkpoint, message)
                    last_checkpoint = threshold
                    batch_since_checkpoint = []
                    paths_since_checkpoint = [seeds_path]

    if do_push and batch_since_checkpoint:
        message = (
            f"Selection-vs-drift sweep: {completed_count}/{len(seeds)} seeds complete "
            f"(final partial batch: seeds {sorted(batch_since_checkpoint)})"
        )
        commit_and_push(paths_since_checkpoint, message)

    print(f"\nAll {completed_count}/{len(seeds)} seeds complete.", flush=True)


if __name__ == "__main__":
    main()
