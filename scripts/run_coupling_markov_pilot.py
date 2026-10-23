"""Coupled (K-model) evolution + periodic genotype probing, for the
K-model Markov-structure investigation -- docs/K_MODEL_MARKOV_DESIGN.md
section 8 (stage 2 pilot).

Reimplements evopersonality_abm.coupling.run_coupled_evolution's
generation loop directly (same reason as scripts/run_lindgren_markov_sweep.py:
the intermediate `genomes` list needs to be available for periodic
census + probing), rather than calling it as a black box -- see design
doc section 2's finding that NEITHER existing coupling run (this one or
the spatial N=1000 one) records per-generation genome identity; this
script is the "new recording-enabled run" that finding calls for.

At every probe age (a multiple of --probe-interval that is also
>= --probe-start, plus the final age), censuses the population's unique
genotypes (evopersonality_abm.k_model_probe.census_genomes -- extended
genotype_key including K_out/K_self exactly, no rounding, per design doc
section 8-3) and probes each one against the 6 fixed opponents
(TFT/AllC/AllD/Random/Gradual/WSLS, decision (b): no K layer, no noise)
plus SELF and POP (both through the K layer, decision 2). Also computes
the forced-bottom recovery-time metric for all 8 "opponents".

Does NOT touch scripts/output/coupling_seed_sweep_5000gen/ or any other
existing coupling run's output -- writes to a new --out-dir only.

Writes, per (seed, probe age):
    seed_X_generation_stats.csv                  (committed, lightweight)
    seed_X_probe_age_G_census.csv.gz              (local only)
    seed_X_probe_age_G_moves.csv.gz               (local only, raw move sequences)
    seed_X_probe_age_G_recovery.csv.gz            (local only, forced-bottom recovery times)
    seed_X_checkpoint.pkl                         (local only, deleted on completion)

Checkpointed every --probe-interval generations (same pattern as
run_lindgren_markov_sweep.py): a seed interrupted resumes from its
checkpoint rather than restarting from generation 0.

Usage (pilot, cloud):
    python3 scripts/run_coupling_markov_pilot.py --seeds 4 --seed-offset 100 \\
        --generations 1200 --probe-start 1000 --probe-interval 50 \\
        --workers 4 --push --out-dir scripts/output/coupling_markov_pilot
"""

import argparse
import csv
import gzip
import os
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.coupling import Genome, mutate_genome, random_genome, round_robin_payoffs_coupled
from evopersonality_abm.population import select_and_reproduce
from evopersonality_abm.k_model_probe import (
    FIXED_OPPONENTS,
    SELFPLAY_OPPONENTS,
    census_genomes,
    draw_partner_genome,
    k_recovery_time_vs_fixed,
    k_recovery_time_vs_genome,
    run_k_selfplay_duel,
    run_k_vs_fixed_duel,
)

ALL_PROBE_OPPONENTS = FIXED_OPPONENTS + SELFPLAY_OPPONENTS  # TFT/AllC/AllD/Random/Gradual/WSLS/SELF/POP


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: Sequence[Path], message: str) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return
    add = _git("add", *existing)
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


def generation_stats_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed_{seed}_generation_stats.csv"


def checkpoint_path_for(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed_{seed}_checkpoint.pkl"


def census_path(out_dir: Path, seed: int, age: int) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"


def moves_path(out_dir: Path, seed: int, age: int) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_moves.csv.gz"


def recovery_path(out_dir: Path, seed: int, age: int) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_recovery.csv.gz"


def _atomic_pickle_dump(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def write_genotype_census(path: Path, census: List[Tuple]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory", "k_out", "k_self"])
        for gid, (key, freq, rep) in enumerate(census):
            w.writerow([
                gid, rep.table.memory_depth, freq,
                ",".join(map(str, rep.table.table)), ",".join(map(str, rep.table.prehistory)),
                repr(rep.k_out), repr(rep.k_self),
            ])


def run_probes_for_age(
    genomes: List[Genome], age: int, seed: int, out_dir: Path,
    probe_repeats: int, probe_rounds: int, free_rounds: int, probe_noise_std: float, master_seed: str,
) -> None:
    """Census + probe every unique genotype against the 6 fixed opponents
    plus SELF/POP. Writes census/moves/recovery files for this (seed, age)."""
    census = census_genomes(genomes)
    write_genotype_census(census_path(out_dir, seed, age), census)

    gids = list(range(len(census)))
    weights = [freq for _key, freq, _rep in census]
    genome_by_gid = {gid: rep for gid, (_key, _freq, rep) in enumerate(census)}

    rng = random.Random(f"{master_seed}_{seed}_{age}")
    moves_rows: List[Tuple] = []
    recovery_rows: List[Tuple] = []

    for gid, (key, freq, rep) in enumerate(census):
        for opponent_name in FIXED_OPPONENTS:
            for repeat in range(probe_repeats):
                own_moves, opp_moves = run_k_vs_fixed_duel(rep, opponent_name, rng, rounds=probe_rounds, noise_std=probe_noise_std)
                for r, (om, pm) in enumerate(zip(own_moves, opp_moves), start=1):
                    moves_rows.append((gid, opponent_name, repeat, r, om, pm))
                rt = k_recovery_time_vs_fixed(rep, opponent_name, rng, free_rounds=free_rounds, noise_std=probe_noise_std)
                recovery_rows.append((gid, opponent_name, repeat, rt if rt is not None else ""))

        for repeat in range(probe_repeats):
            own_moves, opp_moves = run_k_selfplay_duel(rep, rep, rng, rounds=probe_rounds, noise_std=probe_noise_std)
            for r, (om, pm) in enumerate(zip(own_moves, opp_moves), start=1):
                moves_rows.append((gid, "SELF", repeat, r, om, pm))
            rt = k_recovery_time_vs_genome(rep, rep, rng, free_rounds=free_rounds, noise_std=probe_noise_std)
            recovery_rows.append((gid, "SELF", repeat, rt if rt is not None else ""))

        for repeat in range(probe_repeats):
            partner = draw_partner_genome(gids, weights, genome_by_gid, rng)
            own_moves, opp_moves = run_k_selfplay_duel(rep, partner, rng, rounds=probe_rounds, noise_std=probe_noise_std)
            for r, (om, pm) in enumerate(zip(own_moves, opp_moves), start=1):
                moves_rows.append((gid, "POP", repeat, r, om, pm))
            partner2 = draw_partner_genome(gids, weights, genome_by_gid, rng)
            rt = k_recovery_time_vs_genome(rep, partner2, rng, free_rounds=free_rounds, noise_std=probe_noise_std)
            recovery_rows.append((gid, "POP", repeat, rt if rt is not None else ""))

    with gzip.open(moves_path(out_dir, seed, age), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
        w.writerows(moves_rows)

    with gzip.open(recovery_path(out_dir, seed, age), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "opponent", "repeat", "recovery_round"])
        w.writerows(recovery_rows)


def write_generation_stats(path: Path, gen_rows: List[Tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "mean_k_self", "mean_depth", "mean_payoff"])
        for row in gen_rows:
            w.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.4f}", f"{row[4]:.6f}"])


def run_one_seed(
    seed: int, population_size: int, generations: int, rounds_per_duel: int,
    initial_memory_depth: int, point_mutation_rate: float, duplication_rate: float, split_rate: float,
    max_memory_depth: int, min_memory_depth: int, k_mutation_std: float, evolution_noise_std: float,
    probe_start: int, probe_interval: int, probe_repeats: int, probe_rounds: int,
    free_rounds: int, probe_noise_std: float, master_seed: str, out_dir: Path,
) -> Tuple[int, List[Path]]:
    ckpt_path = checkpoint_path_for(out_dir, seed)
    gs_path = generation_stats_path(out_dir, seed)

    resumed = ckpt_path.exists()
    if resumed:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        gen_start = ckpt["gen"]
        genomes = ckpt["genomes"]
        rng = ckpt["rng"]
        gen_rows: List[Tuple] = ckpt["gen_rows"]
    else:
        gen_start = 0
        rng = random.Random(seed)
        genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]
        gen_rows = []

    committed: List[Path] = []
    probe_ages = {a for a in range(probe_start, generations + 1, probe_interval)} | {generations}

    if gen_start in probe_ages:
        run_probes_for_age(genomes, gen_start, seed, out_dir, probe_repeats, probe_rounds, free_rounds, probe_noise_std, master_seed)
        committed.extend([census_path(out_dir, seed, gen_start)])  # local-only; committed list here is just for return bookkeeping

    for gen in range(gen_start, generations):
        payoffs = round_robin_payoffs_coupled(genomes, rounds_per_duel, rng, evolution_noise_std)
        mean_k_out = sum(g.k_out for g in genomes) / len(genomes)
        mean_k_self = sum(g.k_self for g in genomes) / len(genomes)
        mean_depth = sum(g.table.memory_depth for g in genomes) / len(genomes)
        mean_payoff = sum(payoffs) / len(payoffs)
        gen_rows.append((gen, mean_k_out, mean_k_self, mean_depth, mean_payoff))

        genomes = select_and_reproduce(genomes, payoffs, rng)
        genomes = [
            mutate_genome(g, rng, point_mutation_rate, duplication_rate, split_rate, max_memory_depth, min_memory_depth, k_mutation_std)
            for g in genomes
        ]

        age = gen + 1
        write_generation_stats(gs_path, gen_rows)
        if age in probe_ages:
            run_probes_for_age(genomes, age, seed, out_dir, probe_repeats, probe_rounds, free_rounds, probe_noise_std, master_seed)
        _atomic_pickle_dump(ckpt_path, {"gen": age, "genomes": genomes, "rng": rng, "gen_rows": gen_rows})

    committed.append(gs_path)
    ckpt_path.unlink(missing_ok=True)
    return seed, committed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=1200)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--initial-memory-depth", type=int, default=1)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=8)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--k-mutation-std", type=float, default=0.05)
    p.add_argument("--evolution-noise-std", type=float, default=0.05, help="coupling.run_coupled_duel's own noise_std, used during evolution's round-robin duels")
    p.add_argument("--probe-noise-std", type=float, default=0.05, help="K-layer noise_std used during probing (design doc section 8-4: the K model's own noise, not pure Lindgren's noise_rate)")
    p.add_argument("--probe-start", type=int, default=1000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--probe-repeats", type=int, default=20)
    p.add_argument("--probe-rounds", type=int, default=100)
    p.add_argument("--free-rounds", type=int, default=20, help="recovery-time metric's free-play window")
    p.add_argument("--master-seed", type=str, default="k_model_markov_pilot_v1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="scripts/output/coupling_markov_pilot")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_offset + i for i in range(args.seeds)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one_seed, seed, args.population_size, args.generations, args.rounds_per_duel,
                args.initial_memory_depth, args.point_mutation_rate, args.duplication_rate, args.split_rate,
                args.max_memory_depth, args.min_memory_depth, args.k_mutation_std, args.evolution_noise_std,
                args.probe_start, args.probe_interval, args.probe_repeats, args.probe_rounds,
                args.free_rounds, args.probe_noise_std, args.master_seed, out_dir,
            ): seed
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed, committed = fut.result()
            print(f"seed={seed} done, elapsed={time.time() - t0:.0f}s", flush=True)
            if args.push:
                commit_and_push(committed, f"K-model Markov pilot: seed={seed} complete")

    print(f"all seeds done, total elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
