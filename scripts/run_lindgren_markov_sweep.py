"""Evolution + periodic genotype probing for the Lindgren behavioral-
Markov-structure-evolution research line -- see
docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md sections 1-4.

For each seed, this reimplements `population.run_evolution`'s generation
loop directly (rather than calling it as a black box) so the
intermediate `tables` population is available for periodic genotype
census + probing (see `evopersonality_abm/probe.py`). At every "age"
(generations completed so far) that is a multiple of --probe-interval,
including age 0 (the initial population) and the final age after the
last generation, the population's unique genotypes are censused
(`probe.census_genotypes`) and each one is dueled against every
`probe.OPPONENTS` entry (main set + extended set, 7 total), under each
of --probe-noise-rates (default: this run's own evolution noise_rate
["primary"] plus 0.0 ["no-noise sensitivity"], per design doc section
2.4 -- deduplicated, so a noise_rate=0.0 evolution run probes only once).

Usage (cloud session -- keeps the working tree clean turn to turn):
    python3 scripts/run_lindgren_markov_sweep.py --seeds 10 --generations 2000 \\
        --noise-rate 0.01 --workers 4 --push --push-partial-progress \\
        --out-dir scripts/output/markov_sweep_pilot_noise01

Usage (Mac/production -- push only once a seed is genuinely complete):
    python3 scripts/run_lindgren_markov_sweep.py --seeds 50 --generations 5000 \\
        --noise-rate 0.01 --workers 4 --push \\
        --out-dir scripts/output/markov_sweep_production_noise01

Run once per evolution noise_rate condition (e.g. once with
--noise-rate 0.01 for the primary run, once with --noise-rate 0.0 for
the no-noise sensitivity run -- design doc section 1.4).

Writes, per (seed, evolution noise_rate) run (design doc section 4):
    seed_X_generation_stats.csv                         (committed, lightweight)
    seed_X_probe_age_G_noiseN_census.csv.gz              (local only)
    seed_X_probe_age_G_noiseN_moves.csv.gz               (local only, raw move sequences)
    seed_X_probe_age_G_noiseN_summary.csv                (committed, aggregated)
    seed_X_checkpoint.pkl                                (local only, deleted on completion)

Checkpointed every --probe-interval generations: a seed interrupted
(container reclaim, Mac sleep/crash, the cloud session's 2h background
cap) resumes from its checkpoint instead of restarting from generation
0. A (seed) whose seed_X_generation_stats.csv already holds the full
--generations rows is skipped entirely (idempotent/resumable re-run).
"""

import argparse
import csv
import gzip
import os
import pickle
import queue as queue_module
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.population import (
    mutate_individual,
    random_table,
    round_robin_payoffs,
    select_and_reproduce,
)
from evopersonality_abm.probe import (
    EXTENDED_SET,
    MAIN_SET,
    census_genotypes,
    run_probe_duel,
)

PROBE_OPPONENTS = MAIN_SET + EXTENDED_SET  # 7 total, design doc section 2.5


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: Sequence[Path], message: str) -> None:
    """Commit+push the given (already-written, committed-tier) paths,
    retrying push on transient failure. Mirrors run_seed_sweep.py."""
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


def _atomic_pickle_dump(path: Path, obj) -> None:
    """Write to a temp file, then rename over `path`, so a reader never
    sees a half-written checkpoint. Same pattern as
    run_scent_spatial_sweep_n1000.py's _atomic_pickle_dump."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def seed_is_complete(out_dir: Path, seed: int, generations: int) -> bool:
    """generation_stats.csv is rewritten incrementally as the seed runs
    (every probe timepoint, see run_one_seed), so its mere existence does
    NOT mean the seed is done -- a seed stopped mid-run (e.g. by the
    background-execution time cap) leaves a partial file behind. A seed
    is actually complete only once that file holds all `generations` rows."""
    path = generation_stats_path(out_dir, seed)
    if not path.exists():
        return False
    with open(path, newline="") as f:
        return sum(1 for _ in f) - 1 == generations  # -1 for the header row


def _fmt_noise(noise_rate: float) -> str:
    return f"{noise_rate:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def census_path(out_dir: Path, seed: int, age: int) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"


def moves_path(out_dir: Path, seed: int, age: int, noise_rate: float) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_noise{_fmt_noise(noise_rate)}_moves.csv.gz"


def summary_path(out_dir: Path, seed: int, age: int, noise_rate: float) -> Path:
    return out_dir / f"seed_{seed}_probe_age_{age}_noise{_fmt_noise(noise_rate)}_summary.csv"


def write_genotype_census(path: Path, census: List[Tuple]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype_id", "memory_depth", "frequency", "table", "prehistory"])
        for gid, (key, freq, rep) in enumerate(census):
            w.writerow([gid, rep.memory_depth, freq, ",".join(map(str, rep.table)), ",".join(map(str, rep.prehistory))])


def run_probes_for_age(
    tables, age: int, seed: int, noise_rate: float, probe_noise_rates: Sequence[float],
    rng_probe: random.Random, out_dir: Path, probe_repeats: int, probe_rounds: int,
    main_analysis_rounds: int,
) -> List[Path]:
    """Census + duel every unique genotype against every PROBE_OPPONENTS
    entry, under every probe noise rate. Writes census (once per age,
    shared across probe_noise_rates), raw moves (local), and summary
    (committed) files. Returns the committed-tier paths written."""
    census = census_genotypes(tables)
    write_genotype_census(census_path(out_dir, seed, age), census)

    committed: List[Path] = []
    for probe_noise_rate in probe_noise_rates:
        moves_rows: List[Tuple] = []
        summary_rows: List[Tuple] = []
        for gid, (key, freq, rep) in enumerate(census):
            for opponent_name in PROBE_OPPONENTS:
                own_coop_20 = own_coop_full = opp_coop_20 = opp_coop_full = 0
                for repeat in range(probe_repeats):
                    own_moves, opp_moves = run_probe_duel(
                        rep, opponent_name, rng_probe, rounds=probe_rounds, noise_rate=probe_noise_rate
                    )
                    for r, (om, pm) in enumerate(zip(own_moves, opp_moves), start=1):
                        moves_rows.append((gid, opponent_name, repeat, r, om, pm))
                    own_coop_20 += sum(1 for m in own_moves[:main_analysis_rounds] if m == "C")
                    opp_coop_20 += sum(1 for m in opp_moves[:main_analysis_rounds] if m == "C")
                    own_coop_full += sum(1 for m in own_moves if m == "C")
                    opp_coop_full += sum(1 for m in opp_moves if m == "C")
                n = probe_repeats
                summary_rows.append((
                    gid, rep.memory_depth, freq, opponent_name, n,
                    own_coop_20 / (n * main_analysis_rounds),
                    own_coop_full / (n * probe_rounds),
                    opp_coop_20 / (n * main_analysis_rounds),
                    opp_coop_full / (n * probe_rounds),
                ))

        with gzip.open(moves_path(out_dir, seed, age, probe_noise_rate), "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"])
            w.writerows(moves_rows)

        sp = summary_path(out_dir, seed, age, probe_noise_rate)
        with open(sp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "genotype_id", "memory_depth", "frequency", "opponent", "n_repeats",
                "own_coop_rate_20", "own_coop_rate_full", "opp_coop_rate_20", "opp_coop_rate_full",
            ])
            for row in summary_rows:
                w.writerow([row[0], row[1], row[2], row[3], row[4], f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}", f"{row[8]:.4f}"])
        committed.append(sp)

    return committed


def write_generation_stats(path: Path, gen_rows: List[Tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_depth", "std_depth", "min_depth", "max_depth", "mean_payoff"])
        for row in gen_rows:
            w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", row[3], row[4], f"{row[5]:.4f}"])


def run_one_seed(
    seed: int,
    noise_rate: float,
    probe_noise_rates: Tuple[float, ...],
    population_size: int,
    generations: int,
    rounds_per_duel: int,
    initial_memory_depth: int,
    point_mutation_rate: float,
    duplication_rate: float,
    split_rate: float,
    max_memory_depth: int,
    min_memory_depth: int,
    probe_interval: int,
    probe_repeats: int,
    probe_rounds: int,
    main_analysis_rounds: int,
    out_dir: Path,
    progress_queue=None,
    push_partial_progress: bool = True,
) -> Tuple[int, List[Path]]:
    """progress_queue (optional): a multiprocessing Queue. If
    push_partial_progress is True (the default -- needed on the cloud
    session, where the working tree must stay clean between turns), each
    probe timepoint's newly-written committed-tier files (including the
    incrementally-rewritten generation_stats.csv) are put on the queue
    immediately, so the main process can commit+push as the seed runs
    rather than only once this (many-minute) function returns. If False
    (the production/Mac default -- see docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md
    section 1.6/Mac runbook), nothing is reported until the seed is fully
    done, at which point every committed-tier file from the whole run is
    reported together in one batch -- so only a genuinely complete
    seed's files are ever committed/pushed.

    Checkpointed every --probe-interval generations (piggybacking on the
    probe cadence): if seed_X_checkpoint.pkl exists in out_dir, resumes
    from it instead of starting over -- a crash or interruption loses at
    most one probe-interval's worth of generations, not the whole
    (possibly hours-long) seed. The checkpoint is deleted once the seed
    completes."""

    def report(paths: List[Path]) -> None:
        if progress_queue is not None and paths:
            progress_queue.put((seed, paths))

    ckpt_path = checkpoint_path_for(out_dir, seed)
    gs_path = generation_stats_path(out_dir, seed)

    resumed = ckpt_path.exists()
    if resumed:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        age = ckpt["age"]
        tables = ckpt["tables"]
        rng = ckpt["rng"]
        rng_noise = ckpt["rng_noise"]
        rng_probe = ckpt["rng_probe"]
        gen_rows: List[Tuple] = ckpt["gen_rows"]
        committed: List[Path] = ckpt["committed"]
    else:
        age = 0
        rng = random.Random(seed)
        rng_noise = random.Random(rng.random())  # same split-stream convention as population.run_evolution
        rng_probe = random.Random(rng.random())
        tables = [random_table(initial_memory_depth, rng) for _ in range(population_size)]
        gen_rows = []
        committed = []

    def probe_report_and_checkpoint(age: int) -> None:
        new_paths = run_probes_for_age(
            tables, age, seed, noise_rate, probe_noise_rates, rng_probe,
            out_dir, probe_repeats, probe_rounds, main_analysis_rounds,
        )
        write_generation_stats(gs_path, gen_rows)  # keep the stats file in sync with progress so far
        new_paths.append(gs_path)
        committed.extend(new_paths)
        if push_partial_progress:
            report(new_paths)
        _atomic_pickle_dump(ckpt_path, {
            "age": age, "tables": tables, "rng": rng, "rng_noise": rng_noise,
            "rng_probe": rng_probe, "gen_rows": gen_rows, "committed": committed,
        })

    if not resumed:  # a resumed checkpoint's age always already reflects a completed probe
        probe_report_and_checkpoint(age)

    for gen in range(age, generations):
        payoffs = round_robin_payoffs(tables, rounds_per_duel, noise_rate=noise_rate, rng=rng_noise)
        depths = [t.memory_depth for t in tables]
        n = len(depths)
        mean_depth = sum(depths) / n
        std_depth = (sum((d - mean_depth) ** 2 for d in depths) / n) ** 0.5
        mean_payoff = sum(payoffs) / len(payoffs)
        gen_rows.append((gen, mean_depth, std_depth, min(depths), max(depths), mean_payoff))

        tables = select_and_reproduce(tables, payoffs, rng)
        tables = [
            mutate_individual(
                t, rng, point_mutation_rate, duplication_rate, max_memory_depth,
                split_rate=split_rate, min_memory_depth=min_memory_depth,
            )
            for t in tables
        ]
        age = gen + 1
        if age % probe_interval == 0:
            probe_report_and_checkpoint(age)

    if generations % probe_interval != 0:
        # trailing generations past the last probe timepoint -- the stats
        # file needs one more write to cover them. If generations IS a
        # multiple of probe_interval, the loop's last probe already wrote
        # (and, if push_partial_progress, already reported) everything.
        write_generation_stats(gs_path, gen_rows)
        if gs_path not in committed:
            committed.append(gs_path)
        if push_partial_progress:
            report([gs_path])

    if not push_partial_progress:
        report(committed)  # whole-seed batch, now that it's genuinely complete
    ckpt_path.unlink(missing_ok=True)

    return seed, committed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--seed-list", type=str, default=None, help="comma-separated explicit seeds, overrides --seeds/--seed-offset")
    p.add_argument("--population-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=2000)
    p.add_argument("--rounds-per-duel", type=int, default=50)
    p.add_argument("--initial-memory-depth", type=int, default=1)
    p.add_argument("--point-mutation-rate", type=float, default=0.05)
    p.add_argument("--duplication-rate", type=float, default=0.02)
    p.add_argument("--split-rate", type=float, default=0.02)
    p.add_argument("--max-memory-depth", type=int, default=6)
    p.add_argument("--min-memory-depth", type=int, default=0)
    p.add_argument("--noise-rate", type=float, default=0.01, help="evolution-run implementation-error rate (design doc section 1.2)")
    p.add_argument("--probe-noise-rates", type=str, default=None, help="comma-separated override; default is {--noise-rate, 0.0} deduplicated")
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--probe-repeats", type=int, default=30)
    p.add_argument("--probe-rounds", type=int, default=100)
    p.add_argument("--main-analysis-rounds", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--push", action="store_true", help="commit+push committed-tier files")
    p.add_argument(
        "--push-partial-progress", action="store_true", default=False,
        help="push committed-tier files as they're written mid-seed (needed to keep a cloud "
        "session's working tree clean between turns). Default off: push only once a seed is "
        "fully complete, with its whole file set in one batch -- the Mac/production default "
        "(docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md Mac runbook: push only completed files).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_noise_rates is not None:
        probe_noise_rates = tuple(sorted({float(x) for x in args.probe_noise_rates.split(",") if x.strip() != ""}))
    else:
        probe_noise_rates = tuple(sorted({args.noise_rate, 0.0}))

    if args.seed_list is not None:
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    else:
        seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    pending = [s for s in seeds if not seed_is_complete(out_dir, s, args.generations)]
    n_resuming = sum(1 for s in pending if checkpoint_path_for(out_dir, s).exists())
    print(
        f"{len(pending)}/{len(seeds)} seeds pending ({n_resuming} resuming from an in-progress checkpoint) "
        f"(noise_rate={args.noise_rate}, probe_noise_rates={probe_noise_rates})", flush=True,
    )

    # Each seed runs for many minutes; workers report newly-written
    # committed-tier files via this queue as soon as each probe timepoint
    # is done (see run_one_seed's `report`), so the main process can
    # commit+push incrementally instead of only once per (long) seed --
    # this keeps the working tree from accumulating untracked files for
    # the whole duration of a seed's run.
    manager = Manager()
    progress_queue = manager.Queue()
    flush_every_seconds = 20

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one_seed, seed, args.noise_rate, probe_noise_rates,
                args.population_size, args.generations, args.rounds_per_duel,
                args.initial_memory_depth, args.point_mutation_rate, args.duplication_rate,
                args.split_rate, args.max_memory_depth, args.min_memory_depth,
                args.probe_interval, args.probe_repeats, args.probe_rounds,
                args.main_analysis_rounds, out_dir, progress_queue, args.push_partial_progress,
            ): seed
            for seed in pending
        }
        remaining = set(futures.keys())
        accumulated: List[Path] = []
        last_flush = time.time()

        def flush(message: str) -> None:
            nonlocal accumulated, last_flush
            if accumulated and args.push:
                commit_and_push(accumulated, message)
            accumulated = []
            last_flush = time.time()

        while remaining:
            try:
                seed, paths = progress_queue.get(timeout=2)
                accumulated.extend(paths)
            except queue_module.Empty:
                pass

            done_now = {fut for fut in remaining if fut.done()}
            for fut in done_now:
                seed = futures[fut]
                _, committed = fut.result()
                remaining.discard(fut)
                print(f"seed={seed} done ({len(committed)} committed-tier files)", flush=True)

            if accumulated and (time.time() - last_flush >= flush_every_seconds or not remaining):
                flush(f"Markov sweep progress (noise_rate={args.noise_rate}, {len(accumulated)} files)")

        # drain anything still sitting in the queue after all seeds finished
        while True:
            try:
                seed, paths = progress_queue.get_nowait()
                accumulated.extend(paths)
            except queue_module.Empty:
                break
        flush(f"Markov sweep progress (noise_rate={args.noise_rate}, final batch)")

    print("all seeds done", flush=True)


if __name__ == "__main__":
    main()
