"""Local-placement experiment runner (placement="local"), chunked/
checkpointed -- same engine (chunking, checkpoint resume, lock file,
Windows sleep-suppression, periodic git push) as
run_opponent_count_precision_sweep_chunked.py, adapted for this
different research question and output directory.

Only placement="local" is run here. placement="shuffle" comparison data
already exists for k=4,8,16,32 in scripts/output/opponent_count_
precision_sweep/ (the original precision k-sweep, same seeds 0-49, same
PARAMS), so it is reused rather than re-simulated.

Output: scripts/output/local_placement_sweep/k{k}/seed_{s}_timeseries.csv
(per-generation aggregates, including the new cooperation_rate/
k_out_assortativity/k_self_assortativity/same_parent_fraction columns)
and seed_{s}_individual_log.csv (per-agent, every 100 gen, unchanged
schema from the precision sweep).

Usage:
    # pilot (short, cloud): k=8, 10 seeds, 1000 gens
    python3 scripts/run_local_placement_sweep.py --pilot

    # production: k=8,4,16,32 (priority order), 50 seeds, 5000 gens
    python3 scripts/run_local_placement_sweep.py
"""

import argparse
import csv
import ctypes
import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_opponent_count

OUT_ROOT = Path("scripts/output/local_placement_sweep")
K_VALUES = [8, 4, 16, 32]  # priority order, matches existing shuffle-condition k's
SEEDS = list(range(50))
GENERATIONS = 5000
LOG_INTERVAL = 100
CHUNK_SIZE = 500

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

AggRow = Tuple[int, float, float, float, float, float, float, float, float, float]
IndividualRow = Tuple[int, int, float, float, int, float]


def agg_path_for(out_root, k, seed):
    return out_root / f"k{k}" / f"seed_{seed}_timeseries.csv"


def individual_path_for(out_root, k, seed):
    return out_root / f"k{k}" / f"seed_{seed}_individual_log.csv"


def checkpoint_path_for(out_root, k, seed):
    return out_root / f"k{k}" / f"seed_{seed}_checkpoint.pkl"


def lock_path_for(out_root: Path) -> Path:
    return out_root / ".run.lock"


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform liveness check for lock staleness. POSIX:
    os.kill(pid, 0) is a pure existence probe. Windows: os.kill has no
    such special case for signal 0 (it maps to TerminateProcess), so
    this uses OpenProcess + GetExitCodeProcess via ctypes instead, which
    only queries and cannot terminate anything."""
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def acquire_lock(out_root: Path) -> Path:
    lock_path = lock_path_for(out_root)
    if lock_path.exists():
        old_pid = None
        try:
            old_pid = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            pass
        alive = old_pid is not None and _pid_is_alive(old_pid)
        if alive:
            print(
                f"Another run (pid {old_pid}) already holds the lock at {lock_path}.\n"
                f"Refusing to start a second instance against the same output directory.\n"
                f"If you're sure nothing is actually running, delete that file and retry.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Found a stale lock (pid {old_pid} is not running) -- taking over.", flush=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))
    return lock_path


def release_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep_windows() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def allow_sleep_windows() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def run_one_seed_chunked(k: int, seed: int, generations: int, chunk_size: int, out_root_str: str, placement: str = "local"):
    out_root = Path(out_root_str)
    ckpt_path = checkpoint_path_for(out_root, k, seed)

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        next_gen = ckpt["next_generation"]
        genomes = ckpt["genomes"]
        rng_state = ckpt["rng_state"]
        rng_ring_state = ckpt["rng_ring_state"]
        parent_of = ckpt["parent_of"]
        agg_rows: List[AggRow] = ckpt["agg_rows"]
        individual_rows: List[IndividualRow] = ckpt["individual_rows"]
    else:
        next_gen = 0
        genomes = None
        rng_state = None
        rng_ring_state = None
        parent_of = None
        agg_rows = []
        individual_rows = []

    t0 = time.time()
    n_chunks_this_call = 0
    while next_gen < generations:
        chunk_end = min(next_gen + chunk_size, generations)
        chunk_stats, genomes, rng_state, rng_ring_state, parent_of = run_coupled_evolution_opponent_count(
            seed=seed, k=k, generations=chunk_end, start_generation=next_gen,
            placement=placement,
            initial_genomes=genomes, rng_state=rng_state, rng_ring_state=rng_ring_state,
            parent_of_state=parent_of, track_parent_state=True,
            return_state=True, **PARAMS,
        )
        for s in chunk_stats:
            agg_rows.append(
                (
                    s.generation,
                    s.mean_k_out,
                    statistics.pstdev(s.k_out),
                    s.mean_k_self,
                    statistics.pstdev(s.k_self),
                    statistics.fmean(s.memory_depths),
                    s.mean_payoff,
                    s.cooperation_rate,
                    s.k_out_assortativity,
                    s.k_self_assortativity,
                    s.same_parent_fraction,
                )
            )
            if s.generation % LOG_INTERVAL == 0:
                for i in range(len(s.k_out)):
                    individual_rows.append(
                        (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i])
                    )
        next_gen = chunk_end
        n_chunks_this_call += 1

        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "next_generation": next_gen,
                    "genomes": genomes,
                    "rng_state": rng_state,
                    "rng_ring_state": rng_ring_state,
                    "parent_of": parent_of,
                    "agg_rows": agg_rows,
                    "individual_rows": individual_rows,
                },
                f,
            )

    elapsed = time.time() - t0
    return k, seed, agg_rows, individual_rows, elapsed, n_chunks_this_call


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
                    "mean_depth", "mean_payoff", "cooperation_rate",
                    "k_out_assortativity", "k_self_assortativity", "same_parent_fraction"])
        for gen, mko, sko, mks, sks, md, mp, coop, kout_a, kself_a, spf in rows:
            def fmt(x):
                return "" if x != x else f"{x:.6f}"  # NaN -> empty cell
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                        f"{md:.4f}", f"{mp:.6f}", fmt(coop), fmt(kout_a), fmt(kself_a), fmt(spf)])


def write_individual_csv(path: Path, rows: List[IndividualRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff"])
        for gen, idx, k_out, k_self, depth, payoff in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}"])


def status_path_for(out_root: Path) -> Path:
    return Path("progress") / f"{out_root.name}_status.json"


def write_status_json(path: Path, out_root: Path, k_values: List[int], seeds: List[int], generations: int) -> None:
    by_k = {}
    for k in k_values:
        k_dir = out_root / f"k{k}"
        done = sum(
            1 for s in seeds
            if (k_dir / f"seed_{s}_timeseries.csv").exists()
            and (k_dir / f"seed_{s}_individual_log.csv").exists()
        )
        in_progress = []
        for s in seeds:
            ckpt_path = checkpoint_path_for(out_root, k, s)
            if ckpt_path.exists():
                try:
                    with open(ckpt_path, "rb") as f:
                        ckpt = pickle.load(f)
                    in_progress.append({"seed": s, "next_generation": ckpt["next_generation"]})
                except Exception:
                    pass
        by_k[str(k)] = {"done": done, "total": len(seeds), "in_progress": in_progress}

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generations_per_seed": generations,
        "by_k": by_k,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: List[Path], message: str) -> None:
    """Commits and pushes; never raises or stops the run on failure. Pulls
    --rebase first: on a prior run, the remote being ahead (e.g. from a
    push made by a different machine running the same sweep) caused push
    to be rejected repeatedly with nothing to recover it, so this always
    syncs first."""
    add = _git("add", *[p.as_posix() for p in paths])
    if add.returncode != 0:
        print(f"git add failed for {message}: {add.stderr.strip()}", flush=True)
        return
    commit = _git("commit", "-m", message)
    if commit.returncode != 0:
        if "nothing to commit" in (commit.stdout + commit.stderr):
            return
        print(f"git commit failed for {message}: {commit.stderr.strip()}", flush=True)
        return

    pull = _git("pull", "--rebase")
    if pull.returncode != 0:
        print(f"git pull --rebase failed before pushing {message}: {pull.stderr.strip()}", flush=True)
        # still try to push -- if the remote hasn't moved, pull --rebase
        # failing (e.g. no network) doesn't mean push will also fail.

    delay = 2
    for attempt in range(4):
        push = _git("push")
        if push.returncode == 0:
            print(f"pushed: {message}", flush=True)
            return
        print(f"git push failed (attempt {attempt + 1}) for {message}: {push.stderr.strip()}", flush=True)
        if attempt < 3:
            retry_pull = _git("pull", "--rebase")
            if retry_pull.returncode != 0:
                print(f"git pull --rebase retry failed: {retry_pull.stderr.strip()}", flush=True)
        time.sleep(delay)
        delay *= 2
    print(f"git push ultimately failed for: {message}", flush=True)


def push_checkpoint_snapshot_to_scratch_branch(
    paths: List[Path], message: str, branch: str = "checkpoint-scratch-local-placement"
) -> None:
    """Same history-free scratch-branch snapshot mechanism as the
    precision-sweep runner (see its docstring for why), on its own
    branch name so the two experiments' checkpoint snapshots never
    collide."""
    if not paths:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = os.path.join(tmpdir, "index")
        rel_paths = [p.resolve().relative_to(REPO_ROOT).as_posix() for p in paths]

        add = subprocess.run(
            ["git", "add", "-f", "--", *rel_paths],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if add.returncode != 0:
            print(f"scratch-branch git add failed for {message}: {add.stderr.strip()}", flush=True)
            return
        write_tree = subprocess.run(
            ["git", "write-tree"], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if write_tree.returncode != 0:
            print(f"scratch-branch write-tree failed for {message}: {write_tree.stderr.strip()}", flush=True)
            return
        tree_sha = write_tree.stdout.strip()
        commit_tree = subprocess.run(
            ["git", "commit-tree", tree_sha, "-m", message],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        if commit_tree.returncode != 0:
            print(f"scratch-branch commit-tree failed for {message}: {commit_tree.stderr.strip()}", flush=True)
            return
        commit_sha = commit_tree.stdout.strip()

        delay = 2
        for attempt in range(4):
            push = subprocess.run(
                ["git", "push", "--force", "origin", f"{commit_sha}:refs/heads/{branch}"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if push.returncode == 0:
                print(f"pushed checkpoint snapshot to {branch}: {message}", flush=True)
                return
            print(f"scratch-branch push failed (attempt {attempt + 1}) for {message}: {push.stderr.strip()}", flush=True)
            time.sleep(delay)
            delay *= 2
        print(f"scratch-branch push ultimately failed for: {message}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k-values", type=str, default=",".join(str(k) for k in K_VALUES))
    p.add_argument("--n-seeds", type=int, default=len(SEEDS))
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="git commit+push every N newly-completed (k,seed) units")
    p.add_argument("--checkpoint-poll-seconds", type=int, default=600)
    p.add_argument("--push-checkpoints", action="store_true")
    p.add_argument("--status-interval-seconds", type=int, default=3600)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-push", action="store_true", help="disable all git commit/push (local-only)")
    p.add_argument("--pilot", action="store_true",
                   help="k=8 only, 10 seeds, 1000 gens, chunk-size 250, no push, separate out-dir")
    p.add_argument("--placement", choices=["local", "shuffle"], default="local",
                   help="shuffle is only for an apples-to-apples pilot-scale operational-check "
                        "comparison -- production shuffle-condition data already exists in the "
                        "main precision sweep, so this is not meant for production use")
    return p.parse_args()


def main():
    args = parse_args()

    if args.pilot:
        k_values = [8]
        seeds = list(range(10))
        generations = 1000
        chunk_size = 250
        suffix = "_pilot" if args.placement == "local" else "_pilot_shuffle"
        out_root = Path(str(OUT_ROOT) + suffix)
        do_push = False
    else:
        if args.placement != "local":
            print(
                "[ERROR] --placement shuffle is only for the pilot-scale operational-check "
                "comparison (use it together with --pilot). Production shuffle-condition data "
                "already exists in the main precision sweep -- running shuffle here outside "
                "--pilot would write into the same output directory as the real local-placement "
                "production run and overwrite/mix with it.",
                file=sys.stderr,
            )
            sys.exit(1)
        k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
        seeds = list(range(args.n_seeds))
        generations = args.generations
        chunk_size = args.chunk_size
        out_root = OUT_ROOT
        do_push = not args.no_push

    push_checkpoints = do_push and args.push_checkpoints
    status_interval = args.status_interval_seconds if do_push else 0
    status_path = REPO_ROOT / status_path_for(out_root)

    lock_path = acquire_lock(out_root)
    prevent_sleep_windows()
    try:
        units = [(k, seed) for k in k_values for seed in seeds]
        for k in k_values:
            (out_root / f"k{k}").mkdir(parents=True, exist_ok=True)

        pending = [
            (k, s) for k, s in units
            if not (agg_path_for(out_root, k, s).exists() and individual_path_for(out_root, k, s).exists())
        ]
        already_done = len(units) - len(pending)
        n_with_checkpoint = sum(1 for k, s in pending if checkpoint_path_for(out_root, k, s).exists())
        print(f"{already_done}/{len(units)} (k,seed) units already fully done", flush=True)
        print(f"{len(pending)} pending ({n_with_checkpoint} resuming from an in-progress checkpoint)", flush=True)

        if not pending:
            print("All units already complete -- nothing to run.", flush=True)
            return

        completed_count = already_done
        last_checkpoint = (completed_count // args.checkpoint_every) * args.checkpoint_every
        batch_since_checkpoint: List[Tuple[int, int]] = []
        paths_since_checkpoint: List[Path] = []

        t_start = time.time()
        last_checkpoint_poll = time.time()
        last_status_push = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one_seed_chunked, k, seed, generations, chunk_size, str(out_root), args.placement): (k, seed)
                for k, seed in pending
            }
            while futures:
                done, not_done = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
                for future in done:
                    k, seed, agg_rows, individual_rows, elapsed, n_chunks = future.result()
                    agg_path = agg_path_for(out_root, k, seed)
                    ind_path = individual_path_for(out_root, k, seed)
                    write_agg_csv(agg_path, agg_rows)
                    write_individual_csv(ind_path, individual_rows)
                    checkpoint_path_for(out_root, k, seed).unlink(missing_ok=True)

                    completed_count += 1
                    batch_since_checkpoint.append((k, seed))
                    paths_since_checkpoint.extend([agg_path, ind_path])
                    per_gen_ms = elapsed / (n_chunks * chunk_size) * 1000 if n_chunks else 0.0
                    print(f"k={k} seed={seed} done ({completed_count}/{len(units)}): "
                          f"{elapsed:.1f}s this call ({n_chunks} chunk(s)), {per_gen_ms:.1f} ms/gen", flush=True)

                    if do_push:
                        threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                        if threshold > last_checkpoint:
                            message = (
                                f"Local-placement sweep (chunked): {completed_count}/{len(units)} units complete "
                                f"(batch: {sorted(batch_since_checkpoint)})"
                            )
                            commit_and_push(paths_since_checkpoint, message)
                            last_checkpoint = threshold
                            batch_since_checkpoint = []
                            paths_since_checkpoint = []

                futures = {f: futures[f] for f in not_done}

                if push_checkpoints and (time.time() - last_checkpoint_poll) >= args.checkpoint_poll_seconds:
                    ckpt_files = sorted(out_root.glob("k*/seed_*_checkpoint.pkl"))
                    if ckpt_files:
                        push_checkpoint_snapshot_to_scratch_branch(
                            ckpt_files,
                            f"Local-placement sweep (chunked): periodic checkpoint snapshot "
                            f"({len(ckpt_files)} in-progress seed(s))",
                        )
                    last_checkpoint_poll = time.time()

                if status_interval > 0 and (time.time() - last_status_push) >= status_interval:
                    write_status_json(status_path, out_root, k_values, seeds, generations)
                    commit_and_push([status_path], f"Progress snapshot: {completed_count}/{len(units)} units done")
                    last_status_push = time.time()

        if do_push and batch_since_checkpoint:
            message = (
                f"Local-placement sweep (chunked): {completed_count}/{len(units)} units complete "
                f"(final partial batch: {sorted(batch_since_checkpoint)})"
            )
            commit_and_push(paths_since_checkpoint, message)

        if status_interval > 0:
            write_status_json(status_path, out_root, k_values, seeds, generations)
            commit_and_push([status_path], f"Progress snapshot: {completed_count}/{len(units)} units done (final)")

        print(f"\nAll {completed_count}/{len(units)} units complete in {time.time() - t_start:.1f}s wall-clock.",
              flush=True)
    finally:
        allow_sleep_windows()
        release_lock(lock_path)


if __name__ == "__main__":
    main()
