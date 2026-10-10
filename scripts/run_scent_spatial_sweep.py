"""Chapter 3 scent-movement sweep runner, chunked/checkpointed -- same
engine (chunking, checkpoint resume, lock file, Windows sleep-
suppression, periodic git push) as run_local_placement_sweep.py,
adapted for evopersonality_abm.scent_spatial.run_coupled_evolution_scent.

One invocation runs ONE movement condition (--movement none|random|scent)
across --n-seeds seeds. Output/progress/lock paths are split by
experiment (movement) THEN machine name (platform.node(), sanitized),
per the pre-registration's "run operations" section -- so a lock or
in-progress checkpoint written by one physical machine is never
mistaken for stale (or live) state by a different machine happening to
touch the same movement condition's output tree, and the data each
machine produced stays traceable to it.

Usage:
    # pilot (short, cloud): 5 seeds, 1000 gens
    python3 scripts/run_scent_spatial_sweep.py --movement scent --pilot
    python3 scripts/run_scent_spatial_sweep.py --movement none --pilot
    python3 scripts/run_scent_spatial_sweep.py --movement random --p-move 0.3 --pilot

    # production: 50 seeds, 5000 gens
    python3 scripts/run_scent_spatial_sweep.py --movement scent
    python3 scripts/run_scent_spatial_sweep.py --movement random --p-move 0.3
    python3 scripts/run_scent_spatial_sweep.py --movement none
"""

import argparse
import csv
import ctypes
import json
import os
import pickle
import platform
import re
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

from evopersonality_abm.scent_spatial import run_coupled_evolution_scent

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
SEEDS = list(range(50))
GENERATIONS = 5000
LOG_INTERVAL = 100
CHUNK_SIZE = 250

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
    grid_width=13,
    grid_height=13,
    n_phases=5,
    tau=1.0,
    radius=1,
)

AggRow = Tuple[int, float, float, float, float, float, float, float, float, float, float, float, float, float, int, float]
IndividualRow = Tuple[int, int, float, float, int, float, float, float, int, int]


def machine_tag() -> str:
    raw = platform.node() or "unknown-machine"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "unknown-machine"


def out_root_for(movement: str, suffix: str = "") -> Path:
    return EXPERIMENT_ROOT / (movement + suffix) / machine_tag()


def agg_path_for(out_root, seed):
    return out_root / f"seed_{seed}_timeseries.csv"


def individual_path_for(out_root, seed):
    return out_root / f"seed_{seed}_individual_log.csv"


def checkpoint_path_for(out_root, seed):
    return out_root / f"seed_{seed}_checkpoint.pkl"


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


def run_one_seed_chunked(
    seed: int, generations: int, chunk_size: int, out_root_str: str,
    movement: str, p_move: Optional[float], model_params: dict,
):
    out_root = Path(out_root_str)
    ckpt_path = checkpoint_path_for(out_root, seed)

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        next_gen = ckpt["next_generation"]
        genomes = ckpt["genomes"]
        cells = ckpt["cells"]
        rng_state = ckpt["rng_state"]
        rng_spatial_state = ckpt["rng_spatial_state"]
        parent_of = ckpt["parent_of"]
        agg_rows: List[AggRow] = ckpt["agg_rows"]
        individual_rows: List[IndividualRow] = ckpt["individual_rows"]
    else:
        next_gen = 0
        genomes = None
        cells = None
        rng_state = None
        rng_spatial_state = None
        parent_of = None
        agg_rows = []
        individual_rows = []

    t0 = time.time()
    n_chunks_this_call = 0
    while next_gen < generations:
        chunk_end = min(next_gen + chunk_size, generations)
        chunk_stats, genomes, cells, rng_state, rng_spatial_state, parent_of = run_coupled_evolution_scent(
            seed=seed, generations=chunk_end, start_generation=next_gen,
            movement=movement, p_move=p_move,
            initial_genomes=genomes, initial_cells=cells,
            rng_state=rng_state, rng_spatial_state=rng_spatial_state,
            parent_of_state=parent_of, track_parent_state=True,
            return_state=True, **model_params,
        )
        for s in chunk_stats:
            agg_rows.append(
                (
                    s.generation,
                    s.mean_k_out, statistics.pstdev(s.k_out),
                    s.mean_k_self, statistics.pstdev(s.k_self),
                    statistics.fmean(s.memory_depths),
                    s.mean_payoff, s.cooperation_rate,
                    s.k_out_assortativity, s.k_self_assortativity, s.same_parent_fraction,
                    s.move_rate, s.isolated_fraction,
                    s.mean_opponent_count, s.max_opponent_count, s.scent_index_mean,
                )
            )
            if s.generation % LOG_INTERVAL == 0:
                for i in range(len(s.k_out)):
                    pos_x, pos_y = s.positions[i]
                    individual_rows.append(
                        (s.generation, i, s.k_out[i], s.k_self[i], s.memory_depths[i], s.payoffs[i],
                         s.opponent_mean_k_self[i], s.opponent_mean_k_out[i], pos_x, pos_y)
                    )
        next_gen = chunk_end
        n_chunks_this_call += 1

        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "next_generation": next_gen,
                    "genomes": genomes,
                    "cells": cells,
                    "rng_state": rng_state,
                    "rng_spatial_state": rng_spatial_state,
                    "parent_of": parent_of,
                    "agg_rows": agg_rows,
                    "individual_rows": individual_rows,
                },
                f,
            )

    elapsed = time.time() - t0
    return seed, agg_rows, individual_rows, elapsed, n_chunks_this_call


def write_agg_csv(path: Path, rows: List[AggRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self",
            "mean_depth", "mean_payoff", "cooperation_rate",
            "k_out_assortativity", "k_self_assortativity", "same_parent_fraction",
            "move_rate", "isolated_fraction", "mean_opponent_count", "max_opponent_count",
            "scent_index_mean",
        ])
        for (gen, mko, sko, mks, sks, md, mp, coop, kout_a, kself_a, spf,
             move_rate, iso_frac, mean_opp, max_opp, scent_index) in rows:
            def fmt(x):
                return "" if x != x else f"{x:.6f}"  # NaN -> empty cell
            w.writerow([
                gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                f"{md:.4f}", f"{mp:.6f}", fmt(coop), fmt(kout_a), fmt(kself_a), fmt(spf),
                f"{move_rate:.6f}", f"{iso_frac:.6f}", f"{mean_opp:.6f}", max_opp, fmt(scent_index),
            ])


def write_individual_csv(path: Path, rows: List[IndividualRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff",
            "opponent_mean_k_self", "opponent_mean_k_out", "pos_x", "pos_y",
        ])
        for gen, idx, k_out, k_self, depth, payoff, opp_kself, opp_kout, px, py in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}",
                        f"{opp_kself:.6f}", f"{opp_kout:.6f}", px, py])


def write_run_metadata(out_root: Path, movement: str, p_move: Optional[float], model_params: dict) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "machine": platform.node(),
        "os": platform.platform(),
        "python_version": sys.version,
        "movement": movement,
        "p_move": p_move,
        "model_params": model_params,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_metadata.json", "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def status_path_for(out_root: Path) -> Path:
    return Path("progress") / f"scent_spatial_{out_root.parent.name}_{out_root.name}_status.json"


def write_status_json(path: Path, out_root: Path, seeds: List[int], generations: int) -> None:
    done = sum(
        1 for s in seeds
        if agg_path_for(out_root, s).exists() and individual_path_for(out_root, s).exists()
    )
    in_progress = []
    for s in seeds:
        ckpt_path = checkpoint_path_for(out_root, s)
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "rb") as f:
                    ckpt = pickle.load(f)
                in_progress.append({"seed": s, "next_generation": ckpt["next_generation"]})
            except Exception:
                pass
    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generations_per_seed": generations,
        "done": done,
        "total": len(seeds),
        "in_progress": in_progress,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: List[Path], message: str) -> None:
    """Commits and pushes; never raises or stops the run on failure. Pulls
    --rebase first so a remote that moved ahead (e.g. a push from a
    different machine) doesn't repeatedly reject this push."""
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
    paths: List[Path], message: str, branch: str = "checkpoint-scratch-scent-spatial"
) -> None:
    """Same history-free scratch-branch snapshot mechanism as the other
    chunked runners (see run_local_placement_sweep.py's docstring for
    why), on its own branch name so checkpoint snapshots never collide
    across experiments."""
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
    p.add_argument("--movement", choices=["none", "random", "scent"], required=True)
    p.add_argument("--p-move", type=float, default=None,
                   help="required when --movement random (fixed move probability, fit from an M2 pilot)")
    p.add_argument("--n-seeds", type=int, default=len(SEEDS))
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--grid-width", type=int, default=PARAMS["grid_width"])
    p.add_argument("--grid-height", type=int, default=PARAMS["grid_height"])
    p.add_argument("--radius", type=int, default=PARAMS["radius"])
    p.add_argument("--n-phases", type=int, default=PARAMS["n_phases"])
    p.add_argument("--tau", type=float, default=PARAMS["tau"])
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="git commit+push every N newly-completed seeds")
    p.add_argument("--checkpoint-poll-seconds", type=int, default=600)
    p.add_argument("--push-checkpoints", action="store_true")
    p.add_argument("--status-interval-seconds", type=int, default=3600)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-push", action="store_true", help="disable all git commit/push (local-only)")
    p.add_argument("--pilot", action="store_true", help="5 seeds, 1000 gens, chunk-size 250, no push")
    p.add_argument("--out-suffix", type=str, default="",
                   help="extra suffix appended to the output directory name (after _pilot when --pilot "
                        "is set), so a rerun under changed parameters (e.g. a recalibrated tau or "
                        "p_move) doesn't collide with and overwrite previously-recorded data for the "
                        "same movement condition")
    return p.parse_args()


def main():
    args = parse_args()
    if args.movement == "random" and args.p_move is None:
        print("ERROR: --movement random requires --p-move <float>", file=sys.stderr)
        sys.exit(1)

    model_params = dict(PARAMS)
    model_params["grid_width"] = args.grid_width
    model_params["grid_height"] = args.grid_height
    model_params["radius"] = args.radius
    model_params["n_phases"] = args.n_phases
    model_params["tau"] = args.tau

    if args.pilot:
        seeds = list(range(5))
        generations = 1000
        chunk_size = 250
        out_root = out_root_for(args.movement, suffix="_pilot" + args.out_suffix)
        do_push = False
    else:
        seeds = list(range(args.n_seeds))
        generations = args.generations
        chunk_size = args.chunk_size
        out_root = out_root_for(args.movement, suffix=args.out_suffix)
        do_push = not args.no_push

    push_checkpoints = do_push and args.push_checkpoints
    status_interval = args.status_interval_seconds if do_push else 0
    status_path = REPO_ROOT / status_path_for(out_root)

    lock_path = acquire_lock(out_root)
    prevent_sleep_windows()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        write_run_metadata(out_root, args.movement, args.p_move, model_params)

        pending = [
            s for s in seeds
            if not (agg_path_for(out_root, s).exists() and individual_path_for(out_root, s).exists())
        ]
        already_done = len(seeds) - len(pending)
        n_with_checkpoint = sum(1 for s in pending if checkpoint_path_for(out_root, s).exists())
        print(f"{already_done}/{len(seeds)} seeds already fully done", flush=True)
        print(f"{len(pending)} pending ({n_with_checkpoint} resuming from an in-progress checkpoint)", flush=True)

        if not pending:
            print("All seeds already complete -- nothing to run.", flush=True)
            return

        completed_count = already_done
        last_checkpoint = (completed_count // args.checkpoint_every) * args.checkpoint_every
        batch_since_checkpoint: List[int] = []
        paths_since_checkpoint: List[Path] = []

        t_start = time.time()
        last_checkpoint_poll = time.time()
        last_status_push = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one_seed_chunked, seed, generations, chunk_size, str(out_root),
                    args.movement, args.p_move, model_params,
                ): seed
                for seed in pending
            }
            while futures:
                done, not_done = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
                for future in done:
                    seed, agg_rows, individual_rows, elapsed, n_chunks = future.result()
                    agg_path = agg_path_for(out_root, seed)
                    ind_path = individual_path_for(out_root, seed)
                    write_agg_csv(agg_path, agg_rows)
                    write_individual_csv(ind_path, individual_rows)
                    checkpoint_path_for(out_root, seed).unlink(missing_ok=True)

                    completed_count += 1
                    batch_since_checkpoint.append(seed)
                    paths_since_checkpoint.extend([agg_path, ind_path])
                    per_gen_ms = elapsed / (n_chunks * chunk_size) * 1000 if n_chunks else 0.0
                    print(f"seed={seed} done ({completed_count}/{len(seeds)}): "
                          f"{elapsed:.1f}s this call ({n_chunks} chunk(s)), {per_gen_ms:.1f} ms/gen", flush=True)

                    if do_push:
                        threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                        if threshold > last_checkpoint:
                            message = (
                                f"Scent-spatial sweep ({args.movement}, {machine_tag()}, chunked): "
                                f"{completed_count}/{len(seeds)} seeds complete (batch: {sorted(batch_since_checkpoint)})"
                            )
                            commit_and_push(paths_since_checkpoint, message)
                            last_checkpoint = threshold
                            batch_since_checkpoint = []
                            paths_since_checkpoint = []

                futures = {f: futures[f] for f in not_done}

                if push_checkpoints and (time.time() - last_checkpoint_poll) >= args.checkpoint_poll_seconds:
                    ckpt_files = sorted(out_root.glob("seed_*_checkpoint.pkl"))
                    if ckpt_files:
                        push_checkpoint_snapshot_to_scratch_branch(
                            ckpt_files,
                            f"Scent-spatial sweep ({args.movement}, {machine_tag()}): periodic checkpoint snapshot "
                            f"({len(ckpt_files)} in-progress seed(s))",
                        )
                    last_checkpoint_poll = time.time()

                if status_interval > 0 and (time.time() - last_status_push) >= status_interval:
                    write_status_json(status_path, out_root, seeds, generations)
                    commit_and_push([status_path], f"Progress snapshot ({args.movement}, {machine_tag()}): "
                                                    f"{completed_count}/{len(seeds)} seeds done")
                    last_status_push = time.time()

        if do_push and batch_since_checkpoint:
            message = (
                f"Scent-spatial sweep ({args.movement}, {machine_tag()}, chunked): "
                f"{completed_count}/{len(seeds)} seeds complete (final partial batch: {sorted(batch_since_checkpoint)})"
            )
            commit_and_push(paths_since_checkpoint, message)

        if status_interval > 0:
            write_status_json(status_path, out_root, seeds, generations)
            commit_and_push([status_path], f"Progress snapshot ({args.movement}, {machine_tag()}): "
                                            f"{completed_count}/{len(seeds)} seeds done (final)")

        print(f"\nAll {completed_count}/{len(seeds)} seeds complete in {time.time() - t_start:.1f}s wall-clock.",
              flush=True)
    finally:
        allow_sleep_windows()
        release_lock(lock_path)


if __name__ == "__main__":
    main()
