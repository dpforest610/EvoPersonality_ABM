"""第二章 精密kスイープ右側延長(k=99,64,80)の本番ランナー -- 世代チャンク・
チェックポイント版。

背景(チャットで確認済み): このセッションのバックグラウンド実行には
30分前後で強制停止されるケースがあり、かつコンテナ自体の再起動も頻発
する。k=99/80/64は1シードあたり約1.3〜2時間かかるため、シード単位での
チェックポイント(既存のrun_opponent_count_precision_sweep.pyの粒度)
では、中断のたびにそのシードの進捗が丸ごと失われてしまう。

これを解決するため、1シードの実行を --chunk-size(デフォルト500)世代
ごとに区切り、各チャンク終了時に以下をpickleでディスクに保存する:
  - 次に実行すべき世代番号(next_generation)
  - 個体群(genomes)全体
  - rng/rng_ring の内部状態(random.Random.getstate())
  - これまでに蓄積した集計行・個体ログ行

再開時はこのチェックポイントを読み込み、保存された状態から dead-on
(ビット一致で)再開する -- これは
tests/test_spatial.py の
test_opponent_count_chunked_resume_is_bit_identical_to_uninterrupted()
および test_opponent_count_chunked_resume_works_across_three_chunks()
で、中断なし実行との完全一致を確認済み。

チャンクサイズ500世代は、最も重いk=99(約1.46秒/世代)でも1チャンク
約12分で終わり、30分の壁にも、観測されているコンテナ再起動の頻度にも
安全に収まるよう選んだ(k=80で約9.5分、k=64で約8分)。

既存のrun_opponent_count_precision_sweep_extension.pyと出力先
(scripts/output/opponent_count_precision_sweep/k{k}/)を共有するため、
完了済みの最終CSVはそのまま有効。チェックポイントファイルは
scripts/output/opponent_count_precision_sweep/k{k}/seed_{s}_checkpoint.pkl
に置き、両方の最終CSVが揃った時点で削除する。

Usage:
    python3 scripts/run_opponent_count_precision_sweep_chunked.py
    python3 scripts/run_opponent_count_precision_sweep_chunked.py --smoke
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
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.spatial import run_coupled_evolution_opponent_count

OUT_ROOT = Path("scripts/output/opponent_count_precision_sweep")
K_VALUES = [99, 64, 80]  # same priority order as the extension sweep
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

AggRow = Tuple[int, float, float, float, float, float, float]
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
    """Cross-platform liveness check, used only to decide whether a lock
    file is stale. On POSIX, os.kill(pid, 0) sends no actual signal -- it
    just probes whether the pid exists. On Windows, os.kill does NOT have
    that "signal 0" special case: any signal value Windows doesn't treat as
    CTRL_C_EVENT/CTRL_BREAK_EVENT is implemented via TerminateProcess(), so
    calling os.kill(pid, 0) there would actually kill a live process -- the
    opposite of a safe probe. So on Windows this instead uses OpenProcess +
    GetExitCodeProcess via ctypes, which only queries state and can't
    terminate anything."""
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
            return True  # exists under another user -- assume it's real


def acquire_lock(out_root: Path) -> Path:
    """Refuses to start a second instance against the same out_root while
    one is already running (e.g. the user double-clicking the start script,
    or a cloud run and a local run pointed at the same checkpoints). A lock
    held by a pid that is no longer alive (crash, power loss, kill -9) is
    treated as stale and silently taken over -- that is the normal case on
    every "resume after an interruption" restart, not an error."""
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
    """Windows has no equivalent of macOS's `caffeinate` wrapper command, so
    this does the same job from inside the process via the Win32
    SetThreadExecutionState API: it tells Windows this thread needs the
    system to stay awake, without touching the display-sleep timer (we only
    care about the machine not suspending, not the screen staying on). A
    no-op on every other OS."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)


def allow_sleep_windows() -> None:
    """Undoes prevent_sleep_windows() -- must be called when the run ends
    (including on a crash) or Windows would stay awake-pinned until the
    process holding that flag exits anyway. A no-op on every other OS."""
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def run_one_seed_chunked(k: int, seed: int, generations: int, chunk_size: int, out_root_str: str):
    out_root = Path(out_root_str)
    ckpt_path = checkpoint_path_for(out_root, k, seed)

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        next_gen = ckpt["next_generation"]
        genomes = ckpt["genomes"]
        rng_state = ckpt["rng_state"]
        rng_ring_state = ckpt["rng_ring_state"]
        agg_rows: List[AggRow] = ckpt["agg_rows"]
        individual_rows: List[IndividualRow] = ckpt["individual_rows"]
    else:
        next_gen = 0
        genomes = None
        rng_state = None
        rng_ring_state = None
        agg_rows = []
        individual_rows = []

    t0 = time.time()
    n_chunks_this_call = 0
    while next_gen < generations:
        chunk_end = min(next_gen + chunk_size, generations)
        chunk_stats, genomes, rng_state, rng_ring_state = run_coupled_evolution_opponent_count(
            seed=seed, k=k, generations=chunk_end, start_generation=next_gen,
            initial_genomes=genomes, rng_state=rng_state, rng_ring_state=rng_ring_state,
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


def status_path_for(out_root: Path) -> Path:
    return Path("progress") / f"{out_root.name}_status.json"


def write_status_json(path: Path, out_root: Path, k_values: List[int], seeds: List[int], generations: int) -> None:
    """Writes a small summary (per-k done count out of len(seeds), and the
    next_generation of any in-progress checkpoint) -- NOT the checkpoints
    themselves. Meant to be pushed to the main branch cheaply and often (the
    file itself stays tiny; only its content changes between pushes), so
    anyone can check progress by just reading GitHub without needing access
    to whatever machine is actually running the sweep."""
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
    # as_posix() rather than str(): on Windows, str(WindowsPath(...)) renders
    # with backslashes, but git always accepts (and internally normalizes
    # to) forward slashes regardless of platform -- using it unconditionally
    # here is simpler and safer than branching on os.name.
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


def push_checkpoint_snapshot_to_scratch_branch(
    paths: List[Path], message: str, branch: str = "checkpoint-scratch"
) -> None:
    """Pushes in-progress checkpoint .pkl files to a dedicated branch as a
    single history-free root commit (no parent), force-pushed each time.

    This exists because the real git history (committed via commit_and_push)
    should only ever grow from permanent, final results -- not from
    transient in-progress checkpoints that get superseded (and deleted)
    every ~600s. Committing those normally would add a new ~60-240KB blob
    per checkpoint file to history on every poll, across up to 50 seeds x
    3 k-values x many chunks over a 72-hour run, which was estimated to add
    hundreds of MB of permanent, un-reclaimable bloat to the repo.

    Using a throwaway temp index (GIT_INDEX_FILE) instead of the real repo
    index means this never touches the running script's own working-tree
    state or the main branch's staged changes. Building the tree from only
    the given paths (not a full repo snapshot) keeps each pushed commit
    minimal. commit-tree with no -p parent makes a fresh root commit each
    time, so force-pushing it over the branch's previous tip leaves the old
    commit unreferenced (and eventually GC'd) instead of retained in
    history -- the key property that avoids repo-size growth.
    """
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
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="git commit+push every N newly-completed (k,seed) units")
    p.add_argument("--checkpoint-poll-seconds", type=int, default=600,
                   help="if --push-checkpoints is set, push the in-progress chunk checkpoints "
                        "to checkpoint-scratch at least this often")
    p.add_argument("--push-checkpoints", action="store_true",
                   help="also push periodic in-progress checkpoint .pkl snapshots to the "
                        "checkpoint-scratch branch (off by default: a local run's own disk is "
                        "its own backup, so this is mainly for an unattended cloud run with no "
                        "persistent disk of its own)")
    p.add_argument("--status-interval-seconds", type=int, default=3600,
                   help="write progress/<out-dir-name>_status.json (per-k done counts and "
                        "in-progress generations, NOT the checkpoints) and push just that one "
                        "small file to the main branch at least this often. 0 disables. A push "
                        "failure here is logged and retried next cycle -- it never stops the run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true",
                   help="wiring check: k=8 only, 2 seeds x 50 gen, chunk-size 10, no push, separate out-dir")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        k_values = [8]
        seeds = [0, 1]
        generations = 50
        chunk_size = 10
        out_root = Path(str(OUT_ROOT) + "_smoke")
        do_push = False
    else:
        k_values = K_VALUES
        seeds = SEEDS
        generations = GENERATIONS
        chunk_size = args.chunk_size
        out_root = OUT_ROOT
        do_push = True

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
                executor.submit(run_one_seed_chunked, k, seed, generations, chunk_size, str(out_root)): (k, seed)
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
                                f"Opponent-count precision sweep (chunked): {completed_count}/{len(units)} units complete "
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
                            f"Opponent-count precision sweep (chunked): periodic checkpoint snapshot "
                            f"({len(ckpt_files)} in-progress seed(s))",
                        )
                    last_checkpoint_poll = time.time()

                if status_interval > 0 and (time.time() - last_status_push) >= status_interval:
                    write_status_json(status_path, out_root, k_values, seeds, generations)
                    commit_and_push([status_path], f"Progress snapshot: {completed_count}/{len(units)} units done")
                    last_status_push = time.time()

        if do_push and batch_since_checkpoint:
            message = (
                f"Opponent-count precision sweep (chunked): {completed_count}/{len(units)} units complete "
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
