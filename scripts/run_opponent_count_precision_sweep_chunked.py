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
import pickle
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
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="git commit+push every N newly-completed (k,seed) units")
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
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_seed_chunked, k, seed, generations, chunk_size, str(out_root)): (k, seed)
            for k, seed in pending
        }
        for future in as_completed(futures):
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

    if do_push and batch_since_checkpoint:
        message = (
            f"Opponent-count precision sweep (chunked): {completed_count}/{len(units)} units complete "
            f"(final partial batch: {sorted(batch_since_checkpoint)})"
        )
        commit_and_push(paths_since_checkpoint, message)

    print(f"\nAll {completed_count}/{len(units)} units complete in {time.time() - t_start:.1f}s wall-clock.",
          flush=True)


if __name__ == "__main__":
    main()
