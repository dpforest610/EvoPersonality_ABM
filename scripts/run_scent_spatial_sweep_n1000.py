"""Chapter 3 scent-movement sweep runner for the N=1000 replication
(population=1000, 41x41 grid, density ~0.595 -- same as N=100's ~0.59),
checking whether the N=100 results (no significant M2-vs-M1 difference
in beta_ctx(K_out); K_out assortativity drop ~0.37->~0.08 under
movement; all three conditions' observed beta_ctx(K_out) inside the
ring-model-predicted interval) hold at 10x the population, and whether
z*_K_self sits closer to the population mean (weaker drift) than at
N=100. Same chunked/checkpointed engine (lock file, Windows sleep
suppression, periodic git push with pull --rebase first) as
run_scent_spatial_sweep.py.

Record-keeping differs from the N=100 runner (see
docs/PREREGISTRATION_scent_model_N1000.md section on this): the raw
per-agent individual log is an order of magnitude larger here (10x the
rows per snapshot), so it is written LOCALLY ONLY as gzip-compressed
CSV and is never committed (scripts/output/**/*_individual_log.csv.gz
is gitignored). What DOES get committed is the same small
per-generation timeseries.csv as before, plus a new one-row-per-seed
seed_X_summary.csv holding exactly the reduced statistics the final
analysis needs (context-model betas, own-trait Lande-Arnold
coefficients and their seed-mean beta/gamma for the z* bootstrap,
variance/bimodality) -- computed once, from the same in-memory
per-generation-snapshot data that would otherwise only exist in the
(uncommitted) raw log.

Usage:
    # tau calibration pilot (short, cloud): M2 only, one tau at a time
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.1 --pilot --generations 300 --n-seeds 3

    # pilot (short, cloud): 3 seeds, 1000 gens
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --pilot
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement none --pilot
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move 0.5 --pilot

    # production (Windows laptop): 30 seeds, 5000 gens
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --workers 15
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement random --p-move 0.5 --workers 15
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement none --workers 15

    # presentation recording (one seed, a generation range, every phase):
    python3 scripts/run_scent_spatial_sweep_n1000.py --movement scent --tau 0.033333 --pilot \
        --n-seeds 1 --record-phases-seed 0 --record-phases-gens 0-19
"""

import argparse
import csv
import ctypes
import gzip
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
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.scent_spatial import run_coupled_evolution_scent

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep_n1000")
SEEDS = list(range(30))
GENERATIONS = 5000
LOG_INTERVAL = 100
CHUNK_SIZE = 250
STEADY_STATE_MIN_GEN = 2000

PARAMS = dict(
    population_size=1000,
    rounds_per_duel=50,
    point_mutation_rate=0.05,
    duplication_rate=0.02,
    split_rate=0.02,
    max_memory_depth=8,
    min_memory_depth=0,
    k_mutation_std=0.05,
    noise_std=0.05,
    grid_width=41,
    grid_height=41,
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


def individual_gz_path_for(out_root, seed):
    return out_root / f"seed_{seed}_individual_log.csv.gz"


def summary_path_for(out_root, seed):
    return out_root / f"seed_{seed}_summary.csv"


def phase_recording_path_for(out_root, seed):
    return out_root / f"seed_{seed}_phase_recording.pkl.gz"


def checkpoint_path_for(out_root, seed):
    return out_root / f"seed_{seed}_checkpoint.pkl"


def lock_path_for(out_root: Path) -> Path:
    return out_root / ".run.lock"


def _pid_is_alive(pid: int) -> bool:
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


# ---------------------------------------------------------------- seed-summary reduction
# Same math as scripts/analyze_scent_production.py's context_lande_arnold /
# lande_arnold_coeffs / bimodality_coefficient -- duplicated here (not
# imported) so this runner has no dependency on the analysis script, and
# because the reduction here is computed inline from already-in-memory
# chunk_stats rather than re-reading a saved individual_log.csv.

import numpy as np
from scipy.stats import kurtosis, skew


def context_lande_arnold(k_self, k_out, opp_k_self, opp_k_out, payoff):
    def standardize(v):
        v = np.asarray(v)
        sd = np.std(v)
        return (v - np.mean(v)) / sd if sd > 0 else np.zeros_like(v)

    z1_self, z2_self = standardize(k_self), standardize(k_out)
    z1_ctx, z2_ctx = standardize(opp_k_self), standardize(opp_k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1_self), z1_self, z2_self, z1_ctx, z2_ctx])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return coef[1], coef[2], coef[3], coef[4]


def lande_arnold_coeffs(k_self, k_out, payoff):
    z1 = (np.asarray(k_self) - np.mean(k_self)) / np.std(k_self)
    z2 = (np.asarray(k_out) - np.mean(k_out)) / np.std(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2, z1 ** 2, z2 ** 2, z1 * z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    _, beta1, beta2, g11_raw, g22_raw, g12 = coef
    return beta1, beta2, 2 * g11_raw, 2 * g22_raw, g12


def bimodality_coefficient(values):
    values = np.asarray(values)
    n = len(values)
    if n < 4 or np.std(values) == 0:
        return float("nan")
    g1 = skew(values, bias=True)
    g2 = kurtosis(values, fisher=True, bias=True)
    denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 ** 2 + 1) / denom if denom != 0 else float("nan")


SUMMARY_FIELDS = [
    "seed",
    "beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout",
    "beta1", "beta2", "gamma11", "gamma22", "gamma12",
    "mean_beta1", "mean_beta2", "mean_gamma11", "mean_gamma22", "mean_gamma12",
    "var_kout", "var_kself", "bc_kout", "bc_kself",
    "n_qualifying_gens",
]


def compute_seed_summary(seed: int, snapshots_by_gen) -> dict:
    """snapshots_by_gen: {generation: {"k_out": [...], "k_self": [...],
    "payoff": [...], "opp_k_self": [...], "opp_k_out": [...]}}, gen>=
    STEADY_STATE_MIN_GEN only (caller filters)."""
    ctx_names = ["beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout"]
    ctx_vals = {n: [] for n in ctx_names}
    beta1s, beta2s, g11s, g22s, g12s = [], [], [], [], []
    var_kouts, var_kselfs, bc_kouts, bc_kselfs = [], [], [], []

    for gen, g in snapshots_by_gen.items():
        if (len(set(g["k_self"])) < 2 or len(set(g["k_out"])) < 2 or len(set(g["payoff"])) < 2
                or len(set(g["opp_k_self"])) < 2 or len(set(g["opp_k_out"])) < 2):
            continue
        ctx = context_lande_arnold(g["k_self"], g["k_out"], g["opp_k_self"], g["opp_k_out"], g["payoff"])
        for n, v in zip(ctx_names, ctx):
            ctx_vals[n].append(v)
        b1, b2, g11, g22, g12 = lande_arnold_coeffs(g["k_self"], g["k_out"], g["payoff"])
        beta1s.append(b1)
        beta2s.append(b2)
        g11s.append(g11)
        g22s.append(g22)
        g12s.append(g12)
        var_kouts.append(np.var(g["k_out"]))
        var_kselfs.append(np.var(g["k_self"]))
        bc_kouts.append(bimodality_coefficient(g["k_out"]))
        bc_kselfs.append(bimodality_coefficient(g["k_self"]))

    def med(v):
        return float(np.median(v)) if v else float("nan")

    def mean(v):
        return float(np.mean(v)) if v else float("nan")

    return dict(
        seed=seed,
        beta_self_kself=med(ctx_vals["beta_self_kself"]), beta_self_kout=med(ctx_vals["beta_self_kout"]),
        beta_ctx_kself=med(ctx_vals["beta_ctx_kself"]), beta_ctx_kout=med(ctx_vals["beta_ctx_kout"]),
        beta1=med(beta1s), beta2=med(beta2s), gamma11=med(g11s), gamma22=med(g22s), gamma12=med(g12s),
        mean_beta1=mean(beta1s), mean_beta2=mean(beta2s),
        mean_gamma11=mean(g11s), mean_gamma22=mean(g22s), mean_gamma12=mean(g12s),
        var_kout=med(var_kouts), var_kself=med(var_kselfs), bc_kout=med(bc_kouts), bc_kself=med(bc_kselfs),
        n_qualifying_gens=len(beta1s),
    )


def write_summary_csv(path: Path, summary: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_FIELDS)
        row = []
        for field in SUMMARY_FIELDS:
            v = summary[field]
            if field in ("seed", "n_qualifying_gens"):
                row.append(v)
            elif v != v:  # NaN
                row.append("")
            else:
                row.append(f"{v:.8f}")
        w.writerow(row)


def parse_gen_range(spec: Optional[str]) -> Optional[set]:
    if not spec:
        return None
    lo, hi = spec.split("-")
    return set(range(int(lo), int(hi) + 1))


def run_one_seed_chunked(
    seed: int, generations: int, chunk_size: int, out_root_str: str,
    movement: str, p_move: Optional[float], model_params: dict,
    record_phases_seed: Optional[int], record_phases_gens: Optional[set],
    consider_move_prob: Optional[float] = None,
):
    out_root = Path(out_root_str)
    ckpt_path = checkpoint_path_for(out_root, seed)

    record_this_seed = record_phases_seed is not None and record_phases_seed == seed

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
        snapshots_by_gen = ckpt["snapshots_by_gen"]
        phase_frames = ckpt.get("phase_frames", {})
    else:
        next_gen = 0
        genomes = None
        cells = None
        rng_state = None
        rng_spatial_state = None
        parent_of = None
        agg_rows = []
        individual_rows = []
        snapshots_by_gen = {}
        phase_frames = {}

    t0 = time.time()
    n_chunks_this_call = 0
    gens_at_call_start = next_gen
    while next_gen < generations:
        chunk_end = min(next_gen + chunk_size, generations)
        chunk_record_gens = None
        if record_this_seed and record_phases_gens:
            chunk_record_gens = {g for g in record_phases_gens if next_gen <= g < chunk_end}
        phase_recorder_out = {} if chunk_record_gens else None

        chunk_stats, genomes, cells, rng_state, rng_spatial_state, parent_of = run_coupled_evolution_scent(
            seed=seed, generations=chunk_end, start_generation=next_gen,
            movement=movement, p_move=p_move, consider_move_prob=consider_move_prob,
            initial_genomes=genomes, initial_cells=cells,
            rng_state=rng_state, rng_spatial_state=rng_spatial_state,
            parent_of_state=parent_of, track_parent_state=True,
            return_state=True,
            record_phases_for_generations=chunk_record_gens if chunk_record_gens else None,
            phase_recorder_out=phase_recorder_out,
            **model_params,
        )
        if phase_recorder_out:
            for gen, frames in phase_recorder_out.items():
                k_out_vals = next(s.k_out for s in chunk_stats if s.generation == gen)
                k_self_vals = next(s.k_self for s in chunk_stats if s.generation == gen)
                phase_frames[gen] = {"frames": frames, "k_out": k_out_vals, "k_self": k_self_vals}

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
                if s.generation >= STEADY_STATE_MIN_GEN:
                    snapshots_by_gen[s.generation] = {
                        "k_out": list(s.k_out), "k_self": list(s.k_self), "payoff": list(s.payoffs),
                        "opp_k_self": list(s.opponent_mean_k_self), "opp_k_out": list(s.opponent_mean_k_out),
                    }
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
                    "snapshots_by_gen": snapshots_by_gen,
                    "phase_frames": phase_frames,
                },
                f,
            )

    elapsed = time.time() - t0
    gens_this_call = next_gen - gens_at_call_start
    summary = compute_seed_summary(seed, snapshots_by_gen)
    return seed, agg_rows, individual_rows, summary, phase_frames, elapsed, n_chunks_this_call, gens_this_call


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
                return "" if x != x else f"{x:.6f}"
            w.writerow([
                gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}",
                f"{md:.4f}", f"{mp:.6f}", fmt(coop), fmt(kout_a), fmt(kself_a), fmt(spf),
                f"{move_rate:.6f}", f"{iso_frac:.6f}", f"{mean_opp:.6f}", max_opp, fmt(scent_index),
            ])


def write_individual_csv_gz(path: Path, rows: List[IndividualRow]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation", "agent_index", "k_out", "k_self", "memory_depth", "payoff",
            "opponent_mean_k_self", "opponent_mean_k_out", "pos_x", "pos_y",
        ])
        for gen, idx, k_out, k_self, depth, payoff, opp_kself, opp_kout, px, py in rows:
            w.writerow([gen, idx, f"{k_out:.6f}", f"{k_self:.6f}", depth, f"{payoff:.6f}",
                        f"{opp_kself:.6f}", f"{opp_kout:.6f}", px, py])


def write_phase_recording_gz(path: Path, phase_frames: dict) -> None:
    """Pickled+gzipped (not CSV): phase_frames is {generation: {"frames":
    [(phase_idx, positions, moved), ...], "k_out": [...], "k_self":
    [...]}}. Local-only, same as the raw individual log -- this is
    presentation material, not analysis input."""
    with gzip.open(path, "wb") as f:
        pickle.dump(phase_frames, f)


def write_run_metadata(out_root: Path, movement: str, p_move: Optional[float], model_params: dict,
                        consider_move_prob: Optional[float] = None) -> None:
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
        "consider_move_prob": consider_move_prob,
        "model_params": model_params,
        "note": "raw individual_log.csv.gz and phase_recording.pkl.gz are LOCAL ONLY, not pushed "
                "(see .gitignore) -- only timeseries.csv and summary.csv are committed",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "run_metadata.json", "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def status_path_for(out_root: Path) -> Path:
    return Path("progress") / f"scent_spatial_n1000_{out_root.parent.name}_{out_root.name}_status.json"


def write_status_json(path: Path, out_root: Path, seeds: List[int], generations: int) -> None:
    done = sum(
        1 for s in seeds
        if agg_path_for(out_root, s).exists() and summary_path_for(out_root, s).exists()
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
    paths: List[Path], message: str, branch: str = "checkpoint-scratch-scent-spatial-n1000"
) -> None:
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
    p.add_argument("--p-move", type=float, default=None)
    p.add_argument("--n-seeds", type=int, default=len(SEEDS))
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--grid-width", type=int, default=PARAMS["grid_width"])
    p.add_argument("--grid-height", type=int, default=PARAMS["grid_height"])
    p.add_argument("--radius", type=int, default=PARAMS["radius"])
    p.add_argument("--n-phases", type=int, default=PARAMS["n_phases"])
    p.add_argument("--tau", type=float, default=PARAMS["tau"])
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--checkpoint-poll-seconds", type=int, default=600)
    p.add_argument("--push-checkpoints", action="store_true")
    p.add_argument("--status-interval-seconds", type=int, default=3600)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--pilot", action="store_true", help="3 seeds, 1000 gens, chunk-size 250, no push")
    p.add_argument("--out-suffix", type=str, default="")
    p.add_argument("--record-phases-seed", type=int, default=None,
                   help="if set, record every phase (not just per-generation) for this one seed")
    p.add_argument("--record-phases-gens", type=str, default=None,
                   help="generation range 'lo-hi' (inclusive) to phase-record, e.g. '0-19'; "
                        "required if --record-phases-seed is set")
    p.add_argument("--consider-move-prob", type=float, default=None,
                   help="q: 'rarely moves' gate -- each phase, each agent only even considers "
                        "moving with probability q (M1-low/M2-low); omit for the ordinary "
                        "every-phase-considers M0/M1/M2 conditions")
    return p.parse_args()


def main():
    args = parse_args()
    if args.movement == "random" and args.p_move is None:
        print("ERROR: --movement random requires --p-move <float>", file=sys.stderr)
        sys.exit(1)
    if args.record_phases_seed is not None and not args.record_phases_gens:
        print("ERROR: --record-phases-seed requires --record-phases-gens", file=sys.stderr)
        sys.exit(1)
    record_phases_gens = parse_gen_range(args.record_phases_gens)

    model_params = dict(PARAMS)
    model_params["grid_width"] = args.grid_width
    model_params["grid_height"] = args.grid_height
    model_params["radius"] = args.radius
    model_params["n_phases"] = args.n_phases
    model_params["tau"] = args.tau

    if args.pilot:
        seeds = list(range(min(3, args.n_seeds)) if args.n_seeds != len(SEEDS) else range(3))
        generations = args.generations if args.generations != GENERATIONS else 1000
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
        write_run_metadata(out_root, args.movement, args.p_move, model_params, args.consider_move_prob)
        if do_push:
            commit_and_push([out_root / "run_metadata.json"], f"Scent-spatial N=1000 sweep ({args.movement}, "
                                                                f"{machine_tag()}): run metadata")

        pending = [
            s for s in seeds
            if not (agg_path_for(out_root, s).exists() and summary_path_for(out_root, s).exists())
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
                    args.record_phases_seed, record_phases_gens,
                    args.consider_move_prob,
                ): seed
                for seed in pending
            }
            while futures:
                done, not_done = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
                for future in done:
                    seed, agg_rows, individual_rows, summary, phase_frames, elapsed, n_chunks, gens_this_call = future.result()
                    agg_path = agg_path_for(out_root, seed)
                    gz_path = individual_gz_path_for(out_root, seed)
                    summary_path = summary_path_for(out_root, seed)
                    write_agg_csv(agg_path, agg_rows)
                    write_individual_csv_gz(gz_path, individual_rows)
                    write_summary_csv(summary_path, summary)
                    gz_size_mb = gz_path.stat().st_size / (1024 * 1024)
                    if phase_frames:
                        pr_path = phase_recording_path_for(out_root, seed)
                        write_phase_recording_gz(pr_path, phase_frames)
                        print(f"seed={seed}: phase recording saved to {pr_path} "
                              f"({pr_path.stat().st_size / (1024 * 1024):.2f} MB, local only)", flush=True)
                    checkpoint_path_for(out_root, seed).unlink(missing_ok=True)

                    completed_count += 1
                    batch_since_checkpoint.append(seed)
                    paths_since_checkpoint.extend([agg_path, summary_path])  # raw gz is NEVER added here
                    per_gen_ms = elapsed / gens_this_call * 1000 if gens_this_call else 0.0
                    print(f"seed={seed} done ({completed_count}/{len(seeds)}): "
                          f"{elapsed:.1f}s this call ({n_chunks} chunk(s)), {per_gen_ms:.1f} ms/gen, "
                          f"raw individual log = {gz_size_mb:.2f} MB (local only, not pushed)", flush=True)

                    if do_push:
                        threshold = (completed_count // args.checkpoint_every) * args.checkpoint_every
                        if threshold > last_checkpoint:
                            message = (
                                f"Scent-spatial N=1000 sweep ({args.movement}, {machine_tag()}, chunked): "
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
                            f"Scent-spatial N=1000 sweep ({args.movement}, {machine_tag()}): periodic checkpoint "
                            f"snapshot ({len(ckpt_files)} in-progress seed(s))",
                        )
                    last_checkpoint_poll = time.time()

                if status_interval > 0 and (time.time() - last_status_push) >= status_interval:
                    write_status_json(status_path, out_root, seeds, generations)
                    commit_and_push([status_path], f"Progress snapshot N=1000 ({args.movement}, {machine_tag()}): "
                                                    f"{completed_count}/{len(seeds)} seeds done")
                    last_status_push = time.time()

        if do_push and batch_since_checkpoint:
            message = (
                f"Scent-spatial N=1000 sweep ({args.movement}, {machine_tag()}, chunked): "
                f"{completed_count}/{len(seeds)} seeds complete (final partial batch: {sorted(batch_since_checkpoint)})"
            )
            commit_and_push(paths_since_checkpoint, message)

        if status_interval > 0:
            write_status_json(status_path, out_root, seeds, generations)
            commit_and_push([status_path], f"Progress snapshot N=1000 ({args.movement}, {machine_tag()}): "
                                            f"{completed_count}/{len(seeds)} seeds done (final)")

        print(f"\nAll {completed_count}/{len(seeds)} seeds complete in {time.time() - t_start:.1f}s wall-clock.",
              flush=True)
    finally:
        allow_sleep_windows()
        release_lock(lock_path)


if __name__ == "__main__":
    main()
