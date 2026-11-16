"""Run the K-coupled population loop across multiple seeds in parallel,
out to many generations, and ask two questions about the (K_out, K_self)
dynamics seen in the first 1000-generation look
(scripts/output/coupling_reference_1000gen/):

  (1) Does the anti-phase K_out/K_self wobble keep going, or does it
      damp out and settle?
  (2) Does a K_out crash (an "avoidant" phase) coincide with a payoff
      crash every time it happens, across the whole run?

Usage:
    python3 scripts/run_coupling_seed_sweep.py --seed-list 42,1,2,3,4 --generations 5000 --workers 5

Writes one timeseries.csv per seed (generation, mean/std of K_out and
K_self, mean depth, mean payoff), a summary.csv with each seed's
K_out<->payoff correlation, and a comparison plot.
"""

import argparse
import csv
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.coupling import run_coupled_evolution

CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_ORANGE = "#eb6834"
MUTED_INK = "#8f8f88"

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


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)


def commit_and_push(paths: Sequence[Path], message: str) -> None:
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


def write_csv(path: Path, rows: List[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_k_out", "std_k_out", "mean_k_self", "std_k_self", "mean_depth", "mean_payoff"])
        for gen, mko, sko, mks, sks, md, mp in rows:
            w.writerow([gen, f"{mko:.6f}", f"{sko:.6f}", f"{mks:.6f}", f"{sks:.6f}", f"{md:.4f}", f"{mp:.6f}"])


def kout_payoff_correlation(rows: List[Row]) -> float:
    """Pearson correlation between mean K_out and mean payoff across the
    whole run -- a direct number for "does a K_out crash coincide with a
    payoff crash": strongly positive means yes, consistently."""
    kout = [r[1] for r in rows]
    payoff = [r[6] for r in rows]
    try:
        return statistics.correlation(kout, payoff)
    except statistics.StatisticsError:
        return float("nan")


def oscillation_amplitude_by_window(rows: List[Row], window: int) -> List[Tuple[int, float, float]]:
    """Std of mean K_out and mean K_self within each non-overlapping
    generation window -- a shrinking value over time means the anti-phase
    wobble is damping out; roughly flat means it is still going."""
    out = []
    for start in range(0, len(rows), window):
        chunk = rows[start : start + window]
        if len(chunk) < 2:
            continue
        kout_std = statistics.pstdev(r[1] for r in chunk)
        kself_std = statistics.pstdev(r[3] for r in chunk)
        out.append((chunk[0][0], kout_std, kself_std))
    return out


def plot_sweep(all_rows: List[Tuple[int, List[Row]]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_rows)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.2 * n), sharex="col")
    if n == 1:
        axes = [axes]

    for row_axes, (seed, rows) in zip(axes, all_rows):
        ax_k, ax_payoff = row_axes
        gens = [r[0] for r in rows]
        ax_k.plot(gens, [r[1] for r in rows], color=CATEGORICAL_BLUE, linewidth=1, label="mean K_out")
        ax_k.plot(gens, [r[3] for r in rows], color=CATEGORICAL_ORANGE, linewidth=1, label="mean K_self")
        ax_k.set_ylim(-0.02, 1.02)
        ax_k.set_ylabel(f"seed={seed}\nmean value")
        ax_k.spines[["top", "right"]].set_visible(False)

        ax_payoff.plot(gens, [r[6] for r in rows], color=CATEGORICAL_BLUE, linewidth=1)
        ax_payoff.axhline(1.0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax_payoff.axhline(3.0, color=MUTED_INK, linewidth=0.6, linestyle="--")
        ax_payoff.set_ylabel("mean payoff")
        ax_payoff.spines[["top", "right"]].set_visible(False)

        corr = kout_payoff_correlation(rows)
        ax_payoff.set_title(f"corr(K_out, payoff) = {corr:.2f}", fontsize=9)

    axes[0][0].legend(fontsize=8, loc="upper right")
    axes[0][0].set_title("mean K_out / K_self", fontsize=9)
    axes[-1][0].set_xlabel("generation")
    axes[-1][1].set_xlabel("generation")

    fig.suptitle("K_out/K_self and payoff over 5000 generations, per seed", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved comparison plot to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-list", type=str, default="42,1,2,3,4")
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
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="scripts/output/coupling_seed_sweep_5000gen")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    all_rows: List[Tuple[int, List[Row]]] = []

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
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed, rows = fut.result()
            csv_path = out_dir / f"seed_{seed}_timeseries.csv"
            write_csv(csv_path, rows)
            all_rows.append((seed, rows))

            corr = kout_payoff_correlation(rows)
            windows = oscillation_amplitude_by_window(rows, window=500)
            first_amp = (windows[0][1] + windows[0][2]) / 2
            last_amp = (windows[-1][1] + windows[-1][2]) / 2
            print(
                f"seed={seed} done: corr(K_out,payoff)={corr:.3f} "
                f"first-500gen wobble amplitude={first_amp:.3f} "
                f"last-500gen wobble amplitude={last_amp:.3f}",
                flush=True,
            )
            commit_and_push(
                [csv_path],
                f"Coupling seed sweep: seed={seed} result "
                f"(corr={corr:.3f}, wobble {first_amp:.3f}->{last_amp:.3f})",
            )

    all_rows.sort(key=lambda t: t[0])
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "corr_kout_payoff", "wobble_amplitude_first500", "wobble_amplitude_last500"])
        for seed, rows in all_rows:
            corr = kout_payoff_correlation(rows)
            windows = oscillation_amplitude_by_window(rows, window=500)
            first_amp = (windows[0][1] + windows[0][2]) / 2
            last_amp = (windows[-1][1] + windows[-1][2]) / 2
            w.writerow([seed, f"{corr:.4f}", f"{first_amp:.4f}", f"{last_amp:.4f}"])
    commit_and_push([summary_path], "Coupling seed sweep: final summary")

    print("\n=== summary ===")
    print(f"{'seed':>4} {'corr(Kout,payoff)':>18} {'wobble first500':>16} {'wobble last500':>16}")
    for seed, rows in all_rows:
        corr = kout_payoff_correlation(rows)
        windows = oscillation_amplitude_by_window(rows, window=500)
        first_amp = (windows[0][1] + windows[0][2]) / 2
        last_amp = (windows[-1][1] + windows[-1][2]) / 2
        print(f"{seed:>4} {corr:>18.3f} {first_amp:>16.3f} {last_amp:>16.3f}")

    if not args.no_plot:
        plot_path = out_dir / "coupling_seed_sweep_comparison.png"
        plot_sweep(all_rows, plot_path)
        commit_and_push([plot_path], "Coupling seed sweep: comparison plot")


if __name__ == "__main__":
    main()
