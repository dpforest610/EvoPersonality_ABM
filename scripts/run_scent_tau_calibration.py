"""Scent-model tau calibration pilot (revision 1 of
docs/PREREGISTRATION_scent_model.md): M2 (scent) only, 5 candidate
temperatures (tau0 and tau0/3, /10, /30, /100) x 5 seeds x 300
generations each. Short, one-shot (no chunking/checkpointing needed at
this scale).

Writes a per-(tau,seed) timeseries CSV (same columns as the main
runner, including scent_index_mean) plus a summary CSV across all
(tau,seed) pairs, then applies the pre-registered selection rule:
largest tau whose seed-averaged mean scent_index_mean (averaged over
gen>=50, a short burn-in) is >= 0.5.

Usage:
    python3 scripts/run_scent_tau_calibration.py
"""

import csv
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.scent_spatial import run_coupled_evolution_scent

OUT_DIR = Path("scripts/output/scent_tau_calibration")
TAU0 = 1.0
TAUS = [TAU0, TAU0 / 3, TAU0 / 10, TAU0 / 30, TAU0 / 100]
SEEDS = list(range(5))
GENERATIONS = 300
BURN_IN_MIN_GEN = 50

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
    radius=1,
    movement="scent",
)


def run_one(tau: float, seed: int):
    t0 = time.time()
    stats = run_coupled_evolution_scent(seed=seed, generations=GENERATIONS, tau=tau, **PARAMS)
    elapsed = time.time() - t0
    rows = [
        (
            s.generation, s.scent_index_mean, s.move_rate, s.isolated_fraction,
            s.mean_opponent_count, s.max_opponent_count,
            s.k_out_assortativity, s.k_self_assortativity, s.cooperation_rate,
        )
        for s in stats
    ]
    return tau, seed, rows, elapsed


def write_timeseries_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation", "scent_index_mean", "move_rate", "isolated_fraction",
            "mean_opponent_count", "max_opponent_count",
            "k_out_assortativity", "k_self_assortativity", "cooperation_rate",
        ])
        for (gen, idx, mv, iso, mean_opp, max_opp, kout_a, kself_a, coop) in rows:
            def fmt(x):
                return "" if x != x else f"{x:.6f}"
            w.writerow([gen, fmt(idx), f"{mv:.6f}", f"{iso:.6f}", f"{mean_opp:.6f}", max_opp,
                        fmt(kout_a), fmt(kself_a), fmt(coop)])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Calibration pilot: taus={TAUS}, seeds={SEEDS}, generations={GENERATIONS}", flush=True)

    results = {}  # (tau, seed) -> rows
    jobs = [(tau, seed) for tau in TAUS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_one, tau, seed): (tau, seed) for tau, seed in jobs}
        for future in as_completed(futures):
            tau, seed, rows, elapsed = future.result()
            results[(tau, seed)] = rows
            tau_tag = f"{tau:.6f}"
            write_timeseries_csv(OUT_DIR / f"tau{tau_tag}_seed{seed}.csv", rows)
            print(f"tau={tau:.6f} seed={seed} done ({elapsed:.1f}s)", flush=True)

    print("\n" + "=" * 100)
    print(f"Summary (gen>={BURN_IN_MIN_GEN}, seed mean over {len(SEEDS)} seeds)")
    print("=" * 100)
    print(f"\n{'tau':>12}  {'scent_index':>12}  {'move_rate':>10}  {'P(stay)':>9}  "
          f"{'mean_opp_cnt':>13}  {'k_out_assort':>13}  {'coop_rate':>10}")

    summary = {}
    for tau in TAUS:
        idx_means, move_means, opp_means, assort_means, coop_means = [], [], [], [], []
        for seed in SEEDS:
            rows = results[(tau, seed)]
            sel = [r for r in rows if r[0] >= BURN_IN_MIN_GEN]
            idx_vals = [r[1] for r in sel if r[1] == r[1]]
            idx_means.append(statistics.fmean(idx_vals) if idx_vals else float("nan"))
            move_means.append(statistics.fmean(r[2] for r in sel))
            opp_means.append(statistics.fmean(r[4] for r in sel))
            assort_vals = [r[6] for r in sel if r[6] == r[6]]
            assort_means.append(statistics.fmean(assort_vals) if assort_vals else float("nan"))
            coop_vals = [r[8] for r in sel if r[8] == r[8]]
            coop_means.append(statistics.fmean(coop_vals) if coop_vals else float("nan"))

        mean_idx = statistics.fmean(idx_means)
        mean_move = statistics.fmean(move_means)
        mean_opp = statistics.fmean(opp_means)
        mean_assort = statistics.fmean(assort_means)
        mean_coop = statistics.fmean(coop_means)
        summary[tau] = mean_idx
        print(f"{tau:>12.6f}  {mean_idx:>12.6f}  {mean_move:>10.6f}  {1 - mean_move:>9.6f}  "
              f"{mean_opp:>13.6f}  {mean_assort:>13.6f}  {mean_coop:>10.6f}")

    print("\n" + "=" * 100)
    print("Selection (scent-following index only, >= 0.5, largest qualifying tau):")
    print("=" * 100)
    qualifying = [tau for tau in TAUS if summary[tau] >= 0.5]
    if not qualifying:
        print("NONE of the candidate taus reached a mean scent-following index >= 0.5.")
        print("Per PREREGISTRATION_scent_model.md revision 1: report and stop here.")
    else:
        chosen = max(qualifying)
        print(f"Qualifying taus (index >= 0.5): {sorted(qualifying, reverse=True)}")
        print(f"Selected tau = {chosen:.6f} (index = {summary[chosen]:.6f})")


if __name__ == "__main__":
    main()
