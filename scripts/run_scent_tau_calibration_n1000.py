"""N=1000 tau calibration pilot (docs/PREREGISTRATION_scent_model_N1000.md
section 2): M2 only, population=1000, 41x41 grid, 3 candidate
temperatures {0.1, 0.033333, 0.01} x 3 seeds x 300 generations each
(narrowed from the N=100 chapter's 5-candidate sweep, since tau0=1.0
and tau0/3 were already clearly below the 0.5 threshold there).

Applies the SAME selection rule as the N=100 chapter's revision 1:
largest tau whose seed-averaged mean scent_index_mean (gen>=50) is
>= 0.5. Writes scripts/output/scent_tau_calibration_n1000/decision.json
(machine-readable, consumed by scripts/run_n1000_pipeline_mac.sh) and
exits 0 if a tau qualifies, 1 if none do (the pre-registered "report
and stop" case).

Usage:
    python3 scripts/run_scent_tau_calibration_n1000.py
"""

import csv
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.scent_spatial import run_coupled_evolution_scent

OUT_DIR = Path("scripts/output/scent_tau_calibration_n1000")
TAUS = [0.1, 0.033333, 0.01]
SEEDS = list(range(3))
GENERATIONS = 300
BURN_IN_MIN_GEN = 50

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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"N=1000 calibration pilot: taus={TAUS}, seeds={SEEDS}, generations={GENERATIONS}", flush=True)

    results = {}
    jobs = [(tau, seed) for tau in TAUS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, tau, seed): (tau, seed) for tau, seed in jobs}
        for future in as_completed(futures):
            tau, seed, rows, elapsed = future.result()
            results[(tau, seed)] = rows
            write_timeseries_csv(OUT_DIR / f"tau{tau:.6f}_seed{seed}.csv", rows)
            print(f"tau={tau:.6f} seed={seed} done ({elapsed:.1f}s)", flush=True)

    print("\n" + "=" * 100)
    print(f"Summary (gen>={BURN_IN_MIN_GEN}, seed mean over {len(SEEDS)} seeds)")
    print("=" * 100)
    print(f"\n{'tau':>12}  {'scent_index':>12}  {'move_rate':>10}  {'P(stay)':>9}")

    summary = {}
    move_rate_by_tau = {}
    for tau in TAUS:
        idx_means, move_means = [], []
        for seed in SEEDS:
            rows = results[(tau, seed)]
            sel = [r for r in rows if r[0] >= BURN_IN_MIN_GEN]
            idx_vals = [r[1] for r in sel if r[1] == r[1]]
            idx_means.append(statistics.fmean(idx_vals) if idx_vals else float("nan"))
            move_means.append(statistics.fmean(r[2] for r in sel))
        mean_idx = statistics.fmean(idx_means)
        mean_move = statistics.fmean(move_means)
        summary[tau] = mean_idx
        move_rate_by_tau[tau] = mean_move
        print(f"{tau:>12.6f}  {mean_idx:>12.6f}  {mean_move:>10.6f}  {1 - mean_move:>9.6f}")

    qualifying = [tau for tau in TAUS if summary[tau] >= 0.5]
    chosen = max(qualifying) if qualifying else None

    decision = {
        "taus_tested": TAUS,
        "seeds": len(SEEDS),
        "generations": GENERATIONS,
        "scent_index_by_tau": summary,
        "move_rate_by_tau": move_rate_by_tau,
        "qualifying_taus": sorted(qualifying, reverse=True),
        "selected_tau": chosen,
    }
    with open(OUT_DIR / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 100)
    if chosen is None:
        print("NONE of the candidate taus reached a mean scent-following index >= 0.5.")
        print("Per PREREGISTRATION_scent_model_N1000.md: report and stop here.")
        sys.exit(1)
    else:
        print(f"Qualifying taus (index >= 0.5): {sorted(qualifying, reverse=True)}")
        print(f"Selected tau = {chosen:.6f} (index = {summary[chosen]:.6f})")
        sys.exit(0)


if __name__ == "__main__":
    main()
