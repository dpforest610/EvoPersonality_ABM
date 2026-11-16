"""Regime-switch count for the Markov-structure-evolution pilot
(docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 1.4) -- answers the
pilot's calibration question: "with noise, how often/at what timescale
do defection<->cooperation transitions happen within 2000 generations?"
(needed to decide whether production should keep 5000 generations or
be lengthened).

Usage:
    python3 scripts/analyze_markov_sweep_pilot_transitions.py
"""

import csv
from pathlib import Path

COOP_THRESHOLD = 2.0  # midpoint of defect-payoff(1.0) and coop-payoff(3.0), matches run_seed_sweep.py


def load(path: Path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return [(int(row["generation"]), float(row["mean_payoff"])) for row in r]


def regime_switches(rows):
    """Count how many times the regime (coop vs defect, by COOP_THRESHOLD)
    flips across consecutive generations -- captures back-and-forth
    punctuated-equilibrium-style switching, not just a single final
    transition."""
    regimes = ["coop" if p >= COOP_THRESHOLD else "defect" for _, p in rows]
    switches = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            switches.append(rows[i][0])
    return regimes[-1], switches


def main():
    for cond, out_dir in [
        ("noise_rate=0.01 (primary)", "scripts/output/markov_sweep_pilot_noise01"),
        ("noise_rate=0.0 (sensitivity)", "scripts/output/markov_sweep_pilot_noise00"),
    ]:
        print(f"\n=== {cond} ===")
        out_dir_p = Path(out_dir)
        n_with_switch = 0
        total_switches = 0
        for seed in range(10):
            rows = load(out_dir_p / f"seed_{seed}_generation_stats.csv")
            final_regime, switches = regime_switches(rows)
            if switches:
                n_with_switch += 1
                total_switches += len(switches)
            tail = "..." if len(switches) > 8 else ""
            print(
                f"  seed={seed}: final_regime={final_regime:7s} n_switches={len(switches):3d} "
                f"switch_gens={switches[:8]}{tail}"
            )
        print(
            f"  -> {n_with_switch}/10 seeds had >=1 regime switch within 2000 generations; "
            f"total switches={total_switches}"
        )


if __name__ == "__main__":
    main()
