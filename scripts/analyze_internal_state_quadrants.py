"""Measure the (K_out, K_self) internal-state dynamics directly, to
settle -- from the mechanism itself, not the design doc's desk
assignment (HBES2026_ABM骨子_二成分K.md §3.3) -- which corner of the
four (K_out, K_self) quadrants actually produces which qualitative
signature.

Test signal (held fixed for the whole run, not drawn from any real
duel): the opponent consistently defects (a=0 every round -- "a_j"),
the agent's own Lindgren table consistently recommends cooperation
(r=1 every round -- "r_i"), s(0)=0.5. This isolates the pure pull of
the update rule itself:

    s(t+1) = s(t) + k_self*(r - s(t)) + k_out*(a - s(t)) + noise
           = (1 - k_self - k_out)*s(t) + k_self*r + k_out*a + noise

which (ignoring the [0,1] clip and noise) is a linear AR(1) toward the
fixed point s* = (k_self*r + k_out*a) / (k_self + k_out), approached
geometrically at rate (1 - k_self - k_out) per round. With r=1, a=0
this simplifies to s* = k_self / (k_self + k_out) -- the fixed point is
exactly the *share* of k_self in the total coupling weight, independent
of its absolute size; only the convergence rate and noise sensitivity
depend on the total k_self + k_out.

Four corners, HIGH=0.8 / LOW=0.1. The (HIGH, HIGH) corner sums to 1.6,
violating k_out + k_self <= 1, so it is rescaled proportionally to sum
to exactly 1.0 (0.5, 0.5) -- documented explicitly, not silently
dropped -- keeping the two channels equally weighted, which is the
property "high, high" is meant to test here.

  (k_out, k_self):
    LOW,  LOW   = (0.10, 0.10)  sum=0.20
    HIGH, LOW   = (0.80, 0.10)  sum=0.90
    LOW,  HIGH  = (0.10, 0.80)  sum=0.90
    HIGH, HIGH  = (0.50, 0.50)  sum=1.00  (rescaled from 0.8/0.8)

For each corner: the noiseless trajectory (shows the pure deterministic
pull and fixed point) and one noisy realization at noise_std=0.05 (the
model's own default) over 50 rounds (rounds_per_duel's own default),
to see whether the corner's dynamics are visibly noise-dominated at a
duel's actual timescale, not just in the infinite-time limit.

Outputs:
    scripts/output/internal_state_quadrants/quadrant_dynamics.png
    scripts/output/internal_state_quadrants/quadrant_summary.csv

Usage:
    python3 scripts/analyze_internal_state_quadrants.py
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evopersonality_abm.coupling import update_internal_state

OUT_DIR = Path("scripts/output/internal_state_quadrants")
S0 = 0.5
R = 1  # own table: cooperate
A = 0  # opponent: defect
ROUNDS = 50
NOISE_STD = 0.05
SEED = 0

N_REALIZATIONS = 30  # for the across-seed noise-sensitivity check, beyond the single illustrated path
LOW, HIGH = 0.1, 0.8
CORNERS = [
    ("LOW k_out, LOW k_self", LOW, LOW),
    ("HIGH k_out, LOW k_self", HIGH, LOW),
    ("LOW k_out, HIGH k_self", LOW, HIGH),
    ("HIGH k_out, HIGH k_self (rescaled 0.8/0.8 -> 0.5/0.5, sum<=1)", 0.5, 0.5),
]

COLORS = {
    "LOW k_out, LOW k_self": "#8f8f88",
    "HIGH k_out, LOW k_self": "#2a78d6",
    "LOW k_out, HIGH k_self": "#eb6834",
    "HIGH k_out, HIGH k_self (rescaled 0.8/0.8 -> 0.5/0.5, sum<=1)": "#1baf7a",
}


def simulate(k_out: float, k_self: float, rounds: int, noisy: bool, seed: int = SEED):
    rng = random.Random(seed)
    s = S0
    trace = [s]
    for _ in range(rounds):
        noise = rng.gauss(0, NOISE_STD) if noisy else 0.0
        s = update_internal_state(s, k_self, R, k_out, A, noise)
        trace.append(s)
    return trace


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Test signal: r (own table) = {R} [cooperate], a (opponent's last action) = {A} [defect], s(0)={S0}")
    print(f"Fixed point (noiseless, ignoring clip): s* = k_self / (k_self + k_out)   [since r=1, a=0]")
    print(f"Decay rate per round: (1 - k_out - k_self)\n")

    print(f"{'corner':<55} {'k_out':>6} {'k_self':>7} {'sum':>5} {'s* (analytic)':>14} {'decay/round':>12} {'s(50) noiseless':>16} {'s(50) noisy':>12}")
    det_traces = {}
    noisy_traces = {}
    for label, k_out, k_self in CORNERS:
        det = simulate(k_out, k_self, ROUNDS, noisy=False)
        noisy = simulate(k_out, k_self, ROUNDS, noisy=True)
        det_traces[label] = det
        noisy_traces[label] = noisy

        total = k_out + k_self
        s_star = k_self / total if total > 0 else float("nan")
        decay = 1 - total

        # noisy realization's late-window (last 20 rounds) mean/std, as a rough
        # "settled here, this jittery" summary
        late = noisy[-20:]
        late_mean, late_std = float(np.mean(late)), float(np.std(late))

        print(f"{label:<55} {k_out:>6.2f} {k_self:>7.2f} {total:>5.2f} {s_star:>14.4f} {decay:>12.4f} {det[-1]:>16.4f} {noisy[-1]:>12.4f}")
        rows.append(
            {
                "corner": label, "k_out": k_out, "k_self": k_self, "sum": total,
                "analytic_fixed_point": s_star, "decay_per_round": decay,
                "s_final_noiseless": det[-1], "s_final_noisy": noisy[-1],
                "s_late_window_mean_noisy": late_mean, "s_late_window_std_noisy": late_std,
            }
        )

    csv_path = OUT_DIR / "quadrant_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {csv_path}")

    print("\nLate-window (last 20 of 50 rounds, single noisy realization) mean +- std, as an empirical read on 'settled where, how jittery':")
    for r in rows:
        print(f"  {r['corner']:<55} mean={r['s_late_window_mean_noisy']:.4f}  std={r['s_late_window_std_noisy']:.4f}")

    print(f"\nAcross {N_REALIZATIONS} independent noisy realizations, s(round={ROUNDS}) -- the proper "
          f"'how noise-dominated is this corner' check (one path can mislead):")
    for label, k_out, k_self in CORNERS:
        finals = [simulate(k_out, k_self, ROUNDS, noisy=True, seed=s)[-1] for s in range(N_REALIZATIONS)]
        print(f"  {label:<55} mean={np.mean(finals):.4f}  std={np.std(finals):.4f}  "
              f"min={np.min(finals):.4f}  max={np.max(finals):.4f}")

    # --- plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_det, ax_noisy) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    t = np.arange(ROUNDS + 1)
    for label, _, _ in CORNERS:
        color = COLORS[label]
        ax_det.plot(t, det_traces[label], color=color, linewidth=2.2, label=label)
        ax_noisy.plot(t, noisy_traces[label], color=color, linewidth=1.4, alpha=0.85, label=label)

    for ax, title in [(ax_det, "noiseless (pure pull)"), (ax_noisy, f"single noisy realization (noise_std={NOISE_STD})")]:
        ax.axhline(0.5, color="#8f8f88", linewidth=0.8, linestyle=":")
        ax.axhline(0.0, color="#8f8f88", linewidth=0.6, linestyle=":")
        ax.axhline(1.0, color="#8f8f88", linewidth=0.6, linestyle=":")
        ax.set_xlabel("round")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    ax_det.set_ylabel("internal state s")
    ax_det.legend(fontsize=7.5, loc="center right")
    fig.suptitle(f"Internal-state dynamics at the four (K_out, K_self) corners\n"
                 f"(opponent always defects [a=0], own table always recommends cooperate [r=1], s(0)={S0})", y=1.04)
    fig.tight_layout()
    fig_path = OUT_DIR / "quadrant_dynamics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
