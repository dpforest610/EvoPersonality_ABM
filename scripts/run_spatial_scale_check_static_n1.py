"""第二章 experiment A -- disentangle "small neighborhood" from "chopped
short duels" as the driver behind condition (2)'s K_out signal.

Both candidate confounds changed together between well-mixed (chapter
1, K_out ~ 0) and spatial condition (2) (k=8, n_phases=5, K_out
significantly positive at n=50): the opponent count (99 -> 8) AND the
duel format (one continuous 50-round duel per opponent -> five
independent 10-round duels per opponent, each with the internal state
s reset to 0.5). This run isolates the first: n_phases=1 (mobile=False)
means each occupied neighbor pair gets exactly ONE 50-round duel call
per generation -- same duel format as chapter 1's well-mixed pairs,
with only the opponent count reduced to k=8 (Moore) instead of 99.
Positions are still reassigned fresh every generation (place_on_grid is
called once per generation regardless of n_phases -- confirmed by
reading run_coupled_evolution_spatial), so neighbors are a "temporary
small group" here, same as in the n_phases=5 runs, not a persisting
relationship across generations.

Reading:
  - K_out stays significantly positive here -> the small opponent
    count (8 vs 99) is (at least part of) the driver; duel-chopping is
    not required.
  - K_out reverts to ~0 (chapter 1's well-mixed value) -> duel-chopping
    (short duels with frequent internal-state resets) is the driver,
    not neighborhood size per se.

All other parameters identical to the condition (2) k=8 scale check
(run_spatial_scale_check_condition2.py): population=100, grid 12x12
(~30.6% vacant), tau=1.0, move_radius=1, rounds_per_duel=50, and all
non-spatial mutation/game defaults from chapter 1. Not tuned after
results. 50 seeds x 3000 generations, evaluated gen>=1000, same as the
other scale checks.

Usage:
    python3 scripts/run_spatial_scale_check_static_n1.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_spatial_scale_check as base

base.OUT_DIR = Path("scripts/output/spatial_scale_check_k8_n1_static")
base.PARAMS = dict(base.PARAMS, mobile=False, n_phases=1)

if __name__ == "__main__":
    base.main()
