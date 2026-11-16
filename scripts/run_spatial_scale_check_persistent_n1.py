"""第二章 experiment: does letting a lineage keep facing the same
neighborhood (cross-generation position persistence) add anything on
top of experiment A's "small opponent count" effect?

Directly comparable to run_spatial_scale_check_static_n1.py (experiment
A): identical parameters (k=8/Moore, n_phases=1, mobile=False,
population=100, grid 12x12), the ONLY difference is
run_coupled_evolution_spatial_persistent instead of
run_coupled_evolution_spatial -- i.e., positions persist across
generations via parent-cell inheritance (see spatial.py's docstring for
the exact rule: one heir per surviving lineage keeps its parent's cell;
extra offspring and vacated-lineage cells are pooled and scattered
uniformly at random) instead of a fresh random `place_on_grid` every
generation.

Interpretation caveat (explicit, per the chat discussion): this
mechanism produces "a lineage keeps facing a similar neighborhood"
(environmental continuity from staying in place), NOT "an individual
actively seeks out a specific partner" (attachment-style partner
choice). If K_out rises above experiment A's level here, the honest
reading is "a persisting environment adds on top of small-neighborhood
scarcity," not evidence of partner-directed attachment. Also: because
extra offspring scatter randomly, the population is always a MIX of
staying and scattering lineages, not a uniformly persistent one (see
the ad hoc check in chat: ~40-50% of cells occupied in consecutive
generations show a small enough k_self change to be a likely "heir",
the rest are larger jumps consistent with a scattered newcomer).

Reading:
  - K_out here > experiment A's +0.044 -> persisting neighborhood adds
    on top of the "8 vs 99 opponents" scarcity effect.
  - K_out ~= experiment A's +0.044 -> only "how many" opponents matters,
    not whether the same ones keep recurring.

Comparison baseline: run_spatial_scale_check_static_n1.py (experiment
A) with the SAME n_phases=1 and k=8 -- the only variable that differs
is persistent vs. reshuffled-every-generation positions, so any K_out
difference is attributable purely to "does it persist."

50 seeds x 3000 generations, evaluated gen>=1000, same as the other
scale checks. Not tuned after results.

Usage:
    python3 scripts/run_spatial_scale_check_persistent_n1.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_spatial_scale_check as base
from evopersonality_abm.spatial import run_coupled_evolution_spatial_persistent

base.OUT_DIR = Path("scripts/output/spatial_scale_check_k8_n1_persistent")
base.PARAMS = {k: v for k, v in base.PARAMS.items() if k != "move_radius"}
base.PARAMS = dict(base.PARAMS, mobile=False, n_phases=1)
base.run_coupled_evolution_spatial = run_coupled_evolution_spatial_persistent

if __name__ == "__main__":
    base.main()
