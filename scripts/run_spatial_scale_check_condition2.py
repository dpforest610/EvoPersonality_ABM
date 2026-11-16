"""第二章 scale-check run -- condition (2), same scale as the condition-3
scale check (k=8, 10 seeds x 3000 generations), for the measurement-
pipeline calibration step (b) in the chat discussion.

Identical parameter set and phase/duel schedule to
run_spatial_scale_check.py (condition 3), with only mobile=False --
per the chapter-2 design, (2) and (3) must be identical except for the
movement step so the comparison isolates movement's effect. Not tuned
after seeing condition (3)'s results.

Two things this run is for:
  - Does K_self still come through here (same pipeline, no movement)?
    If yes, the calibration isn't an artifact of movement itself.
  - Does K_out stay near zero here, unlike condition (3)'s weak
    (non-significant at n=10) positive tilt?

Usage:
    python3 scripts/run_spatial_scale_check_condition2.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.run_spatial_scale_check as base

base.OUT_DIR = Path("scripts/output/spatial_scale_check_k8_condition2")
base.PARAMS = dict(base.PARAMS, mobile=False)

if __name__ == "__main__":
    base.main()
