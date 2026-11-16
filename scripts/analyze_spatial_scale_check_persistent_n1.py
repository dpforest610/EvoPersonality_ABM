"""Same analysis as analyze_spatial_scale_check.py, pointed at the
cross-generation persistence experiment's output
(spatial_scale_check_k8_n1_persistent/) -- see
run_spatial_scale_check_persistent_n1.py. Compare its K_out against
experiment A's (spatial_scale_check_k8_n1_static/) to isolate the
"persisting neighborhood" effect from the already-confirmed "small
neighborhood" effect.

Usage:
    python3 scripts/analyze_spatial_scale_check_persistent_n1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.analyze_spatial_scale_check as base

base.SWEEP_DIR = Path("scripts/output/spatial_scale_check_k8_n1_persistent")

if __name__ == "__main__":
    base.main()
