"""Same analysis as analyze_spatial_scale_check.py, pointed at condition
(2)'s output (spatial_scale_check_k8_condition2/) instead of condition
(3)'s -- see run_spatial_scale_check_condition2.py.

Usage:
    python3 scripts/analyze_spatial_scale_check_condition2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.analyze_spatial_scale_check as base

base.SWEEP_DIR = Path("scripts/output/spatial_scale_check_k8_condition2")

if __name__ == "__main__":
    base.main()
