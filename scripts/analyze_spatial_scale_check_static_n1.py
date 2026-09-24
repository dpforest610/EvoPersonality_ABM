"""Same analysis as analyze_spatial_scale_check.py, pointed at experiment
A's output (spatial_scale_check_k8_n1_static/) -- see
run_spatial_scale_check_static_n1.py.

Usage:
    python3 scripts/analyze_spatial_scale_check_static_n1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.analyze_spatial_scale_check as base

base.SWEEP_DIR = Path("scripts/output/spatial_scale_check_k8_n1_static")

if __name__ == "__main__":
    base.main()
