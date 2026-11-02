"""Exploratory, outside preregistration: single-timepoint scatter
snapshots of the raw population in the K_self x K_out plane, for one
seed at 2-3 different generations, shown side by side.

Complements scripts/plot_k_model_density_heatmap.py's pooled-across-time
heatmap: pooling many generations together, even when done honestly, can
visually smooth structure into what looks like a single continuous
cloud. This script makes the same caveat claim ("中央値で無理やり分けた
型であり、集団は連続した一つの雲") from the other direction -- plotting
every individual of ONE seed at ONE generation (population_size=100, so
each panel shows the WHOLE population at that moment, not a sample) --
so a reader can see directly that even a single snapshot, not an
average over time, looks like one cloud rather than four separate
clusters near the median split.

Reads raw per-age census data (Mac-local only, gitignored in production)
-- THIS SCRIPT MUST RUN ON THE MAC where it lives. Writes only a PNG
figure, which IS meant to be committed and pushed.

Uses Hiragino (checked against fonts actually installed on this machine)
when available, falling back to IPAGothic with a printed warning
otherwise -- same pattern as the project's other conference figures.

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 14):
    venv/bin/python scripts/plot_k_model_population_snapshot.py \\
        --data-dir scripts/output/coupling_markov_production --seed 200 \\
        --ages 1000 3000 5000 \\
        --out-path scripts/output/figures/k_model_population_snapshot.png
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

_thresh_spec = importlib.util.spec_from_file_location("compute_k_model_type_thresholds", SCRIPTS_DIR / "compute_k_model_type_thresholds.py")
thresh_mod = importlib.util.module_from_spec(_thresh_spec)
sys.modules["compute_k_model_type_thresholds"] = thresh_mod
_thresh_spec.loader.exec_module(thresh_mod)

CAVEAT_TEXT = "中央値で無理やり分けた型であり、集団は連続した一つの雲"

HIRAGINO_CANDIDATES = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Kaku Gothic Pro"]


def select_japanese_font(available_family_names: set) -> Tuple[str, bool]:
    for candidate in HIRAGINO_CANDIDATES:
        if candidate in available_family_names:
            return candidate, True
    return "IPAGothic", False


_FONT_FAMILY, _IS_HIRAGINO = select_japanese_font({f.name for f in fm.fontManager.ttflist})
matplotlib.rcParams["font.family"] = _FONT_FAMILY
if not _IS_HIRAGINO:
    print(
        f"WARNING: no Hiragino font found on this machine -- using {_FONT_FAMILY} instead. "
        "Re-run this script on a Mac (which has Hiragino installed) to regenerate this figure "
        "with the requested font.",
        file=sys.stderr,
    )


def load_snapshot(data_dir: Path, seed: int, age: int) -> List[Tuple[float, float]]:
    """[(k_out, k_self), ...] for every individual of one seed at one
    age -- the whole population at that moment (no pooling across seeds
    or generations), repeated `frequency` times each per genotype row."""
    cpath = thresh_mod.census_path(data_dir, seed, age)
    if not cpath.exists():
        raise SystemExit(f"no census file at {cpath}")
    return thresh_mod.load_k_values(cpath)


def plot_snapshots(snapshots: List[Tuple[int, List[Tuple[float, float]]]], seed: int, out_path: Path) -> None:
    n_panels = len(snapshots)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (age, values) in zip(axes, snapshots):
        k_out = [v[0] for v in values]
        k_self = [v[1] for v in values]
        ax.scatter(k_self, k_out, s=30, alpha=0.75, color="#3b528b", edgecolors="white", linewidths=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("K_self")
        ax.set_title(f"世代 {age}(n={len(values)})")
        ax.set_aspect("equal")

    axes[0].set_ylabel("K_out")

    fig.suptitle(f"Kモデル 個体群の一時点のスナップショット(seed {seed})", y=0.99)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.text(0.5, 0.02, CAVEAT_TEXT, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ages", type=int, nargs="+", required=True, help="2-3 generations to show side by side, e.g. --ages 1000 3000 5000")
    p.add_argument("--out-path", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    snapshots = [(age, load_snapshot(data_dir, args.seed, age)) for age in args.ages]
    for age, values in snapshots:
        print(f"seed={args.seed} age={age} n={len(values)}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_snapshots(snapshots, args.seed, out_path)


if __name__ == "__main__":
    main()
