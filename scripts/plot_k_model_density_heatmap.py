"""Exploratory, outside preregistration: a 2D density heatmap of the
evolved population over the K_self x K_out plane ([0,1] x [0,1], the
genome's own clipped range -- evopersonality_abm.coupling.mutate_genome),
with the fixed median-split threshold lines overlaid (the same
threshold-computation convention as
scripts/compute_k_model_type_thresholds.py: pooled across all given
seeds and every age >= --min-generation).

Purpose: a visual check of the claim this project's own figures already
carry as a caveat ("中央値で無理やり分けた型であり、集団は連続した一つの
雲") -- whether the population actually looks like one continuous cloud
that the median lines cut through, or shows visible structure (e.g.
separated clusters) near the 4-way split.

Reads raw per-age census data (Mac-local only, gitignored in production)
-- THIS SCRIPT MUST RUN ON THE MAC where it lives. Writes only a PNG
figure, which IS meant to be committed and pushed.

Uses Hiragino (checked against fonts actually installed on this machine)
when available, falling back to IPAGothic with a printed warning
otherwise -- same pattern as the project's other conference figures
(scripts/plot_k_model_recovery_figures.py).

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 14):
    venv/bin/python scripts/plot_k_model_density_heatmap.py \\
        --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \\
        --min-generation 1000 --bins 60 \\
        --out-path scripts/output/figures/k_model_density_heatmap.png
"""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

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


def load_pooled_k_values(data_dir: Path, seeds: List[int], min_generation: int, max_generation: int) -> List[Tuple[float, float]]:
    """[(k_out, k_self), ...], repeated `frequency` times each -- same
    pooling convention (seeds x ages >= min_generation) as
    compute_k_model_type_thresholds.py's own fixed-threshold procedure."""
    all_ages = thresh_mod.discover_ages(data_dir, seeds)
    pooled_ages = [a for a in all_ages if min_generation <= a <= max_generation]
    out: List[Tuple[float, float]] = []
    for age in pooled_ages:
        for seed in seeds:
            cpath = thresh_mod.census_path(data_dir, seed, age)
            if not cpath.exists():
                continue
            out.extend(thresh_mod.load_k_values(cpath))
    return out


def plot_density_heatmap(values: List[Tuple[float, float]], bins: int, out_path: Path) -> None:
    k_out = np.array([v[0] for v in values])
    k_self = np.array([v[1] for v in values])
    threshold_kself = statistics.median(k_self.tolist())
    threshold_kout = statistics.median(k_out.tolist())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    h = ax.hist2d(k_self, k_out, bins=bins, range=[[0, 1], [0, 1]], cmap="viridis")
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("個体数")

    ax.axvline(threshold_kself, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axhline(threshold_kout, color="white", linestyle="--", linewidth=1.2, alpha=0.9)

    label_kwargs = dict(fontsize=9, color="white", ha="center", va="center")
    ax.text(threshold_kself / 2, threshold_kout / 2, "K-恐れ型\n(低K_self・低K_out)", **label_kwargs)
    ax.text(threshold_kself / 2, (threshold_kout + 1) / 2, "K-とらわれ型\n(低K_self・高K_out)", **label_kwargs)
    ax.text((threshold_kself + 1) / 2, threshold_kout / 2, "K-拒絶型\n(高K_self・低K_out)", **label_kwargs)
    ax.text((threshold_kself + 1) / 2, (threshold_kout + 1) / 2, "K-安定型\n(高K_self・高K_out)", **label_kwargs)

    ax.set_xlabel(f"K_self(中央値 {threshold_kself:.3f}で点線)")
    ax.set_ylabel(f"K_out(中央値 {threshold_kout:.3f}で点線)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Kモデル個体密度:K_self × K_out平面(n={len(values)})")

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(0.5, 0.012, CAVEAT_TEXT, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--min-generation", type=int, default=1000)
    p.add_argument("--max-generation", type=int, default=10**9, help="large default = no upper bound (pool every age >= --min-generation)")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--out-path", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]

    values = load_pooled_k_values(data_dir, seeds, args.min_generation, args.max_generation)
    if not values:
        raise SystemExit(f"no census data found under {data_dir} for seeds={seeds}, age in [{args.min_generation}, {args.max_generation}]")
    print(f"pooled n = {len(values)} (seeds={seeds}, min_generation={args.min_generation}, max_generation={args.max_generation})")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_density_heatmap(values, args.bins, out_path)


if __name__ == "__main__":
    main()
