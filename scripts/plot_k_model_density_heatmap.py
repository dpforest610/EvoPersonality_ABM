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

Individuals sitting exactly at K_self==0/1 or K_out==0/1 are an artifact
of mutate_genome's clip to [0,1] (a pile-up at the boundary, not part of
the continuous interior cloud). These are excluded from the 2D histogram
(and from its color scale and the two marginal histograms); the excluded
fraction is reported as a caption line instead. The median split lines
are still computed from the FULL (unfiltered) pooled data, matching the
actual thresholds used throughout the rest of this project's analysis
(scripts/compute_k_model_type_thresholds.py) -- only the *display* of
the density excludes the boundary pile-up, not the type-classification
convention. The densest interior bin ("mode") is marked and annotated
separately from the median, so a slide can report either number
correctly.

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


def split_interior_and_boundary(values: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    """Separate individuals pinned exactly at a mutation-clip boundary
    (k_out or k_self == 0.0 or 1.0) from the continuous interior cloud.
    Returns (interior_values, boundary_pct)."""
    interior = [(ko, ks) for ko, ks in values if 0.0 < ko < 1.0 and 0.0 < ks < 1.0]
    n_total = len(values)
    n_boundary = n_total - len(interior)
    boundary_pct = 100.0 * n_boundary / n_total if n_total else 0.0
    return interior, boundary_pct


def find_mode_bin(counts: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> Tuple[float, float]:
    """Center of the most-populated bin of a 2D histogram (counts indexed
    [x_bin, y_bin], matching numpy/matplotlib's own hist2d convention)."""
    ix, iy = np.unravel_index(np.argmax(counts), counts.shape)
    mode_x = (xedges[ix] + xedges[ix + 1]) / 2
    mode_y = (yedges[iy] + yedges[iy + 1]) / 2
    return mode_x, mode_y


def plot_density_heatmap(values: List[Tuple[float, float]], bins: int, out_path: Path) -> None:
    k_out_all = np.array([v[0] for v in values])
    k_self_all = np.array([v[1] for v in values])
    # Median split thresholds: computed from the FULL pooled data (same
    # convention as compute_k_model_type_thresholds.py), NOT the
    # boundary-excluded interior subset below -- these are the actual
    # type-classification thresholds used elsewhere in this project.
    threshold_kself = statistics.median(k_self_all.tolist())
    threshold_kout = statistics.median(k_out_all.tolist())

    interior, boundary_pct = split_interior_and_boundary(values)
    k_out = np.array([v[0] for v in interior])
    k_self = np.array([v[1] for v in interior])

    fig = plt.figure(figsize=(8, 7.6))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[4, 1, 0.18], height_ratios=[1, 4],
        hspace=0.06, wspace=0.08,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    counts, xedges, yedges, image = ax_main.hist2d(k_self, k_out, bins=bins, range=[[0, 1], [0, 1]], cmap="viridis")
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("個体数(端に張り付いた個体を除く)")

    mode_kself, mode_kout = find_mode_bin(counts, xedges, yedges)

    ax_main.axvline(threshold_kself, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    ax_main.axhline(threshold_kout, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    ax_main.plot(mode_kself, mode_kout, marker="*", markersize=16, markerfacecolor="red", markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=5)

    label_kwargs = dict(fontsize=9, color="white", ha="center", va="center")
    ax_main.text(threshold_kself / 2, threshold_kout / 2, "K-恐れ型\n(低K_self・低K_out)", **label_kwargs)
    ax_main.text(threshold_kself / 2, (threshold_kout + 1) / 2, "K-とらわれ型\n(低K_self・高K_out)", **label_kwargs)
    ax_main.text((threshold_kself + 1) / 2, threshold_kout / 2, "K-拒絶型\n(高K_self・低K_out)", **label_kwargs)
    ax_main.text((threshold_kself + 1) / 2, (threshold_kout + 1) / 2, "K-安定型\n(高K_self・高K_out)", **label_kwargs)

    ax_main.set_xlabel(f"K_self(中央値 {threshold_kself:.3f}で点線)")
    ax_main.set_ylabel(f"K_out(中央値 {threshold_kout:.3f}で点線)")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)

    mode_label = f"最頻値(内側のマスで最も密度が高い場所): K_self={mode_kself:.3f}, K_out={mode_kout:.3f}(赤星)"
    ax_main.text(0.02, 0.02, mode_label, transform=ax_main.transAxes, fontsize=7.5, color="white", ha="left", va="bottom")

    # Marginal histograms -- same interior-only (boundary-excluded) data
    # as the main heatmap, so the "one mountain" claim is about the same
    # continuous cloud throughout the whole figure.
    hist_bins = np.linspace(0, 1, bins + 1)
    ax_histx.hist(k_self, bins=hist_bins, color="#3b528b")
    ax_histx.set_xlim(0, 1)
    ax_histx.axvline(threshold_kself, color="gray", linestyle="--", linewidth=1.0)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.set_ylabel("度数")

    ax_histy.hist(k_out, bins=hist_bins, orientation="horizontal", color="#3b528b")
    ax_histy.set_ylim(0, 1)
    ax_histy.axhline(threshold_kout, color="gray", linestyle="--", linewidth=1.0)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xlabel("度数")

    fig.suptitle(f"Kモデル個体密度:K_self × K_out平面(n={len(values)}、内側n={len(interior)})", y=0.98)

    fig.subplots_adjust(left=0.1, right=0.92, top=0.93, bottom=0.11)
    boundary_label = f"端に張り付いた個体{boundary_pct:.1f}%(変異を0〜1に切り詰めたため)"
    fig.text(0.5, 0.03, boundary_label, ha="center", fontsize=8.5, color="black")
    fig.text(0.5, 0.012, CAVEAT_TEXT, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}", flush=True)
    print(
        f"boundary_pct={boundary_pct:.3f}% mode_kself={mode_kself:.4f} mode_kout={mode_kout:.4f} "
        f"median_kself={threshold_kself:.4f} median_kout={threshold_kout:.4f}",
        flush=True,
    )


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
