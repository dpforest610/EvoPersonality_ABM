"""Conference-slide figures for the K-model production results (exploratory
presentation aids, not confirmatory analysis -- the confirmatory numbers are
in docs/K_MODEL_MARKOV_RESULTS.md section 1 / k_model_confirmatory_summary.json).

Figure A (H-B, SELF, by generation): recovered rate
(1 - never_recovered_rate, "20回のうちに和和和へ戻れた割合") for SELF, one
line per K-model type (K-安定型/K-拒絶型/K-とらわれ型/K-恐れ型), each with a
shaded seed-level bootstrap 95% band (resample the 70 seeds WITH
REPLACEMENT --n-bootstrap-seed times, recompute the pooled rate each draw
-- same "resample seeds, not individual duels" convention as the rest of
this project's seed bootstraps).

Figure B (H-C, by depth stratum): recovered rate for SELF, window-averaged
over gen 1000-5000, grouped by effective-memory-depth stratum
(depthLow/depthMid/depthHigh), K-安定型 vs K-恐れ型 only (matching the
confirmatory analysis's H-C1 comparison) with the same seed-bootstrap 95%
band, window-averaged the same way as the confirmatory script's point
estimate (per-age rate, then averaged across ages; the bootstrap band is
built from 2000 window-averaged draws, one draw per bootstrap iteration,
not from flattening ages and iterations together -- see
run_k_model_confirmatory_analysis.py's own module docstring for why that
distinction matters).

Both figures carry the "中央値で無理やり分けた型であり、集団は連続した
一つの雲" caveat.

Reads raw per-age census/recovery data (Mac-local only, gitignored) --
THIS SCRIPT MUST RUN ON THE MAC where the production run's data lives.
Writes only PNG figures, which ARE meant to be committed and pushed.

Uses Hiragino (checked against fonts actually installed on this machine)
when available, falling back to IPAGothic with a printed warning
otherwise -- same pattern as scripts/analyze_selfplay_recovery_by_depth.py
on claude/markov-production-analysis.

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 10):
    venv/bin/python scripts/plot_k_model_recovery_figures.py \\
        --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \\
        --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \\
        --window-start 1000 --window-end 5000 --n-bootstrap-seed 2000 \\
        --out-dir scripts/output/figures
"""

import argparse
import importlib.util
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

_viewer_spec = importlib.util.spec_from_file_location("export_k_model_markov_viewer", SCRIPTS_DIR / "export_k_model_markov_viewer.py")
viewer_mod = importlib.util.module_from_spec(_viewer_spec)
sys.modules["export_k_model_markov_viewer"] = viewer_mod
_viewer_spec.loader.exec_module(viewer_mod)

_confirmatory_spec = importlib.util.spec_from_file_location("run_k_model_confirmatory_analysis", SCRIPTS_DIR / "run_k_model_confirmatory_analysis.py")
confirmatory_mod = importlib.util.module_from_spec(_confirmatory_spec)
sys.modules["run_k_model_confirmatory_analysis"] = confirmatory_mod
_confirmatory_spec.loader.exec_module(confirmatory_mod)

CAVEAT_TEXT = "中央値で無理やり分けた型であり、集団は連続した一つの雲"

TYPES = ["K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"]
# Section 1-1's naming convention: colors are figure-legend-only, mapped
# from the pre-rename colors (白/青/緑/ピンク) -- "白"(K_Secure) is plotted
# as black here since literal white is invisible against a white figure
# background; the others keep their original color.
TYPE_LABEL_JA = {"K_Secure": "K-安定型", "K_Dismissing": "K-拒絶型", "K_Preoccupied": "K-とらわれ型", "K_Fearful": "K-恐れ型"}
TYPE_COLOR = {"K_Secure": "black", "K_Dismissing": "tab:blue", "K_Preoccupied": "tab:green", "K_Fearful": "magenta"}

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
        "Re-run this script on a Mac (which has Hiragino installed) to regenerate these figures "
        "with the requested font.",
        file=sys.stderr,
    )


# ------------------------------------------------------------------ data loading (census + recovery only, no moves)

def load_age_data(data_dir: Path, seeds: List[int], age: int, threshold_kout: float, threshold_kself: float):
    """{seed: census_rows}, {seed: recovery_rows}, true_type_by_seed, depth_by_seed -- skips
    moves.csv.gz entirely (not needed for recovery-rate figures)."""
    census_by_seed = {}
    recovery_by_seed = {}
    for seed in seeds:
        cpath = viewer_mod.pilot_mod.census_path(data_dir, seed, age)
        rpath = viewer_mod.pilot_mod.recovery_path(data_dir, seed, age)
        if not (cpath.exists() and rpath.exists()):
            continue
        census_by_seed[seed] = viewer_mod.load_census_with_k(cpath)
        recovery_by_seed[seed] = viewer_mod.load_recovery(rpath)

    true_type_by_seed = {s: confirmatory_mod.census_true_types(c, threshold_kout, threshold_kself) for s, c in census_by_seed.items()}
    depth_by_seed = {s: confirmatory_mod.gid_effective_depth(c) for s, c in census_by_seed.items()}
    counts_by_gid = confirmatory_mod.recovery_counts_by_gid(recovery_by_seed, depth_by_seed)
    return true_type_by_seed, counts_by_gid


def rate_for_weights(counts_by_gid, true_type_by_seed, key: Tuple[str, str], type_name: str, seed_weights=None) -> Optional[float]:
    tot = confirmatory_mod.recovery_counts_by_label(counts_by_gid, true_type_by_seed, seed_weights=seed_weights)
    c = tot.get((type_name, key))
    if not c or not c[0]:
        return None
    n, n_rec = c
    return n_rec / n


# ------------------------------------------------------------------ Figure A: by generation, 4 types, SELF

def build_figure_a_data(data_dir: Path, seeds: List[int], ages: List[int], threshold_kout: float, threshold_kself: float, n_bootstrap_seed: int, master_seed: str):
    resamples = confirmatory_mod.build_seedboot_resamples(seeds, n_bootstrap_seed, master_seed)
    points: Dict[str, List[Optional[float]]] = {t: [] for t in TYPES}
    bands: Dict[str, List[Tuple[float, float]]] = {t: [] for t in TYPES}

    for age in ages:
        true_type_by_seed, counts_by_gid = load_age_data(data_dir, seeds, age, threshold_kout, threshold_kself)
        for type_name in TYPES:
            points[type_name].append(rate_for_weights(counts_by_gid, true_type_by_seed, ("opp", "SELF"), type_name))

            boot_rates = []
            for resampled in resamples:
                weights: Dict[int, int] = {}
                for s in resampled:
                    weights[s] = weights.get(s, 0) + 1
                r = rate_for_weights(counts_by_gid, true_type_by_seed, ("opp", "SELF"), type_name, seed_weights=weights)
                if r is not None:
                    boot_rates.append(r)
            if boot_rates:
                boot_rates.sort()
                lo = boot_rates[int(0.025 * len(boot_rates))]
                hi = boot_rates[min(int(0.975 * len(boot_rates)), len(boot_rates) - 1)]
                bands[type_name].append((lo, hi))
            else:
                p = points[type_name][-1]
                bands[type_name].append((p, p) if p is not None else (None, None))
        print(f"figure A: age={age} done", flush=True)

    return points, bands


def plot_figure_a(ages: List[int], points, bands, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for type_name in TYPES:
        ys = [v for v in points[type_name] if v is not None]
        xs = [a for a, v in zip(ages, points[type_name]) if v is not None]
        los = [b[0] for b, v in zip(bands[type_name], points[type_name]) if v is not None and b[0] is not None]
        his = [b[1] for b, v in zip(bands[type_name], points[type_name]) if v is not None and b[1] is not None]
        if los and his:
            ax.fill_between(xs, los, his, color=TYPE_COLOR[type_name], alpha=0.15, linewidth=0)
        ax.plot(xs, ys, label=TYPE_LABEL_JA[type_name], color=TYPE_COLOR[type_name], linewidth=2)

    ax.set_xlabel("世代")
    ax.set_ylabel("20回のうちに和和和へ戻れた割合")
    ax.set_title("底からの回復:仲間(SELF)相手、類型ごと")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.025, "帯はseedをまたいだブートストラップの95%範囲", ha="center", fontsize=8, color="gray")
    fig.text(0.5, 0.005, CAVEAT_TEXT, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}", flush=True)


# ------------------------------------------------------------------ Figure B: by depth stratum, K_Secure vs K_Fearful, SELF, window-averaged

def build_figure_b_data(data_dir: Path, seeds: List[int], ages: List[int], threshold_kout: float, threshold_kself: float, n_bootstrap_seed: int, master_seed: str):
    resamples = confirmatory_mod.build_seedboot_resamples(seeds, n_bootstrap_seed, master_seed)
    strata = confirmatory_mod.HC_STRATA
    per_age_rates: Dict[Tuple[str, str], List[float]] = {(t, s): [] for t in ("K_Secure", "K_Fearful") for s in strata}
    per_iteration_rates: Dict[Tuple[str, str], Dict[int, List[float]]] = {(t, s): {i: [] for i in range(len(resamples))} for t in ("K_Secure", "K_Fearful") for s in strata}

    for age in ages:
        true_type_by_seed, counts_by_gid = load_age_data(data_dir, seeds, age, threshold_kout, threshold_kself)
        for type_name in ("K_Secure", "K_Fearful"):
            for stratum in strata:
                r = rate_for_weights(counts_by_gid, true_type_by_seed, ("stratum", stratum), type_name)
                if r is not None:
                    per_age_rates[(type_name, stratum)].append(r)
                for i, resampled in enumerate(resamples):
                    weights: Dict[int, int] = {}
                    for s in resampled:
                        weights[s] = weights.get(s, 0) + 1
                    br = rate_for_weights(counts_by_gid, true_type_by_seed, ("stratum", stratum), type_name, seed_weights=weights)
                    if br is not None:
                        per_iteration_rates[(type_name, stratum)][i].append(br)
        print(f"figure B: age={age} done", flush=True)

    bars = {}
    bands = {}
    for key, ages_rates in per_age_rates.items():
        if not ages_rates:
            continue
        bars[key] = statistics.fmean(ages_rates)
        iter_avgs = [statistics.fmean(vs) for vs in per_iteration_rates[key].values() if vs]
        if iter_avgs:
            iter_avgs.sort()
            lo = iter_avgs[int(0.025 * len(iter_avgs))]
            hi = iter_avgs[min(int(0.975 * len(iter_avgs)), len(iter_avgs) - 1)]
            bands[key] = (lo, hi)
    return bars, bands


def plot_figure_b(bars, bands, out_path: Path) -> None:
    strata = confirmatory_mod.HC_STRATA
    strata_label = {"depthLow": "浅い\n(0〜2)", "depthMid": "中くらい\n(3)", "depthHigh": "深い\n(4以上)"}
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(strata))
    width = 0.35
    for offset, type_name in zip((-1, 1), ("K_Secure", "K_Fearful")):
        heights = [bars.get((type_name, s), 0.0) for s in strata]
        errs = [
            (bars.get((type_name, s), 0.0) - bands[(type_name, s)][0], bands[(type_name, s)][1] - bars.get((type_name, s), 0.0))
            if (type_name, s) in bands else (0.0, 0.0)
            for s in strata
        ]
        err_lo = [e[0] for e in errs]
        err_hi = [e[1] for e in errs]
        positions = [xi + offset * width / 2 for xi in x]
        ax.bar(positions, heights, width=width, label=TYPE_LABEL_JA[type_name], color=TYPE_COLOR[type_name], alpha=0.85, yerr=[err_lo, err_hi], capsize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([strata_label[s] for s in strata])
    ax.set_ylabel("20回のうちに和和和へ戻れた割合(gen 1000〜5000平均)")
    ax.set_title("底からの回復:記憶の深さの層ごと(K-安定型 vs K-恐れ型、SELF相手)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.025, "誤差線はseedをまたいだブートストラップの95%範囲", ha="center", fontsize=8, color="gray")
    fig.text(0.5, 0.005, CAVEAT_TEXT, ha="center", fontsize=8, color="gray")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--thresholds-json", type=str, required=True)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--n-bootstrap-seed", type=int, default=2000)
    p.add_argument("--master-seed", type=str, default="k_model_recovery_figures_v1")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    import json
    with open(args.thresholds_json) as f:
        thresholds = json.load(f)
    threshold_kout = thresholds["threshold_kout"]
    threshold_kself = thresholds["threshold_kself"]

    all_ages = viewer_mod.discover_ages(data_dir)
    ages = [a for a in all_ages if args.window_start <= a <= args.window_end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, bands_a = build_figure_a_data(data_dir, seeds, ages, threshold_kout, threshold_kself, args.n_bootstrap_seed, args.master_seed)
    plot_figure_a(ages, points, bands_a, out_dir / "k_model_recovery_rate_by_generation_SELF.png")

    bars, bands_b = build_figure_b_data(data_dir, seeds, ages, threshold_kout, threshold_kself, args.n_bootstrap_seed, args.master_seed)
    plot_figure_b(bars, bands_b, out_dir / "k_model_recovery_rate_by_depth_stratum_SELF.png")


if __name__ == "__main__":
    main()
