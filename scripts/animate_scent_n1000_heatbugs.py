"""Addition B (docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md): heatbugs-style
4-type animation, same look as heatbugs_style_reference.py (the seed-0,
100-generations-apart prototype) but driven by the dense, every-generation
individual log from Addition A (--individual-log-interval 1), and using
the FIXED pooled-median thresholds from
scripts/compute_scent_n1000_type_thresholds.py instead of the prototype's
seed-0-only ones (0.7835265 / 0.4566165).

Left panel: black background, empty cells colored by the scent field (hot
colormap, fixed color range), agents drawn as colored squares, one of 4
types (median split on K_self/K_out, pooled across all 3 conditions x 30
seeds x gen>=2000 -- fixed, see docs). Right top: type share vs.
generation (line growing as the animation plays). Right bottom: running
sum of (type's mean payoff - population mean payoff) vs. generation.
On-screen note that the split is a forced median cut on one continuous
population, not four natural groups.

Needs scripts/output/.../seed_0_individual_log.csv.gz recorded with
--individual-log-interval 1 (Addition A) -- reads ONLY that file, no
network or production-log access, so it runs on the Mac or in the cloud.

Usage:
    python3 scripts/animate_scent_n1000_heatbugs.py --movement scent \\
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \\
        --speed full      # 5000 gens @ 60fps, ~85s
    python3 scripts/animate_scent_n1000_heatbugs.py --movement scent \\
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \\
        --speed fast       # every 2nd gen, ~40s
"""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scent_spatial_sweep_n1000 import individual_gz_path_for, out_root_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from evopersonality_abm.scent_spatial import moore_offsets  # noqa: E402

# the model's own scent field (_mean_neighbor_payoff, called from
# _run_generation with offsets=moore_offsets(radius)) uses the 8-cell
# Moore neighborhood at the production radius=1, NOT the 4-cell von
# Neumann neighborhood the first prototype used -- see
# test_field_matches_model_mean_neighbor_payoff below.
NEIGHBOR_OFFSETS = moore_offsets(1)

TITLES = {"none": "M0: no movement", "random": "M1: random movement", "scent": "M2: scent-following"}
THRESHOLDS_JSON = Path("scripts/output/scent_spatial_sweep_n1000/analysis/type_thresholds.json")
OUT_DIR = Path("scripts/output/figures")
W = H = 41

TYPES = [
    ("high K_self / high K_out", "#FFFFFF"),
    ("high K_self / low K_out", "#1E90FF"),
    ("low K_self / high K_out", "#32CD32"),
    ("low K_self / low K_out", "#FF4FD8"),
]
# a line/legend-safe substitute for the white swatch on a white chart background
LINE_COLOR = {"#FFFFFF": "#999999", "#1E90FF": "#1E90FF", "#32CD32": "#32CD32", "#FF4FD8": "#FF4FD8"}

# heat-field color range: fixed across the whole animation (payoff per
# round for this game's payoff matrix sits in a narrow band around the
# mutual-cooperation value; widen/narrow with --vmin/--vmax if a
# particular condition clips)
DEFAULT_VMIN, DEFAULT_VMAX = 2.55, 3.15


def load_thresholds():
    with open(THRESHOLDS_JSON) as f:
        d = json.load(f)
    return d["threshold_kself"], d["threshold_kout"]


def load_individual_log(path):
    rows = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            rows[int(r["generation"])].append(
                (int(r["pos_x"]), int(r["pos_y"]), float(r["k_self"]), float(r["k_out"]), float(r["payoff"]))
            )
    gens = sorted(rows)
    return gens, {g: np.array(rows[g]) for g in gens}


def tcode(a, ts, to):
    hs = a[:, 2] >= ts
    ho = a[:, 3] >= to
    return np.where(hs & ho, 0, np.where(hs & ~ho, 1, np.where(~hs & ho, 2, 3)))


def field(a, width=W, height=H, offsets=NEIGHBOR_OFFSETS):
    """Scent field at empty cells: mean payoff of each cell's occupied
    8-cell Moore neighbors -- same neighborhood the model itself scores
    scent candidates with (evopersonality_abm.spatial._mean_neighbor_payoff,
    called with offsets=moore_offsets(radius=1); see
    test_field_matches_model_mean_neighbor_payoff). The first prototype
    (heatbugs_style_reference.py) used the 4-cell von Neumann neighborhood
    instead, which did not match the model. Occupied cells are NaN (drawn
    black, agents drawn over them as squares)."""
    pay = np.full((width, height), np.nan)
    for x, y, *_rest, p in a:
        pay[int(x), int(y)] = p
    f = np.full((height, width), np.nan)
    for x in range(width):
        for y in range(height):
            vals = [pay[(x + dx) % width, (y + dy) % height] for dx, dy in offsets]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                f[y, x] = float(np.mean(vals))
    for x, y, *_rest in a:
        f[int(y), int(x)] = np.nan
    return f


def compute_trajectory(gens, by_gen, ts, to):
    n = len(gens)
    share = np.zeros((n, 4))
    dev = np.zeros((n, 4))
    for i, g in enumerate(gens):
        a = by_gen[g]
        t = tcode(a, ts, to)
        pop_mean = a[:, 4].mean()
        for k in range(4):
            m = t == k
            share[i, k] = m.mean() * 100
            dev[i, k] = (a[m, 4].mean() - pop_mean) if m.any() else 0.0
    return share, np.cumsum(dev, axis=0)


def render(gens, by_gen, share, cum_dev, ts, to, title, out_path, frame_gens, fps, vmin, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("black")

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#C8C8C8")
    axg = fig.add_axes((0.02, 0.06, 0.47, 0.84))
    axg.set_facecolor("black")
    axg.set_xlim(-0.5, W - 0.5)
    axg.set_ylim(-0.5, H - 0.5)
    axg.set_aspect("equal")
    axg.set_xticks([])
    axg.set_yticks([])
    im = axg.imshow(np.full((H, W), np.nan), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                     extent=(-0.5, W - 0.5, -0.5, H - 0.5), interpolation="nearest", zorder=0)
    sq = axg.scatter([], [], marker="s", s=10, linewidths=0, zorder=2)
    fig.text(0.255, 0.93, title, ha="center", fontsize=18, weight="bold")
    gtxt = fig.text(0.255, 0.02, "", ha="center", fontsize=15)

    ax1 = fig.add_axes((0.56, 0.56, 0.27, 0.34))
    ax2 = fig.add_axes((0.56, 0.26, 0.27, 0.26))
    gen_max = gens[-1]
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.set_xlim(0, gen_max)
        ax.tick_params(labelsize=11)
    ax1.set_ylim(0, max(60.0, float(share.max()) * 1.1))
    ax1.set_ylabel("share of agents (%)", fontsize=13)
    ax1.set_title("Type share vs. generation", fontsize=15)
    lo, hi = float(cum_dev.min()), float(cum_dev.max())
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.axhline(0, color="#888", linewidth=0.8)
    ax2.set_ylabel("running sum of payoff\nabove population mean", fontsize=12)
    ax2.set_xlabel("generation", fontsize=13)
    ax2.set_title("Type points vs. generation", fontsize=15)

    l1 = [ax1.plot([], [], color=LINE_COLOR[c], linewidth=2.5)[0] for _, c in TYPES]
    l2 = [ax2.plot([], [], color=LINE_COLOR[c], linewidth=2.5)[0] for _, c in TYPES]
    lg = fig.add_axes((0.845, 0.35, 0.15, 0.3))
    lg.axis("off")
    for k, (name, c) in enumerate(TYPES):
        lg.add_patch(Rectangle((0.0, 0.85 - k * 0.22), 0.12, 0.1, facecolor=c, edgecolor="#999999",
                                linewidth=2, transform=lg.transAxes))
        lg.text(0.17, 0.9 - k * 0.22, name, transform=lg.transAxes, va="center", fontsize=11)
    fig.text(0.56, 0.01,
              f"Types = fixed median split (K_self >= {ts:.4f}, K_out >= {to:.4f}),\npooled across all 3 "
              "conditions x 30 seeds, gen>=2000.\nThe population is one continuous cloud, not four natural "
              "groups.\nEmpty cells: black -> red -> yellow = scent of profit (mean\npayoff of its 8-cell "
              f"neighbours, fixed range [{vmin:.2f}, {vmax:.2f}]).",
              fontsize=8.5, color="#222", va="bottom")

    gen_to_idx = {g: i for i, g in enumerate(gens)}

    def frame(i):
        g = frame_gens[i]
        j = gen_to_idx[g]
        a = by_gen[g]
        im.set_data(field(a))
        sq.set_offsets(a[:, :2])
        sq.set_color([TYPES[k][1] for k in tcode(a, ts, to)])
        for k in range(4):
            l1[k].set_data(gens[: j + 1], share[: j + 1, k])
            l2[k].set_data(gens[: j + 1], cum_dev[: j + 1, k])
        gtxt.set_text(f"Generation {g}")
        return [im, sq, gtxt] + l1 + l2

    anim = animation.FuncAnimation(fig, frame, frames=len(frame_gens))
    anim.save(str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=4500))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--movement", choices=["none", "random", "scent"], required=True)
    p.add_argument("--recording-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", choices=["full", "fast"], required=True,
                   help="'full': every generation @ 60fps (~85s for 5000 gens). "
                        "'fast': every 2nd generation @ 60fps (~40s for 5000 gens)")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    args = p.parse_args()

    ts, to = load_thresholds()
    print(f"thresholds: K_self>={ts:.7f}, K_out>={to:.7f}")

    root = out_root_for(args.movement, experiment_root=args.recording_root)
    path = individual_gz_path_for(root, args.seed)
    print(f"loading {path} ...", flush=True)
    gens, by_gen = load_individual_log(path)
    print(f"{len(gens)} generations loaded (gen {gens[0]}..{gens[-1]})")

    share, cum_dev = compute_trajectory(gens, by_gen, ts, to)

    frame_gens = gens if args.speed == "full" else gens[::2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"scent_n1000_heatbugs_{args.movement}_{args.speed}.mp4"
    print(f"rendering {len(frame_gens)} frames -> {out_path} "
          f"(~{len(frame_gens) / args.fps:.0f}s at {args.fps}fps) ...", flush=True)
    render(gens, by_gen, share, cum_dev, ts, to, TITLES[args.movement], out_path,
           frame_gens, args.fps, args.vmin, args.vmax)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
