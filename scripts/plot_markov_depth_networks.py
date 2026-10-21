"""Exploratory, outside preregistration: one figure per opponent, with a
panel for each effective memory depth 1-6, using the SAME network
definition and coloring convention as the R visualization
(docs/r_scripts/GRA_MAR10feb04Fmarkov_network_with_loops_param_groups.R):
Jaccard >= 0.30 edges (evopersonality_abm.markov_recovery.build_edges),
self-loops for triple-repeat nodes (markov_recovery.self_loop_edges), red
for edges common to every depth panel shown, blue for edges unique to
that panel.

Runs on Mac (reads the large raw probe moves/census files -- Mac-local
only, gitignored, same data as scripts/export_depth_groups_for_r.py).
Reuses that script's census-scanning, selection and series-extraction
logic directly so the two tools agree on what "effective depth d" means
and how a virtual participant is drawn.

Usage: see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section 10.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "IPAGothic"

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import export_depth_groups_for_r as depth_mod  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_apply import OPPONENT_INFO  # noqa: E402
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges  # noqa: E402

OPPONENTS = depth_mod.GROUP_OPPONENTS  # TFT, AllC, AllD, Random, Gradual
DEPTHS = [1, 2, 3, 4, 5, 6]
JACCARD_THRESHOLD = 0.30


def select_depth_group(pool: List[dict], max_size: int, depth: int, rng: random.Random) -> List[dict]:
    """Up to max_size participants at exactly this effective depth --
    reuses export_depth_groups_for_r.select_group's one-per-seed
    preference, but degrades to "use everything available" instead of
    raising when there are fewer than max_size candidates (this is a
    descriptive figure, not a fixed-size group export)."""
    available = [r for r in pool if r["effective_depth"] == depth]
    if len(available) <= max_size:
        return available
    return depth_mod.select_group(pool, max_size, range(depth, depth + 1), rng)


def edges_with_self_loops(series_list: List[str], vocab) -> Dict[Tuple[str, str], float]:
    edges = dict(build_edges(series_list, vocab))
    edges.update({(node, node): w for node, w in self_loop_edges(series_list, vocab).items()})
    return {k: v for k, v in edges.items() if v >= JACCARD_THRESHOLD}


def circular_layout(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    n = len(nodes)
    return {
        node: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n))
        for i, node in enumerate(sorted(nodes))
    }


def draw_panel(ax, edges: Dict[Tuple[str, str], float], common_keys: set, title: str) -> None:
    nodes = sorted({x for e in edges for x in e})
    if not nodes:
        ax.text(0.5, 0.5, "no edges", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return
    pos = circular_layout(nodes)

    for (a, b), j in edges.items():
        color = "red" if (a, b) in common_keys else "blue"
        width = max(j * 3, 0.3)
        if a == b:
            x, y = pos[a]
            loop = plt.Circle((x * 1.15, y * 1.15), 0.06, fill=False, color=color, linewidth=width)
            ax.add_patch(loop)
        else:
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.annotate(
                "", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, shrinkA=8, shrinkB=8,
                                 connectionstyle="arc3,rad=0.15"),
            )

    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=120, color="lightblue", edgecolors="black", zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=6, zorder=4)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_opponent(opponent: str, n_by_depth: Dict[int, int],
                   series_by_depth: Dict[int, Dict[str, Dict[int, str]]], out_path: Path) -> None:
    vocab = load_vocab(OPPONENT_INFO[opponent][1])
    edges_by_depth: Dict[int, Dict[Tuple[str, str], float]] = {}

    for depth in DEPTHS:
        series_list = list(series_by_depth[depth][opponent].values())
        edges_by_depth[depth] = edges_with_self_loops(series_list, vocab) if series_list else {}

    nonempty = [d for d in DEPTHS if edges_by_depth[d]]
    common_keys = set.intersection(*(set(edges_by_depth[d]) for d in nonempty)) if nonempty else set()

    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(4 * len(DEPTHS), 4.5))
    for ax, depth in zip(axes, DEPTHS):
        draw_panel(ax, edges_by_depth[depth], common_keys, f"depth={depth} (n={n_by_depth[depth]})")
    fig.suptitle(f"{opponent}: effective-depth networks (Jaccard >= {JACCARD_THRESHOLD}, red=common across depths shown, blue=depth-specific)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--gradual-reprobe-dir", type=str, required=True)
    p.add_argument("--moves-suffix", type=str, default="noise0p01")
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--group-size", type=int, default=56, help="max participants per depth panel; uses all available if fewer exist")
    p.add_argument("--master-seed", type=str, default="markov_depth_networks_v1")
    p.add_argument("--out-dir", type=str, default="scripts/output/figures")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    gradual_reprobe_dir = Path(args.gradual_reprobe_dir)
    out_dir = Path(args.out_dir)

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.window_start, args.window_end + 1, args.probe_interval))

    print(f"scanning census for {len(seeds)} seeds x {len(ages)} ages...", flush=True)
    pool = depth_mod.collect_census_pool(data_dir, seeds, ages)
    print(f"{len(pool)} census rows collected", flush=True)

    rng = random.Random(args.master_seed)
    participants_by_depth = {depth: select_depth_group(pool, args.group_size, depth, rng) for depth in DEPTHS}
    n_by_depth = {depth: len(participants_by_depth[depth]) for depth in DEPTHS}
    print(f"participants per depth: {n_by_depth}", flush=True)

    series_by_depth = {
        depth: (
            depth_mod.extract_series_for_participants(
                data_dir, gradual_reprobe_dir, args.moves_suffix, participants_by_depth[depth], rng,
            )
            if participants_by_depth[depth] else {opp: {} for opp in OPPONENTS}
        )
        for depth in DEPTHS
    }

    for opponent in OPPONENTS:
        out_path = out_dir / f"markov_depth_networks_{opponent}.png"
        plot_opponent(opponent, n_by_depth, series_by_depth, out_path)


if __name__ == "__main__":
    main()
