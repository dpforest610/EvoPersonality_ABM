"""Prototype animation of the evolving Markov network, one seed -- design
doc section 7.3, exploratory/pilot-scoped per the user's 2026-10-16 request.
One frame per probe age; fixed node positions (option B grid layout,
evopersonality_abm.markov_layout); edges drawn at Jaccard >= .30 (matching
the published human-study figure).

Usage:
    python3 scripts/animate_markov_network_pilot_one_seed.py \\
        --seed 0 --opponent TFT --condition noise01 --out scripts/output/figures/markov_network_anim_seed0_TFT.mp4
"""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evopersonality_abm.markov_apply import (
    load_census,
    load_series_by_genotype_all_opponents,
    resample_virtual_group,
)
from evopersonality_abm.markov_layout import node_position
from evopersonality_abm.markov_recovery import build_edges, load_vocab

EDGE_THRESHOLD = 0.30

CONDITIONS = {
    "noise01": ("scripts/output/markov_sweep_pilot_noise01", "noise0p01"),
    "noise00": ("scripts/output/markov_sweep_pilot_noise00", "noise0"),
}


def average_edges(census_path, moves_path, opponent, vocab, group_size, n_boot, rng):
    """n_boot resampled networks, averaged edge-by-edge (missing -> 0)
    for a smoother per-frame picture than any single noisy draw."""
    frequencies = load_census(census_path)
    series_by_opponent = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=20)
    series_by_genotype = series_by_opponent.get(opponent, {})
    if not series_by_genotype:
        return {}

    sums = {}
    for _ in range(n_boot):
        group = resample_virtual_group(frequencies, series_by_genotype, group_size, rng)
        edges = build_edges(group, vocab)
        for pair, w in edges.items():
            sums[pair] = sums.get(pair, 0.0) + w
    return {pair: total / n_boot for pair, total in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", type=str, default="TFT")
    p.add_argument("--condition", choices=list(CONDITIONS), default="noise01")
    p.add_argument("--group-size", type=int, default=56)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--max-age", type=int, default=2000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    out_dir, moves_suffix = CONDITIONS[args.condition]
    out_dir = Path(out_dir)
    vocab_name = "RAN" if args.opponent == "Random" else ("GRA" if args.opponent == "Gradual" else args.opponent)
    vocab = load_vocab(vocab_name)
    rng = random.Random(args.seed)

    ages = list(range(0, args.max_age + 1, args.probe_interval))
    frames = []
    for age in ages:
        census_path = out_dir / f"seed_{args.seed}_probe_age_{age}_census.csv.gz"
        moves_path = out_dir / f"seed_{args.seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        if not census_path.exists() or not moves_path.exists():
            continue
        edges = average_edges(census_path, moves_path, args.opponent, vocab, args.group_size, args.n_boot, rng)
        frames.append((age, edges))

    print(f"built {len(frames)} frames", flush=True)

    all_nodes = set(vocab)
    positions = {w: node_position(w) for w in all_nodes}

    fig, ax = plt.subplots(figsize=(8, 8))

    def draw_frame(i):
        ax.clear()
        age, edges = frames[i]
        for w, (x, y) in positions.items():
            color = "tab:red" if w == "底底底" else ("tab:green" if w == "和和和" else "lightgray")
            ax.scatter([x], [y], s=40, color=color, zorder=3)
        for (a, b), weight in edges.items():
            if weight < EDGE_THRESHOLD:
                continue
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="steelblue", alpha=min(1.0, weight), linewidth=1 + 3 * weight, zorder=2)
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.2, 4.2)
        ax.set_title(f"seed={args.seed} opponent={args.opponent} condition={args.condition}\ngeneration={age} (edges >= {EDGE_THRESHOLD})")
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, draw_frame, frames=len(frames), interval=1000 / args.fps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), fps=args.fps)
    print(f"saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
