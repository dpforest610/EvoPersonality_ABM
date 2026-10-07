"""Space-time diagrams for the local-placement presentation run:
x=ring position (array index), y=generation, color=K_out / K_self,
shuffle vs local side by side. One PNG per channel.

Usage:
    python3 scripts/plot_local_placement_spacetime.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path("scripts/output/local_placement_presentation")
OUT_DIR = Path("scripts/output/figures")
SEED = 0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shuffle = np.load(DATA_DIR / f"shuffle_seed{SEED}.npz")
    local = np.load(DATA_DIR / f"local_seed{SEED}.npz")

    for channel, label in [("k_out", "K_out"), ("k_self", "K_self")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True, facecolor="white")
        vmin = min(shuffle[channel].min(), local[channel].min())
        vmax = max(shuffle[channel].max(), local[channel].max())
        im = None
        for ax, data, title in [(axes[0], shuffle, "shuffle"), (axes[1], local, "local")]:
            n_gen, n_agents = data[channel].shape
            im = ax.imshow(
                data[channel], aspect="auto", origin="lower",
                extent=[0, n_agents, 0, n_gen],
                cmap="viridis", vmin=vmin, vmax=vmax,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("ring position (array index)")
        axes[0].set_ylabel("generation")
        fig.colorbar(im, ax=list(axes), label=label, fraction=0.025, pad=0.02)
        fig.suptitle(f"Space-time diagram: {label} (k=8, seed={SEED})", fontsize=14)
        fig_path = OUT_DIR / f"local_placement_spacetime_{channel}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
