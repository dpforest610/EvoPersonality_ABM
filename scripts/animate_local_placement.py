"""Conference animation for the local-placement experiment: 100 agents
on a circle, color=K_out (viridis, colorblind-safe, with colorbar),
size=K_self, shuffle (left) vs local (right) synced side by side.
Generation counter at top. White background, English only, 16:9.

Produces both an MP4 (via ffmpeg) and a shorter/lower-res GIF fallback
(via Pillow -- GIF doesn't compress well, so a full-length full-res GIF
would be far too large for slides).

Usage:
    python3 scripts/animate_local_placement.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path("scripts/output/local_placement_presentation")
OUT_DIR = Path("scripts/output/figures")
SEED = 0

MP4_FPS = 15
MP4_STRIDE = 2       # 2000 gens / stride 2 -> 1000 frames / 15fps = 66.7s
GIF_FPS = 10
GIF_STRIDE = 8        # 2000 gens / stride 8 -> 250 frames / 10fps = 25s, smaller file


def ring_positions(n):
    angles = 2 * np.pi * np.arange(n) / n
    return np.cos(angles), np.sin(angles)


def make_animation(out_path, fps, stride, figsize, dpi, writer_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    shuffle = np.load(DATA_DIR / f"shuffle_seed{SEED}.npz")
    local = np.load(DATA_DIR / f"local_seed{SEED}.npz")
    n_gen, n_agents = shuffle["k_out"].shape
    x, y = ring_positions(n_agents)

    frame_gens = list(range(0, n_gen, stride))

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    fig.suptitle("", fontsize=18)
    gen_text = fig.text(0.5, 0.95, "", ha="center", va="top", fontsize=16, color="black")

    scatters = []
    for ax, title in zip(axes, ["shuffle", "local"]):
        ax.set_facecolor("white")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        sc = ax.scatter(x, y, c=np.zeros(n_agents), s=np.full(n_agents, 40.0),
                         cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none")
        scatters.append(sc)

    cbar = fig.colorbar(scatters[0], ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("K_out", fontsize=12)

    # point-size legend (K_self reference sizes)
    for ref_val, ref_size in [(0.0, 20.0), (0.5, 110.0), (1.0, 220.0)]:
        axes[1].scatter([], [], s=ref_size, c="gray", label=f"K_self={ref_val:.1f}")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.45, 1.0), frameon=False, fontsize=9,
                    title="point size", title_fontsize=9)

    def size_from_kself(kself):
        return 20.0 + kself * 200.0

    def update(frame_idx):
        gen = frame_gens[frame_idx]
        scatters[0].set_array(shuffle["k_out"][gen])
        scatters[0].set_sizes(size_from_kself(shuffle["k_self"][gen]))
        scatters[1].set_array(local["k_out"][gen])
        scatters[1].set_sizes(size_from_kself(local["k_self"][gen]))
        gen_text.set_text(f"generation {gen}")
        return scatters + [gen_text]

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(frame_gens), blit=False)

    if writer_name == "ffmpeg":
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp4_path = OUT_DIR / "local_placement_animation.mp4"
    print(f"rendering MP4 ({len(range(0, 2000, MP4_STRIDE))} frames @ {MP4_FPS}fps) -> {mp4_path} ...", flush=True)
    make_animation(mp4_path, MP4_FPS, MP4_STRIDE, figsize=(16, 9), dpi=120, writer_name="ffmpeg")
    print(f"saved {mp4_path}", flush=True)

    gif_path = OUT_DIR / "local_placement_animation.gif"
    print(f"rendering GIF ({len(range(0, 2000, GIF_STRIDE))} frames @ {GIF_FPS}fps) -> {gif_path} ...", flush=True)
    make_animation(gif_path, GIF_FPS, GIF_STRIDE, figsize=(10, 5.6), dpi=80, writer_name="pillow")
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
