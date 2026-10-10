"""Conference animation for the scent-movement pilot: agents on the
13x13 torus grid, color=K_out (viridis, colorblind-safe, with
colorbar), size=K_self, M0 (no movement, left) vs M2 (scent, right)
synced side by side. Generation counter at top. White background,
English only, 16:9.

Built from the PILOT's individual_log.csv (seed 0), which records one
snapshot every 100 generations (LOG_INTERVAL=100 in
run_scent_spatial_sweep.py) -- so this animation has 10 real data
points over the pilot's 1000 generations, not one frame per
generation. Each of those 10 snapshots is held on screen for several
rendered frames (no interpolation of any K_out/K_self/position value --
every rendered frame shows an actually-recorded snapshot, just
repeated) so the pattern is readable at a normal playback speed. A
denser (every-generation) version would need a dedicated presentation
run, same as the local-placement chapter's separate dense-logging
script -- not built here since the pilot data is what this step asks
for.

Produces both an MP4 (via ffmpeg) and a GIF fallback (via Pillow).

Usage:
    python3 scripts/animate_scent_pilot_m0_vs_m2.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path("scripts/output/scent_spatial_sweep")
OUT_DIR = Path("scripts/output/figures")
SEED = 0
GRID_WIDTH = 13
GRID_HEIGHT = 13
MACHINE_TAG = "vm"

MP4_FPS = 10
MP4_HOLD = 15   # each of the 10 snapshots repeated 15x -> 150 frames / 10fps = 15s
GIF_FPS = 5
GIF_HOLD = 8    # 10 * 8 = 80 frames / 5fps = 16s


def load_snapshots(dirname):
    path = EXPERIMENT_ROOT / dirname / MACHINE_TAG / f"seed_{SEED}_individual_log.csv"
    by_gen = defaultdict(lambda: {"k_out": {}, "k_self": {}, "pos": {}})
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            idx = int(row["agent_index"])
            by_gen[gen]["k_out"][idx] = float(row["k_out"])
            by_gen[gen]["k_self"][idx] = float(row["k_self"])
            by_gen[gen]["pos"][idx] = (int(row["pos_x"]), int(row["pos_y"]))
    gens = sorted(by_gen.keys())
    n_agents = len(by_gen[gens[0]]["k_out"])
    k_out = np.array([[by_gen[g]["k_out"][i] for i in range(n_agents)] for g in gens])
    k_self = np.array([[by_gen[g]["k_self"][i] for i in range(n_agents)] for g in gens])
    pos_x = np.array([[by_gen[g]["pos"][i][0] for i in range(n_agents)] for g in gens])
    pos_y = np.array([[by_gen[g]["pos"][i][1] for i in range(n_agents)] for g in gens])
    return gens, k_out, k_self, pos_x, pos_y


def size_from_kself(kself):
    return 20.0 + kself * 200.0


def make_animation(out_path, fps, hold, figsize, dpi, writer_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    gens0, kout0, kself0, px0, py0 = load_snapshots("none_pilot")
    gens2, kout2, kself2, px2, py2 = load_snapshots("scent_pilot")
    assert gens0 == gens2
    n_snapshots = len(gens0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    gen_text = fig.text(0.5, 0.95, "", ha="center", va="top", fontsize=16, color="black")

    scatters = []
    for ax, title in zip(axes, ["M0: no movement", "M2: scent"]):
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
        ax.set_ylim(-0.5, GRID_HEIGHT - 0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        sc = ax.scatter([], [], c=[], s=[], cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none")
        scatters.append(sc)

    cbar = fig.colorbar(scatters[0], ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("K_out", fontsize=12)

    for ref_val, ref_size in [(0.0, 20.0), (0.5, 110.0), (1.0, 220.0)]:
        axes[1].scatter([], [], s=ref_size, c="gray", label=f"K_self={ref_val:.1f}")
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.45, 1.0), frameon=False, fontsize=9,
                    title="point size", title_fontsize=9)

    frame_snapshot_idx = [i for i in range(n_snapshots) for _ in range(hold)]

    def update(frame_idx):
        i = frame_snapshot_idx[frame_idx]
        scatters[0].set_offsets(np.column_stack([px0[i], py0[i]]))
        scatters[0].set_array(kout0[i])
        scatters[0].set_sizes(size_from_kself(kself0[i]))
        scatters[1].set_offsets(np.column_stack([px2[i], py2[i]]))
        scatters[1].set_array(kout2[i])
        scatters[1].set_sizes(size_from_kself(kself2[i]))
        gen_text.set_text(f"generation {gens0[i]}")
        return scatters + [gen_text]

    anim = FuncAnimation(fig, update, frames=len(frame_snapshot_idx), blit=False)

    if writer_name == "ffmpeg":
        writer = FFMpegWriter(fps=fps, bitrate=4000)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp4_path = OUT_DIR / "scent_pilot_m0_vs_m2_animation.mp4"
    print(f"rendering MP4 -> {mp4_path} ...", flush=True)
    make_animation(mp4_path, MP4_FPS, MP4_HOLD, figsize=(16, 9), dpi=120, writer_name="ffmpeg")
    print(f"saved {mp4_path}", flush=True)

    gif_path = OUT_DIR / "scent_pilot_m0_vs_m2_animation.gif"
    print(f"rendering GIF -> {gif_path} ...", flush=True)
    make_animation(gif_path, GIF_FPS, GIF_HOLD, figsize=(10, 5.6), dpi=80, writer_name="pillow")
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
