"""Step 4 of the N=1000 presentation-animation procedure (see
docs/SCENT_MODEL_N1000_PHASE_ANIMATION.md). Reads the phase_recording.pkl.gz
files written by scripts/run_scent_spatial_sweep_n1000.py (one per
condition, each holding ALL recorded generations for its seed -- both the
gen-0-19 and gen-4980-4999 ranges, since recording covers the union of
whatever --record-phases-gens asked for) and renders a 3-panel (M0/M1/M2)
synchronized animation for ONE of the two generation ranges at a time.

Reads ONLY the recording files -- no dependency on the production
individual logs, no network access -- so it runs identically on this cloud
checkout or on the Mac that produced the recording.

One rendered frame = either a held key phase, or one interpolation step
between two consecutive phases. Interpolation is used ONLY to animate
agent position (shortest path across the torus, so a wraparound move is
drawn the short way); K_out, K_self and the background smell field are
never interpolated -- they hold the value of the phase being transitioned
INTO for the whole transition (duels happen before movement within a
phase, so that phase's payoff-so-far is exactly what the movement
responded to). A generation boundary is never interpolated across --
offspring replace parents at the start of a new generation, so "agent 7"
in generation g+1 is not the same individual as "agent 7" in generation g;
the animation hard-cuts there.

Usage:
    python3 scripts/animate_scent_n1000_phase_recording.py \\
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \\
        --gen-range 0-19 --out-name early
    python3 scripts/animate_scent_n1000_phase_recording.py \\
        --recording-root scripts/output/scent_spatial_sweep_n1000_phase_recording \\
        --gen-range 4980-4999 --out-name late
"""

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scent_spatial_sweep_n1000 import out_root_for, phase_recording_path_for  # noqa: E402

CONDITIONS = [("M0: no movement", "none"), ("M1: random movement", "random"), ("M2: scent-following", "scent")]
OUT_DIR = Path("scripts/output/figures")


# ---------------------------------------------------------------- data

def load_phase_recording(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def torus_delta(a: float, b: float, size: int) -> float:
    """Shortest signed delta a -> b on a torus of the given size."""
    d = (b - a) % size
    if d > size / 2:
        d -= size
    return d


def interpolate_positions(pos_from, pos_to, grid_width, grid_height, frac):
    """pos_from/pos_to: list[(x, y)] same length. Returns an (n, 2) float
    array interpolated the torus-shortest way."""
    out = np.empty((len(pos_from), 2))
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pos_from, pos_to)):
        dx = torus_delta(x0, x1, grid_width)
        dy = torus_delta(y0, y1, grid_height)
        out[i, 0] = (x0 + dx * frac) % grid_width
        out[i, 1] = (y0 + dy * frac) % grid_height
    return out


def build_render_frames(phase_frames: dict, n_phases: int, subframes: int, gen_filter=None):
    """phase_frames: {gen: {"frames": [(phase_idx, positions, moved,
    payoff_so_far), ...], "k_out": [...], "k_self": [...]}} as loaded from
    one condition's phase_recording.pkl.gz.

    Returns an ordered list of render-frame dicts:
        {gen, phase, positions (n,2 float array), k_out, k_self,
         payoff_so_far (n array, for the background field),
         moved (n bool array, this phase's movers), is_cut}
    one entry per (held key phase) and one entry per interpolation
    sub-step between consecutive phases within the same generation.
    Generation boundaries are never interpolated across (is_cut=True on
    the first frame of a new generation).
    """
    gens = sorted(g for g in phase_frames if gen_filter is None or gen_filter(g))
    out = []
    prev_positions = None
    for gen in gens:
        rec = phase_frames[gen]
        frames = sorted(rec["frames"], key=lambda t: t[0])
        k_out, k_self = rec["k_out"], rec["k_self"]
        assert len(frames) == n_phases, f"gen {gen}: expected {n_phases} phases, got {len(frames)}"
        for phase_idx, positions, moved, payoff_so_far in frames:
            moved = np.asarray(moved, dtype=bool)
            payoff_so_far = np.asarray(payoff_so_far, dtype=float)
            positions_arr = np.array(positions, dtype=float)
            if prev_positions is None or phase_idx == 0:
                # no meaningful "transition into it" to interpolate from:
                # either the very first frame of the whole animation, or
                # the first phase of a new generation -- offspring replace
                # parents at a generation boundary, so "agent i" in gen g+1
                # is not the same individual as "agent i" in gen g, and
                # interpolating their positions would be meaningless.
                out.append(dict(gen=gen, phase=phase_idx, positions=positions_arr,
                                 k_out=k_out, k_self=k_self, payoff_so_far=payoff_so_far,
                                 moved=moved, is_cut=bool(out)))
            else:
                # interpolation needs the grid size, which this function
                # doesn't take -- left as an unexpanded transition entry
                # for expand_interpolated_frames() to fill in.
                out.append(dict(gen=gen, phase=phase_idx, _interp_from=prev_positions,
                                 _interp_to=positions_arr, subframes=subframes,
                                 k_out=k_out, k_self=k_self, payoff_so_far=payoff_so_far,
                                 moved=moved, is_cut=False))
            prev_positions = positions_arr
    return out


def expand_interpolated_frames(key_frames, grid_width, grid_height):
    """key_frames as returned by build_render_frames: expand each
    transition entry (one holding "_interp_from"/"_interp_to") into its
    `subframes` individual rendered steps, each a dict with a concrete
    (n,2) `positions` array. Pass-through entries without "_interp_from"
    (first-frame-of-a-generation holds) are returned as a single frame."""
    out = []
    for kf in key_frames:
        if "_interp_from" not in kf:
            out.append(kf)
            continue
        subframes = kf["subframes"]
        for step in range(1, subframes + 1):
            frac = step / subframes
            pos = interpolate_positions(kf["_interp_from"], kf["_interp_to"], grid_width, grid_height, frac)
            f = dict(kf)
            f.pop("_interp_from")
            f.pop("_interp_to")
            f.pop("subframes")
            f["positions"] = pos
            # only the final sub-step actually represents "arrival at this phase"
            f["is_arrival"] = step == subframes
            out.append(f)
    return out


def mean_neighbor_payoff_grid(positions, payoff_so_far, grid_width, grid_height, radius=1):
    """Background smell field: for every cell, the mean payoff-so-far of
    its currently-occupied radius-1 Moore neighbors -- same definition as
    evopersonality_abm.spatial._mean_neighbor_payoff (reimplemented
    vectorized over the whole grid for rendering speed). Returns
    (grid, occupied_neighbor_mask): grid[y, x] is NaN where no neighbor is
    occupied (render as white), else the mean payoff-so-far."""
    occ = -np.ones((grid_height, grid_width), dtype=int)
    for i, (x, y) in enumerate(positions):
        occ[int(round(y)) % grid_height, int(round(x)) % grid_width] = i
    offsets = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
               if not (dx == 0 and dy == 0)]
    grid = np.full((grid_height, grid_width), np.nan)
    has_neighbor = np.zeros((grid_height, grid_width), dtype=bool)
    for y in range(grid_height):
        for x in range(grid_width):
            vals = []
            for dx, dy in offsets:
                j = occ[(y + dy) % grid_height, (x + dx) % grid_width]
                if j >= 0 and payoff_so_far[j] == payoff_so_far[j]:  # not NaN
                    vals.append(payoff_so_far[j])
            if vals:
                grid[y, x] = sum(vals) / len(vals)
                has_neighbor[y, x] = True
    return grid, has_neighbor


def payoff_color_range(panel_frames, grid_width, grid_height, lo=2.0, hi=98.0):
    """vmin/vmax for the background field, fixed across the whole video:
    the [lo, hi] percentile of the actually-DISPLAYED grid values (after
    neighbor-averaging), pooled across every frame and panel -- not the
    raw payoff-so-far values (narrower, since averaging compresses the
    range) and not [0, max] (which was dominated by rare high/low
    outliers and left the whole field looking uniformly dark red)."""
    vals = []
    for frames in panel_frames:
        for f in frames:
            grid, has_neighbor = mean_neighbor_payoff_grid(f["positions"], f["payoff_so_far"], grid_width, grid_height)
            vals.append(grid[has_neighbor])
    pooled = np.concatenate(vals)
    return float(np.percentile(pooled, lo)), float(np.percentile(pooled, hi))


# ---------------------------------------------------------------- rendering

def size_from_kself(kself):
    # N=1000 on a 41x41 grid is ~10x denser (on screen) than the N=100/13x13
    # pilot animation, so markers need to be much smaller to stay legible.
    return 2.5 + np.asarray(kself) * 11.0


def render(frames, grid_width, grid_height, n_phases, vmin_payoff, vmax_payoff, out_path, fps, writer_name,
           figsize, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    panel_frames = frames  # list of 3 per-condition frame lists, same length, index-aligned
    n_frames = len(panel_frames[0])
    for pf in panel_frames:
        assert len(pf) == n_frames

    reds = plt.get_cmap("Reds").copy()
    reds.set_bad("white")

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
    # top raised from 0.87 -> 0.92 (less blank space above the panels);
    # bottom lowered from 0.14 -> 0.22 to make room for the new horizontal
    # scent-field colorbar below the legend, without crowding it.
    fig.subplots_adjust(left=0.015, right=0.90, top=0.92, bottom=0.22, wspace=0.06)
    gen_text = fig.text(0.46, 0.975, "", ha="center", va="top", fontsize=16, color="black")
    move_texts = []

    images, scatters = [], []
    for ax, (title, _) in zip(axes, CONDITIONS):
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(-0.5, grid_height - 0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        im = ax.imshow(np.full((grid_height, grid_width), np.nan), origin="lower", cmap=reds,
                        vmin=vmin_payoff, vmax=vmax_payoff, extent=(-0.5, grid_width - 0.5, -0.5, grid_height - 0.5),
                        zorder=1)
        sc = ax.scatter([], [], c=[], s=[], cmap="viridis", vmin=0.0, vmax=1.0,
                         edgecolors="none", linewidths=0.6, zorder=3)
        images.append(im)
        scatters.append(sc)
        move_texts.append(ax.text(0.5, -0.05, "", transform=ax.transAxes, ha="center", va="top", fontsize=11))

    # K_out: the only colorbar on the right edge now, so it gets its own
    # clear space (previously crowded against the scent-field colorbar,
    # whose label overlapped it).
    cbar1 = fig.colorbar(scatters[0], ax=axes, fraction=0.022, pad=0.015, location="right")
    cbar1.set_label("K_out", fontsize=10)

    # scent field: moved to a horizontal colorbar below the panels
    # (instead of a second vertical bar on the right) so the two labels
    # no longer collide.
    cax2 = fig.add_axes((0.30, 0.05, 0.40, 0.016))
    cbar2 = fig.colorbar(ScalarMappable(Normalize(vmin_payoff, vmax_payoff), reds), cax=cax2, orientation="horizontal")
    cbar2.set_label("neighbor payoff-so-far (scent field)", fontsize=10)

    for ref_val, ref_size in [(0.0, 2.5), (0.5, 8.0), (1.0, 13.5)]:
        axes[0].scatter([], [], s=ref_size * 6, c="gray", label=f"K_self={ref_val:.1f}")
    axes[0].scatter([], [], s=40, facecolors="gray", edgecolors="white", linewidths=1.3,
                     label="moved this phase")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.98, -0.11), frameon=False, fontsize=8,
                    ncol=4, handletextpad=0.3, columnspacing=0.8)

    def update(i):
        artists = [gen_text]
        for panel_idx, (im, sc, txt) in enumerate(zip(images, scatters, move_texts)):
            f = panel_frames[panel_idx][i]
            grid, _ = mean_neighbor_payoff_grid(f["positions"], f["payoff_so_far"], grid_width, grid_height)
            im.set_data(grid)
            sc.set_offsets(f["positions"])
            sc.set_array(np.asarray(f["k_out"]))
            sc.set_sizes(size_from_kself(f["k_self"]))
            edge = np.where(f["moved"], "white", "none")
            sc.set_edgecolors(edge)
            move_rate = float(np.mean(f["moved"]))
            txt.set_text(f"move rate this phase: {move_rate:.1%}")
            artists += [im, sc, txt]
        f0 = panel_frames[0][i]
        gen_text.set_text(f"Generation {f0['gen']}, phase {f0['phase'] + 1}/{n_phases}")
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=4500) if writer_name == "ffmpeg" else PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- main

def parse_gen_range(spec):
    lo, hi = spec.split("-")
    return int(lo), int(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-range", required=True, help="'lo-hi' inclusive, e.g. '0-19' or '4980-4999'")
    p.add_argument("--out-name", required=True, help="output file stem, e.g. 'early' or 'late'")
    p.add_argument("--n-phases", type=int, default=5)
    p.add_argument("--grid-width", type=int, default=41)
    p.add_argument("--grid-height", type=int, default=41)
    p.add_argument("--subframes", type=int, default=4, help="interpolation steps per phase transition")
    p.add_argument("--mp4-fps", type=int, default=18)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--color-pctl-lo", type=float, default=2.0,
                   help="lower percentile of the displayed scent-field values used as the fixed vmin")
    p.add_argument("--color-pctl-hi", type=float, default=98.0,
                   help="upper percentile of the displayed scent-field values used as the fixed vmax")
    args = p.parse_args()

    lo, hi = parse_gen_range(args.gen_range)
    gen_filter = lambda g: lo <= g <= hi  # noqa: E731

    panel_frames = []
    for _, movement in CONDITIONS:
        root = out_root_for(movement, experiment_root=args.recording_root)
        path = phase_recording_path_for(root, args.seed)
        print(f"loading {path} ...", flush=True)
        phase_frames = load_phase_recording(path)
        key_frames = build_render_frames(phase_frames, args.n_phases, args.subframes, gen_filter=gen_filter)
        frames = expand_interpolated_frames(key_frames, args.grid_width, args.grid_height)
        panel_frames.append(frames)

    n = min(len(pf) for pf in panel_frames)
    panel_frames = [pf[:n] for pf in panel_frames]

    vmin, vmax = payoff_color_range(panel_frames, args.grid_width, args.grid_height,
                                     lo=args.color_pctl_lo, hi=args.color_pctl_hi)
    print(f"{n} frames/panel, scent-field color range = [{vmin:.3f}, {vmax:.3f}] "
          f"({args.color_pctl_lo:.0f}-{args.color_pctl_hi:.0f} percentile of displayed values)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = OUT_DIR / f"scent_n1000_phase_animation_{args.out_name}.mp4"
    print(f"rendering MP4 -> {mp4_path} ...", flush=True)
    render(panel_frames, args.grid_width, args.grid_height, args.n_phases, vmin, vmax, mp4_path,
           fps=args.mp4_fps, writer_name="ffmpeg", figsize=(16, 9), dpi=120)
    print(f"saved {mp4_path}", flush=True)

    gif_path = OUT_DIR / f"scent_n1000_phase_animation_{args.out_name}.gif"
    print(f"rendering GIF -> {gif_path} ...", flush=True)
    render(panel_frames, args.grid_width, args.grid_height, args.n_phases, vmin, vmax, gif_path,
           fps=args.gif_fps, writer_name="pillow", figsize=(13, 7.3), dpi=85)
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
