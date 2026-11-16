"""Fit a discrete Ornstein-Uhlenbeck (mean-reverting AR(1)) model to
K_out/K_self, to get a single detrend-free "fingerprint" of the process
in place of the retired ~77-gen peak.

Model: x(t+1) - x(t) = -theta*(x(t) - mu) + sigma*eps(t), i.e. an AR(1)
x(t+1) = phi*x(t) + c + sigma*eps(t) with phi = 1-theta, mu = c/theta.
theta (mean-reversion rate), mu (center), sigma (diffusion scale) are
estimated per seed per channel by ordinary least squares regression of
x(t+1) on x(t) -- no moving-average detrend, no PSD peak search.

Read: small theta = weak mean reversion = a long-range wanderer that
takes many generations to relax back toward mu (consistent with the
near-Brownian beta~2 / DFA alpha~1.3 found by
analyze_spectral_exponent_hurst.py -- see the consistency check below).
theta becomes the system's characteristic scale in place of the retired
77-generation period: 1/theta is the AR(1) relaxation time (in
generations), a lower bound on how far apart samples must be to look
roughly independent.

Consistency check: for an AR(1) process, phi close to 1 (small theta)
produces a Lorentzian PSD whose "knee" period is ~2*pi/theta; well
below that period the spectrum is flat (beta~0), well above it
(shorter periods than the knee, i.e. our measurement's 20-300gen band)
the spectrum should look like beta~2 (a random walk's high-frequency
tail). If the fitted theta gives a knee period much longer than
~300 generations, that's exactly consistent with beta~2 being measured
in that band -- this script checks that arithmetic explicitly.

Uses only the existing 50-seed, 5000-generation confirmatory coupling
sweep (scripts/output/coupling_seed_sweep_5000gen/), window
gen[2000,5000) -- no new simulation.

Outputs:
    scripts/output/ou_fit_diagnostic/ou_fit_diagnostic.csv
    scripts/output/ou_fit_diagnostic/ou_parameter_distributions.png

Usage:
    python3 scripts/analyze_ou_fit_diagnostic.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SWEEP_DIR = Path("scripts/output/coupling_seed_sweep_5000gen")
OUT_DIR = Path("scripts/output/ou_fit_diagnostic")
SEEDS = list(range(50))
WINDOW_RANGE = (2000, 5000)
BETA_BAND = (20.0, 300.0)  # the period band beta was measured over, for the consistency check

KOUT_COLOR = "#2a78d6"
KSELF_COLOR = "#eb6834"
MUTED_INK = "#8f8f88"


def load_full_series(seed):
    path = SWEEP_DIR / f"seed_{seed}_timeseries.csv"
    gens, kout, kself = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            kout.append(float(row["mean_k_out"]))
            kself.append(float(row["mean_k_self"]))
    return np.array(gens), np.array(kout), np.array(kself)


def window_mask(gens, lo, hi):
    return (gens >= lo) & (gens < hi)


def fit_ou(x):
    """OLS fit of x(t+1) = phi*x(t) + c + noise. Returns theta, mu, sigma, phi."""
    x_t = x[:-1]
    x_tp1 = x[1:]
    phi, c = np.polyfit(x_t, x_tp1, 1)
    theta = 1.0 - phi
    mu = c / theta if theta != 0 else float("nan")
    resid = x_tp1 - (phi * x_t + c)
    sigma = resid.std(ddof=2)
    return theta, mu, sigma, phi


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        gens, kout, kself = load_full_series(seed)
        mask = window_mask(gens, *WINDOW_RANGE)
        for channel_name, series in [("K_out", kout[mask]), ("K_self", kself[mask])]:
            theta, mu, sigma, phi = fit_ou(series)
            knee_period = 2 * np.pi / theta if theta > 0 else float("inf")
            rows.append(
                {
                    "seed": seed,
                    "channel": channel_name,
                    "theta": theta,
                    "mu": mu,
                    "sigma": sigma,
                    "phi": phi,
                    "knee_period_gen": knee_period,
                }
            )

    csv_path = OUT_DIR / "ou_fit_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path} ({len(rows)} rows)")

    print(f"\n=== OU/AR(1) fit, window gen{WINDOW_RANGE}, n={len(SEEDS)} seeds ===")
    print("model: x(t+1)-x(t) = -theta*(x(t)-mu) + sigma*eps ; phi = 1-theta\n")
    for channel_name in ["K_out", "K_self"]:
        sub = [r for r in rows if r["channel"] == channel_name]
        thetas = np.array([r["theta"] for r in sub])
        mus = np.array([r["mu"] for r in sub])
        sigmas = np.array([r["sigma"] for r in sub])
        phis = np.array([r["phi"] for r in sub])
        knees = np.array([r["knee_period_gen"] for r in sub])

        def q(v):
            return np.percentile(v, [25, 50, 75])

        tq1, tmed, tq3 = q(thetas)
        mq1, mmed, mq3 = q(mus)
        sq1, smed, sq3 = q(sigmas)
        pq1, pmed, pq3 = q(phis)
        kq1, kmed, kq3 = q(knees)

        print(f"{channel_name}:")
        print(f"  theta: median={tmed:.5f}  IQR=[{tq1:.5f},{tq3:.5f}]")
        print(f"  mu:    median={mmed:.4f}  IQR=[{mq1:.4f},{mq3:.4f}]")
        print(f"  sigma: median={smed:.5f}  IQR=[{sq1:.5f},{sq3:.5f}]")
        print(f"  phi=1-theta: median={pmed:.5f}  IQR=[{pq1:.5f},{pq3:.5f}]  ({'close to 1' if pmed > 0.95 else 'not close to 1'})")
        print(f"  implied AR(1) knee period (2*pi/theta): median={kmed:.1f} gen  IQR=[{kq1:.1f},{kq3:.1f}]")
        # AR(1) theory: S(f) is flat (white) for periods >> knee period 2*pi/theta,
        # and rolls off as beta~2 for periods << the knee period. So beta~2 measured
        # over [period_min, period_max] is consistent with the fit iff the knee
        # period sits above period_max -- the whole measurement band must be on the
        # short-period (high-frequency) side of the knee.
        beta_band_hi = BETA_BAND[1]
        consistent = kmed > beta_band_hi
        print(f"  consistency check: knee period ({kmed:.0f} gen) vs. beta measurement band "
              f"({BETA_BAND[0]:.0f}-{BETA_BAND[1]:.0f} gen) -> "
              f"{'CONSISTENT: band sits below the knee, where AR(1) theory predicts beta~2' if consistent else 'INCONSISTENT: band is not below the knee'}")
        print()

    # --- plot: seed distributions of theta, mu, sigma ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, key, title, ylog in [
        (axes[0], "theta", "θ (mean-reversion rate)", True),
        (axes[1], "mu", "μ (center)", False),
        (axes[2], "sigma", "σ (diffusion scale)", False),
    ]:
        for i, (channel_name, color) in enumerate([("K_out", KOUT_COLOR), ("K_self", KSELF_COLOR)]):
            vals = np.array([r[key] for r in rows if r["channel"] == channel_name])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color, alpha=0.5, edgecolor="none")
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.errorbar([i], [med], yerr=[[med - q1], [q3 - med]], marker="o", color=color,
                        markersize=10, linewidth=2.2, capsize=5, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["K_out", "K_self"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title)
        if ylog:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"OU/AR(1) fit, window gen{WINDOW_RANGE}, n={len(SEEDS)} seeds (dots = seeds, marker = median ± IQR)", y=1.02)
    fig.tight_layout()
    fig_path = OUT_DIR / "ou_parameter_distributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
