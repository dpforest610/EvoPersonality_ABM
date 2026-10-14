"""N=1000 scent-model production: FULL ANALYSIS per
docs/PREREGISTRATION_scent_model_N1000.md, in the order the
pre-registration lists it:

    section 0  pilot/production separation (only the 5 production
               directories are read; *_pilot* directories never are)
    section 3  group 2 (M1-low vs M2-low): A2 primary, B(a)/B(b)
               equivalents secondary, movement-frequency (M2 vs M2-low)
               descriptive
    section 5  drift hypothesis: z*_K_self vs the fixed N=100 table
    section 6  group 1 (M0/M1/M2): A primary, B(a)(b)(c) secondary,
               C descriptive

plus, clearly separated and labelled exploratory / NOT pre-registered:
    - an N=100 vs N=1000 side-by-side table (same metrics, same sign
      convention), N=100 values recomputed from the committed N=100 raw
      logs with scripts/analyze_scent_production.py's own functions
    - the ring-model prediction line checked against all 5 N=1000
      conditions at once (residual = observed - predicted)
    - the section-4 ANCOVA (assortativity ~ is_scent + move_rate) refit
      on the N=1000 group-1 and group-2 data

The N=1000 runner never commits the raw per-agent log; the per-seed
reductions (context-model betas, own-trait Lande-Arnold, seed-mean
beta/gamma for z*, variance/bimodality) were computed in-run with the
same math as the N=100 analysis and committed as seed_X_summary.csv.
Operational/environment values come from seed_X_timeseries.csv exactly
as at N=100 (gen>=2000 per-seed mean).

Writes:
    scripts/output/scent_spatial_sweep_n1000/analysis/results.json
    docs/figures/scent_n1000/*.png

Usage:
    python3 scripts/analyze_scent_production_n1000.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_scent_production as n100  # noqa: E402  (N=100 analysis, reused unchanged)

ROOT = Path("scripts/output/scent_spatial_sweep_n1000")
MACHINE = "doeRyanoMacBook-Neo.local"
CONDITIONS = [
    ("M0", ROOT / "none" / MACHINE),
    ("M1", ROOT / "random" / MACHINE),
    ("M2", ROOT / "scent" / MACHINE),
    ("M1-low", ROOT / "random_low" / MACHINE),
    ("M2-low", ROOT / "scent_low" / MACHINE),
]
COND = dict(CONDITIONS)
SEEDS = list(range(30))
STEADY_STATE_MIN_GEN = 2000
N_BOOTSTRAP = 1000

OUT_JSON = ROOT / "analysis" / "results.json"
FIG_DIR = Path("docs/figures/scent_n1000")

# N=100 z*_K_self fixed in the pre-registration, section 5
N100_ZSTAR_KSELF = {"M0": (2.1254, 1.6481, 2.7984), "M1": (2.0160, 1.5424, 2.5726),
                    "M2": (1.8586, 1.2977, 2.5115)}

SUMMARY_NUM = ["beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout",
               "beta1", "beta2", "gamma11", "gamma22", "gamma12",
               "mean_beta1", "mean_beta2", "mean_gamma11", "mean_gamma22", "mean_gamma12",
               "var_kout", "var_kself", "bc_kout", "bc_kself"]


# ---------------------------------------------------------------- loading

def load_summary(root, seed):
    with open(root / f"seed_{seed}_summary.csv") as f:
        row = next(csv.DictReader(f))
    out = {k: (float(row[k]) if row[k] != "" else np.nan) for k in SUMMARY_NUM}
    out["n_qualifying_gens"] = int(row["n_qualifying_gens"])
    return out


def load_condition(root):
    summaries = [load_summary(root, s) for s in SEEDS]
    op = [n100.seed_mean_operational(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
    env = [n100.seed_mean_env(root, s, STEADY_STATE_MIN_GEN) for s in SEEDS]
    return dict(summary=summaries, op=op, env=env)


def check_complete():
    print("Section 0: completeness / pilot exclusion\n")
    ok = True
    for label, root in CONDITIONS:
        n = sum(1 for s in SEEDS if (root / f"seed_{s}_timeseries.csv").exists()
                and (root / f"seed_{s}_summary.csv").exists())
        print(f"  {label:>6}: {n}/30  ({root})")
        ok &= n == 30
    print("  pilot directories (*_pilot*) are not read by this script.\n")
    return ok


# ---------------------------------------------------------------- helpers

def col(data, label, key, src="summary"):
    return np.array([v[key] for v in data[label][src]], dtype=float)


def mw(x, y):
    stat, p, r = n100.mannwhitney_effect(x, y)
    return dict(U=float(stat), p=float(p), r=float(r),
                median_x=float(np.nanmedian(x)), median_y=float(np.nanmedian(y)), stars=n100.sig_stars(p))


def pi_check(x0, y_obs):
    y_hat, lo, hi = n100.predict_interval(x0)
    return dict(x0=float(x0), y_obs=float(y_obs), y_hat=float(y_hat), lo=float(lo), hi=float(hi),
                residual=float(y_obs - y_hat), in_pi=bool(lo <= y_obs <= hi))


def x0_y_obs(data, label):
    x0 = np.nanmedian(col(data, label, "k_out_assortativity", "op"))
    y = np.nanmedian(col(data, label, "beta_ctx_kout"))
    return x0, y


def zstar(summaries, rng):
    r = n100.fitness_peak_for_group(summaries)
    if len(r["valid_betas"]) == 0:
        return dict(n_excluded=r["n_excluded"], n_valid=0)
    gb, gg = r["valid_betas"].mean(axis=0), r["valid_gammas"].mean(axis=0)
    z = -np.linalg.solve(gg, gb)
    # same bootstrap as n100.bootstrap_grand_zstar, but with an explicit rng
    n = len(r["valid_betas"])
    b1, b2 = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        mg = r["valid_gammas"][idx].mean(axis=0)
        if not np.all(np.linalg.eigvalsh(mg) < 0):
            continue
        zz = -np.linalg.solve(mg, r["valid_betas"][idx].mean(axis=0))
        b1.append(zz[0])
        b2.append(zz[1])
    return dict(n_excluded=r["n_excluded"], n_valid=n,
                kself=float(z[0]), kself_ci=[float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))],
                kout=float(z[1]), kout_ci=[float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))])


def summary_to_n100_shape(s):
    """N=1000 summary row -> the dict shape n100.fitness_peak_for_group expects."""
    if np.isnan(s["mean_beta1"]):
        return dict(beta_mean=None, gamma_mean=None)
    return dict(beta_mean=np.array([s["mean_beta1"], s["mean_beta2"]]),
                gamma_mean=np.array([[s["mean_gamma11"], s["mean_gamma12"]],
                                     [s["mean_gamma12"], s["mean_gamma22"]]]))


def ols(y, X, names):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tc = t_dist.ppf(0.975, dof)
    out = {}
    for n, c, e in zip(names, coef, se):
        tv = c / e
        out[n] = dict(coef=float(c), se=float(e), ci=[float(c - tc * e), float(c + tc * e)],
                      t=float(tv), p=float(2 * t_dist.sf(abs(tv), dof)))
    return out


def med_iqr(v):
    v = np.asarray(v, dtype=float)
    return [float(np.nanmedian(v)), float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75))]


# ---------------------------------------------------------------- N=100 recomputation (comparison only)

def compute_n100():
    """Recompute the N=100 production numbers from the committed raw logs
    with the N=100 script's own functions. Order of the bootstrap calls
    matches analyze_scent_production.section_C (M0, M1, M2 on a fresh
    default_rng(0)) so the z* CIs reproduce the pre-registration table."""
    print("Recomputing N=100 values from committed raw logs (comparison only)...", flush=True)
    n100.RNG = np.random.default_rng(0)
    out = {}
    for label, root in n100.CONDITIONS:
        ctx = [n100.seed_median_context_betas(root, s, n100.STEADY_STATE_MIN_GEN) for s in n100.SEEDS]
        op = [n100.seed_mean_operational(root, s, n100.STEADY_STATE_MIN_GEN) for s in n100.SEEDS]
        env = [n100.seed_mean_env(root, s, n100.STEADY_STATE_MIN_GEN) for s in n100.SEEDS]
        summ = [n100.seed_summary(root, s, n100.STEADY_STATE_MIN_GEN) for s in n100.SEEDS]
        vb = [n100.seed_median_variance_bimodality(root, s, n100.STEADY_STATE_MIN_GEN) for s in n100.SEEDS]
        out[label] = dict(ctx=ctx, op=op, env=env, summ=summ, vb=vb)
    for label, _ in n100.CONDITIONS:
        r = n100.fitness_peak_for_group(out[label]["summ"])
        gz = -np.linalg.solve(r["valid_gammas"].mean(axis=0), r["valid_betas"].mean(axis=0))
        b1, b2, _ = n100.bootstrap_grand_zstar(r["valid_betas"], r["valid_gammas"])
        out[label]["zstar"] = dict(
            n_excluded=r["n_excluded"], n_valid=len(r["valid_betas"]),
            kself=float(gz[0]), kself_ci=[float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))],
            kout=float(gz[1]), kout_ci=[float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))])
    return out


def n100_metrics(d):
    res = {}
    for label in ["M0", "M1", "M2"]:
        c = d[label]
        res[label] = dict(
            beta_ctx_kout=float(np.nanmedian([v["beta_ctx_kout"] for v in c["ctx"]])),
            beta_ctx_kself=float(np.nanmedian([v["beta_ctx_kself"] for v in c["ctx"]])),
            beta_self_kout=float(np.nanmedian([v["beta_self_kout"] for v in c["ctx"]])),
            beta_self_kself=float(np.nanmedian([v["beta_self_kself"] for v in c["ctx"]])),
            k_out_assortativity=float(np.nanmedian([v["k_out_assortativity"] for v in c["op"]])),
            same_parent_fraction=float(np.nanmedian([v["same_parent_fraction"] for v in c["op"]])),
            move_rate=float(np.nanmedian([v["move_rate"] for v in c["op"]])),
            cooperation_rate=float(np.nanmedian([v["cooperation_rate"] for v in c["env"]])),
            mean_k_out=float(np.nanmedian([v["mean_k_out"] for v in c["env"]])),
            mean_k_self=float(np.nanmedian([v["mean_k_self"] for v in c["env"]])),
            var_kout=float(np.nanmedian([v["var_kout"] for v in c["vb"]])),
            var_kself=float(np.nanmedian([v["var_kself"] for v in c["vb"]])),
            zstar=c["zstar"],
        )
        bb = d[label]
        res[label]["_assort_per_seed"] = [v["k_out_assortativity"] for v in bb["op"]]
        res[label]["_beta_ctx_kout_per_seed"] = [v["beta_ctx_kout"] for v in bb["ctx"]]
    a = mw([v["beta_ctx_kout"] for v in d["M2"]["ctx"]], [v["beta_ctx_kout"] for v in d["M1"]["ctx"]])
    b = mw([v["k_out_assortativity"] for v in d["M2"]["op"]], [v["k_out_assortativity"] for v in d["M1"]["op"]])
    pis = {}
    for label in ["M0", "M1", "M2"]:
        pis[label] = pi_check(res[label]["k_out_assortativity"], res[label]["beta_ctx_kout"])
    return dict(per_cond=res, A=a, Ba=b, Bb=pis)


# ---------------------------------------------------------------- isolated-agent sensitivity (exploratory)
# An agent with zero duels in a generation has opponent_mean_k_* = NaN.
# In context_lande_arnold, np.std() of a column containing NaN is NaN,
# `NaN > 0` is False, so standardize() returns an all-zero column and
# BOTH context coefficients come out exactly 0 for that generation. This
# is the computation the pre-registration fixed (identical at N=100 and
# N=1000). The variant below drops the isolated agents from that
# generation's regression instead -- exploratory, NOT pre-registered.

CTX_NAMES = ["beta_self_kself", "beta_self_kout", "beta_ctx_kself", "beta_ctx_kout"]
VARIANTS = ["as_registered", "isolated_excluded"]


def _iter_individual_rows(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        yield from csv.DictReader(f)


def seed_ctx_both_variants(path, min_gen=STEADY_STATE_MIN_GEN):
    by_gen = {}
    for row in _iter_individual_rows(path):
        gen = int(row["generation"])
        if gen < min_gen:
            continue
        g = by_gen.setdefault(gen, {k: [] for k in ["k_out", "k_self", "payoff", "opp_k_self", "opp_k_out"]})
        g["k_out"].append(float(row["k_out"]))
        g["k_self"].append(float(row["k_self"]))
        g["payoff"].append(float(row["payoff"]))
        g["opp_k_self"].append(float(row["opponent_mean_k_self"]))
        g["opp_k_out"].append(float(row["opponent_mean_k_out"]))
    keys = ["k_self", "k_out", "payoff", "opp_k_self", "opp_k_out"]

    def degenerate(a):
        # the registered rule (fewer than 2 distinct values in any of the 5
        # variables), applied to whichever arrays that variant regresses on
        return any(len(set(a[k].tolist())) < 2 for k in keys)

    as_reg, excl = [], []
    n_gens_with_isolated = 0
    for gen, g in sorted(by_gen.items()):
        arr = {k: np.asarray(g[k]) for k in keys}
        if not degenerate(arr):
            as_reg.append(n100.context_lande_arnold(arr["k_self"], arr["k_out"], arr["opp_k_self"],
                                                    arr["opp_k_out"], arr["payoff"]))
        keep = ~(np.isnan(arr["opp_k_out"]) | np.isnan(arr["opp_k_self"]))
        if not keep.all():
            n_gens_with_isolated += 1
        kept = {k: v[keep] for k, v in arr.items()}
        if not degenerate(kept):
            excl.append(n100.context_lande_arnold(kept["k_self"], kept["k_out"], kept["opp_k_self"],
                                                  kept["opp_k_out"], kept["payoff"]))
    med = lambda rows: {n: float(np.median([r[i] for r in rows])) if rows else np.nan
                        for i, n in enumerate(CTX_NAMES)}
    return dict(as_registered=med(as_reg), isolated_excluded=med(excl),
                n_gens=len(by_gen), n_gens_as_registered=len(as_reg), n_gens_isolated_excluded=len(excl),
                n_gens_with_isolated=n_gens_with_isolated)


def run_isolated_sensitivity(conditions, seeds, filename):
    """conditions: [(label, root)]; returns per-condition per-seed results,
    or None for a condition whose raw files are not all present."""
    out = {}
    for label, root in conditions:
        paths = [Path(root) / filename.format(seed=s) for s in seeds]
        if not all(p.exists() for p in paths):
            out[label] = None
            continue
        out[label] = [seed_ctx_both_variants(p) for p in paths]
    return out


def deviation_analysis(sens, op_by_cond, pairs, bb_conditions, ancova_pair):
    """Analyses fixed by docs/DEVIATION_isolated_agents.md, both variants side
    by side. sens: {label: [per-seed seed_ctx_both_variants] or None};
    op_by_cond: {label: [per-seed seed_mean_operational]} (same seed order)."""
    out = dict(per_seed={}, per_condition={}, tests={}, Bb={}, ancova_beta_ctx_kout={})
    avail = {l: v for l, v in sens.items() if v is not None}
    out["missing_raw_logs"] = sorted(l for l, v in sens.items() if v is None)
    for label, seeds in avail.items():
        out["per_seed"][label] = seeds
        op = op_by_cond[label]
        out["per_condition"][label] = dict(
            isolated_fraction_mean=float(np.nanmean([o["isolated_fraction"] for o in op])),
            gens_with_isolated_fraction=float(sum(s["n_gens_with_isolated"] for s in seeds)
                                              / sum(s["n_gens"] for s in seeds)),
            **{f"{n}_{v}": float(np.nanmedian([s[v][n] for s in seeds])) for v in VARIANTS for n in CTX_NAMES},
        )
    for a, b in pairs:
        if a in avail and b in avail:
            out["tests"][f"{a} vs {b}"] = {
                v: mw([s[v]["beta_ctx_kout"] for s in avail[a]], [s[v]["beta_ctx_kout"] for s in avail[b]])
                for v in VARIANTS}
    for label in bb_conditions:
        if label in avail:
            x0 = float(np.nanmedian([o["k_out_assortativity"] for o in op_by_cond[label]]))
            out["Bb"][label] = {v: pi_check(x0, np.nanmedian([s[v]["beta_ctx_kout"] for s in avail[label]]))
                                for v in VARIANTS}
    m1, m2 = ancova_pair
    if m1 in avail and m2 in avail:
        mv = np.array([o["move_rate"] for o in op_by_cond[m1]] + [o["move_rate"] for o in op_by_cond[m2]])
        isc = np.concatenate([np.zeros(len(avail[m1])), np.ones(len(avail[m2]))])
        for v in VARIANTS:
            y = np.array([s[v]["beta_ctx_kout"] for s in avail[m1]] + [s[v]["beta_ctx_kout"] for s in avail[m2]])
            out["ancova_beta_ctx_kout"][v] = ols(y, np.column_stack([np.ones_like(y), isc, mv]),
                                                 ["intercept", "is_scent", "move_rate"])
    return out


# ---------------------------------------------------------------- figures

PALETTE = {"M0": "#52514e", "M1": "#2a78d6", "M2": "#eb6834", "M1-low": "#1baf7a", "M2-low": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strip_figure(data, key, src, ylabel, title, path, tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    labels = [l for l, _ in CONDITIONS]
    for i, label in enumerate(labels):
        v = col(data, label, key, src)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=PALETTE[label], alpha=0.75,
                   edgecolors="white", linewidths=0.5, zorder=3)
        m = np.nanmedian(v)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, linewidth=2, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axvline(2.5, color=GRID, linewidth=1)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    _style(ax)
    ymax = ax.get_ylim()[1]
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for k, (i, j, txt) in enumerate(tests):
        y = ymax + span * (0.04 + 0.09 * k)
        ax.plot([i, i, j, j], [y - span * 0.02, y, y, y - span * 0.02], color=INK2, linewidth=1)
        ax.text((i + j) / 2, y + span * 0.01, txt, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(top=ymax + span * (0.06 + 0.09 * len(tests)))
    ax.text(0.99, -0.13, "dots = seeds (n=30 per condition, gen>=2000); bar = median",
            transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def prediction_figure(pis1000, pis100, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    xs = np.linspace(0, 0.75, 200)
    yh, lo, hi = zip(*[n100.predict_interval(x) for x in xs])
    ax.fill_between(xs, lo, hi, color="#d9d8d3", alpha=0.6, linewidth=0, label="ring-model 95% PI")
    ax.plot(xs, yh, color=INK2, linewidth=2, label=r"$\hat{y}(x) = -0.00016799 - 0.00719578x$")
    # hand-placed label offsets (points), so the three y=0 points at x~0.39-0.44 don't collide
    off1000 = {"M0": (8, 10), "M1": (8, 6), "M2": (8, 6), "M1-low": (-4, 22), "M2-low": (-60, 10)}
    off100 = {"M0": (8, -12), "M1": (8, 4), "M2": (-62, -14)}
    for label, r in pis1000.items():
        artifact = abs(r["y_obs"]) == 0.0
        ax.scatter([r["x0"]], [r["y_obs"]], s=60, color=PALETTE[label], edgecolors="white", linewidths=1.5,
                   marker="X" if artifact else "o", zorder=4)
        ax.annotate(f"{label} (N=1000)" + ("*" if artifact else ""), (r["x0"], r["y_obs"]),
                    xytext=off1000.get(label, (6, 4)), textcoords="offset points", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
    for label, r in pis100.items():
        ax.scatter([r["x0"]], [r["y_obs"]], s=50, facecolors="white", edgecolors=PALETTE[label],
                   linewidths=1.8, marker="s", zorder=4)
        ax.annotate(f"{label} (N=100)", (r["x0"], r["y_obs"]), xytext=off100.get(label, (6, -10)),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.text(0.99, 0.02, "* X marker: beta_ctx forced to exactly 0 by isolated agents (NaN context) "
            "in most snapshot generations", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("K_out assortativity (seed median of per-seed means, gen>=2000)", color=INK2, fontsize=9)
    ax.set_ylabel("beta_ctx(K_out) (seed median)", color=INK2, fontsize=9)
    ax.set_title("Ring-model prediction vs observed (filled = N=1000, open square = N=100)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.06))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def zstar_figure(z1000, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for i, label in enumerate(["M0", "M1", "M2"]):
        v, lo, hi = N100_ZSTAR_KSELF[label]
        ax.errorbar([i - 0.12], [v], yerr=[[v - lo], [hi - v]], fmt="s", color=INK2, mfc="white",
                    ms=7, capsize=3, linewidth=1.5, label="N=100 (pre-registered table)" if i == 0 else None)
        z = z1000[label]
        ax.errorbar([i + 0.12], [z["kself"]], yerr=[[z["kself"] - z["kself_ci"][0]], [z["kself_ci"][1] - z["kself"]]],
                    fmt="o", color=PALETTE[label], ms=7, capsize=3, linewidth=1.5,
                    label="N=1000" if i == 0 else None)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["M0", "M1", "M2"])
    ax.set_ylabel("z*_K_self (SD units from population mean)", color=INK2, fontsize=9)
    ax.set_title("Fitness peak z*_K_self, grand mean + bootstrap 95% CI", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _p_label(name, r):
    return f"{name}: p={r['p']:.3g}" if np.isfinite(r["p"]) else f"{name}: not evaluable (all values 0)"


def fmt_mw(name, r):
    return (f"{name}: median {r['median_x']:+.6f} vs {r['median_y']:+.6f}, U={r['U']:.1f}, "
            f"p={r['p']:.6g}, r={r['r']:+.4f} {r['stars']}")


def main():
    if not check_complete():
        print("Not complete. Stopping.")
        return
    data = {label: load_condition(root) for label, root in CONDITIONS}
    res = {}

    # sanity: qualifying generations per seed
    res["n_qualifying_gens"] = {l: sorted({s["n_qualifying_gens"] for s in data[l]["summary"]}) for l in COND}
    print("n_qualifying_gens per seed:", res["n_qualifying_gens"])

    # diagnostic: snapshot generations (gen>=2000, every LOG_INTERVAL=100) with
    # >=1 isolated agent -- in those, the registered context regression returns
    # beta_ctx == 0 exactly (see isolated-agent sensitivity section)
    res["zeroed_snapshot_gens"] = {}
    for label, root in CONDITIONS:
        hit = tot = seeds_majority = 0
        for s in SEEDS:
            rows = n100.load_timeseries(root / f"seed_{s}_timeseries.csv")
            snap = [float(r["isolated_fraction"]) > 0 for r in rows
                    if int(r["generation"]) >= STEADY_STATE_MIN_GEN and int(r["generation"]) % 100 == 0]
            hit += sum(snap)
            tot += len(snap)
            seeds_majority += sum(snap) > len(snap) / 2
        res["zeroed_snapshot_gens"][label] = dict(n=hit, total=tot, fraction=hit / tot,
                                                  seeds_with_majority_zeroed=seeds_majority)
    print("snapshot gens with >=1 isolated agent (beta_ctx forced to 0):", res["zeroed_snapshot_gens"])

    # ---------------- section 3: group 2
    print("\n=== Section 3, group 2 (M1-low vs M2-low) ===")
    a2 = mw(col(data, "M2-low", "beta_ctx_kout"), col(data, "M1-low", "beta_ctx_kout"))
    print("[PRIMARY A2] beta_ctx(K_out) " + fmt_mw("M2-low vs M1-low", a2))
    ba2 = mw(col(data, "M2-low", "k_out_assortativity", "op"), col(data, "M1-low", "k_out_assortativity", "op"))
    print("[SECONDARY B(a)-equiv] K_out assort " + fmt_mw("M2-low vs M1-low", ba2))
    bb2 = {l: pi_check(*x0_y_obs(data, l)) for l in ["M1-low", "M2-low"]}
    for l, r in bb2.items():
        print(f"[SECONDARY B(b)-equiv] {l}: x0={r['x0']:.6f} y={r['y_obs']:+.6f} y_hat={r['y_hat']:+.6f} "
              f"PI=[{r['lo']:+.6f},{r['hi']:+.6f}] in={r['in_pi']}")
    movefreq = {}
    for l in ["M2", "M2-low", "M1", "M1-low"]:
        movefreq[l] = dict(
            k_out_assortativity=med_iqr(col(data, l, "k_out_assortativity", "op")),
            beta_ctx_kout=med_iqr(col(data, l, "beta_ctx_kout")),
            move_rate=med_iqr(col(data, l, "move_rate", "op")),
            cooperation_rate=med_iqr(col(data, l, "cooperation_rate", "env")),
        )
    print("[DESCRIPTIVE] movement frequency:", json.dumps(movefreq, indent=1))
    res["section3"] = dict(A2=a2, Ba=ba2, Bb=bb2, movement_frequency=movefreq)

    # ---------------- section 5 + C z*
    rng = np.random.default_rng(0)
    z = {l: zstar([summary_to_n100_shape(s) for s in data[l]["summary"]], rng) for l, _ in CONDITIONS}
    print("\n=== Section 5: z*_K_self vs N=100 ===")
    for l in ["M0", "M1", "M2"]:
        print(f"  {l}: N=1000 {z[l]['kself']:+.4f} {z[l]['kself_ci']}  vs N=100 {N100_ZSTAR_KSELF[l]}")
    res["zstar"] = z

    # ---------------- section 6: group 1
    print("\n=== Section 6, group 1 ===")
    a = mw(col(data, "M2", "beta_ctx_kout"), col(data, "M1", "beta_ctx_kout"))
    print("[PRIMARY A] beta_ctx(K_out) " + fmt_mw("M2 vs M1", a))
    ba = mw(col(data, "M2", "k_out_assortativity", "op"), col(data, "M1", "k_out_assortativity", "op"))
    print("[SECONDARY B(a)] K_out assort " + fmt_mw("M2 vs M1", ba))
    bb = {l: pi_check(*x0_y_obs(data, l)) for l in ["M0", "M1", "M2"]}
    for l, r in bb.items():
        print(f"[SECONDARY B(b)] {l}: x0={r['x0']:.6f} y={r['y_obs']:+.6f} y_hat={r['y_hat']:+.6f} "
              f"PI=[{r['lo']:+.6f},{r['hi']:+.6f}] in={r['in_pi']}")

    # B(c): ring local + N=100 M0 recomputed
    ring = {}
    for k in n100.RING_K_VALUES:
        rr = n100.RING_LOCAL_ROOT / f"k{k}"
        av, bv = [], []
        for s in n100.SEEDS:
            rows = n100.load_timeseries(rr / f"seed_{s}_timeseries.csv")
            aa = [float(x["k_out_assortativity"]) for x in rows
                  if int(x["generation"]) >= STEADY_STATE_MIN_GEN and x["k_out_assortativity"] != ""]
            av.append(np.median(aa) if aa else np.nan)
            bv.append(n100.seed_median_context_betas(rr, s, STEADY_STATE_MIN_GEN)["beta_ctx_kout"])
        ring[k] = dict(k_out_assortativity=float(np.nanmedian(av)), beta_ctx_kout=float(np.nanmedian(bv)))
    n100d = compute_n100()
    n100m = n100_metrics(n100d)
    bc = dict(
        ring=ring,
        m0_vs_m1={l: dict(k_out_assortativity=bb[l]["x0"], beta_ctx_kout=bb[l]["y_obs"],
                          cooperation_rate=float(np.nanmedian(col(data, l, "cooperation_rate", "env"))))
                  for l in ["M0", "M1"]},
    )
    print("[SECONDARY B(c)]", json.dumps(bc, indent=1))

    # C descriptive
    desc = {}
    for l, _ in CONDITIONS:
        desc[l] = {k: float(np.nanmedian(col(data, l, k))) for k in SUMMARY_NUM if not k.startswith("mean_")}
        desc[l]["env"] = {k: med_iqr(col(data, l, k, "env")) for k in ["cooperation_rate", "mean_k_out", "mean_k_self"]}
        desc[l]["op"] = {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None)
                         for k in n100.OP_COLS for v in [col(data, l, k, "op")]}
    print("[DESCRIPTIVE C]", json.dumps(desc, indent=1))
    res["section6"] = dict(A=a, Ba=ba, Bb=bb, Bc=bc, C=desc)

    # ---------------- exploratory / not pre-registered
    print("\n=== EXPLORATORY (not pre-registered) ===")
    allpi = {l: pi_check(*x0_y_obs(data, l)) for l, _ in CONDITIONS}
    res_vals = np.array([r["residual"] for r in allpi.values()])
    print("ring prediction, all 5:", {l: (round(r["residual"], 6), r["in_pi"]) for l, r in allpi.items()})
    print(f"  mean residual {res_vals.mean():+.6f}, mean |residual| {np.abs(res_vals).mean():.6f}")

    ancova = {}
    for name, (m1, m2) in {"group1": ("M1", "M2"), "group2": ("M1-low", "M2-low")}.items():
        y = np.concatenate([col(data, m1, "k_out_assortativity", "op"), col(data, m2, "k_out_assortativity", "op")])
        mv = np.concatenate([col(data, m1, "move_rate", "op"), col(data, m2, "move_rate", "op")])
        isc = np.concatenate([np.zeros(30), np.ones(30)])
        ancova[name] = ols(y, np.column_stack([np.ones_like(y), isc, mv]), ["intercept", "is_scent", "move_rate"])
        print(f"  ANCOVA {name}:", {k: (round(v['coef'], 6), [round(c, 6) for c in v['ci']], f"{v['p']:.3g}")
                                    for k, v in ancova[name].items()})
    res["exploratory"] = dict(prediction_all5=allpi, residual_mean=float(res_vals.mean()),
                              residual_abs_mean=float(np.abs(res_vals).mean()),
                              ancova=ancova)

    # ---------------- docs/DEVIATION_isolated_agents.md: both variants side by side
    print("\n=== Isolated-agent deviation (docs/DEVIATION_isolated_agents.md) ===")
    dev100 = deviation_analysis(
        run_isolated_sensitivity(n100.CONDITIONS, n100.SEEDS, "seed_{seed}_individual_log.csv"),
        {l: n100d[l]["op"] for l, _ in n100.CONDITIONS},
        pairs=[("M2", "M1")], bb_conditions=["M0", "M1", "M2"], ancova_pair=("M1", "M2"))
    dev1000 = deviation_analysis(
        run_isolated_sensitivity(CONDITIONS, SEEDS, "seed_{seed}_individual_log.csv.gz"),
        {l: data[l]["op"] for l, _ in CONDITIONS},
        pairs=[("M2", "M1"), ("M2-low", "M1-low")], bb_conditions=[l for l, _ in CONDITIONS],
        ancova_pair=("M1", "M2"))
    for name, dev in [("N=100", dev100), ("N=1000", dev1000)]:
        if dev["missing_raw_logs"]:
            print(f"  {name}: raw individual logs not in this checkout for {dev['missing_raw_logs']} "
                  f"(N=1000's are local-only on the run machine) -- rerun this script there.")
        print(f"  {name} per condition:", json.dumps(dev["per_condition"], indent=1))
        for t, r in dev["tests"].items():
            for v in VARIANTS:
                print(f"  {name} [{v}] " + fmt_mw(t, r[v]))
        for l, r in dev["Bb"].items():
            for v in VARIANTS:
                print(f"  {name} B(b) {l} [{v}]: y={r[v]['y_obs']:+.6f} in_pi={r[v]['in_pi']}")
        for v, fit in dev["ancova_beta_ctx_kout"].items():
            print(f"  {name} ANCOVA beta_ctx(K_out) group 1 [{v}]:",
                  {k: (round(x["coef"], 6), [round(c, 6) for c in x["ci"]], f"{x['p']:.3g}") for k, x in fit.items()})
    res["deviation_isolated"] = dict(n100=dev100, n1000=dev1000)

    # N=100 comparison
    n100_clean = {k: v for k, v in n100m.items()}
    for l in n100_clean["per_cond"].values():
        l.pop("_assort_per_seed")
        l.pop("_beta_ctx_kout_per_seed")
    res["n100"] = n100_clean
    print("\nN=100 recomputed:", json.dumps({k: n100_clean[k] for k in ["A", "Ba", "Bb"]}, indent=1))
    for l in ["M0", "M1", "M2"]:
        print(l, json.dumps(n100_clean["per_cond"][l]))

    # ---------------- outputs
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(_json_safe(res), f, indent=2, ensure_ascii=False)
        f.write("\n")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    strip_figure(data, "beta_ctx_kout", "summary", "beta_ctx(K_out) (per-seed median)",
                 "beta_ctx(K_out) by condition, N=1000", FIG_DIR / "fig1_beta_ctx_kout.png",
                 [(1, 2, _p_label("A", a)), (3, 4, _p_label("A2", a2))])
    strip_figure(data, "k_out_assortativity", "op", "K_out assortativity (per-seed mean)",
                 "K_out assortativity of actually-duelled pairs, N=1000", FIG_DIR / "fig2_kout_assortativity.png",
                 [(1, 2, _p_label("B(a)", ba)), (3, 4, _p_label("B(a)-equiv", ba2))])
    prediction_figure(allpi, n100_clean["Bb"], FIG_DIR / "fig3_ring_prediction.png")
    zstar_figure(z, FIG_DIR / "fig4_zstar_kself.png")
    print(f"\nwrote {OUT_JSON} and {FIG_DIR}/*.png")


if __name__ == "__main__":
    main()
