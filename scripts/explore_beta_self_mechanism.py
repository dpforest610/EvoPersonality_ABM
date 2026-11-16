"""Exploratory, NOT pre-registered. Follow-up on
docs/SCENT_MODEL_N1000_RESULTS.md section E1 (isolated-agent sensitivity).

Five questions, tables only, no interpretation:
  1. beta_self(K_self/K_out) change under isolated-exclusion: split into the
     sample-composition effect (fewer individuals) vs. the specification
     effect (context columns going from degenerate-zero to real, i.e. the
     4-variable regression now actually includes the context predictors) by
     also fitting a self-only 2-variable regression on both samples.
  2. beta_ctx(K_self) seed-median, both variants, all 8 conditions.
  3. corr(own, opponent-mean) for K_self and K_out, and VIF of the 4
     context-regression predictors, per condition (isolated-excluded sample
     -- the only one in which opponent-mean is defined for every row).
  4. Queller-type total selection beta_self + r*beta_ctx (r = that trait's
     own assortativity), K_self and K_out, both variants, all 8 conditions.
  5. ring-model local-placement point(s) near assortativity~=0.4, beta_self
     and beta_ctx, lined up against the N=100/N=1000 M0 points.

Writes scripts/output/scent_spatial_sweep_n1000/analysis/beta_self_mechanism.json
and prints all tables.

Usage:
    python3 scripts/explore_beta_self_mechanism.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_scent_production as n100  # noqa: E402
import analyze_scent_production_n1000 as n1000mod  # noqa: E402

OUT_JSON = Path("scripts/output/scent_spatial_sweep_n1000/analysis/beta_self_mechanism.json")

# (population, label, root, seeds, filename_template)
GROUPS = (
    [("N=100", l, r, n100.SEEDS, "seed_{seed}_individual_log.csv") for l, r in n100.CONDITIONS]
    + [("N=1000", l, r, n1000mod.SEEDS, "seed_{seed}_individual_log.csv.gz") for l, r in n1000mod.CONDITIONS]
)
MIN_GEN = n100.STEADY_STATE_MIN_GEN
assert MIN_GEN == n1000mod.STEADY_STATE_MIN_GEN


def standardize(v):
    v = np.asarray(v, dtype=float)
    sd = np.std(v)
    return (v - np.mean(v)) / sd if sd > 0 else np.zeros_like(v)


def self_only_regression(k_self, k_out, payoff):
    z1, z2 = standardize(k_self), standardize(k_out)
    w = np.asarray(payoff) / np.mean(payoff)
    X = np.column_stack([np.ones_like(z1), z1, z2])
    coef, _, _, _ = np.linalg.lstsq(X, w, rcond=None)
    return float(coef[1]), float(coef[2])


def vif_of(cols):
    """cols: list of 4 standardized predictor arrays (already z-scored).
    VIF_j = 1/(1-R2_j), R2_j from regressing predictor j on the other 3 + intercept."""
    X = np.column_stack(cols)
    n = X.shape[0]
    out = []
    for j in range(4):
        y = X[:, j]
        others = np.column_stack([np.ones(n)] + [X[:, k] for k in range(4) if k != j])
        coef, _, _, _ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(float("inf") if r2 >= 1 else 1.0 / (1.0 - r2))
    return out  # [kself, kout, ctx_kself, ctx_kout]


def seed_mechanism(path, min_gen):
    keys = ["k_self", "k_out", "payoff", "opp_k_self", "opp_k_out"]
    by_gen = {}
    for row in n1000mod._iter_individual_rows(path):
        gen = int(row["generation"])
        if gen < min_gen:
            continue
        g = by_gen.setdefault(gen, {k: [] for k in keys})
        g["k_out"].append(float(row["k_out"]))
        g["k_self"].append(float(row["k_self"]))
        g["payoff"].append(float(row["payoff"]))
        g["opp_k_self"].append(float(row["opponent_mean_k_self"]))
        g["opp_k_out"].append(float(row["opponent_mean_k_out"]))

    def degenerate(a):
        return any(len(set(a[k].tolist())) < 2 for k in keys)

    full4, full2, excl4, excl2 = [], [], [], []
    corr_kself, corr_kout, vifs = [], [], []
    for gen, g in sorted(by_gen.items()):
        arr = {k: np.asarray(g[k], dtype=float) for k in keys}
        if not degenerate(arr):
            full4.append(n100.context_lande_arnold(arr["k_self"], arr["k_out"], arr["opp_k_self"],
                                                     arr["opp_k_out"], arr["payoff"]))
            full2.append(self_only_regression(arr["k_self"], arr["k_out"], arr["payoff"]))
        keep = ~(np.isnan(arr["opp_k_out"]) | np.isnan(arr["opp_k_self"]))
        kept = {k: v[keep] for k, v in arr.items()}
        if not degenerate(kept):
            excl4.append(n100.context_lande_arnold(kept["k_self"], kept["k_out"], kept["opp_k_self"],
                                                     kept["opp_k_out"], kept["payoff"]))
            excl2.append(self_only_regression(kept["k_self"], kept["k_out"], kept["payoff"]))
            if np.std(kept["k_self"]) > 0 and np.std(kept["opp_k_self"]) > 0:
                corr_kself.append(float(np.corrcoef(kept["k_self"], kept["opp_k_self"])[0, 1]))
            if np.std(kept["k_out"]) > 0 and np.std(kept["opp_k_out"]) > 0:
                corr_kout.append(float(np.corrcoef(kept["k_out"], kept["opp_k_out"])[0, 1]))
            z1s, z2s = standardize(kept["k_self"]), standardize(kept["k_out"])
            z1c, z2c = standardize(kept["opp_k_self"]), standardize(kept["opp_k_out"])
            vifs.append(vif_of([z1s, z2s, z1c, z2c]))

    def med4(rows, idx):
        return float(np.median([r[idx] for r in rows])) if rows else float("nan")

    def med2(rows, idx):
        return float(np.median([r[idx] for r in rows])) if rows else float("nan")

    vif_arr = np.array(vifs) if vifs else np.full((0, 4), np.nan)
    return dict(
        full4_kself=med4(full4, 0), full4_kout=med4(full4, 1),
        full4_ctx_kself=med4(full4, 2), full4_ctx_kout=med4(full4, 3),
        full2_kself=med2(full2, 0), full2_kout=med2(full2, 1),
        excl4_kself=med4(excl4, 0), excl4_kout=med4(excl4, 1),
        excl4_ctx_kself=med4(excl4, 2), excl4_ctx_kout=med4(excl4, 3),
        excl2_kself=med2(excl2, 0), excl2_kout=med2(excl2, 1),
        corr_kself=float(np.median(corr_kself)) if corr_kself else float("nan"),
        corr_kout=float(np.median(corr_kout)) if corr_kout else float("nan"),
        vif_kself=float(np.median(vif_arr[:, 0])) if len(vif_arr) else float("nan"),
        vif_kout=float(np.median(vif_arr[:, 1])) if len(vif_arr) else float("nan"),
        vif_ctx_kself=float(np.median(vif_arr[:, 2])) if len(vif_arr) else float("nan"),
        vif_ctx_kout=float(np.median(vif_arr[:, 3])) if len(vif_arr) else float("nan"),
    )


def main():
    per_cond = {}
    for pop, label, root, seeds, tmpl in GROUPS:
        key = f"{pop} {label}"
        print(f"computing {key} ...", flush=True)
        rows = [seed_mechanism(Path(root) / tmpl.format(seed=s), MIN_GEN) for s in seeds]
        agg = {f: float(np.median([r[f] for r in rows])) for f in rows[0]}
        per_cond[key] = agg

    print("\n=== Q1: full-population vs isolated-excluded, 4-var vs self-only-2-var ===")
    print(f"{'condition':<14}{'full4_Kself':>12}{'full2_Kself':>12}{'excl2_Kself':>12}{'excl4_Kself':>12}  |  "
          f"{'full4_Kout':>11}{'full2_Kout':>11}{'excl2_Kout':>11}{'excl4_Kout':>11}")
    for key, v in per_cond.items():
        print(f"{key:<14}{v['full4_kself']:>+12.6f}{v['full2_kself']:>+12.6f}{v['excl2_kself']:>+12.6f}"
              f"{v['excl4_kself']:>+12.6f}  |  {v['full4_kout']:>+11.6f}{v['full2_kout']:>+11.6f}"
              f"{v['excl2_kout']:>+11.6f}{v['excl4_kout']:>+11.6f}")

    print("\n=== Q2: beta_ctx(K_self), both variants, all 8 conditions ===")
    for key, v in per_cond.items():
        print(f"{key:<14} as_registered={v['full4_ctx_kself']:+.6f}  isolated_excluded={v['excl4_ctx_kself']:+.6f}")

    print("\n=== Q3: corr(own, opponent-mean) and VIF (isolated-excluded sample) ===")
    for key, v in per_cond.items():
        print(f"{key:<14} corr_Kself={v['corr_kself']:+.4f} corr_Kout={v['corr_kout']:+.4f}  "
              f"VIF[Kself={v['vif_kself']:.3f} Kout={v['vif_kout']:.3f} "
              f"ctxKself={v['vif_ctx_kself']:.3f} ctxKout={v['vif_ctx_kout']:.3f}]")

    # assortativity (r) per condition, both pops -- condition-level median of the op column
    print("\ncomputing assortativity (r) for Queller total selection ...", flush=True)
    r_vals = {}
    for label, root in n100.CONDITIONS:
        op = [n100.seed_mean_operational(root, s, MIN_GEN) for s in n100.SEEDS]
        r_vals[f"N=100 {label}"] = dict(
            r_kself=float(np.nanmedian([o["k_self_assortativity"] for o in op])),
            r_kout=float(np.nanmedian([o["k_out_assortativity"] for o in op])))
    for label, root in n1000mod.CONDITIONS:
        op = [n100.seed_mean_operational(root, s, MIN_GEN) for s in n1000mod.SEEDS]
        r_vals[f"N=1000 {label}"] = dict(
            r_kself=float(np.nanmedian([o["k_self_assortativity"] for o in op])),
            r_kout=float(np.nanmedian([o["k_out_assortativity"] for o in op])))

    print("\n=== Q4: Queller-type total selection beta_self + r*beta_ctx ===")
    total_sel = {}
    for key, v in per_cond.items():
        r = r_vals[key]
        ts = dict(
            as_registered_kself=v["full4_kself"] + r["r_kself"] * v["full4_ctx_kself"],
            isolated_excluded_kself=v["excl4_kself"] + r["r_kself"] * v["excl4_ctx_kself"],
            as_registered_kout=v["full4_kout"] + r["r_kout"] * v["full4_ctx_kout"],
            isolated_excluded_kout=v["excl4_kout"] + r["r_kout"] * v["excl4_ctx_kout"],
        )
        total_sel[key] = ts
        print(f"{key:<14} K_self: as_reg={ts['as_registered_kself']:+.6f} excl={ts['isolated_excluded_kself']:+.6f}"
              f"  (r_Kself={r['r_kself']:.4f})  |  K_out: as_reg={ts['as_registered_kout']:+.6f} "
              f"excl={ts['isolated_excluded_kout']:+.6f} (r_Kout={r['r_kout']:.4f})")

    print("\n=== Q5: ring-model local, assortativity ~ 0.4, vs N=100/N=1000 M0 ===")
    ring = {}
    for k in n100.RING_K_VALUES:
        rr = n100.RING_LOCAL_ROOT / f"k{k}"
        av = []
        betas = []
        for s in n100.SEEDS:
            rows_ts = n100.load_timeseries(rr / f"seed_{s}_timeseries.csv")
            aa = [float(x["k_out_assortativity"]) for x in rows_ts
                  if int(x["generation"]) >= MIN_GEN and x["k_out_assortativity"] != ""]
            av.append(np.median(aa) if aa else np.nan)
            betas.append(n100.seed_median_context_betas(rr, s, MIN_GEN))
        ring[k] = dict(
            k_out_assortativity=float(np.nanmedian(av)),
            beta_self_kself=float(np.nanmedian([b["beta_self_kself"] for b in betas])),
            beta_self_kout=float(np.nanmedian([b["beta_self_kout"] for b in betas])),
            beta_ctx_kself=float(np.nanmedian([b["beta_ctx_kself"] for b in betas])),
            beta_ctx_kout=float(np.nanmedian([b["beta_ctx_kout"] for b in betas])),
        )
        print(f"  ring local k={k:<3} assort={ring[k]['k_out_assortativity']:.6f}  "
              f"beta_self(Kself)={ring[k]['beta_self_kself']:+.6f} beta_self(Kout)={ring[k]['beta_self_kout']:+.6f} "
              f"beta_ctx(Kself)={ring[k]['beta_ctx_kself']:+.6f} beta_ctx(Kout)={ring[k]['beta_ctx_kout']:+.6f}")
    print("  (for comparison, M0 points -- isolated-excluded variant:)")
    for key in ["N=100 M0", "N=1000 M0"]:
        v = per_cond[key]
        r = r_vals[key]
        print(f"  {key:<14} assort(Kout)={r['r_kout']:.6f}  beta_self(Kself)={v['excl4_kself']:+.6f} "
              f"beta_self(Kout)={v['excl4_kout']:+.6f} beta_ctx(Kself)={v['excl4_ctx_kself']:+.6f} "
              f"beta_ctx(Kout)={v['excl4_ctx_kout']:+.6f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(per_cond=per_cond, r_vals=r_vals, total_selection=total_sel, ring=ring), f, indent=2)
        f.write("\n")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
