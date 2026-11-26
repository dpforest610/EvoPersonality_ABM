"""Kモデルの対戦記録から、人と同じ「交代」の分析をする（集計値だけを出力）。"""
import argparse, csv, gzip, json, random, math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OPP = {"Gradual": "GRA", "Random": "RAN", "TFT": "TFT"}
TYPES = ["K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"]
METRICS = ["alt_count", "alt3_any", "retaliation", "alt_longest"]
STATE = {"CC": "和", "CD": "惨", "DC": "搾", "DD": "底"}
ALT = {("惨", "搾"), ("搾", "惨")}
N = 20

def classify(ko, ks, tko, tks):
    hs, ho = ks >= tks, ko >= tko
    return "K_Secure" if hs and ho else "K_Dismissing" if hs else "K_Preoccupied" if ho else "K_Fearful"

def per_series(st):
    alt = sum((st[t], st[t + 1]) in ALT for t in range(len(st) - 1))
    alt3 = float(any("".join(st[t:t + 3]) in ("搾惨搾", "惨搾惨") for t in range(len(st) - 2)))
    after = [st[t + 1] for t in range(len(st) - 1) if st[t] == "惨"]
    ret = sum(s in ("搾", "底") for s in after) / len(after) if after else None
    longest = run = 0
    for t in range(len(st) - 1):
        run = run + 1 if (st[t], st[t + 1]) in ALT else 0
        longest = max(longest, run)
    return (float(alt), alt3, ret, float(longest))

def wls(rows):
    import numpy as np
    w = np.array([r[0] for r in rows], float); x1 = np.array([r[1] for r in rows]); x2 = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
    if len(rows) < 8 or w.sum() <= 0:
        return None
    def z(v):
        m = (w * v).sum() / w.sum(); s = math.sqrt((w * (v - m) ** 2).sum() / w.sum())
        return None if s < 1e-9 else (v - m) / s
    za, zv = z(-x1), z(-x2)
    if za is None or zv is None:
        return None
    X = np.column_stack([np.ones(len(y)), za, zv, za * zv]); sw = np.sqrt(w)
    if np.linalg.matrix_rank(X * sw[:, None]) < 4:
        return None
    b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return b.tolist()

def one_file(data_dir, seed, age, tko, tks):
    cpath = data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
    mpath = data_dir / f"seed_{seed}_probe_age_{age}_moves.csv.gz"
    if not (cpath.exists() and mpath.exists()):
        return seed, age, None, None
    info = {}
    with gzip.open(cpath, "rt", newline="") as f:
        for r in csv.DictReader(f):
            ko, ks = float(r["k_out"]), float(r["k_self"])
            info[int(r["genotype_id"])] = (int(r["frequency"]), ko, ks, classify(ko, ks, tko, tks))
    own, opp = defaultdict(dict), defaultdict(dict)
    with gzip.open(mpath, "rt", newline="") as f:
        rd = csv.reader(f); h = next(rd)
        ig, io, ir, iR, im, iom = (h.index(c) for c in ("genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"))
        for row in rd:
            rnd = int(row[iR])
            if rnd > N or row[io] not in OPP:
                continue
            k = (row[io], int(row[ig]), int(row[ir]))
            own[k][rnd] = row[im]; opp[k][rnd] = row[iom]
    per = defaultdict(lambda: defaultdict(list))
    for (o, g, rep), ob in own.items():
        if g not in info or len(ob) < N:
            continue
        st = [STATE[ob[i] + opp[(o, g, rep)][i]] for i in range(1, N + 1)]
        per[OPP[o]][g].append(per_series(st))
    types, regs = {}, {}
    for o, bygid in per.items():
        gmean = {}
        for g, lst in bygid.items():
            gmean[g] = []
            for mi in range(len(METRICS)):
                vals = [t[mi] for t in lst if t[mi] is not None]
                gmean[g].append(sum(vals) / len(vals) if vals else None)
        for t in TYPES:
            for mi, m in enumerate(METRICS):
                num = den = 0.0
                for g, v in gmean.items():
                    if info[g][3] == t and v[mi] is not None:
                        num += info[g][0] * v[mi]; den += info[g][0]
                if den:
                    types[(t, o, m)] = num / den
        for mi, m in ((0, "alt_count"), (2, "retaliation")):
            rows = [(info[g][0], info[g][2], info[g][1], v[mi]) for g, v in gmean.items() if v[mi] is not None]
            b = wls(rows)
            if b is not None:
                regs[(o, m)] = b
    return seed, age, types, regs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--thresholds-json", required=True)
    ap.add_argument("--age-start", type=int, default=1000); ap.add_argument("--age-end", type=int, default=5000)
    ap.add_argument("--age-step", type=int, default=50); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--out-prefix", default="k_alt")
    a = ap.parse_args()
    th = json.load(open(a.thresholds_json)); tko, tks, seeds = th["threshold_kout"], th["threshold_kself"], th["seeds"]
    ages = list(range(a.age_start, a.age_end + 1, a.age_step)); d = Path(a.data_dir)
    accT = defaultdict(lambda: defaultdict(list)); accR = defaultdict(lambda: defaultdict(list))
    jobs = [(s, g) for s in seeds for g in ages]; done = missing = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(one_file, d, s, g, tko, tks) for s, g in jobs]):
            s, g, types, regs = fu.result(); done += 1
            if types is None:
                missing += 1
            else:
                for k, v in types.items(): accT[k][s].append(v)
                for k, v in regs.items(): accR[k][s].append(v)
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} files (missing {missing})", flush=True)
    rng = random.Random(20261118)
    def boot(sm):
        bs = sorted(sum(rng.choice(sm) for _ in sm) / len(sm) for _ in range(a.n_boot))
        return bs[int(.025 * a.n_boot)], bs[int(.975 * a.n_boot) - 1]
    with open(f"{a.out_prefix}_by_type.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["opponent", "metric", "group", "mean", "n_seeds", "ci_lo", "ci_hi"])
        for o in OPP.values():
            for m in METRICS:
                for t in TYPES:
                    bys = accT.get((t, o, m))
                    if bys:
                        sm = [sum(v) / len(v) for v in bys.values()]; lo, hi = boot(sm)
                        w.writerow([o, m, t, f"{sum(sm)/len(sm):.4f}", len(sm), f"{lo:.4f}", f"{hi:.4f}"])
        for o in OPP.values():
            for m in METRICS:
                S, F = accT.get(("K_Secure", o, m), {}), accT.get(("K_Fearful", o, m), {})
                common = [s for s in S if s in F]
                if common:
                    dd = [sum(S[s]) / len(S[s]) - sum(F[s]) / len(F[s]) for s in common]; lo, hi = boot(dd)
                    w.writerow([o, m, "diff_Secure_minus_Fearful", f"{sum(dd)/len(dd):.4f}", len(dd), f"{lo:.4f}", f"{hi:.4f}"])
    with open(f"{a.out_prefix}_regression.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["opponent", "metric", "term", "b", "n_seeds", "ci_lo", "ci_hi"])
        for (o, m), bys in sorted(accR.items()):
            for j, term in enumerate(["切片", "不安相当(-K_self)", "回避相当(-K_out)", "積"]):
                sm = [sum(v[j] for v in vs) / len(vs) for vs in bys.values()]; lo, hi = boot(sm)
                w.writerow([o, m, term, f"{sum(sm)/len(sm):.4f}", len(sm), f"{lo:.4f}", f"{hi:.4f}"])
    print("wrote", f"{a.out_prefix}_by_type.csv", f"{a.out_prefix}_regression.csv")

if __name__ == "__main__":
    main()
