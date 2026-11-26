"""Kモデルの対戦記録（moves）から、人のデータと同じ3つの指標を類型×相手で集計する。
  協力率 / 入口（底の直後に協力）/ 到達（最初の底のあと和和和）
最初の20回だけを使う（人の実験と同じ）。出力は集計値のCSVだけ。

リポジトリの直下（EvoPersonality_ABM）で実行すること:
    venv/bin/python scripts/exploratory_hbes2026/k_agg.py [--data-dir ...] [--thresholds-json ...]
--data-dir・--thresholds-json は省略時、本番と同じ場所(scripts/output/coupling_markov_production、
その analysis/type_thresholds.json)を使う。
"""
import argparse, csv, gzip, json, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OPP = {"AllC": "AllC", "AllD": "AllD", "Gradual": "GRA", "Random": "RAN", "TFT": "TFT"}
TYPES = ["K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"]
METRICS = ["coop_rate", "entry_C_after_bottom", "reach_CCC_after_first_bottom"]
STATE = {"CC": "和", "CD": "惨", "DC": "搾", "DD": "底"}
N_ROUNDS = 20

def classify(k_out, k_self, tko, tks):
    hs, ho = k_self >= tks, k_out >= tko
    return "K_Secure" if hs and ho else "K_Dismissing" if hs else "K_Preoccupied" if ho else "K_Fearful"

def metrics(st):
    coop = sum(s in ("和", "惨") for s in st) / len(st)
    ent = [st[t + 1] in ("和", "惨") for t in range(len(st) - 1) if st[t] == "底"]
    entry = sum(ent) / len(ent) if ent else None
    reach = None
    if "底" in st[:N_ROUNDS - 3]:
        rest = st[st.index("底") + 1:]
        reach = float(any(rest[i:i + 3] == ["和"] * 3 for i in range(len(rest) - 2)))
    return (coop, entry, reach)

def one_file(data_dir, seed, age, tko, tks):
    cpath = data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
    mpath = data_dir / f"seed_{seed}_probe_age_{age}_moves.csv.gz"
    if not (cpath.exists() and mpath.exists()):
        return seed, age, None
    freq, typ = {}, {}
    with gzip.open(cpath, "rt", newline="") as f:
        for r in csv.DictReader(f):
            g = int(r["genotype_id"]); freq[g] = int(r["frequency"])
            typ[g] = classify(float(r["k_out"]), float(r["k_self"]), tko, tks)
    own, opp = defaultdict(dict), defaultdict(dict)
    with gzip.open(mpath, "rt", newline="") as f:
        rd = csv.reader(f); h = next(rd)
        ig, io, ir, iR, im, iom = (h.index(c) for c in ("genotype_id", "opponent", "repeat", "round", "own_move", "opp_move"))
        for row in rd:
            rnd = int(row[iR])
            if rnd > N_ROUNDS or row[io] not in OPP:
                continue
            k = (row[io], int(row[ig]), int(row[ir]))
            own[k][rnd] = row[im]; opp[k][rnd] = row[iom]
    per = defaultdict(lambda: defaultdict(list))
    for (o, g, rep), ob in own.items():
        if g not in typ or len(ob) < N_ROUNDS:
            continue
        st = [STATE[ob[i] + opp[(o, g, rep)][i]] for i in range(1, N_ROUNDS + 1)]
        per[(typ[g], OPP[o])][g].append(metrics(st))
    out = {}
    for key, bygid in per.items():
        for mi, m in enumerate(METRICS):
            num = den = 0.0
            for g, lst in bygid.items():
                vals = [t[mi] for t in lst if t[mi] is not None]
                if vals:
                    num += freq[g] * sum(vals) / len(vals); den += freq[g]
            if den:
                out[(key[0], key[1], m)] = num / den
    return seed, age, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="scripts/output/coupling_markov_production")
    ap.add_argument("--thresholds-json", default="scripts/output/coupling_markov_production/analysis/type_thresholds.json")
    ap.add_argument("--age-start", type=int, default=1000)
    ap.add_argument("--age-end", type=int, default=5000)
    ap.add_argument("--age-step", type=int, default=50)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="k_by_type_opponent.csv")
    a = ap.parse_args()
    th = json.load(open(a.thresholds_json))
    tko, tks, seeds = th["threshold_kout"], th["threshold_kself"], th["seeds"]
    ages = list(range(a.age_start, a.age_end + 1, a.age_step))
    data_dir = Path(a.data_dir)
    acc = defaultdict(lambda: defaultdict(list))
    jobs = [(s, g) for s in seeds for g in ages]
    done = missing = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_file, data_dir, s, g, tko, tks) for s, g in jobs]
        for fu in as_completed(futs):
            s, g, out = fu.result(); done += 1
            if out is None:
                missing += 1
            else:
                for k, v in out.items():
                    acc[k][s].append(v)
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} files (missing {missing})", flush=True)
    rng = random.Random(20261117)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "metric", "group", "mean", "n_seeds", "ci_lo", "ci_hi"])
        for o in OPP.values():
            for m in METRICS:
                for t in TYPES:
                    bys = acc.get((t, o, m))
                    if not bys:
                        continue
                    sm = [sum(v) / len(v) for v in bys.values()]
                    mean = sum(sm) / len(sm)
                    bs = sorted(sum(rng.choice(sm) for _ in sm) / len(sm) for _ in range(a.n_boot))
                    w.writerow([o, m, t, f"{mean:.4f}", len(sm), f"{bs[int(.025*a.n_boot)]:.4f}", f"{bs[int(.975*a.n_boot)-1]:.4f}"])
    print("wrote", a.out)

if __name__ == "__main__":
    main()
