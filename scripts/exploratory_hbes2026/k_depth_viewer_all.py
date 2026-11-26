"""K モデルの網を「記憶の深さ（実効深さ）」ごとに作る（ビューア用のデータ）。
世代の窓（既定 1000〜5000、--age-step おき）の個体をまとめ、類型×実効深さ×相手ごとに、
人と同じ人数の群を「別々の系統から1体ずつ」抽出して、Jaccard の網を --n-draws 回の平均で作る。
個体の選び方は頻度に比例（重みつきリザーバ抽出で、系統×類型×深さ×相手ごとに最大 R 本の系列を保持）。
出力：k_depth_edges.json / k_depth_nodes.json / k_depth_group_size.csv（集計値のみ）。
リポジトリの直下（EvoPersonality_ABM）で実行すること。
"""
import argparse, csv, gzip, heapq, json, random, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from evopersonality_abm.k_model_cross_lineage import load_series_first_rounds  # noqa: E402
from evopersonality_abm.lindgren import LindgrenTable, effective_memory_depth  # noqa: E402
from evopersonality_abm.markov_recovery import build_edges, load_vocab, self_loop_edges  # noqa: E402

OPPS = {"TFT": "TFT", "AllC": "AllC", "AllD": "AllD", "Random": "RAN", "Gradual": "GRA", "WSLS": "GRA", "SELF": "RAN", "POP": "RAN"}
TYPES = ["K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"]
GROUPINGS = TYPES + ["Mix"]
TARGET = {"K_Secure": 56, "K_Dismissing": 46, "K_Preoccupied": 46, "K_Fearful": 69, "Mix": 46}

def bits(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def classify(ko, ks, tko, tks):
    hs, ho = ks >= tks, ko >= tko
    return "K_Secure" if hs and ho else "K_Dismissing" if hs else "K_Preoccupied" if ho else "K_Fearful"

def one_file(data_dir, seed, age, tko, tks, R, depth_cap):
    c = data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
    m = data_dir / f"seed_{seed}_probe_age_{age}_moves.csv.gz"
    if not (c.exists() and m.exists()):
        return seed, None, None
    rng = random.Random(f"{seed}_{age}")
    info = {}
    with gzip.open(c, "rt", newline="") as f:
        for r in csv.DictReader(f):
            d = effective_memory_depth(LindgrenTable(int(r["memory_depth"]), bits(r["table"]), bits(r["prehistory"])))
            info[int(r["genotype_id"])] = (int(r["frequency"]), classify(float(r["k_out"]), float(r["k_self"]), tko, tks), min(d, depth_cap))
    counts = defaultdict(int)
    for g, (fq, t, d) in info.items():
        counts[(t, d)] += fq
    ser = load_series_first_rounds(m)
    res = defaultdict(list)  # (type, depth, opp) -> [(key, series)]
    for opp in OPPS:
        for g, lst in ser.get(opp, {}).items():
            if g not in info or not lst:
                continue
            fq, t, d = info[g]
            w = fq / len(lst)
            for s in lst:
                res[(t, d, opp)].append((rng.random() ** (1.0 / w), s))
    out = {k: heapq.nlargest(R, v) for k, v in res.items()}
    return seed, out, dict(counts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--thresholds-json", required=True)
    ap.add_argument("--age-start", type=int, default=1000); ap.add_argument("--age-end", type=int, default=5000)
    ap.add_argument("--age-step", type=int, default=100); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--reservoir", type=int, default=40); ap.add_argument("--depth-cap", type=int, default=7)
    ap.add_argument("--n-draws", type=int, default=100); ap.add_argument("--min-group", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    th = json.load(open(a.thresholds_json)); tko, tks, seeds = th["threshold_kout"], th["threshold_kself"], th["seeds"]
    ages = list(range(a.age_start, a.age_end + 1, a.age_step)); dd = Path(a.data_dir)
    pool = defaultdict(list)       # (seed, type, depth, opp) -> [(key, series)]
    counts = defaultdict(int)      # (type, depth) -> individuals (summed over seeds×ages)
    jobs = [(s, g) for s in seeds for g in ages]; done = miss = 0; t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_file, dd, s, g, tko, tks, a.reservoir, a.depth_cap) for s, g in jobs]
        for fu in as_completed(futs):
            s, out, cnt = fu.result(); done += 1
            if out is None:
                miss += 1
            else:
                for (t, d, o), lst in out.items():
                    key = (s, t, d, o)
                    pool[key] = heapq.nlargest(a.reservoir, pool[key] + lst)
                for k, v in cnt.items():
                    counts[k] += v
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} files (missing {miss}) {time.time()-t0:.0f}s", flush=True)
    depths = sorted({d for (_, _, d, _) in pool})
    rng = random.Random("k_depth_viewer_v1")
    edges_out = defaultdict(lambda: defaultdict(dict)); nodes_out = defaultdict(lambda: defaultdict(dict)); gs_rows = []
    for opp, vname in OPPS.items():
        vocab = load_vocab(vname); vs = sorted(vocab)
        for grouping in GROUPINGS:
            types = ["K_Dismissing", "K_Preoccupied"] if grouping == "Mix" else [grouping]
            for d in depths:
                byseed = {}
                for s in seeds:
                    lst = []
                    for t in types:
                        lst += pool.get((s, t, d, opp), [])
                    if lst:
                        byseed[s] = [x[1] for x in heapq.nlargest(a.reservoir, lst)]
                gs = min(TARGET[grouping], len(byseed))
                if opp == "TFT":
                    gs_rows.append({"grouping": grouping, "depth": d, "seeds_available": len(byseed), "group_size": gs,
                                    "individuals": sum(counts.get((t, d), 0) for t in types)})
                if gs < a.min_group:
                    continue
                esum = defaultdict(float); nsum = defaultdict(float); keys = list(byseed)
                for _ in range(a.n_draws):
                    series = [rng.choice(byseed[s]) for s in rng.sample(keys, gs)]
                    e = dict(build_edges(series, vocab)); e.update({(n, n): w for n, w in self_loop_edges(series, vocab).items()})
                    for k, v in e.items():
                        esum[k] += v
                    cnt = defaultdict(int)
                    for sr in series:
                        for wd in set(x.strip().rstrip("。") for x in sr.split(",")):
                            if wd in vocab:
                                cnt[wd] += 1
                    for wd in vs:
                        nsum[wd] += cnt.get(wd, 0) / gs
                edges_out[opp][grouping][str(d)] = [[x, y, round(v / a.n_draws, 4)] for (x, y), v in sorted(esum.items())]
                nodes_out[opp][grouping][str(d)] = {wd: round(nsum[wd] / a.n_draws, 4) for wd in vs}
        print("built", opp, flush=True)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    meta = {"depth_cap": a.depth_cap, "ages": [a.age_start, a.age_end, a.age_step], "n_draws": a.n_draws, "min_group": a.min_group}
    json.dump({"abm": edges_out, "meta": meta}, open(od / "k_depth_edges.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    json.dump({"abm_nodes": nodes_out, "meta": meta}, open(od / "k_depth_nodes.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    with open(od / "k_depth_group_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["grouping", "depth", "seeds_available", "group_size", "individuals"]); w.writeheader(); w.writerows(gs_rows)
    print("wrote", od)

if __name__ == "__main__":
    main()
