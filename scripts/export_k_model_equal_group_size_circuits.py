"""Exploratory, outside preregistration: does the group-size DIFFERENCE
between K-model types (K_Secure target 56, Mix target 46, K_Fearful
target 69 -- docs/DEVIATION_k_model_group_size.md) affect specific
widest-path/circuit values enough to matter, independent of whichever
type's network is actually "closer" to which human group?

For K_Secure, Mix, and K_Fearful, draws a cross-lineage group of the SAME
size for all three at every age -- target 46 (Mix's own target), but
shrunk to the SHARED minimum available across all three types at that
age (docs/DEVIATION_k_model_group_size.md's rule applied jointly, not
per type independently: shared_group_size = min(46, n_avail(K_Secure),
n_avail(Mix), n_avail(K_Fearful))) -- and recomputes the same metric
using each type's OWN human-matched target (56/46/69, independently
shrunk per type, the same rule scripts/export_k_model_markov_viewer.py's
production export already uses) for side-by-side comparison. Both use
200 cross-lineage draws' averaged Jaccard, same as the production viewer
export.

From each resulting averaged-edge network, computes 8 specific values
per (age, group):
  - TFT widest path 底底底->和和和
  - Gradual widest path 底底底->和和和
  - AllC widest path 搾搾搾->和和和
  - AllD widest path 底底底->惨惨惨
  - TFT's cooperation circuit (和和和->和和搾->和搾惨->搾惨和->惨和和->和和和) j*
  - Gradual's cooperation circuit (same walk, Gradual's own network) j*
  - AllC's cooperation circuit (和和和->和和搾->和搾和->搾和和->和和和) j*
  - Gradual's 搾惨搾<->惨搾惨 circuit j* (this exact 2-cycle is one of
    TFT's own named Table-7 circuits, not one of Gradual's -- computed
    here against Gradual's network anyway as a specific requested value,
    not as a claim that it is one of Gradual's named circuits)

Does NOT compute anything for K_Dismissing/K_Preoccupied/WSLS/SELF/POP/
Random -- out of scope for this specific exploratory question.

Reads raw per-age census/moves data (Mac-local only, gitignored) -- THIS
SCRIPT MUST RUN ON THE MAC where the production run's data lives.
Resumable per-age caching, same pattern as
scripts/export_k_model_markov_viewer.py.

Usage (see docs/K_MODEL_MARKOV_MAC_RUNBOOK.md section 13):
    venv/bin/python scripts/export_k_model_equal_group_size_circuits.py \\
        --data-dir scripts/output/coupling_markov_production --seeds 70 --seed-offset 200 \\
        --thresholds-json scripts/output/coupling_markov_production/analysis/type_thresholds.json \\
        --window-start 1000 --window-end 5000 --n-draws 200 --workers 5 \\
        --out-csv scripts/output/coupling_markov_production/viewer/k_model_equal_group_size_circuits.csv
"""

import argparse
import csv
import importlib.util
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

_viewer_spec = importlib.util.spec_from_file_location("export_k_model_markov_viewer", SCRIPTS_DIR / "export_k_model_markov_viewer.py")
viewer_mod = importlib.util.module_from_spec(_viewer_spec)
sys.modules["export_k_model_markov_viewer"] = viewer_mod
_viewer_spec.loader.exec_module(viewer_mod)

from evopersonality_abm.k_model_cross_lineage import (  # noqa: E402
    draw_cross_lineage_identities,
    effective_group_size,
    load_series_first_rounds,
    lookup_series,
    merge_type_membership,
    seeds_with_type,
    type_membership_from_labels,
)
from evopersonality_abm.markov_recovery import build_edges, cycle_jstar, load_vocab, self_loop_edges, widest_path_threshold  # noqa: E402

GROUPS = ["K_Secure", "Mix", "K_Fearful"]
HUMAN_MATCHED_TARGET = {"K_Secure": 56, "Mix": 46, "K_Fearful": 69}
SHARED_TARGET = 46  # Mix's own target -- the size shared across all 3 groups

OPPONENTS = ["TFT", "Gradual", "AllC", "AllD"]
VOCAB_NAME = {"TFT": "TFT", "Gradual": "GRA", "AllC": "AllC", "AllD": "AllD"}

COOPERATION_CIRCUIT_TFT_GRA = ["和和和", "和和搾", "和搾惨", "搾惨和", "惨和和", "和和和"]
COOPERATION_CIRCUIT_ALLC = ["和和和", "和和搾", "和搾和", "搾和和", "和和和"]
GRADUAL_BRIDGE_CIRCUIT = ["搾惨搾", "惨搾惨", "搾惨搾"]  # one of TFT's own named circuits, applied here to Gradual's network

# (label, opponent, kind, params) -- kind is "widest" (params=(start,goal)) or "jstar" (params=circuit walk)
METRICS: List[Tuple[str, str, str, object]] = [
    ("TFT_widest_底底底-和和和", "TFT", "widest", ("底底底", "和和和")),
    ("Gradual_widest_底底底-和和和", "Gradual", "widest", ("底底底", "和和和")),
    ("AllC_widest_搾搾搾-和和和", "AllC", "widest", ("搾搾搾", "和和和")),
    ("AllD_widest_底底底-惨惨惨", "AllD", "widest", ("底底底", "惨惨惨")),
    ("TFT_circuit_協調", "TFT", "jstar", COOPERATION_CIRCUIT_TFT_GRA),
    ("Gradual_circuit_協調", "Gradual", "jstar", COOPERATION_CIRCUIT_TFT_GRA),
    ("AllC_circuit_協調", "AllC", "jstar", COOPERATION_CIRCUIT_ALLC),
    ("Gradual_circuit_搾惨搾-惨搾惨", "Gradual", "jstar", GRADUAL_BRIDGE_CIRCUIT),
]


def average_edges_over_draws(pools, vocab, group_size: int, n_draws: int, opponent: str, series_by_seed, rng: random.Random):
    if group_size == 0:
        return None
    sums: Dict[Tuple[str, str], float] = {}
    for _ in range(n_draws):
        identities = draw_cross_lineage_identities(pools, group_size, rng)
        series = lookup_series(identities, series_by_seed, opponent, rng)
        edges = dict(build_edges(series, vocab))
        edges.update({(n, n): w for n, w in self_loop_edges(series, vocab).items()})
        for k, v in edges.items():
            sums[k] = sums.get(k, 0.0) + v
    return {k: v / n_draws for k, v in sums.items()}


def compute_metrics(edges) -> Dict[str, object]:
    out = {}
    for label, opponent, kind, params in METRICS:
        if opponent not in edges or edges[opponent] is None:
            out[label] = None
            continue
        e = edges[opponent]
        if kind == "widest":
            start, goal = params
            out[label] = widest_path_threshold(e, start=start, goal=goal)
        else:
            out[label] = cycle_jstar(e, params)
    return out


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def process_one_age(data_dir_str: str, age: int, seeds: List[int], threshold_kout: float, threshold_kself: float, n_draws: int, master_seed: str) -> dict:
    data_dir = Path(data_dir_str)
    rng = random.Random(f"{master_seed}_{age}")

    per_seed_census = {}
    per_seed_series = {}
    for seed in seeds:
        cpath = viewer_mod.pilot_mod.census_path(data_dir, seed, age)
        mpath = viewer_mod.pilot_mod.moves_path(data_dir, seed, age)
        if not (cpath.exists() and mpath.exists()):
            continue
        per_seed_census[seed] = viewer_mod.load_census_with_k(cpath)
        per_seed_series[seed] = load_series_first_rounds(mpath)

    true_type_by_seed = {seed: viewer_mod.census_true_types(census, threshold_kout, threshold_kself) for seed, census in per_seed_census.items()}

    base_pools: Dict[str, Dict[int, tuple]] = {}
    for type_name in ("K_Secure", "K_Dismissing", "K_Preoccupied", "K_Fearful"):
        base_pools[type_name] = {
            seed: type_membership_from_labels(census, true_type_by_seed[seed], type_name)
            for seed, census in per_seed_census.items()
        }
    base_pools["Mix"] = {
        seed: merge_type_membership(base_pools["K_Dismissing"][seed], base_pools["K_Preoccupied"][seed])
        for seed in per_seed_census
    }

    n_avail = {g: len(seeds_with_type(base_pools[g])) for g in GROUPS}
    shared_gs = effective_group_size(SHARED_TARGET, min(n_avail.values())) if n_avail else 0

    vocabs = {opp: load_vocab(VOCAB_NAME[opp]) for opp in OPPONENTS}

    rows = []
    for group in GROUPS:
        pools = base_pools[group]
        human_gs = effective_group_size(HUMAN_MATCHED_TARGET[group], n_avail[group])

        shared_edges = {opp: average_edges_over_draws(pools, vocabs[opp], shared_gs, n_draws, opp, per_seed_series, rng) for opp in OPPONENTS}
        human_edges = {opp: average_edges_over_draws(pools, vocabs[opp], human_gs, n_draws, opp, per_seed_series, rng) for opp in OPPONENTS}

        shared_metrics = compute_metrics(shared_edges)
        human_metrics = compute_metrics(human_edges)

        for label, _opponent, _kind, _params in METRICS:
            rows.append({
                "age": age, "group": group, "metric": label,
                "shared46_group_size": shared_gs, "shared46_value": shared_metrics[label],
                "human_matched_group_size": human_gs, "human_matched_value": human_metrics[label],
            })

    return {"age": age, "rows": rows}


def run_all_ages(data_dir: Path, ages: List[int], seeds: List[int], threshold_kout: float, threshold_kself: float, n_draws: int, master_seed: str, workers: int, cache_dir: Path) -> List[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_one_age, str(data_dir), age, seeds, threshold_kout, threshold_kself, n_draws, master_seed): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    results = []
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            results.append(pickle.load(f))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--seed-offset", type=int, required=True)
    p.add_argument("--thresholds-json", type=str, required=True)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--master-seed", type=str, default="k_model_equal_group_size_circuits_v1")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-csv", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    seeds = [args.seed_offset + i for i in range(args.seeds)]
    import json
    with open(args.thresholds_json) as f:
        thresholds = json.load(f)

    all_ages = viewer_mod.discover_ages(data_dir)
    ages = [a for a in all_ages if args.window_start <= a <= args.window_end]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path.parent / "_age_cache_equal_group_size"

    results = run_all_ages(data_dir, ages, seeds, thresholds["threshold_kout"], thresholds["threshold_kself"], args.n_draws, args.master_seed, args.workers, cache_dir)

    rows = []
    for r in results:
        rows.extend(r["rows"])
    rows.sort(key=lambda r: (r["metric"], r["group"], r["age"]))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["age", "group", "metric", "shared46_group_size", "shared46_value", "human_matched_group_size", "human_matched_value"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
