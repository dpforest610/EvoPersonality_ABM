"""第二章: does neighbor DIVERSITY (not just neighbor COUNT) drive K_out
selection? Tested entirely from already-saved individual logs -- no new
simulation -- since both experiment A (spatial_scale_check_k8_n1_static,
positions reshuffled every generation) and the persistence experiment
(spatial_scale_check_k8_n1_persistent, positions inherited across
generations) already log each agent's (x, y) alongside its (k_out,
k_self, payoff) every 50 generations.

Hypothesis (from chat): what actually drove K_out's selection signal in
experiment A was not "few opponents" alone, but "few AND varied
opponents." Cross-generation persistence lets a lineage settle next to
similar neighbors (local homogenization), which would remove exactly
the neighbor-quality variance that makes automatic co-regulation
(K_out) pay off -- explaining why K_out went back toward neutral under
persistence (chat: expA median=+0.044 vs persistent median=+0.005,
Mann-Whitney U p=0.0603).

Three questions, all from existing logs:

  1. Can neighbor diversity be measured from these logs? Yes: for every
     logged snapshot, (x, y) reconstructs that generation's full grid
     occupancy, so each agent's actual Moore (k=8) neighbors are known,
     and the std of their k_out (and k_self) values is a direct,
     cross-sectional diversity measure -- no lineage/residency tracking
     needed.

  2. Within the persistent condition, do agents with MORE diverse
     neighbors show a STRONGER corr(own k_out, own payoff) than agents
     with LESS diverse neighbors? Tested by, per seed and snapshot,
     median-splitting agents on their own neighbors' k_out std, computing
     Spearman rho(k_out, payoff) separately within each half, taking each
     seed's median rho per half over gen>=1000, then a paired Wilcoxon
     (high-diversity half vs low-diversity half, n=50 seeds). Same split
     run on experiment A for comparison.

  3. Is the persistent condition's neighborhood actually more homogeneous
     than experiment A's? Compared via each seed's median neighbor-k_out-std
     (and k_self-std) over gen>=1000, Mann-Whitney U, n=50 each condition.

Usage:
    python3 scripts/analyze_neighbor_diversity_hypothesis.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evopersonality_abm.spatial import MOORE_OFFSETS, torus_neighbors

GRID_WIDTH = 12
GRID_HEIGHT = 12
STEADY_STATE_MIN_GEN = 1000
SEEDS = list(range(50))

CONDITIONS = {
    "experiment A (reshuffled)": Path("scripts/output/spatial_scale_check_k8_n1_static"),
    "persistent": Path("scripts/output/spatial_scale_check_k8_n1_persistent"),
}


def load_individual_log(path):
    """Returns {generation: {agent_index: {'k_out':.., 'k_self':.., 'payoff':.., 'pos':(x,y)}}}."""
    by_gen = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            gen = int(row["generation"])
            idx = int(row["agent_index"])
            by_gen[gen][idx] = {
                "k_out": float(row["k_out"]),
                "k_self": float(row["k_self"]),
                "payoff": float(row["payoff"]),
                "pos": (int(row["x"]), int(row["y"])),
            }
    return by_gen


def neighbor_diversity_per_agent(gen_data):
    """For one generation's {agent_index: {...}} dict, returns
    {agent_index: (neighbor_kout_std, neighbor_kself_std, n_occupied_neighbors)}."""
    agent_at_cell = {rec["pos"]: idx for idx, rec in gen_data.items()}
    result = {}
    for idx, rec in gen_data.items():
        neighbor_cells = torus_neighbors(rec["pos"], GRID_WIDTH, GRID_HEIGHT, MOORE_OFFSETS)
        neighbor_kouts = []
        neighbor_kselfs = []
        for ncell in neighbor_cells:
            j = agent_at_cell.get(ncell)
            if j is not None:
                neighbor_kouts.append(gen_data[j]["k_out"])
                neighbor_kselfs.append(gen_data[j]["k_self"])
        if len(neighbor_kouts) >= 2:
            result[idx] = (np.std(neighbor_kouts), np.std(neighbor_kselfs), len(neighbor_kouts))
    return result


def analyze_condition(path):
    """Returns per-seed, per-snapshot-gen data needed for questions 2 and 3."""
    per_seed_snapshot_diversity = {}  # seed -> gen -> list of (agent_idx, kout_std, kself_std)
    per_seed_snapshot_data = {}  # seed -> gen -> {agent_idx: rec}
    for seed in SEEDS:
        by_gen = load_individual_log(path / f"seed_{seed}_individual_log.csv")
        per_seed_snapshot_diversity[seed] = {}
        per_seed_snapshot_data[seed] = by_gen
        for gen, gen_data in by_gen.items():
            if gen < STEADY_STATE_MIN_GEN:
                continue
            per_seed_snapshot_diversity[seed][gen] = neighbor_diversity_per_agent(gen_data)
    return per_seed_snapshot_data, per_seed_snapshot_diversity


def question3_homogeneity_comparison(all_diversity):
    print("\n=== Question 3: is the persistent condition's neighborhood actually more homogeneous? ===")
    per_seed_median_kout_std = {name: [] for name in CONDITIONS}
    per_seed_median_kself_std = {name: [] for name in CONDITIONS}
    for name in CONDITIONS:
        diversity = all_diversity[name]
        for seed in SEEDS:
            kout_stds, kself_stds = [], []
            for gen, per_agent in diversity[seed].items():
                for idx, (ko_std, ks_std, n) in per_agent.items():
                    kout_stds.append(ko_std)
                    kself_stds.append(ks_std)
            per_seed_median_kout_std[name].append(np.median(kout_stds))
            per_seed_median_kself_std[name].append(np.median(kself_stds))

    for label, per_seed_median in [("neighbor k_out std", per_seed_median_kout_std),
                                    ("neighbor k_self std", per_seed_median_kself_std)]:
        a = np.array(per_seed_median["experiment A (reshuffled)"])
        b = np.array(per_seed_median["persistent"])
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"{label}: expA median={np.median(a):.4f}  persistent median={np.median(b):.4f}  "
              f"Mann-Whitney U={stat:.1f} p={p:.4f}")


def question2_diversity_modulates_correlation(all_data, all_diversity):
    print("\n=== Question 2: within each condition, does neighbor diversity modulate corr(k_out, payoff)? ===")
    for name in CONDITIONS:
        by_gen_data = all_data[name]
        by_gen_diversity = all_diversity[name]
        high_rhos_per_seed = []
        low_rhos_per_seed = []
        for seed in SEEDS:
            high_rhos, low_rhos = [], []
            for gen, per_agent_div in by_gen_diversity[seed].items():
                gen_data = by_gen_data[seed][gen]
                agent_ids = list(per_agent_div.keys())
                if len(agent_ids) < 10:
                    continue
                kout_stds = np.array([per_agent_div[i][0] for i in agent_ids])
                median_split = np.median(kout_stds)
                high_ids = [i for i in agent_ids if per_agent_div[i][0] >= median_split]
                low_ids = [i for i in agent_ids if per_agent_div[i][0] < median_split]

                for ids, bucket in [(high_ids, high_rhos), (low_ids, low_rhos)]:
                    if len(ids) < 5:
                        continue
                    kouts = [gen_data[i]["k_out"] for i in ids]
                    payoffs = [gen_data[i]["payoff"] for i in ids]
                    if len(set(kouts)) < 2 or len(set(payoffs)) < 2:
                        continue
                    rho, _ = spearmanr(kouts, payoffs)
                    bucket.append(rho)

            if high_rhos and low_rhos:
                high_rhos_per_seed.append(np.median(high_rhos))
                low_rhos_per_seed.append(np.median(low_rhos))

        high_arr = np.array(high_rhos_per_seed)
        low_arr = np.array(low_rhos_per_seed)
        n = len(high_arr)
        diffs = high_arr - low_arr
        if np.allclose(diffs, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(high_arr, low_arr)
        print(f"\n[{name}] n={n} seeds with usable high/low splits")
        print(f"  high-diversity-neighbor half: median rho(k_out,payoff)={np.median(high_arr):+.3f}")
        print(f"  low-diversity-neighbor half:  median rho(k_out,payoff)={np.median(low_arr):+.3f}")
        print(f"  paired Wilcoxon (high vs low): stat={stat:.1f} p={p:.4f}  "
              f"median(high-low) diff={np.median(diffs):+.3f}  positive in {int(np.sum(diffs>0))}/{n}")


def main():
    all_data = {}
    all_diversity = {}
    for name, path in CONDITIONS.items():
        print(f"loading and computing neighbor diversity for [{name}]...")
        data, diversity = analyze_condition(path)
        all_data[name] = data
        all_diversity[name] = diversity

    question3_homogeneity_comparison(all_diversity)
    question2_diversity_modulates_correlation(all_data, all_diversity)


if __name__ == "__main__":
    main()
