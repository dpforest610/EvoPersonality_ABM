"""Confirmatory + descriptive Markov-network analysis for the production
run, strictly per docs/PREREGISTRATION_markov_structure_evolution.md
(confirmed commit ccc0b1b7a76f22e150e7c097b1757b1fb8e54e0e).

Reads raw probe data (seed_X_probe_age_G_census.csv.gz /
..._moves.csv.gz, written by scripts/run_lindgren_markov_sweep.py) --
these are Mac-local only (gitignored, never pushed), so THIS SCRIPT MUST
RUN ON THE MAC where they live. It writes only aggregated results
(tables, figures, JSON) to --results-dir, which is what gets committed
and pushed.

Per age in the analysis window (preregistration section 3: gen
1000-5000), one pass over all seeds' probe data computes:
  - scheme (B) [primary] and scheme (A) [descriptive comparison] point
    estimates (recovery/descent widest path, Table 7 j*, largest SCC,
    human-group similarity, group diversity) for all 7 opponents x the
    3 human group sizes (56/46/69) -- same machinery as
    scripts/analyze_markov_sweep_pilot_network.py /
    ..._sampling_comparison.py, reused via evopersonality_abm.markov_apply.
  - the preregistration's seed-level bootstrap (section 5): for the 3
    confirmatory opponents (TFT/AllD/AllC) only, --n-bootstrap-seed
    resamples of the 70 seeds (with replacement), each producing a
    cross-lineage group and a human-group correlation.

After the age loop: point estimates are averaged within the analysis
window (one point estimate per opponent); the seed-bootstrap's
per-iteration values are averaged across the window per iteration,
then turned into the 6 directional comparisons' bootstrap difference
distributions, which evopersonality_abm.markov_confirmatory turns into
one-sided p-values, Holm correction, and H1/H2/H3 verdicts.

Run once per noise condition (--role primary for noise_rate=0.01,
--role sensitivity for noise_rate=0.0 -- preregistration section 6
item 6 reports the sensitivity condition's H1-H3 comparisons
descriptively, not as a second confirmatory test).

Usage (see docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md for the full
Mac command):
    venv/bin/python scripts/run_markov_confirmatory_analysis.py \\
        --data-dir scripts/output/markov_sweep_production_noise01 \\
        --moves-suffix noise0p01 --role primary \\
        --seeds 70 --seed-offset 100 \\
        --results-dir scripts/output/markov_confirmatory_production_noise01 \\
        --workers 5

Time/memory estimate: see the module docstring in
docs/MARKOV_STRUCTURE_EVOLUTION_MAC_RUNBOOK.md section "確証的分析の実行"
(added alongside this script) -- roughly 30-40 minutes on a 6-core Mac
at --workers 5 for the full 81-age production window, well under 1GB
peak memory (one age's full 70-seed pool is a few tens of MB; results
accumulate as small per-iteration sums, not raw rows).
"""

import argparse
import csv
import json
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evopersonality_abm.markov_apply import (
    OPPONENT_INFO,
    bootstrap_aggregate_with_sampler,
    draw_cross_lineage_group_from_resampled_seeds,
    load_census,
    load_genotype_identities,
    load_human_weights,
    load_series_by_genotype_all_opponents,
    resample_cross_seed_group,
    resample_within_seed_group_with_identity,
    seed_bootstrap_resample,
    similarity_to_human,
)
from evopersonality_abm.markov_confirmatory import COMPARISONS, comparison_diff, evaluate_hypotheses
from evopersonality_abm.markov_recovery import build_edges, load_vocab

H_OPPONENTS = ["TFT", "AllD", "AllC"]  # preregistration section 4's confirmatory opponents
ALL_OPPONENTS = list(OPPONENT_INFO)  # 7: main set + extended set


def age_cache_path(cache_dir: Path, age: int) -> Path:
    return cache_dir / f"age_{age}.pkl"


def load_seed_pools_for_age(data_dir: Path, moves_suffix: str, age: int, seeds: List[int]) -> Dict[int, dict]:
    """{seed: {opponent: (frequencies, series_by_genotype, identities)}}
    for one probe age, skipping any seed missing data at this age."""
    pools: Dict[int, dict] = {}
    for seed in seeds:
        census_path = data_dir / f"seed_{seed}_probe_age_{age}_census.csv.gz"
        moves_path = data_dir / f"seed_{seed}_probe_age_{age}_{moves_suffix}_moves.csv.gz"
        if not census_path.exists() or not moves_path.exists():
            continue
        frequencies = load_census(census_path)
        identities = load_genotype_identities(census_path)
        series_by_opponent = load_series_by_genotype_all_opponents(moves_path, main_analysis_rounds=20)
        per_opponent = {}
        for opponent in ALL_OPPONENTS:
            series_by_genotype = series_by_opponent.get(opponent, {})
            if series_by_genotype:
                per_opponent[opponent] = (frequencies, series_by_genotype, identities)
        pools[seed] = per_opponent
    return pools


def process_one_age(
    data_dir_str: str, moves_suffix: str, age: int, seeds: List[int],
    human_weights_path: str, n_boot_point: int, n_boot_descriptive: int,
    n_bootstrap_seed: int, master_seed: str, group_for_human: Dict[str, int],
) -> dict:
    data_dir = Path(data_dir_str)
    human_weights = load_human_weights(Path(human_weights_path))
    pools = load_seed_pools_for_age(data_dir, moves_suffix, age, seeds)

    descriptive_rows: List[dict] = []
    # seed_boot_contributions[(iteration, opponent, human_group)] = corr (or None)
    seed_boot_contributions: List[Tuple[int, str, str, Optional[float]]] = []

    for opponent in ALL_OPPONENTS:
        label, vocab_name = OPPONENT_INFO[opponent]
        vocab = load_vocab(vocab_name)
        human_weights_by_group = (
            {g: human_weights.get((label, g), {}) for g in ("Secure", "Mix", "Fearful")} if label else {}
        )
        seeds_with_opp = {s: pools[s][opponent] for s in pools if opponent in pools[s]}
        n_boot_b = n_boot_point if opponent in H_OPPONENTS else n_boot_descriptive

        for group_size in group_for_human.values():
            if len(seeds_with_opp) >= group_size:
                rng_b = random.Random(f"{master_seed}_pointB_{opponent}_{group_size}_{age}")

                def sample_fn_b(rng, pools_=seeds_with_opp, gs=group_size):
                    return resample_cross_seed_group(pools_, gs, rng)

                result_b = bootstrap_aggregate_with_sampler(
                    sample_fn_b, opponent, n_boot_b, rng_b, human_weights_by_group, vocab, with_diversity=True,
                )
                descriptive_rows.append({"scheme": "B", "age": age, "opponent": opponent, "group_size": group_size, **result_b})

            a_results = []
            for seed, (frequencies, series_by_genotype, identities) in seeds_with_opp.items():
                rng_a = random.Random(f"{master_seed}_pointA_{opponent}_{group_size}_{age}_{seed}")

                def sample_fn_a(rng, freq=frequencies, sbg=series_by_genotype, idn=identities, gs=group_size):
                    return resample_within_seed_group_with_identity(freq, sbg, idn, gs, rng)

                a_results.append(bootstrap_aggregate_with_sampler(
                    sample_fn_a, opponent, n_boot_descriptive, rng_a, human_weights_by_group, vocab, with_diversity=True,
                ))
            if a_results:
                all_keys = {k for r in a_results for k in r}
                avg_a = {}
                for k in all_keys:
                    vals = [r[k] for r in a_results if k in r]
                    avg_a[k] = sum(vals) / len(vals) if vals else float("nan")
                descriptive_rows.append({"scheme": "A", "age": age, "opponent": opponent, "group_size": group_size, **avg_a})

        if opponent in H_OPPONENTS and label is not None:
            seeds_with_data = list(seeds_with_opp.keys())
            for human_group, group_size in group_for_human.items():
                if len(seeds_with_data) < group_size:
                    continue
                human_edges = human_weights.get((label, human_group), {})
                for i in range(n_bootstrap_seed):
                    rng = random.Random(f"{master_seed}_seedboot_{i}_{opponent}_{age}")
                    resampled = seed_bootstrap_resample(seeds_with_data, rng)
                    series, _idents = draw_cross_lineage_group_from_resampled_seeds(
                        seeds_with_opp, resampled, group_size, rng,
                    )
                    edges = build_edges(series, vocab)
                    corr, _jacc = similarity_to_human(edges, human_edges)
                    seed_boot_contributions.append((i, opponent, human_group, corr))

    return {"age": age, "descriptive_rows": descriptive_rows, "seed_boot_contributions": seed_boot_contributions}


def run_all_ages(
    data_dir: Path, moves_suffix: str, ages: List[int], seeds: List[int],
    human_weights_path: str, n_boot_point: int, n_boot_descriptive: int,
    n_bootstrap_seed: int, master_seed: str, workers: int, cache_dir: Path,
    group_for_human: Dict[str, int],
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in ages if not age_cache_path(cache_dir, a).exists()]
    print(f"{len(ages) - len(pending)}/{len(ages)} ages already cached, {len(pending)} pending", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one_age, str(data_dir), moves_suffix, age, seeds,
                human_weights_path, n_boot_point, n_boot_descriptive, n_bootstrap_seed,
                master_seed, group_for_human,
            ): age
            for age in pending
        }
        for fut in as_completed(futures):
            age = futures[fut]
            result = fut.result()
            with open(age_cache_path(cache_dir, age), "wb") as f:
                pickle.dump(result, f)
            print(f"age={age} done, elapsed={time.time() - t0:.0f}s", flush=True)

    descriptive_rows: List[dict] = []
    time_avg_sums: Dict[Tuple[int, str, str], float] = defaultdict(float)
    time_avg_counts: Dict[Tuple[int, str, str], int] = defaultdict(int)
    for age in ages:
        with open(age_cache_path(cache_dir, age), "rb") as f:
            result = pickle.load(f)
        descriptive_rows.extend(result["descriptive_rows"])
        for i, opponent, human_group, corr in result["seed_boot_contributions"]:
            if corr is not None:
                time_avg_sums[(i, opponent, human_group)] += corr
                time_avg_counts[(i, opponent, human_group)] += 1

    return descriptive_rows, time_avg_sums, time_avg_counts


def compute_point_estimates(
    descriptive_rows: List[dict], window: Tuple[int, int], group_for_human: Dict[str, int],
) -> Dict[Tuple[str, str], float]:
    """{(opponent, "S"/"M"/"F"): window-averaged mean_corr_<group>} for
    scheme B, the 3 confirmatory opponents, at each opponent's matching
    human-group-size row."""
    lo, hi = window
    by_key: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    letter = {"Secure": "S", "Mix": "M", "Fearful": "F"}
    for row in descriptive_rows:
        if row["scheme"] != "B" or row["opponent"] not in H_OPPONENTS:
            continue
        if not (lo <= row["age"] <= hi):
            continue
        for human_group, group_size in group_for_human.items():
            if row["group_size"] != group_size:
                continue
            key = f"mean_corr_{human_group}"
            if key in row and row[key] == row[key]:  # not NaN
                by_key[(row["opponent"], letter[human_group])].append(row[key])
    return {k: statistics.fmean(v) for k, v in by_key.items() if v}


def build_bootstrap_diff_distributions(
    time_avg_sums: Dict[Tuple[int, str, str], float], time_avg_counts: Dict[Tuple[int, str, str], int],
    n_bootstrap_seed: int,
) -> Dict[Tuple[str, str], List[float]]:
    letter = {"Secure": "S", "Mix": "M", "Fearful": "F"}
    per_iteration_r: List[Dict[str, Dict[str, float]]] = []
    for i in range(n_bootstrap_seed):
        r_by_opponent: Dict[str, Dict[str, float]] = {}
        ok = True
        for opponent in H_OPPONENTS:
            r_by_group = {}
            for human_group in ("Secure", "Mix", "Fearful"):
                key = (i, opponent, human_group)
                if time_avg_counts.get(key, 0) == 0:
                    ok = False
                    break
                r_by_group[letter[human_group]] = time_avg_sums[key] / time_avg_counts[key]
            if not ok:
                break
            r_by_opponent[opponent] = r_by_group
        if ok:
            per_iteration_r.append(r_by_opponent)

    bootstrap_diffs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r_by_opponent in per_iteration_r:
        for _hyp, _label, opponent, groups in COMPARISONS:
            bootstrap_diffs[(opponent, groups)].append(comparison_diff(r_by_opponent[opponent], groups))
    return dict(bootstrap_diffs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True, help="directory with seed_X_probe_age_G_*.csv.gz (Mac-local raw probe data)")
    p.add_argument("--moves-suffix", type=str, required=True, help="e.g. noise0p01 or noise0, matching run_lindgren_markov_sweep.py's naming")
    p.add_argument("--role", choices=["primary", "sensitivity"], required=True)
    p.add_argument("--seeds", type=int, default=70)
    p.add_argument("--seed-offset", type=int, default=100)
    p.add_argument("--window-start", type=int, default=1000)
    p.add_argument("--window-end", type=int, default=5000)
    p.add_argument("--probe-interval", type=int, default=50)
    p.add_argument("--n-boot-point", type=int, default=1000, help="preregistration section 5: point-estimate repeats for TFT/AllD/AllC")
    p.add_argument("--n-boot-descriptive", type=int, default=200, help="bootstrap repeats for the other 4 opponents + scheme A comparison")
    p.add_argument("--n-bootstrap-seed", type=int, default=2000, help="preregistration section 5: seed-level bootstrap iterations")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--master-seed", type=str, default="markov_confirmatory_v1", help="string seed root for all RNG streams -- change only if re-running with intentionally different randomness")
    p.add_argument("--human-weights", type=str, default="docs/human_network_weights.csv")
    p.add_argument(
        "--group-sizes", type=str, default="Secure:56,Mix:46,Fearful:69",
        help="comma-separated human_group:size overrides -- only change for validating "
        "the pipeline against a dataset too small for the real 56/46/69 (e.g. the "
        "10-seed pilot); production runs must use the preregistered 56/46/69.",
    )
    p.add_argument("--results-dir", type=str, required=True)
    args = p.parse_args()

    group_for_human: Dict[str, int] = {}
    for part in args.group_sizes.split(","):
        name, size = part.split(":")
        group_for_human[name] = int(size)
    if set(group_for_human) != {"Secure", "Mix", "Fearful"}:
        raise ValueError(f"--group-sizes must name exactly Secure, Mix, Fearful; got {group_for_human}")

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / f"_age_cache_{args.role}"

    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    ages = list(range(args.window_start, args.window_end + 1, args.probe_interval))

    t0 = time.time()
    descriptive_rows, time_avg_sums, time_avg_counts = run_all_ages(
        data_dir, args.moves_suffix, ages, seeds, args.human_weights,
        args.n_boot_point, args.n_boot_descriptive, args.n_bootstrap_seed,
        args.master_seed, args.workers, cache_dir, group_for_human,
    )
    print(f"all ages done in {time.time() - t0:.0f}s", flush=True)

    descriptive_path = results_dir / f"markov_confirmatory_descriptive_{args.role}.csv"
    fieldnames = list({k for row in descriptive_rows for k in row})
    with open(descriptive_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(descriptive_rows)
    print(f"wrote {len(descriptive_rows)} descriptive rows to {descriptive_path}", flush=True)

    point_estimates = compute_point_estimates(descriptive_rows, (args.window_start, args.window_end), group_for_human)

    results_payload = {
        "role": args.role,
        "preregistration_commit": "ccc0b1b7a76f22e150e7c097b1757b1fb8e54e0e",
        "window": [args.window_start, args.window_end],
        "n_seeds": args.seeds,
        "n_boot_point": args.n_boot_point,
        "n_bootstrap_seed": args.n_bootstrap_seed,
        "point_estimates": {f"{opp}:{g}": v for (opp, g), v in point_estimates.items()},
    }

    # The same H1-H3-style comparisons are computed for BOTH roles: for
    # role=primary this is the preregistered confirmatory test itself
    # (section 4-5); for role=sensitivity, preregistration section 6
    # item 6 asks for "H1〜H3と同じ比較を行い、記述として報告する" (same
    # comparisons, reported descriptively) -- so the statistics are
    # identical, only the framing/results key differs, making clear the
    # sensitivity numbers are not a second confirmatory test and are not
    # subject to the preregistered stopping rule.
    bootstrap_diffs = build_bootstrap_diff_distributions(time_avg_sums, time_avg_counts, args.n_bootstrap_seed)
    point_estimate_diffs = {
        (opponent, groups): comparison_diff(
            {g: point_estimates[(opponent, g)] for g in ("S", "M", "F") if (opponent, g) in point_estimates},
            groups,
        )
        for _hyp, _label, opponent, groups in COMPARISONS
        if all((opponent, g) in point_estimates for g in groups.split("-"))
    }
    missing = [key for key in [(o, g) for _h, _l, o, g in COMPARISONS] if key not in bootstrap_diffs or key not in point_estimate_diffs]
    results_key = "confirmatory" if args.role == "primary" else "descriptive_h1_h3_replication"
    if missing:
        print(f"WARNING: missing data for comparisons {missing}; {results_key} results incomplete", flush=True)
        results_payload[results_key] = None
    else:
        hypothesis_results = evaluate_hypotheses(point_estimate_diffs, bootstrap_diffs, alpha=args.alpha)
        results_payload[results_key] = [
            {
                "hypothesis": h.hypothesis,
                "opponent": h.opponent,
                "verdict": h.verdict,
                "comparisons": [
                    {
                        "label": c.label,
                        "point_estimate_diff": c.point_estimate_diff,
                        "p_value": c.p_value,
                        "significant_after_holm": c.significant,
                    }
                    for c in h.comparisons
                ],
            }
            for h in hypothesis_results
        ]
    if args.role == "sensitivity":
        results_payload["note"] = (
            "role=sensitivity (noise_rate=0.0): preregistration section 6 item 6 -- "
            "the 'descriptive_h1_h3_replication' block applies the same statistics as the "
            "primary confirmatory test (section 4-5), but is NOT a second confirmatory test "
            "and is not subject to the preregistered stopping rule; report descriptively."
        )

    results_path = results_dir / f"markov_confirmatory_results_{args.role}.json"
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {results_path}", flush=True)
    print(f"total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
