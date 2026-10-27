import importlib.util
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_pilot_spec = importlib.util.spec_from_file_location("run_coupling_markov_pilot", SCRIPTS_DIR / "run_coupling_markov_pilot.py")
pilot_mod = importlib.util.module_from_spec(_pilot_spec)
sys.modules["run_coupling_markov_pilot"] = pilot_mod
_pilot_spec.loader.exec_module(pilot_mod)

_mod_spec = importlib.util.spec_from_file_location("run_k_model_confirmatory_analysis", SCRIPTS_DIR / "run_k_model_confirmatory_analysis.py")
mod = importlib.util.module_from_spec(_mod_spec)
sys.modules["run_k_model_confirmatory_analysis"] = mod
_mod_spec.loader.exec_module(mod)

from evopersonality_abm.coupling import random_genome  # noqa: E402


def make_tiny_production_data(tmp_path, n_seeds=6, n_genomes=12, age=1000):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(n_seeds):
        rng = random.Random(seed)
        genomes = [random_genome(1, rng) for _ in range(n_genomes)]
        pilot_mod.run_probes_for_age(
            genomes, age=age, seed=seed, out_dir=data_dir,
            probe_repeats=3, probe_rounds=10, free_rounds=5, probe_noise_std=0.05, master_seed=f"t{seed}",
        )
    return data_dir


def median_thresholds(data_dir, n_seeds, age):
    import gzip
    import csv
    k_outs, k_selfs = [], []
    for seed in range(n_seeds):
        with gzip.open(pilot_mod.census_path(data_dir, seed, age), "rt", newline="") as f:
            for row in csv.DictReader(f):
                k_outs.append(float(row["k_out"]))
                k_selfs.append(float(row["k_self"]))
    k_outs.sort()
    k_selfs.sort()
    return {"threshold_kout": k_outs[len(k_outs) // 2], "threshold_kself": k_selfs[len(k_selfs) // 2]}


# ------------------------------------------------------------------ pure helper functions

def test_permute_type_labels_preserves_per_type_count():
    gid_to_true_type = {0: "K_Secure", 1: "K_Secure", 2: "K_Fearful", 3: "K_Dismissing"}
    rng = random.Random(0)
    permuted = mod.permute_type_labels(gid_to_true_type, rng)
    assert sorted(permuted.keys()) == sorted(gid_to_true_type.keys())
    from collections import Counter
    assert Counter(permuted.values()) == Counter(gid_to_true_type.values())


def test_merge_type_membership_combines_disjoint_gids():
    a = ({0: 1, 1: 2}, {0: ("id0",), 1: ("id1",)})
    b = ({2: 3}, {2: ("id2",)})
    freq, idents = mod.merge_type_membership(a, b)
    assert freq == {0: 1, 1: 2, 2: 3}
    assert idents == {0: ("id0",), 1: ("id1",), 2: ("id2",)}


def test_draw_cross_lineage_identities_uses_distinct_seeds():
    type_pools = {s: ({0: 1}, {0: (1, "t", "p")}) for s in range(5)}
    rng = random.Random(0)
    identities = mod.draw_cross_lineage_identities(type_pools, 3, rng)
    seeds_used = [s for s, _g in identities]
    assert len(identities) == 3
    assert len(set(seeds_used)) == 3  # distinct seeds


def test_draw_cross_lineage_identities_raises_if_too_few_seeds():
    type_pools = {0: ({0: 1}, {0: (1, "t", "p")})}
    rng = random.Random(0)
    try:
        mod.draw_cross_lineage_identities(type_pools, 5, rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_concatenate_edges_prefixes_by_opponent():
    edges_by_opp = {"TFT": {("a", "b"): 0.5}, "Random": {("a", "b"): 0.2}, "AllC": {}, "AllD": {}, "Gradual": {}}
    concatenated = mod.concatenate_edges(edges_by_opp)
    assert concatenated[("TFT", "a", "b")] == 0.5
    assert concatenated[("Random", "a", "b")] == 0.2


def test_correlation_concatenated_perfect_match():
    v = {("TFT", "a", "b"): 1.0, ("TFT", "c", "d"): 0.5}
    assert abs(mod.correlation_concatenated(v, v) - 1.0) < 1e-9


def test_window_average_rate_averages_per_age_rates():
    assert mod.window_average_rate([(10, 5), (10, 3)]) == 0.4  # (0.5 + 0.3) / 2


# ------------------------------------------------------------------ end-to-end smoke

def test_process_one_age_and_build_results_smoke(tmp_path):
    n_seeds = 6
    age = 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=12, age=age)
    thresholds = median_thresholds(data_dir, n_seeds, age)

    seeds = list(range(n_seeds))
    seedboot_resamples = mod.build_seedboot_resamples(seeds, 3, "test")
    tiny_group_size = {"K_Secure": 2, "Mix": 2, "K_Fearful": 2}

    result = mod.process_one_age(
        str(data_dir), age, seeds, thresholds["threshold_kout"], thresholds["threshold_kself"],
        str(REPO_ROOT / "docs" / "human_network_weights.csv"),
        n_draws_main=2, seedboot_resamples=seedboot_resamples,
        n_permutations=2, permutation_draws_ha=2, master_seed="test",
        ha_group_size=tiny_group_size,
    )
    assert result["age"] == age
    assert set(result["ha_point_edges"].keys()) == {"K_Secure", "Mix", "K_Fearful"}

    human_vectors = mod.build_human_concatenated_vectors(REPO_ROOT / "docs" / "human_network_weights.csv")
    summary = mod.build_results([result], human_vectors)
    for key in ("ha", "hb", "hc"):
        assert key in summary
    assert "testable_strata" in summary["hc"]


def test_main_writes_summary_file(tmp_path):
    n_seeds = 6
    age = 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=12, age=age)
    thresholds = median_thresholds(data_dir, n_seeds, age)
    thresholds_path = tmp_path / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f)

    results_dir = tmp_path / "out"
    sys_argv_backup = sys.argv
    sys.argv = [
        "run_k_model_confirmatory_analysis.py",
        "--data-dir", str(data_dir), "--seeds", str(n_seeds), "--seed-offset", "0",
        "--thresholds-json", str(thresholds_path),
        "--window-start", str(age), "--window-end", str(age),
        "--n-draws-main", "2", "--n-bootstrap-seed", "3", "--n-permutations", "2",
        "--permutation-draws-ha", "2", "--workers", "1",
        "--group-size-secure", "2", "--group-size-mix", "2", "--group-size-fearful", "2",
        "--results-dir", str(results_dir),
    ]
    try:
        mod.main()
    finally:
        sys.argv = sys_argv_backup

    out_path = results_dir / "k_model_confirmatory_summary.json"
    assert out_path.exists()
    with open(out_path) as f:
        summary = json.load(f)
    assert set(summary.keys()) == {"ha", "hb", "hc"}


def test_gid_effective_depth_handles_memory_depth_zero():
    # Regression: a memory-depth-0 genotype has an empty prehistory, written
    # to census.csv.gz as "" -- this crashed the production run with
    # ValueError: invalid literal for int() with base 10: ''.
    from evopersonality_abm.lindgren import LindgrenTable
    t0 = LindgrenTable(0, [1], [])
    t2 = LindgrenTable(1, [0, 1, 1, 0], [1, 0])
    census = []
    for gid, t in ((1, t0), (2, t2)):
        census.append((gid, 1, t.memory_depth, ",".join(map(str, t.table)), ",".join(map(str, t.prehistory)), 0.5, 0.5))
    assert census[0][4] == ""
    depths = mod.gid_effective_depth(census)
    assert set(depths) == {1, 2}
    assert depths[1] == mod.effective_memory_depth(t0)
    assert mod._parse_bits("") == []
    assert mod._parse_bits("1,0,1") == [1, 0, 1]


def test_recovery_counts_by_label_matches_naive_row_scan():
    # The pre-aggregated counts must equal a direct scan of every record,
    # for true labels, weighted (bootstrap) seeds and permuted labels.
    rng = random.Random(3)
    types = ["K_Secure", "K_Fearful", "K_Dismissing", "K_Preoccupied"]
    recovery_by_seed, depth_by_seed, label_by_seed = {}, {}, {}
    for seed in range(5):
        gids = list(range(15))
        depth_by_seed[seed] = {g: rng.randint(0, 5) for g in gids}
        label_by_seed[seed] = {g: rng.choice(types) for g in gids}
        recovery_by_seed[seed] = [
            (rng.choice(gids), rng.choice(["SELF", "POP", "TFT"]), r, rng.choice([None, 2, 5]))
            for r in range(200)
        ]
    counts = mod.recovery_counts_by_gid(recovery_by_seed, depth_by_seed)
    weights = {0: 2, 1: 0, 3: 1, 4: 3}
    for w in (None, weights):
        got = mod.recovery_counts_by_label(counts, label_by_seed, w)
        for t in ("K_Secure", "K_Fearful"):
            for opp in ("SELF", "POP"):
                n = nr = 0
                for seed, rows in recovery_by_seed.items():
                    ww = 1 if w is None else w.get(seed, 0)
                    for gid, o, _r, rr in rows:
                        if o == opp and label_by_seed[seed][gid] == t:
                            n += ww
                            nr += ww * (rr is not None)
                assert got.get((t, ("opp", opp)), (0, 0)) == (n, nr)
            for stratum in mod.HC_STRATA:
                n = nr = 0
                for seed, rows in recovery_by_seed.items():
                    ww = 1 if w is None else w.get(seed, 0)
                    for gid, o, _r, rr in rows:
                        if o == "SELF" and label_by_seed[seed][gid] == t and mod.depth_stratum(depth_by_seed[seed][gid]) == stratum:
                            n += ww
                            nr += ww * (rr is not None)
                assert got.get((t, ("stratum", stratum)), (0, 0)) == (n, nr)


def test_load_series_first_rounds_matches_full_loader(tmp_path):
    data_dir = make_tiny_production_data(tmp_path, n_seeds=2, n_genomes=6, age=1000)
    for seed in range(2):
        path = pilot_mod.moves_path(data_dir, seed, 1000)
        # probe_rounds=10 in the tiny data: also check a cut below the duel length
        for n in (20, 5):
            assert mod.load_series_first_rounds(path, n) == mod.viewer_mod.load_series_by_genotype_all_opponents(path, n)


def test_process_one_age_uses_available_lineages_when_a_type_is_scarce(tmp_path):
    # docs/DEVIATION_k_model_group_size.md: a group size larger than the
    # number of lineages carrying the type must shrink to that number
    # (point estimate, seed bootstrap and permutations alike) instead of
    # raising, and bootstrap positions without the type must be skipped.
    n_seeds = 6
    age = 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=3, age=age)
    thresholds = median_thresholds(data_dir, n_seeds, age)
    seeds = list(range(n_seeds))
    big = {"K_Secure": 50, "Mix": 50, "K_Fearful": 50}
    result = mod.process_one_age(
        str(data_dir), age, seeds, thresholds["threshold_kout"], thresholds["threshold_kself"],
        str(REPO_ROOT / "docs" / "human_network_weights.csv"),
        n_draws_main=3, seedboot_resamples=mod.build_seedboot_resamples(seeds, 20, "t"),
        n_permutations=5, permutation_draws_ha=2, master_seed="t", ha_group_size=big,
    )
    census = {s: mod.viewer_mod.load_census_with_k(pilot_mod.census_path(data_dir, s, age)) for s in seeds}
    for type_name, gs in result["ha_effective_group_size"].items():
        labels = {s: mod.census_true_types(c, thresholds["threshold_kout"], thresholds["threshold_kself"]) for s, c in census.items()}
        members = {"Mix": {"K_Dismissing", "K_Preoccupied"}}.get(type_name, {type_name})
        n_lineages = sum(1 for s in seeds if any(t in members for t in labels[s].values()))
        assert gs == min(50, n_lineages)
    assert any(gs < n_seeds for gs in result["ha_effective_group_size"].values())


def test_hb_hc_statistics_are_never_recovered_fearful_minus_secure(tmp_path):
    # Regression for the sign error: the preregistered H-B/H-C statistic is
    # never_recovered(K_Fearful) - never_recovered(K_Secure). Build it by
    # hand from the per-age counts and compare with what the pipeline reports.
    n_seeds, age = 6, 1000
    data_dir = make_tiny_production_data(tmp_path, n_seeds=n_seeds, n_genomes=12, age=age)
    th = median_thresholds(data_dir, n_seeds, age)
    seeds = list(range(n_seeds))
    r = mod.process_one_age(
        str(data_dir), age, seeds, th["threshold_kout"], th["threshold_kself"],
        str(REPO_ROOT / "docs" / "human_network_weights.csv"),
        n_draws_main=2, seedboot_resamples=mod.build_seedboot_resamples(seeds, 3, "s"),
        n_permutations=2, permutation_draws_ha=2, master_seed="s",
        ha_group_size={"K_Secure": 2, "Mix": 2, "K_Fearful": 2},
    )
    assert r["recovery_diff_convention"] == "never_recovered"
    for hyp, label, opp in mod.HB_COMPARISONS:
        if ("K_Fearful", opp) in r["hb_point"] and ("K_Secure", opp) in r["hb_point"]:
            never = {t: 1 - r["hb_point"][(t, opp)][1] / r["hb_point"][(t, opp)][0] for t in ("K_Secure", "K_Fearful")}
            expected = never["K_Fearful"] - never["K_Secure"]
            assert abs(r["hb_perm_true"][label] - expected) < 1e-12
            human_vectors = mod.build_human_concatenated_vectors(REPO_ROOT / "docs" / "human_network_weights.csv")
            summary = mod.build_results([r], human_vectors)
            row = [x for x in summary["hb"]["rows"] if x["comparison"] == label][0]
            assert abs(row["point_diff"] - expected) < 1e-12


def test_normalize_cached_result_negates_old_recovery_diffs():
    old = {
        "hb_seedboot_diffs": {"a": {0: 0.1}}, "hc_seedboot_diffs": {"b": {0: -0.2}},
        "hb_perm_null": {"a": {0: 0.3}}, "hc_perm_null": {"b": {1: 0.4}},
        "hb_perm_true": {"a": 0.5}, "hc_perm_true": {"b": -0.6},
        "ha_perm_true": {"x": 0.7},
    }
    new = mod.normalize_cached_result(old)
    assert new["hb_seedboot_diffs"] == {"a": {0: -0.1}} and new["hc_seedboot_diffs"] == {"b": {0: 0.2}}
    assert new["hb_perm_null"] == {"a": {0: -0.3}} and new["hc_perm_null"] == {"b": {1: -0.4}}
    assert new["hb_perm_true"] == {"a": -0.5} and new["hc_perm_true"] == {"b": 0.6}
    assert new["ha_perm_true"] == {"x": 0.7}  # H-A untouched
    assert mod.normalize_cached_result(new) is new  # idempotent on new caches
