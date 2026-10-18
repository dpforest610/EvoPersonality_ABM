import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "analyze_markov_confirmatory_regime_correspondence",
    REPO_ROOT / "scripts" / "analyze_markov_confirmatory_regime_correspondence.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_markov_confirmatory_regime_correspondence"] = mod
spec.loader.exec_module(mod)


def test_regime_switches_counts_transitions():
    # payoff sequence: defect, defect, coop, coop, defect -- 2 switches
    rows = [(0, 1.0, 1.0), (1, 1.0, 1.5), (2, 1.0, 2.5), (3, 1.0, 3.0), (4, 1.0, 1.0)]
    regimes, switches = mod.regime_switches(rows)
    assert regimes == ["defect", "defect", "coop", "coop", "defect"]
    assert switches == [2, 4]


def test_regime_switches_no_switch_when_constant_regime():
    rows = [(g, 1.0, 3.0) for g in range(5)]
    _regimes, switches = mod.regime_switches(rows)
    assert switches == []


def test_regime_switches_threshold_is_inclusive_of_coop_threshold():
    rows = [(0, 1.0, 1.0), (1, 1.0, mod.COOP_THRESHOLD)]
    regimes, _switches = mod.regime_switches(rows)
    assert regimes[1] == "coop"  # exactly at threshold counts as cooperation


def test_mean_ignoring_nan_basic():
    assert mod.mean_ignoring_nan([1.0, 3.0]) == 2.0


def test_load_generation_stats_parses_rows(tmp_path):
    path = tmp_path / "seed_0_generation_stats.csv"
    path.write_text("generation,mean_depth,std_depth,min_depth,max_depth,mean_payoff\n0,1.5,0.1,1,2,2.3\n1,1.6,0.1,1,2,2.4\n")
    rows = mod.load_generation_stats(path)
    assert rows == [(0, 1.5, 2.3), (1, 1.6, 2.4)]
