"""Probe (virtual-participant) infrastructure for the Lindgren behavioral-
Markov-structure research line -- see
docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md sections 2-2.5.

Two pieces:
  - `census_genotypes`: dedupe a population of LindgrenTable genomes by
    (memory_depth, table, prehistory) and count frequencies (2.2).
  - `OPPONENTS` / `make_opponent_agent` / `run_probe_duel`: the fixed
    main-set and extended-set opponents (2.3), each dueled against with
    the genome's OWN prehistory (2.4) and the same noise machinery as the
    evolution run (duel.run_duel's noise_rate/rng).
"""

import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .agent import Agent
from .duel import run_duel
from .game import COOPERATE, DEFECT
from .lindgren import (
    LindgrenTable,
    always_cooperate_table,
    always_defect_table,
    pavlov_table,
    tit_for_tat_table,
)
from .strategies import make_gradual_strategy, random_strategy_factory

GenotypeKey = Tuple[int, Tuple[int, ...], Tuple[int, ...]]


def genotype_key(table: LindgrenTable) -> GenotypeKey:
    return (table.memory_depth, table.table, table.prehistory)


def census_genotypes(tables: Sequence[LindgrenTable]) -> List[Tuple[GenotypeKey, int, LindgrenTable]]:
    """Dedupe a population by (memory_depth, table, prehistory); returns
    [(key, frequency, a_representative_LindgrenTable), ...], one entry per
    unique genotype, sorted by descending frequency (ties broken by key,
    for a deterministic order)."""
    counts: "Counter[GenotypeKey]" = Counter(genotype_key(t) for t in tables)
    representative: Dict[GenotypeKey, LindgrenTable] = {}
    for t in tables:
        representative.setdefault(genotype_key(t), t)
    return sorted(
        ((k, n, representative[k]) for k, n in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )


# Opponent "factories": each returns a fresh Agent.strategy-compatible
# callable (or a LindgrenTable, for the fixed-table opponents reused
# unchanged from lindgren.py) plus the markov_recovery vocabulary name to
# use for that opponent. A fresh callable per duel matters for the
# stateful ones (Gradual) -- see strategies.make_gradual_strategy.
# vocab "GRA" is the full 64-word (unrestricted) vocabulary (design doc
# section 3.1) -- used for Random, Gradual and the extended-set opponents
# (GradualAppeasement, WSLS) alike, per design doc section 3.4.
OPPONENTS: Dict[str, Dict] = {
    "TFT": {"table": tit_for_tat_table, "vocab": "TFT"},
    "AllC": {"table": always_cooperate_table, "vocab": "AllC"},
    "AllD": {"table": always_defect_table, "vocab": "AllD"},
    "Random": {"strategy_factory": random_strategy_factory, "vocab": "RAN"},
    "Gradual": {"strategy_factory": lambda rng: make_gradual_strategy(appeasement=False), "vocab": "GRA"},
    "GradualAppeasement": {"strategy_factory": lambda rng: make_gradual_strategy(appeasement=True), "vocab": "GRA"},
    "WSLS": {"table": pavlov_table, "vocab": "GRA"},
}

MAIN_SET = ["TFT", "AllC", "AllD", "Random", "Gradual"]
EXTENDED_SET = ["GradualAppeasement", "WSLS"]


def make_opponent_strategy(name: str, rng: random.Random) -> Callable:
    """A fresh Agent.strategy-compatible callable for one probe duel."""
    spec = OPPONENTS[name]
    if "table" in spec:
        return spec["table"]().as_strategy()
    return spec["strategy_factory"](rng)


def run_probe_duel(
    genome_table: LindgrenTable,
    opponent_name: str,
    rng: random.Random,
    rounds: int = 100,
    noise_rate: float = 0.0,
) -> Tuple[List[str], List[str]]:
    """One probe duel: genome_table (using its OWN prehistory, same as in
    the population's own evolution duels -- design doc section 2.4) vs the
    named fixed opponent. Returns (own_moves, opp_moves), each length
    `rounds`."""
    focal = Agent("focal", genome_table.as_strategy())
    opponent = Agent("opponent", make_opponent_strategy(opponent_name, rng))
    run_duel(focal, opponent, rounds=rounds, noise_rate=noise_rate, rng=rng)
    return focal.history_self, opponent.history_self


def run_selfplay_duel(
    focal_table: LindgrenTable,
    partner_table: LindgrenTable,
    rng: random.Random,
    rounds: int = 100,
    noise_rate: float = 0.0,
) -> Tuple[List[str], List[str]]:
    """Exploratory, outside preregistration (docs/EXPLORATORY_selfplay_recovery.md):
    one duel between two EVOLVED LindgrenTables -- focal_table (using its
    own prehistory, same convention as run_probe_duel) against
    partner_table (likewise its own prehistory; for a "SELF" duel,
    partner_table IS focal_table -- two independent Agent instances
    wrapping the same table, which is safe since LindgrenTable.decide()
    is a pure function of the history each Agent tracks itself; for a
    "POP" duel, partner_table is a different genotype drawn from the same
    (seed, age) population). Returns (own_moves, opp_moves), each length
    `rounds`, from focal_table's point of view."""
    focal = Agent("focal", focal_table.as_strategy())
    partner = Agent("partner", partner_table.as_strategy())
    run_duel(focal, partner, rounds=rounds, noise_rate=noise_rate, rng=rng)
    return focal.history_self, partner.history_self


def recovery_time_from_forced_bottom(
    focal_table: LindgrenTable,
    opponent_table: LindgrenTable,
    rng: random.Random,
    free_rounds: int = 20,
    noise_rate: float = 0.01,
) -> Optional[int]:
    """Exploratory, outside preregistration, NOT one of the human study's
    own measures (docs/EXPLORATORY_selfplay_recovery.md section 3): forces
    the first 3 rounds to mutual defection (DD, i.e. the 底底底 state),
    then lets focal_table and opponent_table play `free_rounds` more
    rounds freely (same per-round "trembling hand" noise convention as
    duel.run_duel). Returns the 1-indexed round, WITHIN the free phase,
    at which 2 CONSECUTIVE rounds of mutual cooperation (CC) first occur
    -- "recovery time from the bottom" -- or None if it never happens
    within free_rounds (right-censored).

    Both sides are plain LindgrenTables here (TFT/WSLS/a population
    partner/the focal table itself for SELF), decided directly via
    `decide()` rather than through Agent/run_duel, since this needs a
    hand-forced initial history that run_duel has no hook for."""
    own_history = [DEFECT, DEFECT, DEFECT]
    opp_history = [DEFECT, DEFECT, DEFECT]

    for round_i in range(1, free_rounds + 1):
        own_action = focal_table.decide(own_history, opp_history)
        opp_action = opponent_table.decide(opp_history, own_history)
        if noise_rate and rng.random() < noise_rate:
            own_action = DEFECT if own_action == COOPERATE else COOPERATE
        if noise_rate and rng.random() < noise_rate:
            opp_action = DEFECT if opp_action == COOPERATE else COOPERATE

        own_history.append(own_action)
        opp_history.append(opp_action)

        if (
            own_action == COOPERATE and opp_action == COOPERATE
            and own_history[-2] == COOPERATE and opp_history[-2] == COOPERATE
        ):
            return round_i

    return None
