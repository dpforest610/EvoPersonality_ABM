"""Probe infrastructure for the K-model (two-channel coupling) Markov-
structure investigation -- docs/K_MODEL_MARKOV_DESIGN.md section 8
(stage 2, Annen's decisions of 2026-10-23).

Mirrors evopersonality_abm/probe.py's pure-Lindgren probe design (fixed
opponent set, genotype census, forced-bottom recovery time) but adapted
for Genome (table + k_out + k_self), per Annen's decision 2:

  - Fixed opponents (TFT/AllC/AllD/Random/Gradual/WSLS) are read
    DIRECTLY and deterministically -- LindgrenTable.decide() or the
    strategies.py closure -- with NO K layer and NO noise of their own.
  - SELF (the focal genome against an independent copy of itself) and
    POP (the focal genome against another genome drawn from the same
    population) both go through the K layer on BOTH sides, i.e. the
    genuine evopersonality_abm.coupling update_internal_state dynamics.

coupling.run_coupled_duel is NOT reused or modified here: it returns
payoffs only (no move histories, needed for the 3-gram network), and
assumes both sides are symmetric Genomes (the fixed-opponent case is
asymmetric). This module reimplements the same per-round update
(`update_internal_state`, `coupling.py`'s own tested function, IS
reused unchanged) in two duel loops that capture full move histories
instead.

Noise (docs/K_MODEL_MARKOV_DESIGN.md section 8-4): the K-model's own
noise_std (default 0.05, a Gaussian term inside the internal-state
update) is used for every K-layer side (focal in all duels, partner in
SELF/POP). It is NOT the same kind of noise as pure Lindgren's
noise_rate (a post-hoc action flip) and the two are not set equal. A
fixed opponent (decision (b)) carries no noise of its own at all.
"""

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .coupling import INITIAL_STATE, NEUTRAL_PRIOR, Genome, update_internal_state
from .game import COOPERATE, DEFECT
from .lindgren import (
    LindgrenTable,
    always_cooperate_table,
    always_defect_table,
    pavlov_table,
    tit_for_tat_table,
)
from .strategies import make_gradual_strategy, random_strategy_factory

_BIT_OF_ACTION = {COOPERATE: 1, DEFECT: 0}
_ACTION_OF_BIT = {1: COOPERATE, 0: DEFECT}

GenotypeKey = Tuple[int, Tuple[int, ...], Tuple[int, ...], float, float]

# The 6 fixed opponents Annen's decision 2 names (TFT/AllC/AllD/Random/
# Gradual/WSLS -- NOT GradualAppeasement, unlike probe.py's extended
# set). Same vocab convention as probe.OPPONENTS (GRA = the full 64-word
# unrestricted vocabulary, used for Random/Gradual/WSLS alike).
OPPONENTS: Dict[str, Dict] = {
    "TFT": {"table": tit_for_tat_table, "vocab": "TFT"},
    "AllC": {"table": always_cooperate_table, "vocab": "AllC"},
    "AllD": {"table": always_defect_table, "vocab": "AllD"},
    "Random": {"strategy_factory": random_strategy_factory, "vocab": "RAN"},
    "Gradual": {"strategy_factory": lambda rng: make_gradual_strategy(appeasement=False), "vocab": "GRA"},
    "WSLS": {"table": pavlov_table, "vocab": "GRA"},
}

FIXED_OPPONENTS = list(OPPONENTS)  # ["TFT", "AllC", "AllD", "Random", "Gradual", "WSLS"]
SELFPLAY_OPPONENTS = ["SELF", "POP"]


def genotype_key(genome: Genome) -> GenotypeKey:
    """(memory_depth, table, prehistory, k_out, k_self) -- the pure-
    Lindgren genotype_key (evopersonality_abm.probe) extended with exact
    (unrounded) K_out/K_self, per docs/K_MODEL_MARKOV_DESIGN.md section
    8-3's adopted option 1. Because coupling.mutate_genome applies an
    UNCONDITIONAL per-generation Gaussian nudge to K_out/K_self (unlike
    the table's probabilistic point mutation), almost every individual
    is its own distinct genotype under this key -- expected, not a bug;
    see section 8-3 for the reasoning."""
    t = genome.table
    return (t.memory_depth, t.table, t.prehistory, genome.k_out, genome.k_self)


def census_genomes(genomes: Sequence[Genome]) -> List[Tuple[GenotypeKey, int, Genome]]:
    """[(key, frequency, a_representative_Genome), ...], sorted by
    descending frequency (ties broken by key) -- same contract as
    probe.census_genotypes, generalized to Genome."""
    counts: Dict[GenotypeKey, int] = {}
    representative: Dict[GenotypeKey, Genome] = {}
    for g in genomes:
        k = genotype_key(g)
        counts[k] = counts.get(k, 0) + 1
        representative.setdefault(k, g)
    return sorted(
        ((k, n, representative[k]) for k, n in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )


def make_fixed_opponent_strategy(name: str, rng: random.Random) -> Callable[[List[str], List[str]], str]:
    """A fresh (own_history, opp_history) -> action callable for one
    probe duel, from the OPPONENT's own point of view (its own_history
    is ITS actions, opp_history is the K-model focal individual's
    actions) -- same calling convention as probe.make_opponent_strategy.
    Deterministic for TFT/AllC/AllD/WSLS (LindgrenTable.decide()); Random
    has its own independent Bernoulli draw (not K-layer noise -- this is
    what "Random" means); Gradual is a fresh stateful closure."""
    spec = OPPONENTS[name]
    if "table" in spec:
        return spec["table"]().as_strategy()
    return spec["strategy_factory"](rng)


def run_k_vs_fixed_duel(
    genome: Genome,
    opponent_name: str,
    rng: random.Random,
    rounds: int = 100,
    noise_std: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """One probe duel: the K-model genome (focal, through the K layer)
    against a fixed, deterministic opponent (no K layer, no noise --
    Annen's decision 2 / design doc section 1-4 option (b)). Returns
    (own_moves, opp_moves) from the focal genome's point of view, each
    length `rounds`."""
    own_history: List[str] = []
    opp_history: List[str] = []
    opponent_strategy = make_fixed_opponent_strategy(opponent_name, rng)

    s = INITIAL_STATE
    last_opp_action: Optional[str] = None

    for _ in range(rounds):
        r = _BIT_OF_ACTION[genome.table.decide(own_history, opp_history)]
        a = NEUTRAL_PRIOR if last_opp_action is None else _BIT_OF_ACTION[last_opp_action]
        s = update_internal_state(s, genome.k_self, r, genome.k_out, a, rng.gauss(0, noise_std))
        own_action = _ACTION_OF_BIT[1 if rng.random() < s else 0]
        opp_action = opponent_strategy(opp_history, own_history)

        own_history.append(own_action)
        opp_history.append(opp_action)
        last_opp_action = opp_action

    return own_history, opp_history


def run_k_selfplay_duel(
    genome_a: Genome,
    genome_b: Genome,
    rng: random.Random,
    rounds: int = 100,
    noise_std: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """One probe duel between two K-model genomes, BOTH through the K
    layer (SELF: genome_a is genome_b, two independent local states --
    safe, since all mutable state here is local to this call, not stored
    on the frozen Genome; POP: genome_b is a different genome from the
    same population). Reimplements coupling.run_coupled_duel's per-round
    update (same update_internal_state call, reused unchanged) but
    returns move histories instead of payoffs -- coupling.py itself is
    not modified. Returns (own_moves, opp_moves) from genome_a's point
    of view."""
    history_self_a: List[str] = []
    history_opp_a: List[str] = []
    history_self_b: List[str] = []
    history_opp_b: List[str] = []

    s_a = INITIAL_STATE
    s_b = INITIAL_STATE
    last_action_a: Optional[str] = None
    last_action_b: Optional[str] = None

    for _ in range(rounds):
        r_a = _BIT_OF_ACTION[genome_a.table.decide(history_self_a, history_opp_a)]
        r_b = _BIT_OF_ACTION[genome_b.table.decide(history_self_b, history_opp_b)]

        a_for_a = NEUTRAL_PRIOR if last_action_b is None else _BIT_OF_ACTION[last_action_b]
        a_for_b = NEUTRAL_PRIOR if last_action_a is None else _BIT_OF_ACTION[last_action_a]

        s_a = update_internal_state(s_a, genome_a.k_self, r_a, genome_a.k_out, a_for_a, rng.gauss(0, noise_std))
        s_b = update_internal_state(s_b, genome_b.k_self, r_b, genome_b.k_out, a_for_b, rng.gauss(0, noise_std))

        action_a = _ACTION_OF_BIT[1 if rng.random() < s_a else 0]
        action_b = _ACTION_OF_BIT[1 if rng.random() < s_b else 0]

        history_self_a.append(action_a)
        history_opp_a.append(action_b)
        history_self_b.append(action_b)
        history_opp_b.append(action_a)
        last_action_a, last_action_b = action_a, action_b

    return history_self_a, history_opp_a


def draw_partner_genome(gids: List, weights: List[int], genomes: Dict, rng: random.Random):
    """One frequency-weighted, with-replacement POP partner draw (may
    coincidentally redraw the focal genotype itself -- same convention
    as export_markov_selfplay_recovery.draw_partner_table)."""
    partner_gid = rng.choices(gids, weights=weights, k=1)[0]
    return genomes[partner_gid]


def k_recovery_time_vs_fixed(
    genome: Genome,
    opponent_name: str,
    rng: random.Random,
    free_rounds: int = 20,
    noise_std: float = 0.05,
    forced_rounds: int = 3,
) -> Optional[int]:
    """Exploratory (mirrors probe.recovery_time_from_forced_bottom, NOT
    one of the human study's own measures -- see docs/EXPLORATORY_
    selfplay_recovery.md section 3 for the pure-Lindgren original this
    is adapted from): forces the first `forced_rounds` rounds to mutual
    defection (底底底), running update_internal_state through them (so
    the focal's internal state s genuinely reflects having just
    experienced 3 rounds at the bottom, rather than being reset to a
    neutral 0.5 at the start of free play -- the action emitted in a
    forced round is overridden to DEFECT regardless of what s/the
    Bernoulli draw would have produced, but s itself, and the table's
    own recommendation r, are computed exactly as in normal play from
    the [forced] history so far), then lets `free_rounds` more rounds
    play out normally against a fixed opponent (no K layer, no noise,
    decision (b)). Returns the 1-indexed round, WITHIN the free phase,
    at which 2 CONSECUTIVE rounds of mutual cooperation (和和和) first
    occur, or None if it never happens within free_rounds."""
    own_history: List[str] = []
    opp_history: List[str] = []
    opponent_strategy = make_fixed_opponent_strategy(opponent_name, rng)

    s = INITIAL_STATE
    last_opp_action: Optional[str] = None

    for round_i in range(1, forced_rounds + free_rounds + 1):
        r = _BIT_OF_ACTION[genome.table.decide(own_history, opp_history)]
        a = NEUTRAL_PRIOR if last_opp_action is None else _BIT_OF_ACTION[last_opp_action]
        s = update_internal_state(s, genome.k_self, r, genome.k_out, a, rng.gauss(0, noise_std))

        if round_i <= forced_rounds:
            own_action = DEFECT
            opp_action = DEFECT
        else:
            own_action = _ACTION_OF_BIT[1 if rng.random() < s else 0]
            opp_action = opponent_strategy(opp_history, own_history)

        own_history.append(own_action)
        opp_history.append(opp_action)
        last_opp_action = opp_action

        if round_i > forced_rounds:
            free_round_i = round_i - forced_rounds
            if (
                own_action == COOPERATE and opp_action == COOPERATE
                and own_history[-2] == COOPERATE and opp_history[-2] == COOPERATE
            ):
                return free_round_i

    return None


def k_recovery_time_vs_genome(
    genome_a: Genome,
    genome_b: Genome,
    rng: random.Random,
    free_rounds: int = 20,
    noise_std: float = 0.05,
    forced_rounds: int = 3,
) -> Optional[int]:
    """Same as k_recovery_time_vs_fixed, but the opponent is ALSO a
    K-model genome (SELF/POP) -- both sides' internal states are forced
    through mutual defection, then both free-play normally. Returns the
    1-indexed free-phase round of the first 2-consecutive-CC from
    genome_a's point of view, or None."""
    history_self_a: List[str] = []
    history_opp_a: List[str] = []
    history_self_b: List[str] = []
    history_opp_b: List[str] = []

    s_a = INITIAL_STATE
    s_b = INITIAL_STATE
    last_action_a: Optional[str] = None
    last_action_b: Optional[str] = None

    for round_i in range(1, forced_rounds + free_rounds + 1):
        r_a = _BIT_OF_ACTION[genome_a.table.decide(history_self_a, history_opp_a)]
        r_b = _BIT_OF_ACTION[genome_b.table.decide(history_self_b, history_opp_b)]
        a_for_a = NEUTRAL_PRIOR if last_action_b is None else _BIT_OF_ACTION[last_action_b]
        a_for_b = NEUTRAL_PRIOR if last_action_a is None else _BIT_OF_ACTION[last_action_a]

        s_a = update_internal_state(s_a, genome_a.k_self, r_a, genome_a.k_out, a_for_a, rng.gauss(0, noise_std))
        s_b = update_internal_state(s_b, genome_b.k_self, r_b, genome_b.k_out, a_for_b, rng.gauss(0, noise_std))

        if round_i <= forced_rounds:
            action_a = DEFECT
            action_b = DEFECT
        else:
            action_a = _ACTION_OF_BIT[1 if rng.random() < s_a else 0]
            action_b = _ACTION_OF_BIT[1 if rng.random() < s_b else 0]

        history_self_a.append(action_a)
        history_opp_a.append(action_b)
        history_self_b.append(action_b)
        history_opp_b.append(action_a)
        last_action_a, last_action_b = action_a, action_b

        if round_i > forced_rounds:
            free_round_i = round_i - forced_rounds
            if (
                action_a == COOPERATE and action_b == COOPERATE
                and history_self_a[-2] == COOPERATE and history_opp_a[-2] == COOPERATE
            ):
                return free_round_i

    return None
