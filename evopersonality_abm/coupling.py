"""Two-channel coupling (K_out, K_self) layered on the Lindgren table --
HBES2026_ABM骨子_二成分K.md §3.

Each agent's genome grows a `LindgrenTable` (the deliberate strategy,
untouched -- see `lindgren.py`) plus two coupling genes in [0,1]:

  - K_self: deliberate-channel weight -- how strongly the agent follows
    its own Lindgren table's recommendation.
  - K_out: automatic-channel weight -- how strongly it is pulled toward
    the opponent's most recently observed action, independent of any
    strategic evaluation (co-regulation).

Per round, an internal state s_i(t) in [0,1] (current cooperation
propensity) blends the two channels (§3.2):

    s_i(t) = s_i(t-1) + K_self*(r_i - s_i(t-1)) + K_out*(a_j - s_i(t-1)) + noise

  - r_i: the agent's own Lindgren table decision, looked up against its
    *actual* play history so far (same `decide()` used by the pure
    Lindgren duel).
  - a_j: the opponent's most recently *observed* action (0.5, a neutral
    prior, before any round has been played).

s is clipped to [0,1]; the actual action played is a Bernoulli draw,
a_i(t) ~ Bernoulli(s_i(t)) -- so what an agent actually does can diverge
from what its table recommends, and that divergence is what "co-regulate
without a strategy behind it" (K_out) vs. "hold the line" (K_self) means.

No spatial structure yet -- this sits directly on top of the existing
round-robin population loop in `population.py`.
"""

import itertools
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .game import COOPERATE, DEFECT, payoff
from .lindgren import LindgrenTable
from .population import mutate_individual, random_table, select_and_reproduce

Bit = int

_BIT_OF_ACTION = {COOPERATE: 1, DEFECT: 0}
_ACTION_OF_BIT = {1: COOPERATE, 0: DEFECT}

INITIAL_STATE = 0.5
NEUTRAL_PRIOR = 0.5  # a_j before any real opponent action has been observed


@dataclass(frozen=True)
class Genome:
    """One agent's genotype: a Lindgren strategy plus its coupling genes."""

    table: LindgrenTable
    k_out: float
    k_self: float


def random_genome(memory_depth: int, rng: random.Random) -> Genome:
    """A genome with a random table and (K_out, K_self) drawn uniformly
    from [0,1]^2 -- the starting condition §5 asks for."""
    return Genome(
        table=random_table(memory_depth, rng),
        k_out=rng.random(),
        k_self=rng.random(),
    )


def update_internal_state(
    s_prev: float, k_self: float, r: Bit, k_out: float, a: float, noise: float
) -> float:
    """One step of the §3.2 update, clipped to [0,1]."""
    s = s_prev + k_self * (r - s_prev) + k_out * (a - s_prev) + noise
    return min(1.0, max(0.0, s))


def run_coupled_duel(
    genome_a: Genome,
    genome_b: Genome,
    rounds: int,
    rng: random.Random,
    noise_std: float = 0.05,
    track_cooperation: bool = False,
) -> Tuple[float, float]:
    """Iterated PD where each side's action is drawn from its coupled
    internal state rather than read directly off its Lindgren table.
    Returns (total_payoff_a, total_payoff_b) by default; with
    track_cooperation=True, returns a 4-tuple with two more ints appended
    (cooperate_count_a, cooperate_count_b) -- opt-in and appended rather
    than inserted, so every existing 2-tuple-unpacking caller is
    unaffected."""
    history_self_a: List[str] = []
    history_opp_a: List[str] = []
    history_self_b: List[str] = []
    history_opp_b: List[str] = []

    s_a = INITIAL_STATE
    s_b = INITIAL_STATE
    last_action_a: Optional[str] = None
    last_action_b: Optional[str] = None

    total_a = 0.0
    total_b = 0.0
    coop_count_a = 0
    coop_count_b = 0

    for _ in range(rounds):
        r_a = _BIT_OF_ACTION[genome_a.table.decide(history_self_a, history_opp_a)]
        r_b = _BIT_OF_ACTION[genome_b.table.decide(history_self_b, history_opp_b)]

        a_for_a = NEUTRAL_PRIOR if last_action_b is None else _BIT_OF_ACTION[last_action_b]
        a_for_b = NEUTRAL_PRIOR if last_action_a is None else _BIT_OF_ACTION[last_action_a]

        s_a = update_internal_state(
            s_a, genome_a.k_self, r_a, genome_a.k_out, a_for_a, rng.gauss(0, noise_std)
        )
        s_b = update_internal_state(
            s_b, genome_b.k_self, r_b, genome_b.k_out, a_for_b, rng.gauss(0, noise_std)
        )

        action_a = _ACTION_OF_BIT[1 if rng.random() < s_a else 0]
        action_b = _ACTION_OF_BIT[1 if rng.random() < s_b else 0]

        payoff_a, payoff_b = payoff(action_a, action_b)
        total_a += payoff_a
        total_b += payoff_b
        if action_a == COOPERATE:
            coop_count_a += 1
        if action_b == COOPERATE:
            coop_count_b += 1

        history_self_a.append(action_a)
        history_opp_a.append(action_b)
        history_self_b.append(action_b)
        history_opp_b.append(action_a)
        last_action_a, last_action_b = action_a, action_b

    if track_cooperation:
        return total_a, total_b, coop_count_a, coop_count_b
    return total_a, total_b


def round_robin_payoffs_coupled(
    genomes: List[Genome], rounds_per_duel: int, rng: random.Random, noise_std: float
) -> List[float]:
    """Play every pair once; return each agent's mean payoff per round."""
    n = len(genomes)
    total_payoff = [0.0] * n
    for i, j in itertools.combinations(range(n), 2):
        payoff_i, payoff_j = run_coupled_duel(
            genomes[i], genomes[j], rounds_per_duel, rng, noise_std
        )
        total_payoff[i] += payoff_i
        total_payoff[j] += payoff_j

    n_opponents = n - 1
    if n_opponents <= 0:
        return [0.0] * n
    return [total / (n_opponents * rounds_per_duel) for total in total_payoff]


def mutate_genome(
    genome: Genome,
    rng: random.Random,
    point_mutation_rate: float,
    duplication_rate: float,
    split_rate: float,
    max_memory_depth: int,
    min_memory_depth: int,
    k_mutation_std: float,
) -> Genome:
    """Mutate the table exactly as in the pure Lindgren loop, plus a small
    independent Gaussian nudge to each of K_out/K_self, clipped to [0,1]
    ("微小変異" -- small mutation, not a rare discrete event like
    duplication/split)."""
    new_table = mutate_individual(
        genome.table,
        rng,
        point_mutation_rate,
        duplication_rate,
        max_memory_depth,
        split_rate=split_rate,
        min_memory_depth=min_memory_depth,
    )
    new_k_out = min(1.0, max(0.0, genome.k_out + rng.gauss(0, k_mutation_std)))
    new_k_self = min(1.0, max(0.0, genome.k_self + rng.gauss(0, k_mutation_std)))
    return Genome(table=new_table, k_out=new_k_out, k_self=new_k_self)


@dataclass
class CoupledGenerationStats:
    generation: int
    k_out: List[float]
    k_self: List[float]
    memory_depths: List[int]
    payoffs: List[float]

    @property
    def mean_payoff(self) -> float:
        return statistics.fmean(self.payoffs)

    @property
    def mean_k_out(self) -> float:
        return statistics.fmean(self.k_out)

    @property
    def mean_k_self(self) -> float:
        return statistics.fmean(self.k_self)


def run_coupled_evolution(
    population_size: int = 100,
    generations: int = 300,
    rounds_per_duel: int = 50,
    initial_memory_depth: int = 1,
    point_mutation_rate: float = 0.05,
    duplication_rate: float = 0.02,
    split_rate: float = 0.02,
    max_memory_depth: int = 8,
    min_memory_depth: int = 0,
    k_mutation_std: float = 0.05,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> List[CoupledGenerationStats]:
    """Run the K-coupled population loop and return per-generation stats.

    Same generational structure as `population.run_evolution` (round-robin
    -> record -> fitness-proportional selection -> mutate), but genomes
    carry (K_out, K_self) alongside the Lindgren table, starting from a
    uniform distribution over [0,1]^2 (§5).
    """
    rng = random.Random(seed)
    genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]

    stats: List[CoupledGenerationStats] = []
    for gen in range(generations):
        payoffs = round_robin_payoffs_coupled(genomes, rounds_per_duel, rng, noise_std)
        stats.append(
            CoupledGenerationStats(
                generation=gen,
                k_out=[g.k_out for g in genomes],
                k_self=[g.k_self for g in genomes],
                memory_depths=[g.table.memory_depth for g in genomes],
                payoffs=payoffs,
            )
        )

        genomes = select_and_reproduce(genomes, payoffs, rng)
        genomes = [
            mutate_genome(
                g,
                rng,
                point_mutation_rate,
                duplication_rate,
                split_rate,
                max_memory_depth,
                min_memory_depth,
                k_mutation_std,
            )
            for g in genomes
        ]

    return stats
