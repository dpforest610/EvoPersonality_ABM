"""Population-level evolution of Lindgren memory-m strategies.

The first step from a single duel to a population: N agents play a
round-robin (all-pairs) iterated Prisoner's Dilemma, payoff determines
how many offspring each genome leaves in the next generation
(fitness-proportional selection), and offspring are mutated with the two
operators already defined in `lindgren.py` -- neutral memory-depth
duplication and point mutation on the table.

No spatial structure, no odor gradient, no K_out/K_self coupling yet
(see HBES2026_ABM骨子_二成分K.md section 4): this module is the
population-level foundation those are layered onto later.
"""

import itertools
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .agent import Agent
from .duel import run_duel
from .lindgren import LindgrenTable, duplicate_memory, point_mutate, split_memory


def random_table(memory_depth: int, rng: random.Random) -> LindgrenTable:
    """A uniformly random lookup table at a fixed memory depth."""
    table_len = 2 ** (2 * memory_depth)
    prehistory_len = 2 * memory_depth
    table = [rng.randrange(2) for _ in range(table_len)]
    prehistory = [rng.randrange(2) for _ in range(prehistory_len)]
    return LindgrenTable(memory_depth=memory_depth, table=table, prehistory=prehistory)


def round_robin_payoffs(
    tables: Sequence[LindgrenTable], rounds_per_duel: int,
    noise_rate: float = 0.0, rng: Optional[random.Random] = None,
) -> List[float]:
    """Play every pair once; return each agent's mean payoff per round.

    Mean-per-round (not total) so payoffs stay comparable regardless of
    population size or duel length.

    noise_rate/rng: passed straight through to duel.run_duel (implementation-
    error noise -- see docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 1.2).
    0.0 (the default) reproduces the original fully deterministic behavior.
    """
    n = len(tables)
    total_payoff = [0.0] * n
    for i, j in itertools.combinations(range(n), 2):
        agent_i = Agent(f"{i}", tables[i].as_strategy())
        agent_j = Agent(f"{j}", tables[j].as_strategy())
        run_duel(agent_i, agent_j, rounds=rounds_per_duel, noise_rate=noise_rate, rng=rng)
        total_payoff[i] += agent_i.score
        total_payoff[j] += agent_j.score

    n_opponents = n - 1
    if n_opponents <= 0:
        return [0.0] * n
    return [total / (n_opponents * rounds_per_duel) for total in total_payoff]


def select_and_reproduce(
    tables: Sequence[LindgrenTable], payoffs: Sequence[float], rng: random.Random
) -> List[LindgrenTable]:
    """Fitness-proportional (roulette-wheel) selection with replacement.

    Higher payoff -> more expected offspring; population size is held
    constant. Payoffs are non-negative (PD payoffs are >= 0), which
    roulette-wheel selection requires.
    """
    return list(rng.choices(tables, weights=payoffs, k=len(tables)))


def mutate_individual(
    table: LindgrenTable,
    rng: random.Random,
    point_mutation_rate: float,
    duplication_rate: float,
    max_memory_depth: int,
    split_rate: float = 0.0,
    min_memory_depth: int = 0,
) -> LindgrenTable:
    """Apply at most one memory-depth mutation and one point mutation.

    Duplication (grow) and split (shrink, its exact inverse -- see
    `lindgren.split_memory`) are mutually exclusive per offspring per
    generation: duplication is tried first, and split is only tried if
    duplication didn't fire. `max_memory_depth`/`min_memory_depth` cap
    how far depth can wander (table size is exponential in depth, so an
    unbounded max is not computationally safe).
    """
    if table.memory_depth < max_memory_depth and rng.random() < duplication_rate:
        table = duplicate_memory(table)
    elif table.memory_depth > min_memory_depth and rng.random() < split_rate:
        table = split_memory(table)
    if rng.random() < point_mutation_rate:
        index = rng.randrange(len(table.table))
        table = point_mutate(table, index)
    return table


@dataclass
class GenerationStats:
    generation: int
    memory_depths: List[int]
    payoffs: List[float]

    @property
    def mean_depth(self) -> float:
        return statistics.fmean(self.memory_depths)

    @property
    def std_depth(self) -> float:
        return statistics.pstdev(self.memory_depths)

    @property
    def min_depth(self) -> int:
        return min(self.memory_depths)

    @property
    def max_depth(self) -> int:
        return max(self.memory_depths)

    @property
    def mean_payoff(self) -> float:
        return statistics.fmean(self.payoffs)


def run_evolution(
    population_size: int = 50,
    generations: int = 40,
    rounds_per_duel: int = 150,
    initial_memory_depth: int = 1,
    point_mutation_rate: float = 0.05,
    duplication_rate: float = 0.02,
    split_rate: float = 0.0,
    max_memory_depth: int = 6,
    min_memory_depth: int = 0,
    seed: Optional[int] = None,
    noise_rate: float = 0.0,
) -> List[GenerationStats]:
    """Run the population loop and return per-generation summary stats.

    Each generation: round-robin tournament -> record stats -> select
    next generation proportional to payoff -> mutate offspring. With
    `split_rate` > 0, memory depth can shrink as well as grow, so it need
    not simply ratchet up to `max_memory_depth` and stay there.

    noise_rate: implementation-error probability for every duel this
    generation (see duel.run_duel). 0.0 (the default) reproduces the
    original fully deterministic behavior exactly. Uses its own RNG stream
    derived from the main one (same split-stream convention as
    spatial.py/scent_spatial.py: `random.Random(rng.random())`).
    """
    rng = random.Random(seed)
    rng_noise = random.Random(rng.random())
    tables = [random_table(initial_memory_depth, rng) for _ in range(population_size)]

    stats: List[GenerationStats] = []
    for gen in range(generations):
        payoffs = round_robin_payoffs(tables, rounds_per_duel, noise_rate=noise_rate, rng=rng_noise)
        depths = [t.memory_depth for t in tables]
        stats.append(GenerationStats(generation=gen, memory_depths=depths, payoffs=payoffs))

        tables = select_and_reproduce(tables, payoffs, rng)
        tables = [
            mutate_individual(
                t,
                rng,
                point_mutation_rate,
                duplication_rate,
                max_memory_depth,
                split_rate=split_rate,
                min_memory_depth=min_memory_depth,
            )
            for t in tables
        ]

    return stats
