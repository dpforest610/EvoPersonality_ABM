"""Spatial layer for the two-channel K model -- 第二章 design memo (chat).

Adds a grid: N agents occupy distinct cells of a W x H torus (periodic
boundary), a strict subset of W*H so there is room to move. A generation
is split into `n_phases` phases; each phase, occupied cells duel their
currently-occupied grid neighbors for `rounds_per_duel // n_phases`
rounds (via the existing `run_coupled_duel`, unmodified), then --
only in the "mobile" condition -- each agent may relocate to an
unoccupied neighboring cell, pulled toward whichever candidate cell's
occupied neighbors have the highest mean payoff-so-far ("individual
scent": an empty cell's attractiveness comes entirely from who
currently lives next to it, never from the cell's own history, so
"good soil" and "good neighbor" can't be conflated -- see design memo
point 1).

Reproduction (`select_and_reproduce`) and mutation (`mutate_genome`)
stay the population-level, non-spatial operators from `coupling.py`
unchanged: spatial structure only decides who duels whom, not who
reproduces. Positions are reassigned fresh every generation
(`place_on_grid`) since reproduction is global -- there is no parent
cell for an offspring to inherit.

Three conditions this module supports:
  - mobile=False is condition (2): spatial, phased, no movement.
  - mobile=True  is condition (3): spatial, phased, individual-based
    softmax movement.
Condition (1) (well-mixed) is `coupling.run_coupled_evolution`,
untouched by this module.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .coupling import CoupledGenerationStats, mutate_genome, random_genome, run_coupled_duel
from .population import select_and_reproduce

Coord = Tuple[int, int]

VON_NEUMANN_OFFSETS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MOORE_OFFSETS: Tuple[Coord, ...] = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def torus_neighbors(
    cell: Coord, width: int, height: int, offsets: Sequence[Coord] = VON_NEUMANN_OFFSETS
) -> List[Coord]:
    """Neighboring cells of `cell` on a W x H periodic (toroidal) grid."""
    x, y = cell
    return [((x + dx) % width, (y + dy) % height) for dx, dy in offsets]


def place_on_grid(n_agents: int, width: int, height: int, rng: random.Random) -> List[Coord]:
    """Assign each of n_agents to a distinct random cell (n_agents <= width*height).

    Fresh placement every call -- see module docstring: reproduction is
    global/population-level, so there is no position for an offspring to
    inherit from a parent.
    """
    if n_agents > width * height:
        raise ValueError(f"{n_agents} agents do not fit on a {width}x{height} grid")
    all_cells = [(x, y) for x in range(width) for y in range(height)]
    return rng.sample(all_cells, n_agents)


def _softmax_sample(options: List[Coord], scores: List[float], rng: random.Random, tau: float) -> Coord:
    """Sample one of `options` with probability proportional to exp(score / tau)."""
    max_score = max(scores)
    weights = [math.exp((s - max_score) / tau) for s in scores]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for option, w in zip(options, weights):
        cumulative += w
        if r <= cumulative:
            return option
    return options[-1]


def _mean_neighbor_payoff(
    cell: Coord,
    width: int,
    height: int,
    agent_at_cell: Dict[Coord, int],
    total_payoff: List[float],
    rounds_played: List[int],
    offsets: Sequence[Coord] = VON_NEUMANN_OFFSETS,
) -> float:
    """Attractiveness of `cell`: mean per-round payoff-so-far of its
    currently occupied neighbors. 0.0 (a neutral floor -- PD payoffs are
    non-negative) if none of its neighbors are occupied, or an occupied
    neighbor hasn't played a round yet this generation."""
    vals = []
    for ncell in torus_neighbors(cell, width, height, offsets):
        j = agent_at_cell.get(ncell)
        if j is not None and rounds_played[j] > 0:
            vals.append(total_payoff[j] / rounds_played[j])
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class SpatialGenerationStats(CoupledGenerationStats):
    positions: List[Coord] = field(default_factory=list)
    rounds_played: List[int] = field(default_factory=list)


def run_coupled_evolution_spatial(
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
    grid_width: int = 12,
    grid_height: int = 12,
    n_phases: int = 5,
    tau: float = 1.0,
    move_radius: int = 1,
    mobile: bool = True,
    neighbor_offsets: Sequence[Coord] = VON_NEUMANN_OFFSETS,
) -> List[SpatialGenerationStats]:
    """Spatial version of `coupling.run_coupled_evolution`.

    `mobile=False` is condition (2); `mobile=True` is condition (3).
    Both run the identical phase/duel schedule -- only the movement step
    differs -- so the only variable between them is whether an agent can
    relocate (design memo point 2: keeps (2) vs (3) the clean comparison).

    neighbor_offsets selects the neighborhood topology: VON_NEUMANN_OFFSETS
    (k=4, the module default) or MOORE_OFFSETS (k=8).

    move_radius is currently fixed at 1 (candidates = the moving cell's
    own grid neighbors); kept as an explicit parameter rather than
    hardcoded so a >1 variant can be added later without changing the
    call signature.
    """
    if move_radius != 1:
        raise NotImplementedError("move_radius > 1 not yet implemented")
    if rounds_per_duel % n_phases != 0:
        raise ValueError(f"rounds_per_duel={rounds_per_duel} must be divisible by n_phases={n_phases}")
    rounds_per_phase = rounds_per_duel // n_phases

    rng = random.Random(seed)
    rng_spatial = random.Random(rng.random())  # separate stream for placement/movement

    genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]

    stats: List[SpatialGenerationStats] = []
    for gen in range(generations):
        positions = place_on_grid(population_size, grid_width, grid_height, rng_spatial)
        agent_at_cell: Dict[Coord, int] = {positions[i]: i for i in range(population_size)}
        cell_of_agent: Dict[int, Coord] = {i: positions[i] for i in range(population_size)}

        total_payoff = [0.0] * population_size
        rounds_played = [0] * population_size

        for _phase in range(n_phases):
            seen_pairs = set()
            for cell, i in list(agent_at_cell.items()):
                for ncell in torus_neighbors(cell, grid_width, grid_height, neighbor_offsets):
                    j = agent_at_cell.get(ncell)
                    if j is None:
                        continue
                    pair = (min(i, j), max(i, j))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    payoff_i, payoff_j = run_coupled_duel(
                        genomes[i], genomes[j], rounds_per_phase, rng, noise_std
                    )
                    total_payoff[i] += payoff_i
                    total_payoff[j] += payoff_j
                    rounds_played[i] += rounds_per_phase
                    rounds_played[j] += rounds_per_phase

            if mobile:
                order = list(range(population_size))
                rng_spatial.shuffle(order)
                for i in order:
                    cell = cell_of_agent[i]
                    empty_neighbors = [
                        c
                        for c in torus_neighbors(cell, grid_width, grid_height, neighbor_offsets)
                        if c not in agent_at_cell
                    ]
                    options = [cell] + empty_neighbors  # "stay" is always an option
                    scores = [
                        _mean_neighbor_payoff(
                            c, grid_width, grid_height, agent_at_cell, total_payoff, rounds_played,
                            neighbor_offsets,
                        )
                        for c in options
                    ]
                    chosen = _softmax_sample(options, scores, rng_spatial, tau)
                    if chosen != cell:
                        del agent_at_cell[cell]
                        agent_at_cell[chosen] = i
                        cell_of_agent[i] = chosen

        final_positions = [cell_of_agent[i] for i in range(population_size)]
        payoffs = [
            total_payoff[i] / rounds_played[i] if rounds_played[i] > 0 else 0.0
            for i in range(population_size)
        ]

        stats.append(
            SpatialGenerationStats(
                generation=gen,
                k_out=[g.k_out for g in genomes],
                k_self=[g.k_self for g in genomes],
                memory_depths=[g.table.memory_depth for g in genomes],
                payoffs=payoffs,
                positions=final_positions,
                rounds_played=rounds_played,
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
