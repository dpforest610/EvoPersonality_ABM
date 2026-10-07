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
import statistics
from collections import defaultdict
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


def _run_generation_phases(
    genomes,
    agent_at_cell: Dict[Coord, int],
    cell_of_agent: Dict[int, Coord],
    grid_width: int,
    grid_height: int,
    n_phases: int,
    rounds_per_duel: int,
    noise_std: float,
    tau: float,
    mobile: bool,
    neighbor_offsets: Sequence[Coord],
    rng: random.Random,
    rng_spatial: random.Random,
) -> Tuple[List[float], List[int]]:
    """Runs one generation's phase/duel(+movement) loop, shared by
    `run_coupled_evolution_spatial` and `run_coupled_evolution_spatial_persistent`.
    Mutates agent_at_cell/cell_of_agent in place when mobile=True. Returns
    (total_payoff, rounds_played), indexed by agent index."""
    population_size = len(cell_of_agent)
    rounds_per_phase = rounds_per_duel // n_phases
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

    return total_payoff, rounds_played


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

        total_payoff, rounds_played = _run_generation_phases(
            genomes, agent_at_cell, cell_of_agent, grid_width, grid_height, n_phases,
            rounds_per_duel, noise_std, tau, mobile, neighbor_offsets, rng, rng_spatial,
        )

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


def run_coupled_evolution_spatial_persistent(
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
    n_phases: int = 1,
    tau: float = 1.0,
    mobile: bool = False,
    neighbor_offsets: Sequence[Coord] = VON_NEUMANN_OFFSETS,
) -> List[SpatialGenerationStats]:
    """Cross-generation neighbor persistence variant, for isolating
    "a lineage keeps facing the same neighborhood" from the plain
    neighborhood-size effect that `run_coupled_evolution_spatial`
    (fresh `place_on_grid` every generation) already established.

    Positions are set ONCE at generation 0 (`place_on_grid`), then
    persist across generations by construction: reproduction stays the
    same global/population-level fitness-proportional selection as
    everywhere else in this project (mathematically identical to
    `population.select_and_reproduce` -- this function reimplements it
    over indices rather than genome objects only so it can also track
    which parent each offspring came from), but each of the
    population_size offspring is given a grid cell by:

      1. every parent that left >=1 offspring keeps its own cell for
         exactly one of them (the "heir", picked uniformly at random
         among that parent's offspring when it has more than one --
         they are genetically identical before mutation, so this choice
         carries no information);
      2. every other offspring (extra children of a lineage with >1
         offspring, or an offspring with no direct claim because its
         parent's cell was already claimed by a sibling) is scattered
         uniformly at random over the leftover cell pool: cells whose
         previous occupant left NO offspring (an extinct lineage's
         vacated territory) plus the grid's permanent vacancy.

    This keeps population size and grid vacancy constant every
    generation without introducing a movement mechanism of its own --
    `mobile` still only controls the WITHIN-generation phase movement
    from `run_coupled_evolution_spatial`, unrelated to this cross-
    generation inheritance step. Interpretation caveat (see chat): this
    produces "a lineage keeps facing a similar neighborhood" (environmental
    continuity), not "an individual seeks out a specific partner"
    (active partner choice) -- and because extra offspring scatter
    randomly, the population is a mix of staying and scattering
    lineages, not a uniformly persistent one.
    """
    if rounds_per_duel % n_phases != 0:
        raise ValueError(f"rounds_per_duel={rounds_per_duel} must be divisible by n_phases={n_phases}")

    rng = random.Random(seed)
    rng_spatial = random.Random(rng.random())

    genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]
    positions = place_on_grid(population_size, grid_width, grid_height, rng_spatial)
    cell_of_agent: Dict[int, Coord] = {i: positions[i] for i in range(population_size)}
    agent_at_cell: Dict[Coord, int] = {positions[i]: i for i in range(population_size)}

    all_cells = [(x, y) for x in range(grid_width) for y in range(grid_height)]

    stats: List[SpatialGenerationStats] = []
    for gen in range(generations):
        total_payoff, rounds_played = _run_generation_phases(
            genomes, agent_at_cell, cell_of_agent, grid_width, grid_height, n_phases,
            rounds_per_duel, noise_std, tau, mobile, neighbor_offsets, rng, rng_spatial,
        )

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

        # global fitness-proportional selection (same weighting as select_and_reproduce),
        # done over indices so each offspring's parent (and hence parent's cell) is known
        parent_indices = rng.choices(range(population_size), weights=payoffs, k=population_size)

        offspring_by_parent: Dict[int, List[int]] = defaultdict(list)
        for offspring_idx, parent_idx in enumerate(parent_indices):
            offspring_by_parent[parent_idx].append(offspring_idx)

        new_cell_of_offspring: Dict[int, Coord] = {}
        leftover_offspring: List[int] = []
        for parent_idx, offspring_list in offspring_by_parent.items():
            heir = rng_spatial.choice(offspring_list)
            new_cell_of_offspring[heir] = cell_of_agent[parent_idx]
            leftover_offspring.extend(o for o in offspring_list if o != heir)

        reserved_cells = set(new_cell_of_offspring.values())
        available_cells = [c for c in all_cells if c not in reserved_cells]
        rng_spatial.shuffle(available_cells)
        for offspring_idx, cell in zip(leftover_offspring, available_cells):
            new_cell_of_offspring[offspring_idx] = cell

        genomes = [
            mutate_genome(
                genomes[parent_indices[i]],
                rng,
                point_mutation_rate,
                duplication_rate,
                split_rate,
                max_memory_depth,
                min_memory_depth,
                k_mutation_std,
            )
            for i in range(population_size)
        ]

        cell_of_agent = {i: new_cell_of_offspring[i] for i in range(population_size)}
        agent_at_cell = {cell: i for i, cell in cell_of_agent.items()}

    return stats


@dataclass
class LocalPlacementGenerationStats(CoupledGenerationStats):
    """CoupledGenerationStats plus the operational-check/presentation
    fields the placement="local" experiment (and its "shuffle" baseline,
    for comparison) needs:

    - cooperation_rate: population mean fraction of COOPERATE actions
      this generation, pooled across all duels.
    - k_out_assortativity / k_self_assortativity: Pearson correlation of
      the two endpoints' K_out/K_self values, taken over all ACTUALLY-
      DUELED pairs this generation -- a simplified Moran's-I-like /
      network-assortativity measure of "do individuals who duel each
      other tend to be similar". NaN if there are fewer than 2 edges or
      the values are constant (no variance to correlate).
    - same_parent_fraction: share of duel edges whose two endpoints
      share this generation's immediate parent (from the most recent
      reproduction step). NaN at generation 0 (nothing has reproduced
      yet) or if there are no edges.

    All four are computed identically regardless of placement, so they
    are the direct operational check for whether actually-dueled pairs
    are more similar / more often siblings under "local" than
    "shuffle"."""

    cooperation_rate: float = float("nan")
    k_out_assortativity: float = float("nan")
    k_self_assortativity: float = float("nan")
    same_parent_fraction: float = float("nan")
    parent_of: Optional[List[Optional[int]]] = None
    """This generation's per-individual immediate-parent array (old ring
    index of the parent that produced each current individual), only
    populated when record_parent_history=True -- None otherwise, to
    avoid the extra memory/copy cost for callers (like the main
    production sweep) that only need the scalar same_parent_fraction."""


def _edge_assortativity(values: Sequence[float], edges: Sequence[Tuple[int, int]]) -> float:
    if len(edges) < 2:
        return float("nan")
    x = [values[i] for i, j in edges]
    y = [values[j] for i, j in edges]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return statistics.correlation(x, y)


def fixed_ring_duel_edges(population_size: int, k: int) -> List[Tuple[int, int]]:
    """Same k-nearest-neighbor connection rule as ring_duel_edges, but
    using the array's own index order as the ring directly, with no
    per-generation reshuffle -- for placement="local", where array index
    IS the lineage-persistent ring position produced by
    local_placement_order. No rng is needed since nothing is randomized
    here."""
    if k < 1 or k > population_size - 1:
        raise ValueError(f"k={k} must be in [1, {population_size - 1}]")
    if k % 2 == 1 and k != population_size - 1:
        raise NotImplementedError(
            f"fixed_ring_duel_edges only supports odd k when k == population_size - 1 "
            f"(complete graph); k={k} is not exactly k-regular with this construction"
        )
    half_before = k // 2
    half_after = k - half_before
    offsets = list(range(1, half_after + 1)) + list(range(-half_before, 0))
    edges = set()
    for i in range(population_size):
        for offset in offsets:
            j = (i + offset) % population_size
            edges.add((min(i, j), max(i, j)))
    return list(edges)


def local_placement_order(parent_indices: Sequence[int], population_size: int) -> List[int]:
    """Returns a permutation `order` of selection-slot indices such that
    offspring sharing the same parent end up ring-adjacent ("blocks"),
    blocks ordered by the parent's position on the OLD ring (0 ..
    population_size-1); a parent with no offspring simply contributes no
    block (no gap left behind, blocks are packed). order[p] = the
    original selection-slot index of the offspring placed at new ring
    position p. Scanning old ring positions starting at 0 is one valid
    rotation of the ring (any starting point works equally, since it's a
    ring) -- see run_coupled_evolution_opponent_count's docstring."""
    by_parent: Dict[int, List[int]] = defaultdict(list)
    for selection_slot, parent_idx in enumerate(parent_indices):
        by_parent[parent_idx].append(selection_slot)
    order: List[int] = []
    for old_ring_pos in range(population_size):
        order.extend(by_parent.get(old_ring_pos, []))
    return order


def ring_duel_edges(population_size: int, k: int, rng: random.Random) -> List[Tuple[int, int]]:
    """A fresh random circular ordering ("ring") of all agents, each
    agent duels its k nearest ring-neighbors (roughly half on each side;
    one extra ahead for odd k) -- symmetric, exactly k-regular. k =
    population_size-1 makes every agent adjacent to everyone, i.e. the
    complete graph (well-mixed's exact pairing structure)."""
    if k < 1 or k > population_size - 1:
        raise ValueError(f"k={k} must be in [1, {population_size - 1}]")
    if k % 2 == 1 and k != population_size - 1:
        # Odd k (other than the complete-graph special case) can't be made
        # exactly symmetric with this construction: the boundary offset at
        # distance k//2+1 only fits on one side, and since edge existence
        # is checked from EITHER endpoint, every node ends up receiving an
        # extra reciprocal edge -- degree k+1, not k. Not needed for any k
        # in the planned sweep (4/8/16/32/50 are even, 99 = population-1),
        # so this is a validation, not a fix.
        raise NotImplementedError(
            f"ring_duel_edges only supports odd k when k == population_size - 1 "
            f"(complete graph); k={k} is not exactly k-regular with this construction"
        )
    perm = list(range(population_size))
    rng.shuffle(perm)
    half_before = k // 2
    half_after = k - half_before
    offsets = list(range(1, half_after + 1)) + list(range(-half_before, 0))
    edges = set()
    for p in range(population_size):
        agent_i = perm[p]
        for offset in offsets:
            agent_j = perm[(p + offset) % population_size]
            edges.add((min(agent_i, agent_j), max(agent_i, agent_j)))
    return list(edges)


def run_coupled_evolution_opponent_count(
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
    k: int = 8,
    placement: str = "shuffle",
    start_generation: int = 0,
    initial_genomes=None,
    rng_state=None,
    rng_ring_state=None,
    parent_of_state: Optional[List[Optional[int]]] = None,
    track_parent_state: bool = False,
    record_parent_history: bool = False,
    return_state: bool = False,
):
    """Pure "opponent count" sweep, stripped of 2D grid geometry -- see
    chat: after ruling out movement, duel-chopping, cross-generation
    persistence, and neighbor diversity as the driver behind experiment
    A's K_out signal, the next question is whether the phenomenon itself
    is real (does it scale monotonically with opponent count k?) rather
    than a k=8/Moore-topology-specific artifact. A 2D torus grid can't
    cleanly represent arbitrary k (wraparound collisions for k close to
    the grid size), so this uses `ring_duel_edges` instead: every
    generation, agents are placed on a freshly-shuffled ring and each
    duels its k nearest ring-neighbors for one continuous
    rounds_per_duel-round duel (n_phases=1 -- duel-chopping already
    ruled out; no movement -- already ruled out; fresh ring every
    generation -- matches experiment A's reshuffled-every-generation
    baseline, not the cross-generation persistence variant, which
    weakened rather than strengthened the effect).

    k = population_size - 1 reduces to the complete graph, i.e. chapter
    1's exact well-mixed pairing -- used as a built-in consistency
    anchor, though chapter 1's own already-collected data is the
    reference point in practice rather than re-simulating this case.

    placement="shuffle" (default) is byte-for-byte the original
    implementation: every generation, duel-neighbors come from a
    freshly-shuffled ring (rng_ring), independent of array layout --
    kept unchanged on purpose and checked against a golden reference
    captured before this parameter existed (see
    test_opponent_count_placement_shuffle_matches_golden_reference).

    placement="local" instead makes the array index itself a
    persistent, lineage-based ring position: each generation's
    offspring are placed in contiguous blocks by parent (same parent's
    children end up ring-adjacent), blocks ordered by the parent's OLD
    ring position (see local_placement_order), and duel-neighbors are
    this generation's k nearest neighbors in THAT fixed array order --
    no reshuffle (see fixed_ring_duel_edges). rng_ring is still
    accepted/threaded through for call-signature symmetry with
    "shuffle" but is never drawn from when placement="local", so its
    state does not advance in that mode.

    Both placement modes return LocalPlacementGenerationStats (a
    CoupledGenerationStats subclass) with four extra fields computed
    identically for both modes -- cooperation_rate, k_out_assortativity/
    k_self_assortativity (similarity between ACTUALLY-DUELED pairs), and
    same_parent_fraction (share of duel edges whose two endpoints share
    this generation's immediate parent) -- see the dataclass's
    docstring. These are the operational check for whether "local"
    actually makes similar individuals duel each other more often than
    "shuffle" does.

    Chunked-checkpoint resume: pass initial_genomes/rng_state/
    rng_ring_state (as returned by a previous call with
    return_state=True) together with start_generation to continue a run
    exactly where it left off -- random.Random's getstate()/setstate()
    captures the full internal generator state, so resuming reproduces
    bit-identical results to an uninterrupted run (verified by test).
    With return_state=True, returns (stats, final_genomes, rng_state,
    rng_ring_state) instead of just stats, so a caller can checkpoint --
    unless track_parent_state=True, in which case a 5th element
    (the current population's per-individual immediate-parent array,
    needed to score the FIRST post-resume generation's
    same_parent_fraction correctly) is appended. track_parent_state
    defaults to False so this is purely additive: every existing caller
    that unpacks the original 4-tuple is unaffected.
    """
    if placement not in ("shuffle", "local"):
        raise ValueError(f"placement must be 'shuffle' or 'local', got {placement!r}")

    if rng_state is not None and rng_ring_state is not None:
        rng = random.Random()
        rng.setstate(rng_state)
        rng_ring = random.Random()
        rng_ring.setstate(rng_ring_state)
    else:
        rng = random.Random(seed)
        rng_ring = random.Random(rng.random())

    if initial_genomes is not None:
        genomes = initial_genomes
    else:
        genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]

    parent_of: List[Optional[int]] = (
        list(parent_of_state) if parent_of_state is not None else [None] * population_size
    )

    stats: List[LocalPlacementGenerationStats] = []
    for gen in range(start_generation, generations):
        if placement == "shuffle":
            edges = ring_duel_edges(population_size, k, rng_ring)
        else:
            edges = fixed_ring_duel_edges(population_size, k)

        total_payoff = [0.0] * population_size
        rounds_played = [0] * population_size
        coop_count = [0] * population_size
        for i, j in edges:
            payoff_i, payoff_j, coop_i, coop_j = run_coupled_duel(
                genomes[i], genomes[j], rounds_per_duel, rng, noise_std, track_cooperation=True
            )
            total_payoff[i] += payoff_i
            total_payoff[j] += payoff_j
            coop_count[i] += coop_i
            coop_count[j] += coop_j
            rounds_played[i] += rounds_per_duel
            rounds_played[j] += rounds_per_duel

        payoffs = [
            total_payoff[i] / rounds_played[i] if rounds_played[i] > 0 else 0.0
            for i in range(population_size)
        ]

        total_rounds = sum(rounds_played)
        cooperation_rate = sum(coop_count) / total_rounds if total_rounds > 0 else float("nan")

        k_out_vals = [g.k_out for g in genomes]
        k_self_vals = [g.k_self for g in genomes]

        if any(p is None for p in parent_of) or not edges:
            same_parent_fraction = float("nan")
        else:
            same_parent_fraction = sum(1 for i, j in edges if parent_of[i] == parent_of[j]) / len(edges)

        stats.append(
            LocalPlacementGenerationStats(
                generation=gen,
                k_out=k_out_vals,
                k_self=k_self_vals,
                memory_depths=[g.table.memory_depth for g in genomes],
                payoffs=payoffs,
                cooperation_rate=cooperation_rate,
                k_out_assortativity=_edge_assortativity(k_out_vals, edges),
                k_self_assortativity=_edge_assortativity(k_self_vals, edges),
                same_parent_fraction=same_parent_fraction,
                parent_of=list(parent_of) if record_parent_history else None,
            )
        )

        parent_indices = list(rng.choices(range(population_size), weights=payoffs, k=population_size))
        selected_genomes = [genomes[idx] for idx in parent_indices]
        mutated_genomes = [
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
            for g in selected_genomes
        ]

        order = list(range(population_size)) if placement == "shuffle" else local_placement_order(
            parent_indices, population_size
        )
        genomes = [mutated_genomes[slot] for slot in order]
        parent_of = [parent_indices[slot] for slot in order]

    if return_state:
        if track_parent_state:
            return stats, genomes, rng.getstate(), rng_ring.getstate(), parent_of
        return stats, genomes, rng.getstate(), rng_ring.getstate()
    return stats
