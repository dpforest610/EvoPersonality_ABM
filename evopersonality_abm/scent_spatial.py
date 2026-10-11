"""Chapter 3: scent-seeking movement on a 2D torus, with lineage-based
offspring placement so a movement-built neighborhood can actually persist
into the next generation (unlike `run_coupled_evolution_spatial`, which
re-places everyone randomly every generation -- see that module's
docstring -- so any structure movement built within a generation was
discarded before it could matter for who duels whom next generation).

This module is additive only: it imports from `spatial.py`, `coupling.py`,
`population.py` and `game.py` but changes none of them.

Grid: a W x H torus, `population_size` agents on distinct cells, a
strict subset of W*H (room to move/place into). Default 13x13 (density
~0.59). A generation is split into `n_phases` phases; each phase,
occupied cells duel their currently-occupied radius-r Moore neighbors
(radius 1 = the usual 8-cell Moore neighborhood, radius 2 = 24 cells)
for rounds_per_duel // n_phases rounds via the existing
`run_coupled_duel` (unmodified), then -- condition-dependent -- a
movement step:

  - movement="none" (M0): no movement.
  - movement="scent" (M2): each agent (random order per phase) picks,
    via softmax, among {stay} + {empty radius-r neighbor cells}, scored
    by each candidate cell's `_mean_neighbor_payoff` (reused from
    spatial.py unchanged) -- pulled toward whoever's radius-r neighbors
    currently have the highest per-round payoff-so-far.
  - movement="random" (M1, a yoked control for M2): each agent (random
    order per phase) moves with a fixed probability p_move to a
    uniformly random empty radius-r neighbor cell, else stays. p_move
    is supplied by the caller (fit from M2's observed per-phase move
    rate), not re-derived here.

An agent with zero duels across the whole generation (empty
neighborhood every phase) gets no payoff signal; its per-round payoff
for this generation is set to the mutual-defection payoff P (from
game.PAYOFFS) as a provisional placeholder -- report how often this
actually happens (isolated_fraction) rather than assuming it's
negligible.

Reproduction is the same global fitness-proportional selection as
`population.select_and_reproduce` (implemented over indices, as in
`run_coupled_evolution_spatial_persistent` and
`run_coupled_evolution_opponent_count`, so each offspring's parent is
known). Offspring placement: per parent, the first offspring takes the
parent's own old cell; any further offspring go to the nearest empty
cell to that same old cell (toroidal Chebyshev distance, ties broken by
the RNG), placed in parent-index round-robin order (one offspring per
still-needing parent per round) so no single parent's extra children
get first pick of empty cells across the whole generation. Generation 0
is placed uniformly at random (`spatial.place_on_grid`, reused
unchanged).
"""

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .coupling import mutate_genome, random_genome, run_coupled_duel
from .game import DEFECT, PAYOFFS
from .spatial import (
    Coord,
    LocalPlacementGenerationStats,
    _edge_assortativity,
    _mean_neighbor_payoff,
    _opponent_means,
    _softmax_sample,
    place_on_grid,
    torus_neighbors,
)

MUTUAL_DEFECT_PAYOFF = PAYOFFS[(DEFECT, DEFECT)][0]

MOVEMENT_MODES = ("none", "random", "scent")


def moore_offsets(radius: int) -> Tuple[Coord, ...]:
    """All (dx, dy) with Chebyshev distance <= radius from (0, 0),
    excluding (0, 0) itself. radius=1 is the usual 8-cell Moore
    neighborhood; radius=2 is 24 cells."""
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    return tuple(
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if (dx, dy) != (0, 0)
    )


def _chebyshev_ring_offsets(d: int) -> List[Coord]:
    """All (dx, dy) with max(|dx|, |dy|) == d exactly (the square ring
    of cells at Chebyshev distance d), d=0 giving [(0, 0)]."""
    if d == 0:
        return [(0, 0)]
    offsets = []
    for dx in range(-d, d + 1):
        for dy in range(-d, d + 1):
            if max(abs(dx), abs(dy)) == d:
                offsets.append((dx, dy))
    return offsets


def nearest_empty_cell(
    target: Coord, width: int, height: int, occupied_cells, rng: random.Random
) -> Coord:
    """The empty cell closest to `target` by toroidal Chebyshev distance
    (ties broken uniformly at random by `rng`), searching outward ring
    by ring. Raises if no empty cell exists within the grid (shouldn't
    happen given this module's population/grid-size invariant, but
    fails loudly instead of looping forever)."""
    max_d = max(width, height)
    for d in range(0, max_d + 1):
        seen = set()
        candidates = []
        for dx, dy in _chebyshev_ring_offsets(d):
            cell = ((target[0] + dx) % width, (target[1] + dy) % height)
            if cell in seen:
                continue
            seen.add(cell)
            if cell not in occupied_cells:
                candidates.append(cell)
        if candidates:
            return rng.choice(candidates)
    raise RuntimeError(f"no empty cell found near {target} on a {width}x{height} grid")


def assign_offspring_cells(
    parent_indices: Sequence[int],
    old_cell_of_agent: Sequence[Coord],
    grid_width: int,
    grid_height: int,
    rng: random.Random,
) -> List[Coord]:
    """Returns `new_cells` (length == len(parent_indices)) such that
    new_cells[offspring_slot] is where that offspring is placed:
    - each parent's first offspring (in slot order) takes the parent's
      own old cell;
    - every further offspring of that parent goes to the then-nearest
      empty cell to the parent's old cell, assigned in parent-index
      round-robin order (one offspring per still-needing parent per
      pass) so no parent's extra children get first claim on nearby
      empty cells ahead of another parent's."""
    offspring_by_parent: Dict[int, List[int]] = defaultdict(list)
    for offspring_slot, parent_idx in enumerate(parent_indices):
        offspring_by_parent[parent_idx].append(offspring_slot)

    population_size = len(parent_indices)
    new_cells: List[Optional[Coord]] = [None] * population_size
    occupied_cells = set()

    for parent_idx, offspring_slots in offspring_by_parent.items():
        first_slot = offspring_slots[0]
        cell = old_cell_of_agent[parent_idx]
        new_cells[first_slot] = cell
        occupied_cells.add(cell)

    remaining: Dict[int, List[int]] = {
        p: slots[1:] for p, slots in offspring_by_parent.items() if len(slots) > 1
    }
    while remaining:
        for parent_idx in sorted(remaining.keys()):
            slots = remaining[parent_idx]
            if not slots:
                continue
            offspring_slot = slots.pop(0)
            target = old_cell_of_agent[parent_idx]
            chosen = nearest_empty_cell(target, grid_width, grid_height, occupied_cells, rng)
            new_cells[offspring_slot] = chosen
            occupied_cells.add(chosen)
        remaining = {p: s for p, s in remaining.items() if s}

    assert all(c is not None for c in new_cells)
    return new_cells  # type: ignore[return-value]


@dataclass
class ScentGenerationStats(LocalPlacementGenerationStats):
    """LocalPlacementGenerationStats (cooperation_rate,
    k_out_assortativity, k_self_assortativity, same_parent_fraction,
    parent_of, opponent_mean_k_self, opponent_mean_k_out -- see that
    class's docstring; all computed identically here, over this
    generation's ACTUALLY-dueled pairs pooled across all n_phases)
    plus this model's own fields:

    - positions: this generation's final (post-movement) cell per
      agent.
    - move_rate: fraction of (agent, phase) slots in which the agent
      actually changed cell, pooled across phases this generation --
      "1フェーズあたりの移動率".
    - isolated_fraction: fraction of agents with zero duels (any phase)
      this generation -- see module docstring on the provisional P
      payoff given to these agents.
    - mean_opponent_count / var_opponent_count / max_opponent_count:
      degree distribution (population variance) of the actual-duel
      graph this generation -- "対戦相手数の分布".
    - scent_index_mean: mean "scent-following index" over this
      generation's movement decisions (movement="scent" only; NaN
      otherwise). Per decision: (chosen candidate's score - mean of
      candidate scores) / (max of candidate scores - mean of candidate
      scores), skipping decisions where max == mean (no information in
      the candidate scores to follow). ~0 for a uniform/random choice
      among candidates, 1 for always picking the best-scoring candidate
      -- see docs/PREREGISTRATION_scent_model.md revision 1 for why
      this calibrates tau instead of any outcome metric (beta,
      cooperation rate, assortativity, ...).
    """

    positions: List[Coord] = field(default_factory=list)
    move_rate: float = float("nan")
    isolated_fraction: float = float("nan")
    mean_opponent_count: float = float("nan")
    var_opponent_count: float = float("nan")
    max_opponent_count: int = 0
    scent_index_mean: float = float("nan")


def _run_generation(
    genomes,
    cells: Sequence[Coord],
    grid_width: int,
    grid_height: int,
    n_phases: int,
    rounds_per_duel: int,
    noise_std: float,
    tau: float,
    movement: str,
    radius: int,
    p_move: Optional[float],
    rng: random.Random,
    rng_spatial: random.Random,
    phase_callback=None,
):
    """phase_callback, if given, is called once per phase (after that
    phase's duels AND movement step) as
    phase_callback(phase_index, positions_snapshot, moved_this_phase)
    -- positions_snapshot is a list[Coord] indexed by agent (same
    convention as the generation-level `positions` field) and
    moved_this_phase a list[bool], both length population_size.
    K_out/K_self are not included since genomes (and hence K_out/K_self)
    are fixed for the whole generation -- a caller recording a
    presentation trajectory already has them once per generation.
    Opt-in and side-effect-only: when None (the default, every existing
    caller), this changes nothing about the simulation itself --
    identical duels, identical movement decisions, identical RNG
    consumption."""
    population_size = len(genomes)
    offsets = moore_offsets(radius)
    rounds_per_phase = rounds_per_duel // n_phases

    agent_at_cell: Dict[Coord, int] = {c: i for i, c in enumerate(cells)}
    cell_of_agent: Dict[int, Coord] = {i: c for i, c in enumerate(cells)}

    total_payoff = [0.0] * population_size
    rounds_played = [0] * population_size
    coop_count = [0] * population_size
    all_edges = set()
    total_move_slots = 0
    scent_index_sum = 0.0
    scent_index_count = 0

    for _phase in range(n_phases):
        moved_this_phase = [False] * population_size
        seen_pairs = set()
        for cell, i in list(agent_at_cell.items()):
            for ncell in torus_neighbors(cell, grid_width, grid_height, offsets):
                j = agent_at_cell.get(ncell)
                if j is None:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                payoff_i, payoff_j, coop_i, coop_j = run_coupled_duel(
                    genomes[i], genomes[j], rounds_per_phase, rng, noise_std, track_cooperation=True
                )
                total_payoff[i] += payoff_i
                total_payoff[j] += payoff_j
                coop_count[i] += coop_i
                coop_count[j] += coop_j
                rounds_played[i] += rounds_per_phase
                rounds_played[j] += rounds_per_phase
        all_edges |= seen_pairs

        if movement == "none":
            pass
        elif movement == "scent":
            order = list(range(population_size))
            rng_spatial.shuffle(order)
            for i in order:
                cell = cell_of_agent[i]
                empty_neighbors = [
                    c for c in torus_neighbors(cell, grid_width, grid_height, offsets)
                    if c not in agent_at_cell
                ]
                options = [cell] + empty_neighbors
                scores = [
                    _mean_neighbor_payoff(
                        c, grid_width, grid_height, agent_at_cell, total_payoff, rounds_played, offsets
                    )
                    for c in options
                ]
                chosen = _softmax_sample(options, scores, rng_spatial, tau)

                mean_score = statistics.fmean(scores)
                max_score = max(scores)
                if max_score != mean_score:
                    chosen_score = scores[options.index(chosen)]
                    scent_index_sum += (chosen_score - mean_score) / (max_score - mean_score)
                    scent_index_count += 1

                if chosen != cell:
                    del agent_at_cell[cell]
                    agent_at_cell[chosen] = i
                    cell_of_agent[i] = chosen
                    total_move_slots += 1
                    moved_this_phase[i] = True
        elif movement == "random":
            if p_move is None:
                raise ValueError("p_move must be given when movement='random'")
            order = list(range(population_size))
            rng_spatial.shuffle(order)
            for i in order:
                if rng_spatial.random() >= p_move:
                    continue
                cell = cell_of_agent[i]
                empty_neighbors = [
                    c for c in torus_neighbors(cell, grid_width, grid_height, offsets)
                    if c not in agent_at_cell
                ]
                if not empty_neighbors:
                    continue
                chosen = rng_spatial.choice(empty_neighbors)
                del agent_at_cell[cell]
                agent_at_cell[chosen] = i
                cell_of_agent[i] = chosen
                total_move_slots += 1
                moved_this_phase[i] = True
        else:
            raise ValueError(f"movement must be one of {MOVEMENT_MODES}, got {movement!r}")

        if phase_callback is not None:
            positions_snapshot = [cell_of_agent[i] for i in range(population_size)]
            phase_callback(_phase, positions_snapshot, moved_this_phase)

    final_cells = [cell_of_agent[i] for i in range(population_size)]
    payoffs = [
        total_payoff[i] / rounds_played[i] if rounds_played[i] > 0 else MUTUAL_DEFECT_PAYOFF
        for i in range(population_size)
    ]
    total_rounds = sum(rounds_played)
    cooperation_rate = sum(coop_count) / total_rounds if total_rounds > 0 else float("nan")
    isolated_fraction = sum(1 for r in rounds_played if r == 0) / population_size
    move_rate = total_move_slots / (population_size * n_phases)

    degree = [0] * population_size
    for i, j in all_edges:
        degree[i] += 1
        degree[j] += 1
    mean_opponent_count = statistics.fmean(degree)
    var_opponent_count = statistics.pvariance(degree)
    max_opponent_count = max(degree)
    scent_index_mean = scent_index_sum / scent_index_count if scent_index_count > 0 else float("nan")

    return (
        payoffs, final_cells, list(all_edges), cooperation_rate, isolated_fraction, move_rate,
        mean_opponent_count, var_opponent_count, max_opponent_count, scent_index_mean,
    )


def run_coupled_evolution_scent(
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
    grid_width: int = 13,
    grid_height: int = 13,
    n_phases: int = 5,
    tau: float = 1.0,
    radius: int = 1,
    movement: str = "none",
    p_move: Optional[float] = None,
    start_generation: int = 0,
    initial_genomes=None,
    initial_cells: Optional[List[Coord]] = None,
    rng_state=None,
    rng_spatial_state=None,
    parent_of_state: Optional[List[Optional[int]]] = None,
    track_parent_state: bool = False,
    record_parent_history: bool = False,
    return_state: bool = False,
    record_phases_for_generations: Optional[Set[int]] = None,
    phase_recorder_out: Optional[Dict[int, List[Tuple[int, List[Coord], List[bool]]]]] = None,
):
    """Chapter-3 scent-movement model -- see module docstring.

    movement: "none" (M0), "random" (M1, needs p_move), or "scent" (M2).

    Chunked-checkpoint resume: pass initial_genomes/initial_cells/
    rng_state/rng_spatial_state (as returned by a previous call with
    return_state=True) together with start_generation, same pattern as
    run_coupled_evolution_opponent_count. With return_state=True,
    returns (stats, final_genomes, final_cells, rng_state,
    rng_spatial_state) -- or a 6th element (parent_of) appended if
    track_parent_state=True.

    record_phases_for_generations: an optional set of generation
    indices for which per-phase (not just per-generation) snapshots are
    wanted -- e.g. for a presentation animation showing movement within
    a generation. When given, phase_recorder_out (a plain dict) must
    also be given; for each requested generation it is populated with
    phase_recorder_out[gen] = [(phase_index, positions_snapshot,
    moved_this_phase), ...], one tuple per phase (see
    `_run_generation`'s phase_callback docstring for the snapshot
    format). K_out/K_self are not duplicated per phase since genomes
    are fixed for the whole generation -- read them once from that
    generation's ScentGenerationStats. This changes nothing about the
    simulation for any generation not in the set, and nothing at all
    when record_phases_for_generations is None (the default, every
    existing caller)."""
    if movement not in MOVEMENT_MODES:
        raise ValueError(f"movement must be one of {MOVEMENT_MODES}, got {movement!r}")
    if rounds_per_duel % n_phases != 0:
        raise ValueError(f"rounds_per_duel={rounds_per_duel} must be divisible by n_phases={n_phases}")
    if population_size > grid_width * grid_height:
        raise ValueError(
            f"population_size={population_size} does not fit on a {grid_width}x{grid_height} grid"
        )

    if rng_state is not None and rng_spatial_state is not None:
        rng = random.Random()
        rng.setstate(rng_state)
        rng_spatial = random.Random()
        rng_spatial.setstate(rng_spatial_state)
    else:
        rng = random.Random(seed)
        rng_spatial = random.Random(rng.random())

    if initial_genomes is not None:
        genomes = initial_genomes
    else:
        genomes = [random_genome(initial_memory_depth, rng) for _ in range(population_size)]

    if initial_cells is not None:
        cells = list(initial_cells)
    else:
        cells = place_on_grid(population_size, grid_width, grid_height, rng_spatial)

    parent_of: List[Optional[int]] = (
        list(parent_of_state) if parent_of_state is not None else [None] * population_size
    )

    stats: List[ScentGenerationStats] = []
    for gen in range(start_generation, generations):
        phase_callback = None
        if record_phases_for_generations is not None and gen in record_phases_for_generations:
            if phase_recorder_out is None:
                raise ValueError("phase_recorder_out must be given when record_phases_for_generations is set")
            gen_frames: List[Tuple[int, List[Coord], List[bool]]] = []
            phase_recorder_out[gen] = gen_frames

            def phase_callback(phase_idx, positions_snapshot, moved_this_phase, _frames=gen_frames):
                _frames.append((phase_idx, list(positions_snapshot), list(moved_this_phase)))

        (
            payoffs, final_cells, edges, cooperation_rate, isolated_fraction, move_rate,
            mean_opp, var_opp, max_opp, scent_index_mean,
        ) = _run_generation(
            genomes, cells, grid_width, grid_height, n_phases, rounds_per_duel,
            noise_std, tau, movement, radius, p_move, rng, rng_spatial,
            phase_callback=phase_callback,
        )

        k_out_vals = [g.k_out for g in genomes]
        k_self_vals = [g.k_self for g in genomes]

        if any(p is None for p in parent_of) or not edges:
            same_parent_fraction = float("nan")
        else:
            same_parent_fraction = sum(1 for i, j in edges if parent_of[i] == parent_of[j]) / len(edges)

        stats.append(
            ScentGenerationStats(
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
                opponent_mean_k_self=_opponent_means(k_self_vals, edges, population_size),
                opponent_mean_k_out=_opponent_means(k_out_vals, edges, population_size),
                positions=final_cells,
                move_rate=move_rate,
                isolated_fraction=isolated_fraction,
                mean_opponent_count=mean_opp,
                var_opponent_count=var_opp,
                max_opponent_count=max_opp,
                scent_index_mean=scent_index_mean,
            )
        )

        parent_indices = list(rng.choices(range(population_size), weights=payoffs, k=population_size))
        genomes = [
            mutate_genome(
                genomes[parent_idx],
                rng,
                point_mutation_rate,
                duplication_rate,
                split_rate,
                max_memory_depth,
                min_memory_depth,
                k_mutation_std,
            )
            for parent_idx in parent_indices
        ]
        cells = assign_offspring_cells(parent_indices, final_cells, grid_width, grid_height, rng_spatial)
        parent_of = list(parent_indices)

    if return_state:
        if track_parent_state:
            return stats, genomes, cells, rng.getstate(), rng_spatial.getstate(), parent_of
        return stats, genomes, cells, rng.getstate(), rng_spatial.getstate()
    return stats
