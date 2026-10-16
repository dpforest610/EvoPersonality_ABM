"""3-step Markov network and resilience/structure analysis, ported from the
human-study code (Annen, JOEP manuscript) -- see
docs/MARKOV_STRUCTURE_EVOLUTION_DESIGN.md section 3 and 7 for the full
design and provenance of each function.

`series_from_moves`, `build_edges`, `widest_path_threshold` are ported
UNCHANGED from the attached `markov3_widest_path_reference.py` (verified
2026-10-15 against the human data, N=171: reproduces all 9 recovery
thresholds in Table 15 and the 3 TFT descent thresholds in Table 8 to
3 decimal places). Do not change these definitions.

`cycle_jstar` is ported UNCHANGED from `bootstrap_persistence.R`'s
`cycle_jstar` function (OSF code archive). `strongly_connected_components`
is a standard Tarjan's-algorithm implementation of the same definition the
human-side analysis used (networkx.strongly_connected_components) for
Table 10. `count_simple_cycles` is a NEW definition (not ported -- the
original R `generate_all_cycles` function used for Table 12 could not be
recovered; see the design doc section 7.1), matching the specification:
simple cycles (no repeated nodes), counted once regardless of rotation,
with a maximum cycle length to keep the count tractable on the 64-node
networks.
"""

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

STATE = {"CC": "和", "CD": "惨", "DC": "搾", "DD": "底"}

VOCAB_DIR = Path(__file__).resolve().parent / "markov_vocab"
VOCAB_FILES = {"TFT": "tft.txt", "GRA": "gra.txt", "RAN": "ran.txt", "AllC": "allc.txt", "AllD": "alld.txt"}


def load_vocab(opponent: str) -> frozenset:
    """The reserved-word vocabulary (3-gram nodes) for one opponent, loaded
    from evopersonality_abm/markov_vocab/<opponent>.txt (one word per
    line). TFT has 16 words, GRA/RAN the full 64 (all 3-symbol
    combinations -- i.e. no restriction), AllC/AllD 8 each (restricted to
    the joint states reachable when the opponent never changes action)."""
    path = VOCAB_DIR / VOCAB_FILES[opponent]
    with open(path, encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


# ------------------------------------------------------------------ ported, unchanged

def series_from_moves(focal: Sequence[str], opp: Sequence[str], n_rounds: int = 20) -> str:
    """focal/opp: sequences of 'C'/'D'. Returns the Excel-style 推移3系列 string.

    Ported unchanged from markov3_widest_path_reference.py."""
    s = [STATE[a + b] for a, b in zip(focal[:n_rounds], opp[:n_rounds])]
    grams = ["".join(s[i - 2:i + 1]) for i in range(2, len(s))]
    return ",".join(grams) + "。"


def build_edges(series_list: Sequence[str], vocab) -> Dict[Tuple[str, str], float]:
    """Ported unchanged from markov3_widest_path_reference.py.

    Per participant: the SET of 3-grams present (restricted to vocab).
    Edge a->b exists if a and b co-occur in at least one participant's set
    (NOT sequence adjacency) and a[-2:] == b[:2]. Weight = Jaccard over
    participants: |lines(a) & lines(b)| / |lines(a) | lines(b)|."""
    word_lines = defaultdict(set)
    trans = set()
    for ln, s in enumerate(series_list):
        if not s:
            continue
        seen = []
        for n in (x.strip().rstrip("。") for x in s.split(",")):
            if n in vocab and n not in seen:
                seen.append(n)
        for n in seen:
            word_lines[n].add(ln)
        for a, b in itertools.combinations(seen, 2):
            if a[-2:] == b[:2]:
                trans.add((a, b))
            if b[-2:] == a[:2]:
                trans.add((b, a))
    return {(a, b): len(word_lines[a] & word_lines[b]) / len(word_lines[a] | word_lines[b])
            for a, b in trans}


def widest_path_threshold(edges: Dict[Tuple[str, str], float], start: str = "底底底", goal: str = "和和和"):
    """Ported unchanged from markov3_widest_path_reference.py.

    Descending over the distinct Jaccard values, the first threshold t at
    which `goal` is reachable from `start` using only edges with J >= t.
    None if start or goal is absent from the network."""
    nodes = {x for e in edges for x in e}
    if start not in nodes or goal not in nodes:
        return None
    for t in sorted(set(edges.values()), reverse=True):
        adj = defaultdict(list)
        for (a, b), j in edges.items():
            if j >= t:
                adj[a].append(b)
        seen, stack = {start}, [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if goal in seen:
            return t
    return None


# ------------------------------------------------------------------ structure metrics (7.1)

def cycle_jstar(edges: Dict[Tuple[str, str], float], seq: Sequence[str]) -> float:
    """Ported unchanged from bootstrap_persistence.R's cycle_jstar.

    `seq` is a closed walk given as a list of nodes, e.g.
    ["底惨底", "惨底惨", "底惨底"] (first == last for an actual cycle).
    Returns the minimum edge Jaccard along the walk's consecutive pairs;
    a missing edge counts as weight 0 (matches the R version's
    `if(is.null(...)) 0`)."""
    weights = [edges.get((seq[i], seq[i + 1]), 0.0) for i in range(len(seq) - 1)]
    return min(weights) if weights else 0.0


def strongly_connected_components(edges: Dict[Tuple[str, str], float], threshold: float) -> List[frozenset]:
    """Standard Tarjan's-algorithm strongly-connected-components on the
    directed graph of edges with jaccard >= threshold -- the same
    definition as networkx.strongly_connected_components, used for
    Table 10 (confirmed 2026-10-16 to match the human-side analysis for
    the Secure/Fearful groups; Mix differed there, attributed to the
    paper side, not the definition -- the model uses this standard
    definition unconditionally)."""
    adj = defaultdict(list)
    nodes = set()
    for (a, b), j in edges.items():
        if j >= threshold:
            adj[a].append(b)
            nodes.add(a)
            nodes.add(b)

    index_counter = [0]
    stack: List[str] = []
    lowlink: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    result: List[frozenset] = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj[v]:
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(frozenset(comp))

    for v in sorted(nodes):
        if v not in index:
            strongconnect(v)
    return result


def count_simple_cycles(edges: Dict[Tuple[str, str], float], threshold: float, max_length: int = 6) -> int:
    """NEW definition for Table 12 (the original R `generate_all_cycles`
    could not be recovered -- see design doc section 7.1): the number of
    distinct simple cycles (no repeated nodes except the closing one) in
    the directed graph of edges with jaccard >= threshold, with at most
    `max_length` edges, counting rotations of the same cycle once.

    A self-loop can never occur here: build_edges only ever creates edges
    between two DISTINCT 3-grams (itertools.combinations), so (a, a) is
    never a key of `edges`."""
    adj = defaultdict(list)
    nodes = set()
    for (a, b), j in edges.items():
        if j >= threshold:
            adj[a].append(b)
            nodes.add(a)
            nodes.add(b)
    ordered = sorted(nodes)
    node_index = {n: i for i, n in enumerate(ordered)}

    count = 0
    for start in ordered:
        start_idx = node_index[start]

        def dfs(path):
            nonlocal count
            if len(path) > max_length:
                return
            u = path[-1]
            for v in adj[u]:
                if v == start and len(path) >= 2:
                    count += 1
                elif node_index.get(v, -1) > start_idx and v not in path:
                    dfs(path + [v])

        dfs([start])
    return count
