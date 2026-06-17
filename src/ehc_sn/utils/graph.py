"""DAG generation and permutation utilities for the seqmaze probe.

All algorithms are pure-Python with no external graph library dependency.
"""

from __future__ import annotations

import random
from typing import Sequence


# =============================================================================
def count_shortest_paths(
    adjacency: Sequence[list[int]],
    start: int,
    goal: int,
) -> int:
    """Count the number of distinct shortest paths from start to goal in a DAG.

    Uses BFS with path-counting: when the goal is reached at depth d, every
    node at depth d-1 that can reach the goal contributes its path count.

    Args:
        adjacency: Graph adjacency list.
        start: Start node index.
        goal: Goal node index.

    Returns:
        Number of distinct shortest paths.  0 if no path exists.
    """
    if start == goal:
        return 1

    from collections import deque

    # BFS to find shortest distance to each node and path count.
    # distances[i] = None if unvisited, else shortest distance from start.
    distances: list[int | None] = [None] * len(adjacency)
    counts: list[int] = [0] * len(adjacency)
    distances[start] = 0
    counts[start] = 1

    q: deque[int] = deque([start])

    while q:
        node = q.popleft()
        d = distances[node]
        if d is None:
            continue
        assert d is not None
        nd = d + 1
        for succ in adjacency[node]:
            # If succ is goal and we've already found a path of same length,
            # we can still count this one (don't early-exit).
            if distances[succ] is None:
                distances[succ] = nd
                counts[succ] = counts[node]
                q.append(succ)
            elif distances[succ] == nd:
                counts[succ] += counts[node]

    return counts[goal]


# =============================================================================
def generate_transition_dag(
    n_nodes: int,
    max_out_degree: int,
    seed: int,
    min_path_length: int = 1,
    require_unique_shortest_path: bool = True,
) -> list[list[int]]:
    """Generate a random directed acyclic graph adjacency list.

    Each node i can have outgoing edges only to nodes j > i, guaranteeing a DAG.
    For each node, we randomly sample up to ``max_out_degree`` successors from
    higher-indexed nodes.

    The DAG is guaranteed to have a path from node 0 to node n_nodes-1 whose
    length in nodes is at least ``min_path_length`` (i.e., at least
    ``min_path_length - 1`` edges).

    Args:
        n_nodes: Number of nodes in the graph (N).
        max_out_degree: Maximum out-degree per node (K).
        seed: Random seed for reproducibility.
        min_path_length: Minimum acceptable path length in nodes
            (default 1, meaning any path is acceptable).

    Returns:
        adjacency list of length n_nodes where adj[i] is a sorted list of
        successor node indices.

    Raises:
        ValueError: If min_path_length > n_nodes (impossible to satisfy).
        RuntimeError: If retry budget is exhausted without finding a DAG
            satisfying the min_path_length constraint.
    """
    if min_path_length > n_nodes:
        raise ValueError(
            f"min_path_length ({min_path_length}) cannot exceed "
            f"n_nodes ({n_nodes})."
        )

    # First, try random generation with retries.
    max_attempts = 200
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        adjacency: list[list[int]] = [[] for _ in range(n_nodes)]

        for i in range(n_nodes - 1):
            # Eligible successors: nodes with index > i
            candidates = list(range(i + 1, n_nodes))
            if not candidates:
                continue
            k = rng.randint(0, min(max_out_degree, len(candidates)))
            chosen = rng.sample(candidates, k)
            adjacency[i] = sorted(chosen)

        # Ensure at least one path from start (0) to goal (n_nodes - 1)
        if not _has_path(adjacency, 0, n_nodes - 1):
            if len(adjacency[0]) < max_out_degree:
                adjacency[0].append(n_nodes - 1)
                adjacency[0].sort()
            else:
                adjacency[0][-1] = n_nodes - 1
                adjacency[0].sort()

        # Check minimum path length and unique shortest path
        sp = shortest_path(adjacency, 0, n_nodes - 1)
        if len(sp) >= min_path_length:
            if (
                not require_unique_shortest_path
                or count_shortest_paths(adjacency, 0, n_nodes - 1) == 1
            ):
                return adjacency

    # If retries exhausted, synthesise a deterministic chain that satisfies
    # the constraint, then add random extra edges to non-spine nodes only.
    # When require_unique_shortest_path is True, the spine construction
    # naturally guarantees uniqueness because extra edges go to spine nodes
    # only (can't create equal-length alternatives to the spine path).
    rng = random.Random(seed + max_attempts)
    adjacency = [[] for _ in range(n_nodes)]
    # Build a spine: 0 -> 1 -> ... -> min_path_length-2 -> n_nodes-1.
    spine_set = set(range(min_path_length - 1))
    for i in range(min_path_length - 2):
        adjacency[i].append(i + 1)
    if min_path_length >= 2:
        adjacency[min_path_length - 2].append(n_nodes - 1)
    # Add random extra edges only from non-spine nodes or from spine nodes
    # to other spine nodes (which won't create shortcuts since BFS finds
    # the shortest path).
    for i in range(n_nodes - 1):
        existing = set(adjacency[i])
        candidates = [j for j in range(i + 1, n_nodes) if j not in existing]
        if not candidates:
            continue
        # Only allow edges from spine nodes to other spine nodes (safe).
        if i in spine_set:
            candidates = [j for j in candidates if j in spine_set]
        if not candidates:
            continue
        k = rng.randint(
            0, min(max_out_degree - len(adjacency[i]), len(candidates))
        )
        if k > 0:
            chosen = rng.sample(candidates, k)
            adjacency[i].extend(chosen)
            adjacency[i].sort()
    return adjacency


# =============================================================================
def shortest_path(
    adjacency: Sequence[list[int]],
    start: int,
    goal: int,
) -> list[int]:
    """BFS shortest path in a DAG adjacency list.

    Args:
        adjacency: Graph adjacency list.
        start: Start node index.
        goal: Goal node index.

    Returns:
        List of node indices forming the shortest path from start to goal
        (inclusive). Empty list if no path exists.
    """
    if start == goal:
        return [start]

    visited = {start}
    queue: list[tuple[int, list[int]]] = [(start, [start])]

    while queue:
        node, path = queue.pop(0)
        for succ in adjacency[node]:
            if succ == goal:
                return path + [goal]
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [succ]))
    return []


# =============================================================================
def _has_path(
    adjacency: Sequence[list[int]],
    start: int,
    goal: int,
) -> bool:
    """Check if a path exists from start to goal."""
    return bool(shortest_path(adjacency, start, goal))


# =============================================================================
def remap_obs_ids(n_nodes: int, seed: int) -> list[int]:
    """Generate a random permutation of observation IDs [0, n_nodes).

    Each call with a different seed produces a different remapping, preventing
    the model from memorizing fixed observation id → candidate index mappings.

    Args:
        n_nodes: Number of nodes.
        seed: Random seed for reproducibility.

    Returns:
        List of length n_nodes where result[i] = remapped_obs_id for node i.
    """
    rng = random.Random(seed)
    ids = list(range(n_nodes))
    rng.shuffle(ids)
    return ids


# =============================================================================
def permute_candidate_order(
    n_nodes: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Generate a permutation of candidate indices and its inverse.

    Args:
        n_nodes: Number of candidate nodes.
        seed: Random seed.

    Returns:
        (permutation, inverse) where:
            permutation[t] = the candidate index assigned to original position t
            inverse[permuted_index] = original position
    """
    rng = random.Random(seed)
    perm = list(range(n_nodes))
    rng.shuffle(perm)
    inv = [0] * n_nodes
    for orig, permuted in enumerate(perm):
        inv[permuted] = orig
    return perm, inv
