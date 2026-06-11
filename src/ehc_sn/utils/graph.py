"""DAG generation and permutation utilities for the seqmaze probe.

All algorithms are pure-Python with no external graph library dependency.
"""

from __future__ import annotations

import random
from typing import Sequence


# =============================================================================
def generate_transition_dag(
    n_nodes: int,
    max_out_degree: int,
    seed: int,
) -> list[list[int]]:
    """Generate a random directed acyclic graph adjacency list.

    Each node i can have outgoing edges only to nodes j > i, guaranteeing a DAG.
    For each node, we randomly sample up to ``max_out_degree`` successors from
    higher-indexed nodes.

    Args:
        n_nodes: Number of nodes in the graph (N).
        max_out_degree: Maximum out-degree per node (K).
        seed: Random seed for reproducibility.

    Returns:
        adjacency list of length n_nodes where adj[i] is a sorted list of
        successor node indices.
    """
    rng = random.Random(seed)
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
    # If no path exists, add a direct edge 0 -> n_nodes-1 (if under max_out_degree)
    if not _has_path(adjacency, 0, n_nodes - 1):
        if len(adjacency[0]) < max_out_degree:
            adjacency[0].append(n_nodes - 1)
            adjacency[0].sort()
        else:
            # Replace the last edge with the goal edge
            adjacency[0][-1] = n_nodes - 1
            adjacency[0].sort()

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
