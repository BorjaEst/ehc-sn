"""DAG generation and permutation utilities for the seqmaze probe.

All algorithms are pure-Python with no external graph library dependency.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# =============================================================================
def compute_layered_dag_positions(
    *,
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> NDArray:
    """Compute deterministic layered DAG positions using longest predecessor depth.

    Assigns each node an x-coordinate equal to its layer (longest path from
    any source node), guaranteeing that every directed edge moves strictly
    left-to-right.  Within each layer, nodes are sorted by their canonical
    index (deterministic y-spacing).

    Args:
        num_nodes: Number of valid (non-padded) nodes.
        edges: Directed edges as ``(source, target)`` pairs in compact
            (non-padded) index space.

    Returns:
        Float array of shape ``(num_nodes, 2)`` with columns ``(x, y)``.
        ``x`` is the layer index; ``y`` is the within-layer rank.

    Raises:
        ValueError: If *edges* contains a cycle (detected when layer
            computation fails to converge after ``num_nodes`` iterations).
    """
    # Build adjacency
    successors: list[list[int]] = [[] for _ in range(num_nodes)]
    predecessors: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            raise ValueError(
                f"Edge ({u}, {v}) has out-of-range node index "
                f"(num_nodes={num_nodes})."
            )
        successors[u].append(v)
        predecessors[v].append(u)

    # Longest predecessor depth (DP over DAG topological order)
    layer = [0] * num_nodes
    for iteration in range(num_nodes + 1):
        changed = False
        for v in range(num_nodes):
            if predecessors[v]:
                new_layer = 1 + max(layer[u] for u in predecessors[v])
                if new_layer != layer[v]:
                    layer[v] = new_layer
                    changed = True
        if not changed:
            break
    else:
        raise ValueError(
            "Cycle detected: longest-predecessor-depth iteration did not "
            f"converge after {num_nodes + 1} passes over {num_nodes} nodes."
        )

    # Group nodes by layer, sort within each layer by node index
    n_layers = max(layer) + 1
    nodes_by_layer: list[list[int]] = [[] for _ in range(n_layers)]
    for v in range(num_nodes):
        nodes_by_layer[layer[v]].append(v)

    positions = np.zeros((num_nodes, 2), dtype=np.float64)
    for x, group in enumerate(nodes_by_layer):
        for y_offset, v in enumerate(sorted(group)):
            positions[v, 0] = float(x)
            # Center within layer
            positions[v, 1] = float(y_offset - (len(group) - 1) / 2.0)

    return positions


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
def generate_hamiltonian_dag(
    n_nodes: int,
    max_out_degree: int,
    seed: int,
    *,
    target_edges: int | None = None,
    min_extra_edges_per_node: int = 0,
    short_edge_prob: float = 0.6,
    medium_edge_prob: float = 0.30,
    long_edge_prob: float = 0.10,
    short_span: tuple[int, int] = (2, 5),
    medium_span: tuple[int, int] = (6, 15),
    long_span: tuple[int, int] = (16, 44),
) -> list[list[int]]:
    """Generate a DAG with mandatory Hamiltonian backbone and controlled
    shortcut edges in three rank-separation bands.

    Every node ``i`` has the mandatory forward edge ``i → i+1``, guaranteeing
    that all non-terminal nodes have out-degree ≥ 1 and that every lower-rank
    node can reach every higher-rank node.  Additional forward edges are
    sampled probabilistically in three span bands (short, medium, long) to
    create forks, merges, bypasses, and hierarchy while preserving the
    backbone connectivity guarantee.

    Args:
        n_nodes: Number of nodes (topological ranks 0..n_nodes-1).
        max_out_degree: Maximum out-degree per node (must be ≥ 2).
        seed: Deterministic seed.
        target_edges: Desired total edge count.  When set, edges are sampled
            until the target is reached or each node's out-degree budget is
            exhausted (soft target; no error if unreachable).
        min_extra_edges_per_node: Minimum extra edges beyond the mandatory
            backbone edge for each non-terminal node.  Nodes near the end of
            the chain with few forward candidates may receive fewer.
        short_edge_prob: Per-candidate probability for short-span edges.
        medium_edge_prob: Per-candidate probability for medium-span edges.
        long_edge_prob: Per-candidate probability for long-span edges.
        short_span: (min, max) inclusive separation for short edges.
        medium_span: (min, max) inclusive separation for medium edges.
        long_span: (min, max) inclusive separation for long edges.

    Returns:
        Adjacency list where ``adj[i]`` is a sorted list of successor
        indices.  Length equals ``n_nodes``.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be ≥ 2, got {n_nodes}")
    if max_out_degree < 2:
        raise ValueError(f"max_out_degree must be ≥ 2, got {max_out_degree}")

    rng = random.Random(seed)
    adj: list[list[int]] = [[] for _ in range(n_nodes)]

    # 1. Mandatory backbone: i → i+1
    for i in range(n_nodes - 1):
        adj[i].append(i + 1)

    backbone_edges = n_nodes - 1

    # If min_extra_edges_per_node > 0, ensure each non-terminal node
    # gets at least that many extra edges (may be impossible near end).
    if min_extra_edges_per_node > 0:
        for i in range(n_nodes - 1):
            needed = min_extra_edges_per_node
            candidates = [j for j in range(i + 2, n_nodes) if j not in adj[i]]
            while needed > 0 and candidates:
                j = rng.choice(candidates)
                if len(adj[i]) < max_out_degree:
                    adj[i].append(j)
                    candidates.remove(j)
                    needed -= 1
                else:
                    break

    extras_needed = (
        None
        if target_edges is None
        else max(0, target_edges - sum(len(a) for a in adj))
    )
    extras_added = 0

    # 2. Define span bands with eligibility: (lo, hi, prob)
    bands = [
        ("short", short_span[0], short_span[1], short_edge_prob),
        ("medium", medium_span[0], medium_span[1], medium_edge_prob),
        ("long", long_span[0], long_span[1], long_edge_prob),
    ]

    # 3. Process bands in priority order, shuffling candidates within each band
    for band_name, lo, hi, prob in bands:
        # Build candidate list for this band
        candidates: list[tuple[int, int]] = []
        for i in range(n_nodes - 1):
            lo_j = max(i + 1, i + lo)
            hi_j = min(n_nodes - 1, i + hi)
            for j in range(lo_j, hi_j + 1):
                if j not in adj[i]:
                    candidates.append((i, j))
        rng.shuffle(candidates)
        for i, j in candidates:
            if extras_needed is not None and extras_added >= extras_needed:
                break
            if len(adj[i]) >= max_out_degree:
                continue
            if j in adj[i]:
                continue
            if rng.random() < prob:
                adj[i].append(j)
                extras_added += 1
        if extras_needed is not None and extras_added >= extras_needed:
            break

    # 4. Sort for determinism
    for i in range(n_nodes):
        adj[i].sort()

    total = backbone_edges + extras_added
    if extras_needed is not None and extras_added < extras_needed:
        import warnings

        warnings.warn(
            f"generate_hamiltonian_dag: target_edges={target_edges} but only "
            f"added {extras_added} extra edges (budget exhausted). "
            f"Total edges: {total}."
        )

    return adj


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


# =============================================================================
def canonical_dag_digest(
    adjacency: list[list[int]],
    obs_ids: np.ndarray,
    *,
    schema_version: int = 1,
) -> str:
    """Return ``"sha256:..."`` for a canonical DAG serialization.

    The canonical representation hashes only the **public-graph content**:
    the sorted set of public observation IDs and the sorted set of directed
    ``(source_obs_id, dest_obs_id)`` edges.  Storage order, Python object
    order, and task-specific metadata are excluded.

    The digest identifies the labeled public observation graph.  It changes
    when the public-ID assignment changes the public edge relation.  Two DAGs
    produce the same digest **iff** they have the same set of public
    observation IDs and the same set of public directed edges.  Because the
    rank-to-public-ID permutation determines the mapping from internal
    adjacency to public edges, changing the permutation (on any non-trivial
    graph) changes the digest.

    Args:
        adjacency: Adjacency list in **rank-index space**.
            ``adj[i]`` is a sorted list of successor **rank** indices for
            rank ``i``.  This is the canonical index space — all producers
            and consumers that compute the digest MUST use this convention.
        obs_ids: ``(n_nodes,)`` int32 array of public observation IDs in
            **rank-index order** (``obs_ids[i]`` = public ID of rank ``i``).
        schema_version: Canonical serialization schema version (default ``1``).

    Returns:
        String ``"sha256:<64-hex-char>"``.

    Raises:
        ValueError: When ``len(adjacency) != len(obs_ids)``.
    """
    n = len(adjacency)
    if n != len(obs_ids):
        raise ValueError(
            f"adjacency length ({n}) does not match obs_ids length ({len(obs_ids)})."
        )

    # Build edge list and sort by (source_obs_id, dest_obs_id) — iterating
    # rank order does NOT guarantee public-ID order because obs_ids is a
    # random permutation.
    edges_set: set[tuple[int, int]] = set()
    for u in range(n):
        u_obs = int(obs_ids[u])
        for v in adjacency[u]:
            v_obs = int(obs_ids[v])
            edges_set.add((u_obs, v_obs))
    edges_sorted = sorted(edges_set)

    # Sorted set of public node IDs (not rank-index ordered).
    node_ids_sorted = sorted(int(obs_ids[i]) for i in range(n))

    canonical = {
        "schema": f"ehc-sn.dag.v{schema_version}",
        "n_nodes": n,
        "nodes": node_ids_sorted,
        "edges": edges_sorted,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def construction_dag_digest(
    adjacency: list[list[int]],
    obs_ids: np.ndarray,
    seed: int,
    params: dict | None = None,
    *,
    schema_version: int = 1,
) -> str:
    """Return ``"sha256:..."`` for DAG construction provenance.

    Unlike :func:`canonical_dag_digest`, this digest includes the
    explicit ``rank_to_obs_id`` mapping, the generation seed, and optional
    generation parameters.  Both digests change when the permutation
    changes — this digest records the provenance directly, while
    :func:`canonical_dag_digest` changes because the public edge set
    changes.

    Args:
        adjacency: Adjacency list in rank-index space.
        obs_ids: Public observation IDs in rank-index order.
        seed: Generation seed.
        params: Optional dict of resolved generation parameters.
        schema_version: Canonical serialization schema version (default ``1``).

    Returns:
        String ``"sha256:<64-hex-char>"``.
    """
    n = len(adjacency)
    edges_as_seen: list[list[int]] = []
    for u in range(n):
        for v in sorted(adjacency[u]):
            edges_as_seen.append([int(obs_ids[u]), int(obs_ids[v])])

    record: dict = {
        "schema": f"ehc-sn.dag.provenance.v{schema_version}",
        "n_nodes": n,
        "rank_to_obs_id": [int(obs_ids[i]) for i in range(n)],
        "edges": edges_as_seen,
        "seed": seed,
    }
    if params is not None:
        record["params"] = dict(params)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
