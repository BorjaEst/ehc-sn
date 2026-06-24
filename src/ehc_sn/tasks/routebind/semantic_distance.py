"""DAG all-pairs shortest-path table in public observation ID space.

Routebind precomputes this table to support semantic-distance-based
query selection.  The table is computed over the public observation ID
adjacency list (``pub_adjacency``) — the same adjacency that the product-
state oracle uses.  Crucially, public observation IDs are a random
permutation of topological ranks, so the algorithm never assumes
``obs_id < obs_id'`` implies topological order.

BFS from every source over the public-ID adjacency works correctly
regardless of ID ordering because it follows directed edges in public
ID space.

Semantic distance is defined in **edges** (not waypoints):
    ``semantic_shortest_path_edges(o_s, o_g)`` = minimum number of DAG
    edges from ``o_s`` to ``o_g``.

The guaranteed lower bound on waypoint count is:
    ``minimum_waypoint_count = semantic_shortest_path_edges + 1``
(because the start observation is already accepted).
"""

from __future__ import annotations

from collections import deque

import numpy as np

INF = np.iinfo(np.int32).max


def compute_semantic_distance_table(
    pub_adjacency: list[list[int]],
    n_obs: int,
) -> np.ndarray:
    """Compute all-pairs shortest-path distances in the public DAG.

    Operates on the same public observation ID adjacency that the routebind
    oracle uses.  Never assumes IDs encode topological rank.

    Args:
        pub_adjacency:
            ``pub_adjacency[src]`` is a sorted list of successor public
            observation IDs for node ``src`` (0..n_obs-1).
        n_obs:
            Number of observation nodes.

    Returns:
        ``(n_obs, n_obs)`` int32 array where ``dist[src, dst]`` is the
        minimum number of directed edges from ``src`` to ``dst``, or
        ``INF`` when no path exists.  ``dist[i, i] == 0`` for all ``i``.
    """
    dist = np.full((n_obs, n_obs), INF, dtype=np.int32)

    for src in range(n_obs):
        dist[src, src] = 0
        queue: deque[int] = deque()
        visited = {src}

        for succ in pub_adjacency[src]:
            if succ not in visited:
                visited.add(succ)
                dist[src, succ] = 1
                queue.append(succ)

        while queue:
            u = queue.popleft()
            nd = dist[src, u] + 1
            for v in pub_adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    dist[src, v] = nd
                    queue.append(v)

    return dist


def compute_diameter(dist: np.ndarray) -> int:
    """Return the maximum finite edge-distance from a distance table.

    Args:
        dist: ``(n_obs, n_obs)`` int32 array from
            :func:`compute_semantic_distance_table`.

    Returns:
        Maximum finite value, or 0 if the table contains no finite
        positive distances (single-node graph).
    """
    finite = dist[dist < INF]
    return int(finite.max()) if finite.size > 0 else 0
