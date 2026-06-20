"""Geometric substrate + relational topology combination for Goaltrace.

This module generates a hidden spatial environment, places observation anchors,
computes traversable distances, applies a truncated exponential kernel, and
masks the result with DAG adjacency to produce a sparse directed relational
weight matrix.

Four public functions:

    1. generate_anchor_grid        — place distinct observation anchors on a grid
    2. compute_all_pairs_grid_distances  — BFS distances between anchors
    3. distance_kernel             — truncated exponential kernel K(d) in [0, 1]
    4. combine_relational_topology — mask geometric support with DAG adjacency
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ehc_sn.data.layout.openfield import rectangle_adjacency


# =============================================================================
def generate_anchor_grid(
    n_observations: int,
    grid_width: int,
    grid_height: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place *n_observations* distinct anchor cells on a rectangular grid
    and return the grid adjacency matrix.

    Each observation gets exactly one canonical anchor.  Anchors are randomly
    selected without replacement from all grid cells.

    Args:
        n_observations: Number of observations (N).
        grid_width: Grid width in cells.
        grid_height: Grid height in cells.
        seed: Deterministic seed for anchor placement.

    Returns:
        (anchors, grid_adj) where:

        - anchors: (n_observations, 2) int32 array of (row, col).
        - grid_adj: (C, C) bool adjacency matrix, no self-loops.

    Raises:
        ValueError: If n_observations exceeds the number of grid cells.
    """
    n_cells = grid_width * grid_height
    if n_observations > n_cells:
        raise ValueError(
            f"n_observations ({n_observations}) exceeds grid cells "
            f"({grid_width}x{grid_height} = {n_cells})."
        )

    rng = np.random.default_rng(seed)
    all_cells = np.arange(n_cells, dtype=np.int32)
    chosen = rng.permuted(all_cells)[:n_observations]
    rows = chosen // grid_width
    cols = chosen % grid_width
    anchors = np.column_stack([rows, cols]).astype(np.int32)

    grid_adj = rectangle_adjacency(grid_width, grid_height, stay_still=False)

    return anchors, grid_adj


# =============================================================================
def compute_all_pairs_grid_distances(
    anchors: np.ndarray,
    grid_adj: np.ndarray,
    grid_width: int,
) -> np.ndarray:
    """Compute shortest traversable grid distances between all anchor pairs.

    Runs BFS from each anchor cell to find distances to all other anchor cells.

    Args:
        anchors: (N, 2) int32 array of (row, col) positions.
        grid_adj: (C, C) bool adjacency matrix (no self-loops).
        grid_width: Number of columns in the grid.

    Returns:
        (N, N) float64 distance matrix.  Symmetric, diagonal zero.
    """
    N = anchors.shape[0]
    n_cells = grid_adj.shape[0]

    # Build adjacency list once
    succ: list[list[int]] = [[] for _ in range(n_cells)]
    for u in range(n_cells):
        succ[u] = [v for v in range(n_cells) if grid_adj[u, v]]

    # Convert anchors to linear indices: idx = row * width + col
    anchor_linear = anchors[:, 0] * grid_width + anchors[:, 1]

    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        start = int(anchor_linear[i])
        dist = np.full(n_cells, np.iinfo(np.int32).max, dtype=np.int32)
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            d = dist[u] + 1
            for v in succ[u]:
                if d < dist[v]:
                    dist[v] = d
                    q.append(v)
        for j in range(N):
            D[i, j] = float(dist[int(anchor_linear[j])])

    return D


# =============================================================================
def distance_kernel(
    D: np.ndarray,
    tau: float,
    D_max: float,
) -> np.ndarray:
    """Apply the truncated exponential kernel to a distance matrix.

    K(d) = (exp(-d/tau) - exp(-D_max/tau)) / (1 - exp(-D_max/tau)),  d < D_max
    K(d) = 0, d >= D_max

    Args:
        D: (N, N) float64 distance matrix.
        tau: Temperature / distance scale (must be > 0).
        D_max: Maximum effective distance (must be > 0).

    Returns:
        (N, N) float64 support values in [0, 1].

    Raises:
        ValueError: If tau <= 0 or D_max <= 0.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if D_max <= 0:
        raise ValueError(f"D_max must be > 0, got {D_max}")

    K = np.zeros_like(D, dtype=np.float64)
    mask = D < D_max

    exp_neg_d_tau = np.exp(-D[mask] / tau)
    exp_neg_max_tau = np.exp(-D_max / tau)
    denom = 1.0 - exp_neg_max_tau

    if denom <= 0:
        K[mask] = 1.0
    else:
        K[mask] = (exp_neg_d_tau - exp_neg_max_tau) / denom

    return K


# =============================================================================
def combine_relational_topology(
    geo_support: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """Mask geometric support with DAG adjacency to produce a directed
    relational weight matrix.

    For each ordered pair (i, j):

        W[i,j] = geo_support[i,j]  if j in adjacency[i], else 0.0

    Args:
        geo_support: (N, N) float64 kernel output in [0, 1].
        adjacency: DAG adjacency list adj[i] = list of successor indices.

    Returns:
        (N, N) float64 weight matrix W.

    Raises:
        ValueError: If the positive-weight-iff-edge invariant is violated.
    """
    N = geo_support.shape[0]
    W = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        if not adjacency[i]:
            continue
        for j in adjacency[i]:
            W[i, j] = geo_support[i, j]

    # Validate: positive weight iff edge
    for i in range(N):
        has_adj = i < len(adjacency)
        for j in range(N):
            has_edge = has_adj and j in adjacency[i]
            has_weight = W[i, j] > 0
            if has_edge and not has_weight:
                raise ValueError(
                    f"Edge ({i} -> {j}) exists but weight is zero. "
                    f"geo_support[{i}, {j}] = {geo_support[i, j]:.6f}"
                )
            if has_weight and not has_edge:
                raise ValueError(
                    f"Weight positive at ({i}, {j}) but no edge."
                )

    return W


# =============================================================================
__all__ = [
    "generate_anchor_grid",
    "compute_all_pairs_grid_distances",
    "distance_kernel",
    "combine_relational_topology",
]
