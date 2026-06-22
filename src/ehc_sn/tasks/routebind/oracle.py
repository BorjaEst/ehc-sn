"""Routebind product-state oracle — canonical task truth.

Owns the product-state 0-1 BFS, uniqueness detection, physical-route
projection, accepted-waypoint reconstruction, and first-direction
derivation.

This module provides two API styles:

- ``compute_goal_distance_table`` — compute a reverse 0-1 BFS table for
  one goal observation (reusable across many start queries).
- ``solve_product_state_route`` — per-query wrapper that calls the
  kernel and returns an ``OracleResult`` (preserved for backward
  compatibility).

The computational core is a Numba-accelerated reverse 0-1 BFS over the
product graph ``(position, accepted_observation)``.  The kernel stores
distance, policy, and optimal-count arrays that can be reused for many
start queries sharing the same goal.

Two confirmed bugs in the forward+reverse+DP design are fixed here:
1. Missing ``node_at_position`` guard in reverse semantic propagation.
2. Non-topological sort order in the DP counting pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ehc_sn.tasks.routebind._oracle_kernel import (
    _ACCEPT,
    _GOAL,
    _MOVE,
    _UNSET,
    INF,
    reverse_01_bfs_kernel,
)
from ehc_sn.tasks.routebind.contracts import DELTA_TO_DIRECTION, Direction

# =============================================================================
# Oracle result type
# =============================================================================


@dataclass(frozen=True)
class OracleResult:
    """Canonical output of the routebind product-state oracle.

    Attributes:
        product_path: Ordered list of flat product-state indices
            ``(pos * n_obs + dag_node)`` from start to goal.
        physical_route: Ordered spatial positions of the projected route
            (zero-cost semantic transitions removed).
        waypoints: Accepted semantic waypoint events as
            ``(product_state_idx, position, dag_node)`` tuples.
        next_dir: First physical movement direction as ``Direction`` enum
            value, or ``-1`` when the route has fewer than 2 positions.
        next_obs: DAG node index of the first post-start accepted
            observation, or ``-1`` if no such observation exists.
        cost: Total physical movement cost (number of physical steps).
    """

    product_path: tuple[int, ...]
    physical_route: tuple[int, ...]
    waypoints: tuple[tuple[int, int, int], ...]
    next_dir: int
    next_obs: int
    cost: int


# =============================================================================
# Internal helpers
# =============================================================================


def _build_dag_csr(
    adjacency: list[list[int]], n_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR-format predecessor arrays for the Numba kernel.

    Args:
        adjacency: ``adjacency[o]`` is the list of successor observation IDs
            for node ``o`` (public observation ID space).
        n_obs: Number of observation nodes.

    Returns:
        Tuple of ``(pred_offsets, pred_nodes)``:
        - pred_offsets: ``(n_obs + 1,)`` int32, CSR offsets.
        - pred_nodes: ``(total_pred_edges,)`` int32, predecessor IDs sorted
          ascending per destination.
    """
    pred_lists: list[list[int]] = [[] for _ in range(n_obs)]
    for src in range(n_obs):
        for dst in adjacency[src]:
            pred_lists[dst].append(src)
    for lst in pred_lists:
        lst.sort()

    offsets = np.zeros(n_obs + 1, dtype=np.int32)
    for i in range(n_obs):
        offsets[i + 1] = offsets[i] + len(pred_lists[i])

    nodes = np.zeros(int(offsets[-1]), dtype=np.int32)
    for i in range(n_obs):
        for j, pred in enumerate(pred_lists[i]):
            nodes[int(offsets[i]) + j] = pred

    return offsets, nodes


def _dag_ctx_to_csr(
    dag_ctx: dict[str, Any], n_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the legacy dag_ctx dict to CSR predecessor arrays.

    This bridge preserves backward compatibility for callers that pass
    the ``dag_ctx`` dict with ``pred`` and ``pred_counts`` keys.
    """
    pred_flat = dag_ctx["pred"]
    pred_counts = dag_ctx["pred_counts"]

    offsets = np.zeros(n_obs + 1, dtype=np.int32)
    for i in range(n_obs):
        offsets[i + 1] = offsets[i] + int(pred_counts[i])

    nodes = np.zeros(int(offsets[-1]), dtype=np.int32)
    for i in range(n_obs):
        cnt = int(pred_counts[i])
        for j in range(cnt):
            nodes[int(offsets[i]) + j] = int(pred_flat[i, j])

    return offsets, nodes


def reconstruct_from_policy(
    start_pos: int,
    start_obs: int,
    goal_node_idx: int,
    n_obs: int,
    distance: np.ndarray,
    policy_kind: np.ndarray,
    policy_next: np.ndarray,
    opt_count: np.ndarray,
    row_col: np.ndarray,
    node_at_position: np.ndarray,
) -> OracleResult | None:
    """Walk the policy chain from a start state and return an ``OracleResult``.

    Parameters
    ----------
    start_pos : int
        Physical start position index.
    start_obs : int
        Public observation ID at the start position (already accepted).
    goal_node_idx : int
        Goal observation ID.
    n_obs : int
        Number of DAG observation nodes.
    distance : (n_slots * n_obs,) int32
        Distance array from the kernel.
    policy_kind : (n_slots * n_obs,) int8
        Policy transition kind array from the kernel.
    policy_next : (n_slots * n_obs,) int32
        Policy next-state array from the kernel.
    opt_count : (n_slots * n_obs,) uint8
        Optimal-count array from the kernel.
    row_col : (n_slots, 2) int32
        Row, col per position.
    node_at_position : (n_slots,) int32
        Observation ID per position, ``-1`` for non-observation.

    Returns
    -------
    OracleResult or None
        ``None`` if the state is unreachable, ambiguous, or the chain is
        broken.
    """
    n_slots = row_col.shape[0]
    start_state = start_pos * n_obs + start_obs

    # --- Check reachability ---
    if distance[start_state] >= INF:
        return None

    # --- Check uniqueness ---
    if opt_count[start_state] >= 2:
        return None

    # --- Walk the policy chain ---
    product_path: list[int] = [start_state]
    u = start_state
    while policy_kind[u] != _GOAL:
        u = int(policy_next[u])
        if u < 0:
            return None  # broken chain
        product_path.append(u)

    # --- Extract physical route and waypoints ---
    physical_route: list[int] = []
    waypoints: list[tuple[int, int, int]] = []
    for ps_idx, ps in enumerate(product_path):
        p_pos = ps // n_obs
        p_obs = ps % n_obs
        if ps_idx == 0:
            physical_route.append(p_pos)
            waypoints.append((ps_idx, p_pos, p_obs))
        else:
            prev_ps = product_path[ps_idx - 1]
            prev_pos = prev_ps // n_obs
            prev_obs = prev_ps % n_obs
            if p_pos != prev_pos:
                physical_route.append(p_pos)
            if p_obs != prev_obs:
                waypoints.append((ps_idx, p_pos, p_obs))

    cost = int(distance[start_state])

    # --- First physical step direction ---
    next_dir = -1
    if len(physical_route) >= 2:
        dr = int(row_col[physical_route[1], 0]) - int(
            row_col[physical_route[0], 0]
        )
        dc = int(row_col[physical_route[1], 1]) - int(
            row_col[physical_route[0], 1]
        )
        delta = (dr, dc)
        if delta in DELTA_TO_DIRECTION:
            next_dir = int(DELTA_TO_DIRECTION[delta])

    # --- First post-start accepted observation ---
    next_obs = -1
    if len(waypoints) >= 2:
        next_obs = waypoints[1][2]

    return OracleResult(
        product_path=tuple(product_path),
        physical_route=tuple(physical_route),
        waypoints=tuple(waypoints),
        next_dir=next_dir,
        next_obs=next_obs,
        cost=cost,
    )


# =============================================================================
# Public API
# =============================================================================


def compute_goal_distance_table(
    physical_neighbors: np.ndarray,
    node_at_position: np.ndarray,
    pred_offsets: np.ndarray,
    pred_nodes: np.ndarray,
    goal_occurrences: np.ndarray,
    goal_node_idx: int,
    n_slots: int,
    n_obs: int,
    _workspace: dict | None = None,
) -> dict:
    """Compute the reverse distance table for one goal observation.

    Runs a single reverse 0-1 BFS over the product graph from all
    terminal states ``(p, goal_node_idx)`` where ``p`` is a goal
    occurrence.  The resulting distance, policy, and optimal-count
    arrays can be queried for any start state in O(1).

    When ``_workspace`` is provided the caller supplies pre-allocated
    arrays (same shape and dtype as returned by the no-workspace path).
    The caller owns initialisation and reset between successive calls.
    Workspace keys: ``"distance"``, ``"policy_kind"``, ``"policy_next"``,
    ``"opt_count"``, ``"deque_buf"``.

    Returns a dict with keys:
        distance: ``(n_slots * n_obs,)`` int32
        policy_kind: ``(n_slots * n_obs,)`` int8
        policy_next: ``(n_slots * n_obs,)`` int32
        opt_count: ``(n_slots * n_obs,)`` uint8
        n_states_processed: int
        deque_overflow: bool

    Raises
    ------
    ValueError
        If ``goal_occurrences`` is empty, or ``goal_node_idx`` is out of
        ``[0, n_obs)``.
    KeyError
        If ``_workspace`` is provided but missing a required key.
    """
    if goal_occurrences.shape[0] == 0:
        raise ValueError("goal_occurrences is empty.")
    if not (0 <= goal_node_idx < n_obs):
        raise ValueError(
            f"goal_node_idx={goal_node_idx} out of range [0, {n_obs})."
        )

    n_states = n_slots * n_obs
    INF = np.iinfo(np.int32).max

    if _workspace is not None:
        required = (
            "distance",
            "policy_kind",
            "policy_next",
            "opt_count",
            "deque_buf",
        )
        for k in required:
            if k not in _workspace:
                raise KeyError(
                    f"compute_goal_distance_table: _workspace missing key {k!r}."
                )
        distance = _workspace["distance"]
        policy_kind = _workspace["policy_kind"]
        policy_next = _workspace["policy_next"]
        opt_count = _workspace["opt_count"]
        deque_buf = _workspace["deque_buf"]
    else:
        distance = np.full(n_states, INF, dtype=np.int32)
        policy_kind = np.zeros(n_states, dtype=np.int8)
        policy_next = np.full(n_states, -1, dtype=np.int32)
        opt_count = np.zeros(n_states, dtype=np.uint8)
        deque_buf = np.zeros(2 * n_states, dtype=np.int32)

    n_processed = reverse_01_bfs_kernel(
        physical_neighbors=physical_neighbors,
        node_at_position=node_at_position,
        pred_offsets=pred_offsets,
        pred_nodes=pred_nodes,
        goal_occurrences=goal_occurrences,
        goal_node_idx=goal_node_idx,
        n_slots=n_slots,
        n_obs=n_obs,
        distance=distance,
        policy_kind=policy_kind,
        policy_next=policy_next,
        opt_count=opt_count,
        deque_buf=deque_buf,
    )

    return {
        "distance": distance,
        "policy_kind": policy_kind,
        "policy_next": policy_next,
        "opt_count": opt_count,
        "n_states_processed": max(n_processed, 0),
        "deque_overflow": n_processed < 0,
    }


def solve_product_state_route(
    physical_neighbors: np.ndarray,
    dag_ctx: dict[str, Any],
    node_at_position: np.ndarray,
    row_col: np.ndarray,
    n_slots: int,
    n_obs: int,
    start_pos: int,
    start_node_idx: int,
    goal_node_idx: int,
    max_route_length: int,
) -> tuple[OracleResult | None, bool, str]:
    """Solve the routebind product-state oracle.

    Finds the unique optimal product-state route from
    ``(start_pos, start_node_idx)`` to any terminal state
    ``(p, goal_node_idx)`` where position ``p`` contains
    ``goal_node_idx``.

    The initial semantic state is ``start_node_idx`` (the DAG node
    corresponding to the observation physically at ``start_pos``).
    DAG node ``0`` is NOT a universal start — the caller must pass
    the actual start observation.

    Args:
        physical_neighbors: ``(n_slots, 4)`` int32 — neighbor position
            index per direction (UP=0, RIGHT=1, DOWN=2, LEFT=3),
            ``-1`` for none.
        dag_ctx: DAG context dict with keys ``succ``, ``succ_counts``,
            ``pred``, ``pred_counts``, ``rank``.
        node_at_position: ``(n_slots,)`` int32 — DAG node index per
            position, ``-1`` for non-observation cells.
        row_col: ``(n_slots, 2)`` int32 — row, col per position.
        n_slots: Number of spatial positions.
        n_obs: Number of DAG observation nodes.
        start_pos: Physical start position index.
        start_node_idx: DAG node index of the observation at start_pos.
        goal_node_idx: DAG node index of the goal observation.
        max_route_length: Maximum allowed physical route length.

    Returns:
        ``(OracleResult, True, "")`` on success,
        ``(None, False, rejection_reason)`` on failure.

    Raises:
        ValueError: If ``start_node_idx`` or ``goal_node_idx`` is out of
            range ``[0, n_obs)``.
    """
    if not (0 <= start_node_idx < n_obs):
        raise ValueError(
            f"start_node_idx={start_node_idx} out of range [0, {n_obs})."
        )
    if not (0 <= goal_node_idx < n_obs):
        raise ValueError(
            f"goal_node_idx={goal_node_idx} out of range [0, {n_obs})."
        )
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots}.")

    # --- Convert legacy dag_ctx dict to CSR predecessor arrays ---
    pred_offsets, pred_nodes = _dag_ctx_to_csr(dag_ctx, n_obs)

    # --- Find all goal occurrences ---
    goal_positions = np.where(node_at_position == goal_node_idx)[0]
    if goal_positions.shape[0] == 0:
        return (None, False, "no_path")

    # --- Run the reverse kernel ---
    table = compute_goal_distance_table(
        physical_neighbors=physical_neighbors,
        node_at_position=node_at_position,
        pred_offsets=pred_offsets,
        pred_nodes=pred_nodes,
        goal_occurrences=goal_positions,
        goal_node_idx=goal_node_idx,
        n_slots=n_slots,
        n_obs=n_obs,
    )

    if table["deque_overflow"]:
        return (None, False, "deque_overflow")

    # --- Reconstruct from start state ---
    distance = table["distance"]
    policy_kind = table["policy_kind"]
    policy_next = table["policy_next"]
    opt_count = table["opt_count"]

    start_cost = int(distance[start_pos * n_obs + start_node_idx])
    if start_cost >= INF:
        return (None, False, "no_path")

    result = reconstruct_from_policy(
        start_pos=start_pos,
        start_obs=start_node_idx,
        goal_node_idx=goal_node_idx,
        n_obs=n_obs,
        distance=distance,
        policy_kind=policy_kind,
        policy_next=policy_next,
        opt_count=opt_count,
        row_col=row_col,
        node_at_position=node_at_position,
    )

    if result is None:
        # Check which reason
        if opt_count[start_pos * n_obs + start_node_idx] >= 2:
            return (None, False, "multiple_optimal_solutions")
        return (None, False, "reconstruction_failed")

    # --- Post-reconstruction validation ---
    physical_route = result.physical_route
    waypoints = result.waypoints

    if len(set(physical_route)) != len(physical_route):
        return (None, False, "non_simple_spatial_projection")
    if len(physical_route) > max_route_length:
        return (None, False, "route_too_long")
    if len(waypoints) < 2:
        return (None, False, "trivial_waypoint_sequence")

    return (result, True, "")


__all__ = [
    "OracleResult",
    "compute_goal_distance_table",
    "reconstruct_from_policy",
    "solve_product_state_route",
]
