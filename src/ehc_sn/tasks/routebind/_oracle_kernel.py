"""Numba-accelerated reverse 0-1 BFS kernel for the routebind oracle.

This module exports one ``@njit`` compiled function that operates on
pre-allocated primitive NumPy arrays.  No Python objects, dicts, or
dataclasses appear inside the compiled function.

Algorithm
---------
For one layout and one goal observation ``g``, the kernel computes the
minimum physical-move cost from every product state ``(p, o)`` to the
closest physical occurrence of ``g`` (with the goal already accepted).
The cost is defined by:

- physical moves: cost 1, preserves semantic state.
- semantic acceptance: cost 0, requires the current cell to contain
  a valid DAG successor of the current accepted observation.

The kernel stores a policy table alongside distances, enabling
reconstruction of the optimal product-state path from any start state.

See ``spec/designs/routebind-oracle-reverse-bfs.md`` for the full
design rationale.
"""

from __future__ import annotations

import numpy as np
from numba import njit

INF = np.iinfo(np.int32).max

# Transition-kind constants matching PolicyTransition (hard-coded so that
# the Numba kernel does not import from contracts.py).
_UNSET: int = 0
_MOVE: int = 1
_ACCEPT: int = 2
_GOAL: int = 3


@njit(cache=True)
def reverse_01_bfs_kernel(
    physical_neighbors: np.ndarray,
    node_at_position: np.ndarray,
    pred_offsets: np.ndarray,
    pred_nodes: np.ndarray,
    goal_occurrences: np.ndarray,
    goal_node_idx: int,
    n_slots: int,
    n_obs: int,
    distance: np.ndarray,
    policy_kind: np.ndarray,
    policy_next: np.ndarray,
    opt_count: np.ndarray,
    deque_buf: np.ndarray,
) -> int:
    """Run one reverse 0-1 BFS over the product graph.

    Mutates *distance*, *policy_kind*, *policy_next*, and *opt_count*
    in-place.  Returns the number of states processed (>= 0) on success,
    or ``-1`` if the deque overflowed.

    Parameters
    ----------
    physical_neighbors : (n_slots, 4) int32
        Neighbour position index per direction (UP=0, RIGHT=1, DOWN=2,
        LEFT=3).  ``-1`` for invalid/blocked.
    node_at_position : (n_slots,) int32
        Public observation ID at each position, or ``-1`` for non-
        observation cells.  Read-only.
    pred_offsets : (n_obs + 1,) int32
        CSR offsets into *pred_nodes*.  ``pred_offsets[o']`` .. ``pred_offsets[o'+1]``
        are the predecessors of ``o'`` in the DAG.
    pred_nodes : (total_pred_edges,) int32
        Flat predecessor observation IDs, sorted ascending per destination.
    goal_occurrences : (n_goal_occs,) int32
        Physical positions containing the goal observation.
    goal_node_idx : int
        Goal observation ID (public).
    n_slots : int
        Number of spatial positions.
    n_obs : int
        Number of DAG observation nodes.
    distance : (n_slots * n_obs,) int32
        Workspace; initialised to ``INF`` by caller.
    policy_kind : (n_slots * n_obs,) int8
        Workspace; initialised to ``_UNSET`` by caller.
    policy_next : (n_slots * n_obs,) int32
        Workspace; initialised to ``-1`` by caller.
    opt_count : (n_slots * n_obs,) uint8
        Workspace; initialised to ``0`` by caller.
    deque_buf : (2 * n_slots * n_obs,) int32
        Ring-buffer deque workspace.

    Returns
    -------
    int
        Number of states popped from the deque, or ``-1`` on overflow.
    """
    n_states = n_slots * n_obs
    deque_capacity = deque_buf.shape[0]

    head = 0
    tail = 0

    # --- Step 1: seed all goal-occurrence states ---
    for gi in range(goal_occurrences.shape[0]):
        p = goal_occurrences[gi]
        s = p * n_obs + goal_node_idx
        distance[s] = 0
        policy_kind[s] = _GOAL
        opt_count[s] = 1
        deque_buf[tail] = s
        tail += 1
        if tail == deque_capacity:
            tail = 0
        # overflow check after first push
        if head == tail:
            return -1

    n_processed = 0

    # --- Step 2: 0-1 BFS ---
    while head != tail:
        u = deque_buf[head]
        head += 1
        if head == deque_capacity:
            head = 0
        n_processed += 1

        u_pos = u // n_obs
        u_obs = u % n_obs
        du = distance[u]

        # --- Reverse physical propagation (cost 1, push BACK) ---
        for k in range(4):
            p = physical_neighbors[u_pos, k]
            if p < 0:
                continue
            v = p * n_obs + u_obs
            nd = du + 1
            if nd < distance[v]:
                distance[v] = nd
                policy_kind[v] = _MOVE
                policy_next[v] = u
                opt_count[v] = opt_count[u]
                # push BACK
                deque_buf[tail] = v
                tail += 1
                if tail == deque_capacity:
                    tail = 0
                if head == tail:
                    return -1
            elif nd == distance[v] and nd < INF:
                new_ct = opt_count[v] + opt_count[u]
                if new_ct > 2:
                    new_ct = 2
                opt_count[v] = new_ct

        # --- Reverse semantic propagation (cost 0, push FRONT) ---
        # Guard: position must contain the currently-accepted observation o'.
        # This ensures forward acceptance o -> o' is physically possible.
        obs_here = node_at_position[u_pos]
        if obs_here == u_obs and obs_here >= 0:
            start_off = pred_offsets[obs_here]
            end_off = pred_offsets[obs_here + 1]
            for j in range(start_off, end_off):
                prev_obs = pred_nodes[j]
                v = u_pos * n_obs + prev_obs
                if du < distance[v]:
                    distance[v] = du
                    policy_kind[v] = _ACCEPT
                    policy_next[v] = u
                    opt_count[v] = opt_count[u]
                    # push FRONT
                    head -= 1
                    if head < 0:
                        head = deque_capacity - 1
                    deque_buf[head] = v
                    if head == tail:
                        return -1
                elif du == distance[v] and du < INF:
                    new_ct = opt_count[v] + opt_count[u]
                    if new_ct > 2:
                        new_ct = 2
                    opt_count[v] = new_ct

    return n_processed
