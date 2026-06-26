"""Routebind prediction-to-behavior decoding.

Owns the canonical greedy decoder for extracting discrete routes from
predicted trajectory fields, waypoint extraction, and direction-logit
decoding.

This is the inverse of ``targets.py``: targets encodes oracle truth into
fields; decoding recovers interpretable behavior from model predictions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# =============================================================================
# Route extraction
# =============================================================================


def extract_route_from_trajectory_field(
    trajectory_field: np.ndarray,
    start_idx: int,
    cell_type: np.ndarray | None = None,
    max_length: int = 150,
    grid_width: int | None = None,
) -> list[int]:
    """Extract a discrete route by greedy max-activation walk.

    At each step, examines the four neighbors of the current position in
    direction order UP (0), RIGHT (1), DOWN (2), LEFT (3).  Selects the
    traversable, unvisited neighbor with the highest predicted activation.
    Ties are broken by the fixed direction order (UP > RIGHT > DOWN > LEFT).

    When *grid_width* is ``None``, the function assumes a square grid and
    infers width from ``sqrt(n_slots)``.  This is the legacy behaviour;
    explicit *grid_width* is preferred for non-square canvases.

    Args:
        trajectory_field: ``(S,)`` float32 predicted activations.
        start_idx: Index of the start position.
        cell_type: Optional ``(S,)`` int32 cell types.  When provided,
            ``CELL_WALL`` (0) positions are rejected.
        max_length: Maximum extracted route length.
        grid_width: Width of the grid (number of columns).  When ``None``,
            inferred from assuming a square grid.

    Returns:
        Ordered list of position indices, or empty list if activation at
        *start_idx* is zero.
    """
    n_slots = len(trajectory_field)
    if trajectory_field[start_idx] <= 0:
        return []
    if grid_width is not None:
        width = grid_width
        height = n_slots // width
        if height * width != n_slots:
            raise ValueError(
                f"n_slots={n_slots} is not divisible by grid_width={width}."
            )
    else:
        width = int(math.sqrt(n_slots))
        if width * width != n_slots:
            raise ValueError(
                f"n_slots={n_slots} is not a perfect square and "
                f"grid_width was not provided."
            )
        height = width

    route: list[int] = [start_idx]
    visited: set[int] = {start_idx}

    # Direction order: UP, RIGHT, DOWN, LEFT
    dirs: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for _ in range(min(max_length, n_slots)):
        pos = route[-1]
        r, c = divmod(pos, width)
        best_nbr = -1
        best_val = -1.0
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                npos = nr * width + nc
                if npos in visited:
                    continue
                if cell_type is not None and int(cell_type[npos]) == 0:
                    continue  # WALL
                val = float(trajectory_field[npos])
                if val > best_val:
                    best_val = val
                    best_nbr = npos
        if best_nbr < 0 or best_val <= 0:
            break
        route.append(best_nbr)
        visited.add(best_nbr)

    return route


# =============================================================================
# Waypoint extraction
# =============================================================================


def extract_waypoints_from_field(
    waypoint_field: np.ndarray,
    route: list[int],
    threshold: float = 0.01,
) -> list[tuple[int, float]]:
    """Extract predicted waypoints along *route* by threshold.

    Returns ``(position, activation)`` pairs sorted by descending
    activation.
    """
    waypoints: list[tuple[int, float]] = []
    for pos in route:
        val = float(waypoint_field[pos])
        if val > threshold:
            waypoints.append((pos, val))
    waypoints.sort(key=lambda x: -x[1])
    return waypoints


def extract_waypoint_sequence(
    waypoint_field: np.ndarray,
    route: list[int],
    observation_id: np.ndarray,
    threshold: float = 0.01,
) -> list[tuple[int, int, float]]:
    """Extract accepted waypoints in route order with observation IDs.

    Unlike :func:`extract_waypoints_from_field`, this function preserves
    route traversal order and includes the observation identity at each
    waypoint position.

    Args:
        waypoint_field: ``(S,)`` float32 target or predicted waypoint field.
        route: Ordered list of position indices from
            :func:`extract_route_from_trajectory_field`.
        observation_id: ``(S,)`` int32 observation identity per position.
        threshold: Minimum activation to count as a waypoint.

    Returns:
        List of ``(position, observation_id, activation)`` tuples in route
        order.  Only positions on *route* with activation > *threshold*
        are included.
    """
    seq: list[tuple[int, int, float]] = []
    for pos in route:
        val = float(waypoint_field[pos])
        if val > threshold:
            obs = int(observation_id[pos])
            seq.append((pos, obs, val))
    return seq


def extract_waypoint_events_from_support(
    waypoint_support: np.ndarray,
    waypoint_semantic_depth: np.ndarray,
    observation_id: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Extract waypoint events from support channels without a physical route.

    Returns ``(position, observation_id, semantic_depth)`` tuples sorted
    by ascending semantic depth.  When multiple positions share the same
    semantic depth (ambiguous optimal acceptance), all are included.

    Unlike :func:`extract_waypoint_sequence`, this function does not
    require a physically extracted route — it reads directly from the
    oracle's waypoint support channels.

    Args:
        waypoint_support: ``(S,)`` bool — ``True`` for positions where an
            optimal semantic acceptance occurs (plus the start position).
        waypoint_semantic_depth: ``(S,)`` int16 — minimum number of
            semantic-acceptance events from start per waypoint position.
            Sentinel ``-1`` for unsupported positions.
        observation_id: ``(S,)`` int32 — observation identity per position.

    Returns:
        List of ``(position, observation_id, semantic_depth)`` tuples
        sorted by ascending semantic depth.
    """
    if waypoint_support.shape != waypoint_semantic_depth.shape:
        raise ValueError(
            f"waypoint_support shape {waypoint_support.shape} does not match "
            f"waypoint_semantic_depth shape {waypoint_semantic_depth.shape}."
        )
    events: list[tuple[int, int, int]] = []
    for p in range(len(waypoint_support)):
        if waypoint_support[p]:
            sd = int(waypoint_semantic_depth[p])
            if sd >= 0:
                events.append((p, int(observation_id[p]), sd))
    events.sort(key=lambda x: x[2])
    return events


# =============================================================================
# Direction decoding
# =============================================================================


def decode_next_direction(
    logits: np.ndarray,
) -> int:
    """Return the direction enum value from logit argmax.

    Args:
        logits: ``(4,)`` float32 logits for {UP, RIGHT, DOWN, LEFT}.

    Returns:
        ``Direction`` enum value (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT).
    """
    return int(np.argmax(logits).item())


__all__ = [
    "decode_next_direction",
    "extract_route_from_trajectory_field",
    "extract_waypoint_events_from_support",
    "extract_waypoint_sequence",
    "extract_waypoints_from_field",
]
