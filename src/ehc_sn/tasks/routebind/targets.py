"""Routebind oracle-result-to-target encoding.

Converts a solved ``OracleResult`` into training supervision targets:
trajectory field, waypoint field, and consistency validation.

This module does NOT search or reconstruct.  It receives an already-computed
oracle result and encodes it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# =============================================================================
# Target encoding
# =============================================================================


def encode_trajectory_field(
    physical_route: tuple[int, ...] | list[int],
    n_slots: int,
    decay: float,
) -> np.ndarray:
    """Encode the spatial trajectory field.

    Args:
        physical_route: Ordered positions of the projected route.
        n_slots: Total number of spatial positions (S = H * W).
        decay: Spatial decay factor ``gamma_space`` in ``(0, 1)``.

    Returns:
        ``(n_slots,)`` float32 array with ``decay ** k`` at route position k.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1), got {decay}.")
    field = np.zeros(n_slots, dtype=np.float32)
    for k, pos in enumerate(physical_route):
        field[pos] = float(decay**k)
    return field


def encode_waypoint_field(
    waypoints: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    n_slots: int,
    decay: float,
) -> np.ndarray:
    """Encode the semantic waypoint field.

    Args:
        waypoints: Accepted semantic events as
            ``(product_state_idx, position, observation_id)``.
            First waypoint is the start (already accepted, receives 1.0).
        n_slots: Total number of spatial positions (S = H * W).
        decay: Semantic decay factor ``gamma_sem`` in ``(0, 1)``.

    Returns:
        ``(n_slots,)`` float32 array with ``decay ** m`` at each waypoint
        position.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1), got {decay}.")
    field = np.zeros(n_slots, dtype=np.float32)
    for m, (_, pos, _) in enumerate(waypoints):
        field[pos] = float(decay**m)
    return field


# =============================================================================
# Optimal-support target encoders
# =============================================================================


def encode_trajectory_support(
    support: np.ndarray,
    forward_depth: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Encode the decayed trajectory field from support and depth arrays.

    ``target_trajectory[p] = support[p] * decay ** forward_depth[p]``

    Args:
        support: ``(n_slots,)`` bool — trajectory support mask.
        forward_depth: ``(n_slots,)`` int16 — minimum forward depth per
            supported position (0 for unsupported).
        decay: Spatial decay factor ``gamma_space`` in ``(0, 1)``.

    Returns:
        ``(n_slots,)`` float32 array.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1), got {decay}.")
    field = np.zeros(len(support), dtype=np.float32)
    for p in range(len(support)):
        if support[p]:
            field[p] = float(decay ** int(forward_depth[p]))
    return field


def encode_waypoint_support(
    support: np.ndarray,
    semantic_depth: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Encode the decayed waypoint field from support and depth arrays.

    ``target_waypoint[p] = support[p] * decay ** semantic_depth[p]``

    Args:
        support: ``(n_slots,)`` bool — waypoint support mask.
        semantic_depth: ``(n_slots,)`` int16 — minimum semantic depth per
            waypoint position (0 for non-waypoint).
        decay: Semantic decay factor ``gamma_sem`` in ``(0, 1)``.

    Returns:
        ``(n_slots,)`` float32 array.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError(f"decay must be in (0, 1), got {decay}.")
    field = np.zeros(len(support), dtype=np.float32)
    for p in range(len(support)):
        if support[p]:
            field[p] = float(decay ** int(semantic_depth[p]))
    return field


def encode_optimal_direction_mask(
    direction_mask: np.ndarray,
) -> np.ndarray:
    """Encode the multi-label optimal direction mask.

    Args:
        direction_mask: ``(4,)`` bool — one per {UP, RIGHT, DOWN, LEFT}.

    Returns:
        ``(4,)`` bool array (passthrough).
    """
    return direction_mask.copy()


def encode_optimal_observation_mask(
    observation_mask: np.ndarray,
) -> np.ndarray:
    """Encode the multi-label optimal first-observation mask.

    Args:
        observation_mask: ``(n_obs,)`` bool.

    Returns:
        ``(n_obs,)`` bool array (passthrough).
    """
    return observation_mask.copy()


# =============================================================================
# Decay consistency
# =============================================================================


def validate_decay_consistency(
    field: np.ndarray,
    route: tuple[int, ...] | list[int],
    decay: float,
    tol: float = 1e-5,
) -> list[str]:
    """Check that *field* matches ``decay**k`` at each route position.

    Returns list of violation descriptions.  Empty list means consistent.
    """
    violations: list[str] = []
    for k, pos in enumerate(route):
        expected = decay**k
        actual = float(field[pos])
        if abs(actual - expected) > tol:
            violations.append(
                f"pos {pos} step {k}: field={actual:.6f} expected={expected:.6f}"
            )
    return violations


__all__ = [
    "encode_trajectory_field",
    "encode_trajectory_support",
    "encode_waypoint_field",
    "encode_waypoint_support",
    "encode_optimal_direction_mask",
    "encode_optimal_observation_mask",
    "validate_decay_consistency",
]
