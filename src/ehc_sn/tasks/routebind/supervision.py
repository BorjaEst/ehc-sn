"""Routebind task supervision — target coercion for field prediction.

This module owns the conversion of routebind batch tensors into
learning-ready supervision structs.  It is task-owned and reusable
across model families (HRM v1, any future field-prediction model).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class RoutebindSupervision:
    """Typed supervision struct for routebind field prediction.

    Field names match ``RoutebindCorpusSchema`` exactly.

    Attributes:
        target_trajectory: Oracle spatial trajectory field, shape
            ``(B, S)`` float32, values in ``[0, 1]``.
        target_waypoint: Oracle semantic waypoint field, shape
            ``(B, S)`` float32, values in ``[0, 1]``.
        target_optimal_directions: Multi-label mask over optimal first
            physical directions, shape ``(B, 4)`` bool.
        target_optimal_next_observations: Multi-label mask over optimal
            first post-start observations, shape ``(B, N_obs)`` bool.
        spatial_mask: Valid-cell mask, shape ``(B, S)`` bool.  True
            inside the natural spatial domain, False for storage padding.
    """

    target_trajectory: Tensor
    target_waypoint: Tensor
    target_optimal_directions: Tensor
    target_optimal_next_observations: Tensor
    spatial_mask: Tensor


# =============================================================================
def build_routebind_supervision(
    executed_batch: Batch,
) -> RoutebindSupervision:
    """Build routebind field supervision from an executed batch frame.

    Args:
        executed_batch: Must contain ``"target_trajectory"``,
            ``"target_waypoint"``, ``"target_optimal_directions"``,
            ``"target_optimal_next_observations"``, ``"spatial_mask"``.

    Returns:
        Typed supervision struct with corpus-matching field names.

    Raises:
        KeyError: If any required key is missing from *executed_batch*.
    """
    required_keys = (
        "target_trajectory",
        "target_waypoint",
        "target_optimal_directions",
        "target_optimal_next_observations",
        "spatial_mask",
    )
    for key in required_keys:
        if key not in executed_batch:
            raise KeyError(
                f"build_routebind_supervision: '{key}' missing from "
                f"executed_batch."
            )
    return RoutebindSupervision(
        target_trajectory=executed_batch["target_trajectory"].to(
            dtype=torch.float32
        ),
        target_waypoint=executed_batch["target_waypoint"].to(
            dtype=torch.float32
        ),
        target_optimal_directions=executed_batch[
            "target_optimal_directions"
        ].to(dtype=torch.bool),
        target_optimal_next_observations=executed_batch[
            "target_optimal_next_observations"
        ].to(dtype=torch.bool),
        spatial_mask=executed_batch["spatial_mask"].to(dtype=torch.bool),
    )


# =============================================================================
__all__ = [
    "RoutebindSupervision",
    "build_routebind_supervision",
]
