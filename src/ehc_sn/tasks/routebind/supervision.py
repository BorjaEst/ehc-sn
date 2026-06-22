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
class RoutebindFieldSupervision:
    """Typed supervision struct for routebind field prediction.

    Attributes:
        target_trajectory: Oracle spatial trajectory field, shape
            ``(B, S)`` float32, values in ``[0, 1]``.
        target_waypoint: Oracle semantic waypoint field, shape
            ``(B, S)`` float32, values in ``[0, 1]``.
        target_next_dir: Ground-truth first movement direction, shape
            ``(B,)`` int64.
        target_next_obs: Ground-truth first post-start accepted
            observation identity, shape ``(B,)`` int64.
        mask: Valid-cell mask, shape ``(B, S)`` bool.
    """

    target_trajectory: Tensor
    target_waypoint: Tensor
    target_next_dir: Tensor
    target_next_obs: Tensor
    mask: Tensor


# =============================================================================
def build_routebind_supervision(
    executed_batch: Batch,
) -> RoutebindFieldSupervision:
    """Build routebind field supervision from an executed batch frame.

    Args:
        executed_batch: Must contain ``"target_trajectory"``,
            ``"target_waypoint"``, ``"target_next_dir"``,
            ``"target_next_obs"``, ``"spatial_mask"``.

    Returns:
        Typed supervision struct.

    Raises:
        KeyError: If any required key is missing from *executed_batch*.
    """
    required_keys = (
        "target_trajectory",
        "target_waypoint",
        "target_next_dir",
        "target_next_obs",
        "spatial_mask",
    )
    for key in required_keys:
        if key not in executed_batch:
            raise KeyError(
                f"build_routebind_supervision: '{key}' missing from "
                f"executed_batch."
            )
    mask = executed_batch["spatial_mask"].to(dtype=torch.bool)
    return RoutebindFieldSupervision(
        target_trajectory=executed_batch["target_trajectory"].to(
            dtype=torch.float32
        ),
        target_waypoint=executed_batch["target_waypoint"].to(
            dtype=torch.float32
        ),
        target_next_dir=executed_batch["target_next_dir"].to(dtype=torch.int64),
        target_next_obs=executed_batch["target_next_obs"].to(dtype=torch.int64),
        mask=mask,
    )


# =============================================================================
__all__ = [
    "RoutebindFieldSupervision",
    "build_routebind_supervision",
]
