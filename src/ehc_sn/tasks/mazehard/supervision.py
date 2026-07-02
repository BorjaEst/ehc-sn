"""MazeHard task supervision — target coercion and token weighting.

This module owns the conversion of MazeHard batch tensors into
learning-ready supervision structs.  It is task-owned and reusable
across model families (HRM v1, HRM v2, EHP).

PATH upweighting logic moved here from :mod:`ehp_sn.adapters.hrm.objectives`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ehc_sn.tasks.mazehard.contracts import MAZE_HARD_IGNORE_LABEL_ID
from ehc_sn.tasks.mazehard.runtime import PATH_ID
from ehc_sn.types import Batch


# =============================================================================
@dataclass(frozen=True)
class MazeHardTokenSupervision:
    """Typed supervision struct for MazeHard token prediction.

    Attributes:
        labels: Integer labels with ignore positions set to
            :data:`MAZE_HARD_IGNORE_LABEL_ID`, shape ``(B, S)``.
        weights: Per-token loss weights, shape ``(B, S)`` float32.
    """

    labels: Tensor
    weights: Tensor


# =============================================================================
def build_mazehard_weights(labels: Tensor) -> Tensor:
    """Build per-token weights that upweight PATH tokens.

    PATH labels (``PATH_ID = 5``) receive weight 2.0.
    Ignore positions (``MAZE_HARD_IGNORE_LABEL_ID = -100``) receive weight 0.0.
    All other positions receive weight 1.0.

    Args:
        labels: Integer label tensor, shape ``(B, S)``.

    Returns:
        Weight tensor, shape ``(B, S)`` float32.
    """
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights = torch.where(
        labels == PATH_ID,
        torch.full_like(weights, 2.0),
        weights,
    )
    weights = torch.where(
        labels == MAZE_HARD_IGNORE_LABEL_ID,
        torch.zeros_like(weights),
        weights,
    )
    return weights


# =============================================================================
def build_mazehard_supervision(
    executed_batch: Batch,
) -> MazeHardTokenSupervision:
    """Build MazeHard token supervision from an executed batch frame.

    Args:
        executed_batch: Must contain ``"labels"``.

    Returns:
        Typed supervision struct with labels and PATH-upweighted weights.

    Raises:
        KeyError: If ``"labels"`` is missing from *executed_batch*.
    """
    if "labels" not in executed_batch:
        raise KeyError(
            "build_mazehard_supervision: 'labels' missing from executed_batch."
        )
    labels = executed_batch["labels"]
    weights = build_mazehard_weights(labels)
    return MazeHardTokenSupervision(labels=labels, weights=weights)


# =============================================================================
__all__ = [
    "MazeHardTokenSupervision",
    "build_mazehard_supervision",
    "build_mazehard_weights",
]
