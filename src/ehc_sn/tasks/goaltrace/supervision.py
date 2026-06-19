"""Goaltrace task supervision — target coercion for field prediction.

This module owns the conversion of Goaltrace batch tensors into
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
class GoaltraceFieldSupervision:
    """Typed supervision struct for Goaltrace field prediction.

    Attributes:
        target_field: Oracle goal-conditioned prospective firing field,
            shape ``(B, N)`` float32, values in ``[0, 1]``.
        node_mask: Valid-node mask, shape ``(B, N)`` bool.
    """

    target_field: Tensor
    node_mask: Tensor


# =============================================================================
def build_goaltrace_supervision(
    executed_batch: Batch,
) -> GoaltraceFieldSupervision:
    """Build Goaltrace field supervision from an executed batch frame.

    Args:
        executed_batch: Must contain ``"target_field"`` and ``"node_mask"``.

    Returns:
        Typed supervision struct.

    Raises:
        KeyError: If any required key is missing from *executed_batch*.
    """
    if "target_field" not in executed_batch:
        raise KeyError(
            "build_goaltrace_supervision: 'target_field' missing "
            "from executed_batch."
        )
    if "node_mask" not in executed_batch:
        raise KeyError(
            "build_goaltrace_supervision: 'node_mask' missing "
            "from executed_batch."
        )
    return GoaltraceFieldSupervision(
        target_field=executed_batch["target_field"].to(dtype=torch.float32),
        node_mask=executed_batch["node_mask"].to(dtype=torch.bool),
    )


# =============================================================================
__all__ = [
    "GoaltraceFieldSupervision",
    "build_goaltrace_supervision",
]
