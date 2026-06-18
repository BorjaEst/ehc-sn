"""Goaltrace task-level runtime helpers and typed batch extraction.

Owns typed extraction of task input and target dataclasses from generic batch
mappings.  Goaltrace is a single-step field-prediction task — there is no
multi-step TaskRuntime implementation.
"""

from __future__ import annotations

from typing import Final

import torch

from ehc_sn.tasks.goaltrace.contracts import (
    GoaltraceTargets,
    GoaltraceTaskInput,
)
from ehc_sn.types import Batch

# Canonical batch keys for goaltrace field prediction.
GOALTRACE_BATCH_KEYS: Final[tuple[str, ...]] = (
    "observation_id",
    "weight",
    "current_flag",
    "goal_flag",
    "node_mask",
    "target_field",
)


# =============================================================================
def _validate_goaltrace_batch(batch: Batch) -> None:
    """Ensure all required keys are present."""
    missing = [key for key in GOALTRACE_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(
            "Goaltrace batch is missing required keys: "
            + ", ".join(missing)
            + "."
        )


# =============================================================================
def extract_goaltrace_task_input(batch: Batch) -> GoaltraceTaskInput:
    """Extract goaltrace task input fields from one generic batch mapping.

    Args:
        batch: Dict mapping channel names to tensors.  Must contain all
            keys in :data:`GOALTRACE_BATCH_KEYS`.

    Returns:
        Typed :class:`GoaltraceTaskInput` with dtype coercion applied.

    Raises:
        KeyError: When any required key is absent from *batch*.
    """
    _validate_goaltrace_batch(batch)
    return GoaltraceTaskInput(
        observation_id=batch["observation_id"].to(dtype=torch.int64),
        weight=batch["weight"].to(dtype=torch.float32),
        current_flag=batch["current_flag"].to(dtype=torch.bool),
        goal_flag=batch["goal_flag"].to(dtype=torch.bool),
        node_mask=batch["node_mask"].to(dtype=torch.bool),
    )


# =============================================================================
def extract_goaltrace_targets(batch: Batch) -> GoaltraceTargets:
    """Extract goaltrace supervision targets from one generic batch mapping.

    Args:
        batch: Dict mapping channel names to tensors.

    Returns:
        Typed :class:`GoaltraceTargets` with dtype coercion applied.

    Raises:
        KeyError: When ``"target_field"`` is absent from *batch*.
    """
    if "target_field" not in batch:
        raise KeyError(
            "Goaltrace batch is missing required target key: 'target_field'."
        )
    return GoaltraceTargets(
        target_field=batch["target_field"].to(dtype=torch.float32),
    )


# =============================================================================
__all__ = [
    "GOALTRACE_BATCH_KEYS",
    "extract_goaltrace_task_input",
    "extract_goaltrace_targets",
]
