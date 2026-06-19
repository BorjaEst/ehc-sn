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
    "successor_indices",
    "successor_mask",
)
"""All keys present in a collated goaltrace batch.

Distinct from :data:`GOALTRACE_MODEL_INPUT_CHANNELS` which is the
adapter-facing subset.  Keys here also include supervision targets
and evaluation metadata.
"""


# =============================================================================
def _require_key(batch: Batch, key: str) -> None:
    """Raise ``KeyError`` if *key* is absent from *batch*."""
    if key not in batch:
        raise KeyError(f"Goaltrace batch is missing required key: {key!r}.")


# =============================================================================
def extract_goaltrace_task_input(batch: Batch) -> GoaltraceTaskInput:
    """Extract goaltrace task input fields from one generic batch mapping.

    Explicitly projects only the five model-input keys.  Extra batch keys
    (targets, evaluation metadata) are silently ignored.

    Args:
        batch: Dict mapping channel names to tensors.

    Returns:
        Typed :class:`GoaltraceTaskInput` with dtype coercion applied.

    Raises:
        KeyError: When any model-input key is absent from *batch*.
    """
    for k in (
        "observation_id",
        "weight",
        "current_flag",
        "goal_flag",
        "node_mask",
    ):
        _require_key(batch, k)
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
