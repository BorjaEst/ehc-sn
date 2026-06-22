"""Routebind task-level runtime helpers and typed batch extraction.

Owns typed extraction of task input and target dataclasses from generic batch
mappings.  Routebind is a single-step field-prediction task — there is no
multi-step TaskRuntime implementation.
"""

from __future__ import annotations

from typing import Final

import torch

from ehc_sn.tasks.routebind.contracts import (
    RoutebindTargets,
    RoutebindTaskInput,
)
from ehc_sn.types import Batch

# Canonical batch keys for routebind field prediction.
ROUTEBIND_BATCH_KEYS: Final[tuple[str, ...]] = (
    "cell_type",
    "observation_id",
    "start_flag",
    "goal_flag",
    "cell_mask",
    "target_trajectory",
    "target_waypoint",
    "target_next_dir",
    "target_next_obs",
)
"""All keys present in a collated routebind batch.

Distinct from :data:`ROUTEBIND_MODEL_INPUT_CHANNELS` which is the
adapter-facing subset.  Keys here also include supervision targets.
"""


# =============================================================================
def _require_key(batch: Batch, key: str) -> None:
    """Raise ``KeyError`` if *key* is absent from *batch*."""
    if key not in batch:
        raise KeyError(f"Routebind batch is missing required key: {key!r}.")


# =============================================================================
def extract_routebind_task_input(batch: Batch) -> RoutebindTaskInput:
    """Extract routebind task input fields from one generic batch mapping.

    Explicitly projects only the five model-input keys.  Extra batch keys
    (targets, evaluation metadata) are silently ignored.

    Args:
        batch: Dict mapping channel names to tensors.

    Returns:
        Typed :class:`RoutebindTaskInput` with dtype coercion applied.

    Raises:
        KeyError: When any model-input key is absent from *batch*.
    """
    for k in (
        "cell_type",
        "observation_id",
        "start_flag",
        "goal_flag",
        "cell_mask",
    ):
        _require_key(batch, k)
    return RoutebindTaskInput(
        cell_type=batch["cell_type"].to(dtype=torch.int32),
        observation_id=batch["observation_id"].to(dtype=torch.int64),
        start_flag=batch["start_flag"].to(dtype=torch.bool),
        goal_flag=batch["goal_flag"].to(dtype=torch.bool),
        cell_mask=batch["cell_mask"].to(dtype=torch.bool),
    )


# =============================================================================
def extract_routebind_targets(batch: Batch) -> RoutebindTargets:
    """Extract routebind supervision targets from one generic batch mapping.

    Args:
        batch: Dict mapping channel names to tensors.

    Returns:
        Typed :class:`RoutebindTargets` with dtype coercion applied.

    Raises:
        KeyError: When any target key is absent from *batch*.
    """
    for k in (
        "target_trajectory",
        "target_waypoint",
        "target_next_dir",
        "target_next_obs",
    ):
        _require_key(batch, k)
    return RoutebindTargets(
        target_trajectory=batch["target_trajectory"].to(dtype=torch.float32),
        target_waypoint=batch["target_waypoint"].to(dtype=torch.float32),
        target_next_dir=batch["target_next_dir"].to(dtype=torch.int64),
        target_next_obs=batch["target_next_obs"].to(dtype=torch.int64),
    )


# =============================================================================
__all__ = [
    "ROUTEBIND_BATCH_KEYS",
    "extract_routebind_targets",
    "extract_routebind_task_input",
]
