"""Arena task-level batch helpers.

Task-level coercion helpers that are needed outside any single execution mode.
These functions operate on generic batch or carry mappings and produce typed
task contracts without depending on replay-mode internals.
"""

from __future__ import annotations

from typing import Mapping

from torch import Tensor

from .contracts import ArenaTargets, ArenaTaskInput


# =============================================================================
def coerce_arena_targets(
    data: Mapping[str, Tensor],
) -> ArenaTargets:
    """Extract :class:`ArenaTargets` from a carry data mapping.

    Args:
        data: Carry data dict, must contain ``observation_id``.

    Returns:
        :class:`ArenaTargets` with ``observation_id`` and optional
        ``is_revisit``.

    Raises:
        KeyError: If ``observation_id`` is absent.
    """
    if "observation_id" not in data:
        raise KeyError("Arena carry data must provide 'observation_id' to build ArenaTargets.")
    return ArenaTargets(
        observation_id=data["observation_id"],
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
def coerce_arena_task_input(
    data: Mapping[str, Tensor],
) -> ArenaTaskInput:
    """Coerce a step payload mapping to a typed :class:`ArenaTaskInput`.

    Args:
        data: Step payload dict.  Must contain ``observation``,
            ``observation_id``, ``previous_action``, and ``location_id``.

    Returns:
        :class:`ArenaTaskInput` with all required and optional fields set.

    Raises:
        KeyError: If any required field is absent.
    """
    required = ("observation", "observation_id", "previous_action", "location_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Arena task-input payload is missing required fields: {', '.join(missing)}.")
    return ArenaTaskInput(
        observation=data["observation"],
        observation_id=data["observation_id"],
        previous_action=data["previous_action"],
        location_id=data["location_id"],
        valid_action_mask=data.get(
            "valid_action_mask",
            data["observation"].new_zeros(data["observation"].shape[0]),
        ),
        step_count=data.get(
            "step_count",
            data["observation"].new_zeros(data["observation"].shape[0], 1),
        ),
        region_id=data.get("region_id"),
        landmark_id=data.get("landmark_id"),
        episode_start=data.get("episode_start"),
        is_revisit=data.get("is_revisit"),
    )


# =============================================================================
__all__ = [
    "coerce_arena_targets",
    "coerce_arena_task_input",
]
