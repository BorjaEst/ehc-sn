"""Dungeon task-level runtime helpers.

Generic task coercion helpers that are needed outside any single execution mode.
These functions operate on generic step payload mappings and produce typed
task contracts without depending on any capability internals.
"""

from __future__ import annotations

from typing import Mapping

from torch import Tensor

from .contracts import DungeonTaskInput


# =============================================================================
def coerce_dungeon_task_input(
    data: Mapping[str, Tensor],
) -> DungeonTaskInput:
    """Coerce a step payload mapping to a typed :class:`DungeonTaskInput`.

    Args:
        data: Step payload dict.  Must contain ``observation``,
            ``observation_id``, ``previous_action``, and ``location_id``.

    Returns:
        :class:`DungeonTaskInput` with all required and optional fields set.

    Raises:
        KeyError: If any required field is absent.
    """
    required = ("observation", "observation_id", "previous_action", "location_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dungeon task-input payload is missing required fields: {', '.join(missing)}.")
    return DungeonTaskInput(
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
    "coerce_dungeon_task_input",
]
