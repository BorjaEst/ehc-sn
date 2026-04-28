"""Arena task-level runtime helpers.

Generic task coercion helpers and batch schema constants that operate on
generic batch or carry mappings and produce typed task contracts without
depending on any execution-binding internals.

Replay batch schema constants and helpers live here (not in the capability
module) because data-generation scripts, lightning runtimes, and capability
code all reference the same schema.
"""

from __future__ import annotations

from typing import Final, Mapping

from torch import Tensor

from ehc_sn.types import Batch

from .contracts import ArenaTargets, ArenaTaskInput

# =============================================================================
# Replay batch schema constants
# =============================================================================

ARENA_REPLAY_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "topology",
    "observations",
    "mask_valid",
    "trajectory_row",
    "trajectory_col",
    "trajectory_previous_action",
    "trajectory_episode_start",
    "trajectory_valid_step",
    "trajectory_length",
)

ARENA_REPLAY_OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    "regions",
    "landmarks",
)

ARENA_STEP_KEYS: Final[tuple[str, ...]] = (
    "observation",
    "observation_id",
    "previous_action",
    "location_id",
    "region_id",
    "landmark_id",
    "valid_action_mask",
    "step_count",
    "episode_start",
    "is_revisit",
)


# =============================================================================
def batch_size_from_arena_batch(batch: Batch) -> int:
    """Return the leading batch dimension from an arena replay batch."""
    return int(batch[ARENA_REPLAY_REQUIRED_KEYS[0]].shape[0])


# =============================================================================
def infer_arena_replay_batch_keys(batch: Batch) -> tuple[str, ...]:
    """Return the replay batch keys present in ``batch``, validating required keys."""
    missing = [key for key in ARENA_REPLAY_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError("Arena replay batch is missing required keys: " + ", ".join(missing) + ".")
    return ARENA_REPLAY_REQUIRED_KEYS + tuple(k for k in ARENA_REPLAY_OPTIONAL_KEYS if k in batch)


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
    "ARENA_REPLAY_OPTIONAL_KEYS",
    "ARENA_REPLAY_REQUIRED_KEYS",
    "ARENA_STEP_KEYS",
    "batch_size_from_arena_batch",
    "coerce_arena_targets",
    "coerce_arena_task_input",
    "infer_arena_replay_batch_keys",
]
