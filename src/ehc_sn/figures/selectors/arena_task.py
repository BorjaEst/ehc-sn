"""Selector for the ``arena_task_layout`` figure.

Reads ``arena/*`` trace keys from a persisted eval artifact and produces a
typed data object for the overview template.  The data includes the
environment layout (wall mask, observation map), the agent trajectory, and
the revisit mask.

Required trace keys:
    - ``arena/wall_mask`` ``(H, W)`` bool — passable cells (True) vs walls (False)
    - ``arena/observation_ids`` ``(H, W)`` int — observation ID per cell (0 = wall)
    - ``arena/trajectory_locations`` ``(T,)`` int — visited location indices
    - ``arena/revisit_mask`` ``(T,)`` bool — True at revisit steps

Optional trace keys:
    - ``arena/actions`` ``(T-1,)`` int — action taken at each transition
    - ``arena/valid_mask`` ``(H, W)`` bool — valid (reachable) cell mask
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures._contracts import AnyWorld
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

# =============================================================================
# Canonical trace key constants
# =============================================================================

ARENA_TRACE_KEY_WALL_MASK = "arena/wall_mask"
ARENA_TRACE_KEY_OBSERVATION_IDS = "arena/observation_ids"
ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS = "arena/trajectory_locations"
ARENA_TRACE_KEY_REVISIT_MASK = "arena/revisit_mask"
TRACE_KEY_ACTIONS = "arena/actions"
TRACE_KEY_VALID_MASK = "arena/valid_mask"

_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        ARENA_TRACE_KEY_WALL_MASK,
        ARENA_TRACE_KEY_OBSERVATION_IDS,
        ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS,
        ARENA_TRACE_KEY_REVISIT_MASK,
    }
)


# =============================================================================
@dataclass
class ArenaTaskOverviewData:
    """Prepared data for the ``arena_task_layout`` figure template.

    Attributes:
        wall_mask: ``(H, W)`` bool — True for passable cells.
        observation_ids: ``(H, W)`` int32 — observation ID per location.
        trajectory_locations: ``(T,)`` int32 — sequence of visited location ids.
        revisit_mask: ``(T,)`` bool — True at revisit steps.
        actions: ``(T-1,)`` int8 or None — action at each transition.
        valid_mask: ``(H, W)`` bool or None — reachable cell mask.
        world: Reconstructed ``AnyWorld`` for figure plotting primitives.
    """

    wall_mask: NDArray
    observation_ids: NDArray
    trajectory_locations: NDArray
    revisit_mask: NDArray
    actions: NDArray | None = None
    valid_mask: NDArray | None = None
    world: AnyWorld | None = None


# =============================================================================
def _build_world_from_mask_and_obs(
    wall_mask: NDArray,
    observation_ids: NDArray,
    valid_mask: NDArray | None = None,
) -> dict:
    """Build a minimal world dict from wall_mask and observation_ids.

    The constructed world has one location per grid cell, ordered
    row-major.  Wall cells get an invalid marker (NaN value in maps).
    The world conforms to the ``WorldLike`` protocol so it can be passed
    to ``plot_map()`` and ``plot_time_colored_trajectory()``.
    """
    H, W = wall_mask.shape
    locations: list[dict] = []

    for row in range(H):
        for col in range(W):
            loc: dict = {
                "o": float(col),
                "y": float(row),
            }
            if valid_mask is not None:
                loc["valid"] = bool(valid_mask[row, col])
            else:
                loc["valid"] = bool(wall_mask[row, col])
            loc["shiny"] = False
            loc["actions"] = []
            locations.append(loc)

    return {
        "locations": locations,
        "n_locations": H * W,
        "n_actions": 4,
        "spatial_geometry": "grid2d",
    }


# =============================================================================
def select_arena_task_overview(
    trace: TraceTree,
    ctx: FigureContext,  # noqa: ARG001
) -> ArenaTaskOverviewData:
    """Extract Arena task overview data from a persisted eval trace.

    Args:
        trace: Loaded ``TraceTree`` containing ``arena/*`` keys.
        ctx: Figure context (unused — the overview is static).

    Returns:
        :class:`ArenaTaskOverviewData` with all required fields populated.

    Raises:
        KeyError: If any required ``arena/*`` trace key is missing.
    """
    # Validate required keys.
    for key in _REQUIRED_KEYS:
        if key not in trace.path_to_index:
            raise KeyError(
                f"Arena task overview requires trace key {key!r}; "
                f"the eval artifact does not contain it. "
                f"Available keys: {sorted(trace.path_strs)}"
            )

    wall_mask = np.asarray(trace.get(ARENA_TRACE_KEY_WALL_MASK))
    observation_ids = np.asarray(trace.get(ARENA_TRACE_KEY_OBSERVATION_IDS))
    trajectory_locations = np.asarray(
        trace.get(ARENA_TRACE_KEY_TRAJECTORY_LOCATIONS)
    )
    revisit_mask = np.asarray(trace.get(ARENA_TRACE_KEY_REVISIT_MASK))

    valid_mask: NDArray | None = None
    if TRACE_KEY_VALID_MASK in trace.path_to_index:
        valid_mask = np.asarray(trace.get(TRACE_KEY_VALID_MASK))

    actions: NDArray | None = None
    if TRACE_KEY_ACTIONS in trace.path_to_index:
        actions = np.asarray(trace.get(TRACE_KEY_ACTIONS))

    world = _build_world_from_mask_and_obs(
        wall_mask,
        observation_ids,
        valid_mask=valid_mask,
    )

    return ArenaTaskOverviewData(
        wall_mask=wall_mask,
        observation_ids=observation_ids,
        trajectory_locations=trajectory_locations,
        revisit_mask=revisit_mask,
        actions=actions,
        valid_mask=valid_mask,
        world=world,
    )


# =============================================================================
__all__ = ["ArenaTaskOverviewData", "select_arena_task_overview"]
