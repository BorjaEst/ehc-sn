"""Routebind task-overview figure selector.

Reads ``routebind/*`` meta keys from a persisted eval artifact and produces
a typed data object for the overview template.

Required meta keys (always checked):
    - ``routebind/cell_type`` ``(S,)`` int32 — 0=WALL, 1=FREE, 2=OBSERVATION
    - ``routebind/observation_id`` ``(S,)`` int32 — observation ID or sentinel
    - ``routebind/start_flag`` ``(S,)`` bool — True at start position
    - ``routebind/goal_flag`` ``(S,)`` bool — True at goal positions
    - ``routebind/target_trajectory`` ``(S,)`` float32 — oracle trajectory field

Optional meta keys (read when present, fallback to array-derived estimates):
    - ``routebind/n_observations`` — corpus-wide observation vocabulary size.
    - ``routebind/canvas_width`` — grid width in cells.
    - ``routebind/canvas_height`` — grid height in cells (must equal width).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.keys import (
    ROUTEBIND_META_KEY_CANVAS_HEIGHT,
    ROUTEBIND_META_KEY_CANVAS_WIDTH,
    ROUTEBIND_META_KEY_CELL_TYPE,
    ROUTEBIND_META_KEY_GOAL_FLAG,
    ROUTEBIND_META_KEY_N_OBSERVATIONS,
    ROUTEBIND_META_KEY_OBSERVATION_ID,
    ROUTEBIND_META_KEY_START_FLAG,
    ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
)
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass
class RoutebindTaskOverviewData:
    """Prepared data for the ``routebind_task_overview`` figure template.

    Attributes:
        cell_type: ``(S,)`` int32 — cell categories.
        observation_id: ``(S,)`` int32 — per-position observation identity.
        start_flag: ``(S,)`` bool — True at the unique start.
        goal_flag: ``(S,)`` bool — True at goal positions.
        target_trajectory: ``(S,)`` float32 — oracle trajectory field.
        grid_width: Grid width in cells (derived from explicit canvas
            metadata when available, else from sqrt of S).
        n_observations: Corpus-wide observation vocabulary size (from
            explicit ``routebind/n_observations`` metadata when available,
            else derived from the max observation ID present in the sample).
    """

    cell_type: NDArray[np.int32]
    observation_id: NDArray[np.int32]
    start_flag: NDArray[np.bool_]
    goal_flag: NDArray[np.bool_]
    target_trajectory: NDArray[np.float32]
    grid_width: int
    n_observations: int


# =============================================================================
def _to_1d(arr: np.ndarray) -> np.ndarray:
    """Squeeze to 1-D for single-sample access."""
    while arr.ndim > 1:
        arr = arr.squeeze(0)
    return arr


def select_routebind_task_overview(
    trace: TraceTree,
    ctx: FigureContext,
) -> RoutebindTaskOverviewData:
    """Extract routebind task overview data from a persisted trace.

    Reads routebind meta keys from *trace* and returns a typed data object.
    The first valid sample index from *ctx* is used (default: 0).
    """
    sample_idx = ctx.sample_idx

    def _meta(key: str) -> np.ndarray:
        arr = np.asarray(trace.get_meta_path(key))
        if arr.ndim >= 2:
            arr = arr[sample_idx]
        return _to_1d(arr)

    cell_type = _meta(ROUTEBIND_META_KEY_CELL_TYPE).astype(np.int32)
    observation_id = _meta(ROUTEBIND_META_KEY_OBSERVATION_ID).astype(np.int32)
    start_flag = _meta(ROUTEBIND_META_KEY_START_FLAG).astype(bool)
    goal_flag = _meta(ROUTEBIND_META_KEY_GOAL_FLAG).astype(bool)
    target_trajectory = _meta(ROUTEBIND_META_KEY_TARGET_TRAJECTORY).astype(
        np.float32
    )

    # Cross-validate array lengths
    _lengths = {
        "cell_type": len(cell_type),
        "observation_id": len(observation_id),
        "start_flag": len(start_flag),
        "goal_flag": len(goal_flag),
        "target_trajectory": len(target_trajectory),
    }
    if len(set(_lengths.values())) != 1:
        raise ValueError(
            f"Routebind meta arrays have mismatched lengths: {_lengths}"
        )

    S = len(cell_type)

    # Grid dimensions: explicit metadata takes precedence
    if trace.has_meta_path(
        ROUTEBIND_META_KEY_CANVAS_WIDTH
    ) and trace.has_meta_path(ROUTEBIND_META_KEY_CANVAS_HEIGHT):
        canvas_w = int(np.asarray(trace.get_meta_path(ROUTEBIND_META_KEY_CANVAS_WIDTH)).flat[0])  # fmt: skip
        canvas_h = int(np.asarray(trace.get_meta_path(ROUTEBIND_META_KEY_CANVAS_HEIGHT)).flat[0])  # fmt: skip
        if canvas_h != canvas_w:
            raise ValueError(
                f"Routebind task overview requires a square grid, "
                f"got canvas_width={canvas_w}, canvas_height={canvas_h}"
            )
        if canvas_w * canvas_h != S:
            raise ValueError(
                f"Routebind canvas dimensions {canvas_w}×{canvas_h} "
                f"give {canvas_w * canvas_h} cells but arrays have length {S}"
            )
        grid_width = canvas_w
    else:
        # Fallback: infer square grid from array length
        grid_width = int(math.sqrt(S))
        if grid_width * grid_width != S:
            raise ValueError(
                f"Routebind meta arrays have length {S} which is not a perfect square. "
                "Attach routebind/canvas_width and routebind/canvas_height metadata "
                "to support non-square grids."
            )

    # Observation vocabulary size: explicit metadata takes precedence
    if trace.has_meta_path(ROUTEBIND_META_KEY_N_OBSERVATIONS):
        n_observations = int(
            np.asarray(
                trace.get_meta_path(ROUTEBIND_META_KEY_N_OBSERVATIONS)
            ).flat[0]
        )
    else:
        # Fallback: derive from max observation ID present
        valid_obs = observation_id[(cell_type == 2) & (observation_id >= 0)]
        n_observations = int(valid_obs.max()) + 1 if len(valid_obs) > 0 else 0

    return RoutebindTaskOverviewData(
        cell_type=cell_type,
        observation_id=observation_id,
        start_flag=start_flag,
        goal_flag=goal_flag,
        target_trajectory=target_trajectory,
        grid_width=grid_width,
        n_observations=n_observations,
    )


__all__ = [
    "RoutebindTaskOverviewData",
    "select_routebind_task_overview",
]
