"""Routebind figure selectors.

Provides semantic sample types and adapters for task and evaluation
figures.  The task-overview selector consumes a ``RoutebindTaskSample``
built from a ``TraceTree`` via ``build_routebind_task_sample``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    ROUTEBIND_META_KEY_TARGET_WAYPOINT,
)
from ehc_sn.traces.trace_tree import TraceTree

# =============================================================================
# Semantic source views
# =============================================================================


@dataclass
class RoutebindTaskSample:
    """One Routebind dataset sample without model involvement."""

    cell_type: NDArray  # (S,) int32
    observation_id: NDArray  # (S,) int32
    start_flag: NDArray  # (S,) bool
    goal_flag: NDArray  # (S,) bool
    target_trajectory: NDArray  # (S,) float32
    target_waypoint: NDArray  # (S,) float32
    grid_width: int = 30
    sample_id: str = ""


@dataclass
class RoutebindEvaluationSample:
    """One Routebind evaluated sample with batch dim already indexed out."""

    task: RoutebindTaskSample
    prediction_trajectory: NDArray  # (T, S,) float32
    prediction_waypoint: NDArray  # (T, S,) float32
    halted: NDArray  # (T,) bool
    steps: NDArray  # (T,)
    halt_step: int | None = None
    truncated: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


def build_routebind_task_sample(
    trace: TraceTree, *, sample_idx: int = 0
) -> RoutebindTaskSample:
    """Build a Routebind task sample from a TraceTree."""
    from ehc_sn.traces.keys import (
        ROUTEBIND_META_KEY_CANVAS_HEIGHT,
        ROUTEBIND_META_KEY_CANVAS_WIDTH,
        ROUTEBIND_META_KEY_CELL_TYPE,
        ROUTEBIND_META_KEY_GOAL_FLAG,
        ROUTEBIND_META_KEY_OBSERVATION_ID,
        ROUTEBIND_META_KEY_START_FLAG,
        ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
        ROUTEBIND_META_KEY_TARGET_WAYPOINT,
    )

    def _sample(key: str) -> np.ndarray:
        arr = np.asarray(trace.get_meta_path(key))
        if arr.ndim >= 2:
            arr = arr[sample_idx]
        while arr.ndim > 1:
            arr = arr.squeeze(0)
        return arr

    width = (
        int(_sample(ROUTEBIND_META_KEY_CANVAS_WIDTH))
        if trace.has(ROUTEBIND_META_KEY_CANVAS_WIDTH)
        else 30
    )

    return RoutebindTaskSample(
        cell_type=_sample(ROUTEBIND_META_KEY_CELL_TYPE),
        observation_id=_sample(ROUTEBIND_META_KEY_OBSERVATION_ID),
        start_flag=_sample(ROUTEBIND_META_KEY_START_FLAG),
        goal_flag=_sample(ROUTEBIND_META_KEY_GOAL_FLAG),
        target_trajectory=_sample(ROUTEBIND_META_KEY_TARGET_TRAJECTORY),
        target_waypoint=_sample(ROUTEBIND_META_KEY_TARGET_WAYPOINT),
        grid_width=width,
    )


# =============================================================================
@dataclass
class RoutebindTaskOverviewData:
    """Prepared data for the ``task_overview_routebind`` figure template.

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
        waypoint_obs_sequence: Ordered observation IDs along the oracle
            route.
    """

    cell_type: NDArray[np.int32]
    observation_id: NDArray[np.int32]
    start_flag: NDArray[np.bool_]
    goal_flag: NDArray[np.bool_]
    target_trajectory: NDArray[np.float32]
    grid_width: int
    n_observations: int
    waypoint_obs_sequence: list[int]


# =============================================================================
def _to_1d(arr: np.ndarray) -> np.ndarray:
    """Squeeze to 1-D for single-sample access."""
    while arr.ndim > 1:
        arr = arr.squeeze(0)
    return arr


def select_task_overview_routebind(
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

    # Waypoint observation sequence (always present in routebind traces)
    target_waypoint = _meta(ROUTEBIND_META_KEY_TARGET_WAYPOINT).astype(
        np.float32
    )
    if len(target_waypoint) != S:
        raise ValueError(
            f"target_waypoint length {len(target_waypoint)} != {S}"
        )

    # Read waypoints directly from waypoint support channels.
    # When the trace carries `waypoint_support` and `waypoint_semantic_depth`
    # meta keys, use those for ordering.  Otherwise fall back to the
    # greedy-route-based extraction for backward compat with v1 traces.
    if trace.has_meta_path("routebind/waypoint_support"):
        wp_sup = np.asarray(trace.get_meta_path("routebind/waypoint_support"))
        if wp_sup.ndim >= 2:
            wp_sup = wp_sup[sample_idx]
        wp_sup = _to_1d(wp_sup)
        wsd_key = "routebind/waypoint_semantic_depth"
        if trace.has_meta_path(wsd_key):
            wsd = np.asarray(trace.get_meta_path(wsd_key))
            if wsd.ndim >= 2:
                wsd = wsd[sample_idx]
            wsd = _to_1d(wsd)
        else:
            wsd = np.full(S, -1, dtype=np.int16)
        raw = _extract_waypoint_events_from_support(wp_sup, wsd, observation_id)
        waypoint_obs_sequence = [obs for _, obs, _ in raw]
    else:
        route = _extract_oracle_route(
            target_trajectory, start_flag, cell_type, grid_width
        )
        waypoint_obs_sequence = _extract_waypoint_obs_sequence(
            target_waypoint, route, observation_id
        )

    return RoutebindTaskOverviewData(
        cell_type=cell_type,
        observation_id=observation_id,
        start_flag=start_flag,
        goal_flag=goal_flag,
        target_trajectory=target_trajectory,
        grid_width=grid_width,
        n_observations=n_observations,
        waypoint_obs_sequence=waypoint_obs_sequence,
    )


# =============================================================================
# Inline decoding helpers (Layer-2 compatible — no import from tasks/)
# =============================================================================


def _extract_oracle_route(
    trajectory_field: np.ndarray,
    start_flag: np.ndarray,
    cell_type: np.ndarray,
    grid_width: int,
    max_length: int = 150,
) -> list[int]:
    """Greedy max-activation route extraction from the oracle trajectory field.

    Semantics match ``tasks.routebind.decoding.extract_route_from_trajectory_field``.
    Direction order: UP (0), RIGHT (1), DOWN (2), LEFT (3).
    """
    n_slots = len(trajectory_field)
    height = n_slots // grid_width
    start_idx = int(np.argmax(start_flag))

    if trajectory_field[start_idx] <= 0:
        return []

    route: list[int] = [start_idx]
    visited: set[int] = {start_idx}
    dirs: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for _ in range(min(max_length, n_slots)):
        pos = route[-1]
        r, c = divmod(pos, grid_width)
        best_nbr = -1
        best_val = -1.0
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < grid_width:
                npos = nr * grid_width + nc
                if npos in visited:
                    continue
                if int(cell_type[npos]) == 0:
                    continue  # WALL
                val = float(trajectory_field[npos])
                if val > best_val:
                    best_val = val
                    best_nbr = npos
        if best_nbr < 0 or best_val <= 0:
            break
        route.append(best_nbr)
        visited.add(best_nbr)

    return route


def _extract_waypoint_events_from_support(
    waypoint_support: np.ndarray,
    waypoint_semantic_depth: np.ndarray,
    observation_id: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Extract waypoint events from support arrays, ordered by semantic depth.

    Semantics match ``tasks.routebind.decoding.extract_waypoint_events_from_support``
    but with no dependency on that module (Layer-2 compatible).
    """
    events: list[tuple[int, int, int]] = []
    for p in range(len(waypoint_support)):
        if waypoint_support[p]:
            sd = int(waypoint_semantic_depth[p])
            if sd >= 0:
                events.append((p, int(observation_id[p]), sd))
    events.sort(key=lambda x: x[2])
    return events


def _extract_waypoint_obs_sequence(
    waypoint_field: np.ndarray,
    route: list[int],
    observation_id: np.ndarray,
    threshold: float = 0.01,
) -> list[int]:
    """Extract observation IDs along *route* where waypoint activation exceeds threshold.

    Semantics match ``tasks.routebind.decoding.extract_waypoint_sequence``.
    Returns only the observation IDs in route order.
    """
    obs_seq: list[int] = []
    for pos in route:
        if float(waypoint_field[pos]) > threshold:
            obs_seq.append(int(observation_id[pos]))
    return obs_seq


__all__ = [
    "RoutebindTaskOverviewData",
    "select_task_overview_routebind",
]
