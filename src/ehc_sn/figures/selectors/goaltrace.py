"""Goaltrace figure selectors.

Produces geometry, scalar field data, and diagnostics from goaltrace trace
metadata and dense prediction traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.keys import (
    GOALTRACE_META_KEY_CURRENT_FLAG,
    GOALTRACE_META_KEY_GOAL_FLAG,
    GOALTRACE_META_KEY_NODE_MASK,
    GOALTRACE_META_KEY_OBSERVATION_ID,
    GOALTRACE_META_KEY_SUCCESSOR_INDICES,
    GOALTRACE_META_KEY_SUCCESSOR_MASK,
    GOALTRACE_META_KEY_TARGET_FIELD,
    GOALTRACE_META_KEY_WEIGHT,
    GOALTRACE_TRACE_KEY_FIRING_FIELD,
)
from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.utils.graph import compute_layered_dag_positions


# =============================================================================
@dataclass(frozen=True)
class GoaltraceGraphGeometry:
    """Precomputed graph layout ready for rendering.

    All indices are in compact (non-padded) rendered-node space.
    Edges are in compact index space.  Labels are observation IDs.
    """

    observation_ids: NDArray  # (N,) int — observation label for each node
    edges: tuple[tuple[int, int], ...]  # directed edges in compact space
    positions: NDArray  # (N, 2) float64 — deterministic layered layout
    current_node_index: int  # compact index of the current node
    goal_node_index: int  # compact index of the goal node


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTaskOverviewFigureData:
    """Prepared data for the goaltrace task-overview figure."""

    geometry: GoaltraceGraphGeometry
    weight: NDArray  # (N,) float32 — oracle input weights
    target_field: NDArray  # (N,) float32 — oracle target firing field
    n_valid: int  # number of valid (non-padded) nodes


# =============================================================================
@dataclass(frozen=True)
class GoaltracePredictionDiagnostics:
    """Sample-level diagnostic text values for the prediction figure."""

    selected_trace_step: int
    selection_reason: Literal["halt", "final"]
    field_mse: float
    current_observation_id: int
    goal_observation_id: int
    current_target: float
    current_prediction: float
    goal_target: float
    goal_prediction: float
    largest_error_observation_id: int
    largest_absolute_error: float


# =============================================================================
@dataclass(frozen=True)
class GoaltracePredictionExampleFigureData:
    """Prepared data for the goaltrace prediction-example figure."""

    geometry: GoaltraceGraphGeometry
    weight: NDArray  # (N,) float32 — oracle input weights
    target_field: NDArray  # (N,) float32 — oracle target firing field
    predicted_field: NDArray  # (N,) float32 — model predicted field
    diagnostics: GoaltracePredictionDiagnostics


# =============================================================================
# Internal helpers
# =============================================================================


def _to_np(value: object) -> np.ndarray:
    """Return a numpy array, moving from CPU if needed."""
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _require_meta(trace: TraceTree, key: str) -> np.ndarray:
    """Fetch a required meta key or raise."""
    return np.asarray(_to_np(trace.get_meta_path(key)))


def _build_geometry(
    observation_id: NDArray,
    node_mask: NDArray,
    successor_indices: NDArray,
    successor_mask: NDArray,
    current_flag: NDArray,
    goal_flag: NDArray,
) -> GoaltraceGraphGeometry:
    """Build ``GoaltraceGraphGeometry`` from raw padded metadata arrays.

    Strips padded nodes and masked successor entries, then computes
    deterministic layered positions on the valid induced subgraph.
    """
    # Identify valid nodes
    valid_indices = np.where(node_mask)[0]  # padded-space indices
    n_valid = len(valid_indices)
    padded_to_compact = {int(p): c for c, p in enumerate(valid_indices)}

    # Map observation IDs for valid nodes
    obs_ids = observation_id[valid_indices]

    # Find current and goal in compact space
    current_padded = int(np.where(current_flag)[0][0])
    goal_padded = int(np.where(goal_flag)[0][0])
    current_compact = padded_to_compact[current_padded]
    goal_compact = padded_to_compact[goal_padded]

    # Build edges in compact space from successor data
    edges: list[tuple[int, int]] = []
    for c, p in enumerate(valid_indices):
        succ_row = successor_indices[p]
        mask_row = successor_mask[p]
        for k in range(succ_row.shape[0]):
            if mask_row[k]:
                succ_padded = int(succ_row[k])
                if succ_padded in padded_to_compact:
                    edges.append((c, padded_to_compact[succ_padded]))

    # Compute positions on the valid induced subgraph
    positions = compute_layered_dag_positions(
        num_nodes=n_valid, edges=tuple(edges)
    )

    return GoaltraceGraphGeometry(
        observation_ids=obs_ids,
        edges=tuple(edges),
        positions=positions,
        current_node_index=current_compact,
        goal_node_index=goal_compact,
    )


def _select_sample_arrays_1d(
    trace: TraceTree, ctx: FigureContext
) -> dict[str, np.ndarray]:
    """Extract and squeeze per-sample 1D arrays from trace metadata.

    Follows the same ``(B, N) → (N,)`` contract as the routebind selector:
    indexes into the leading batch dimension via ``ctx.sample_idx`` and
    squeezes per-node arrays to 1-D.

    Successor arrays (``successor_indices``, ``successor_mask``) become
    ``(N, K)`` — squeezed only beyond the second dimension.
    """
    sample_idx = ctx.sample_idx

    def _sample_arr(key: str) -> np.ndarray:
        arr = np.asarray(_require_meta(trace, key))
        if arr.ndim >= 2:
            arr = arr[sample_idx]
        while arr.ndim > 1:
            arr = arr.squeeze(0)
        return arr

    def _sample_arr_2d(key: str) -> np.ndarray:
        """Extract a 2-D array (e.g. successor_indices: (B, N, K) → (N, K))."""
        arr = np.asarray(_require_meta(trace, key))
        if arr.ndim >= 3:
            arr = arr[sample_idx]
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr

    return {
        "observation_id": _sample_arr(GOALTRACE_META_KEY_OBSERVATION_ID),
        "weight": _sample_arr(GOALTRACE_META_KEY_WEIGHT),
        "node_mask": _sample_arr(GOALTRACE_META_KEY_NODE_MASK),
        "current_flag": _sample_arr(GOALTRACE_META_KEY_CURRENT_FLAG),
        "goal_flag": _sample_arr(GOALTRACE_META_KEY_GOAL_FLAG),
        "target_field": _sample_arr(GOALTRACE_META_KEY_TARGET_FIELD),
        "successor_indices": _sample_arr_2d(
            GOALTRACE_META_KEY_SUCCESSOR_INDICES
        ),
        "successor_mask": _sample_arr_2d(GOALTRACE_META_KEY_SUCCESSOR_MASK),
    }


def _compact_arrays(
    arrays: dict[str, np.ndarray],
    geometry: GoaltraceGraphGeometry,
) -> dict[str, np.ndarray]:
    """Slice per-node arrays from padded to compact valid-node space."""
    valid_indices = np.where(arrays["node_mask"])[0]
    compact: dict[str, np.ndarray] = {}
    for key in ("weight", "target_field"):
        compact[key] = arrays[key][valid_indices]
    return compact


# =============================================================================
# Public selectors
# =============================================================================


def select_goaltrace_task_overview(
    trace: TraceTree,
    ctx: FigureContext,
) -> GoaltraceTaskOverviewFigureData:
    """Extract task-overview data from a goaltrace trace (meta keys only).

    Args:
        trace: Trace tree with all goaltrace meta keys populated.
        ctx: Figure context (``sample_idx`` selects the batch element).

    Returns:
        Prepared figure data with geometry and scalar arrays.

    Raises:
        ValueError: If any required meta key is absent.
    """
    arrays = _select_sample_arrays_1d(trace, ctx)
    geometry = _build_geometry(
        arrays["observation_id"],
        arrays["node_mask"],
        arrays["successor_indices"],
        arrays["successor_mask"],
        arrays["current_flag"],
        arrays["goal_flag"],
    )
    compact = _compact_arrays(arrays, geometry)
    n_valid = len(geometry.observation_ids)
    return GoaltraceTaskOverviewFigureData(
        geometry=geometry,
        weight=compact["weight"],
        target_field=compact["target_field"],
        n_valid=n_valid,
    )


# =============================================================================
def select_goaltrace_prediction_example(
    trace: TraceTree,
    ctx: FigureContext,
) -> GoaltracePredictionExampleFigureData:
    """Extract prediction-example data from a goaltrace evaluation trace.

    Reads both meta keys and the dense ``goaltrace/firing_field`` trace.
    Normalises the predicted field to ``[trace_step, batch, node]`` and
    selects the halting or final step.

    Args:
        trace: Trace tree with goaltrace meta keys and ``goaltrace/firing_field``.
        ctx: Figure context (``sample_idx`` selects the batch element).

    Returns:
        Prepared figure data with geometry, scalar arrays, and diagnostics.

    Raises:
        ValueError: If ``goaltrace/firing_field`` or any required meta key
            is absent.
    """
    arrays = _select_sample_arrays_1d(trace, ctx)

    geometry = _build_geometry(
        arrays["observation_id"],
        arrays["node_mask"],
        arrays["successor_indices"],
        arrays["successor_mask"],
        arrays["current_flag"],
        arrays["goal_flag"],
    )
    compact = _compact_arrays(arrays, geometry)

    # Read dense prediction trace
    firing_field_raw = np.asarray(
        _to_np(trace.get(GOALTRACE_TRACE_KEY_FIRING_FIELD))
    )
    if firing_field_raw.ndim < 2:
        raise ValueError(
            f"{GOALTRACE_TRACE_KEY_FIRING_FIELD} must have at least 2 dims "
            f"(got {firing_field_raw.ndim})."
        )

    # Normalise to [trace_step, batch, node]
    while firing_field_raw.ndim > 3:
        firing_field_raw = firing_field_raw.squeeze(0)

    sample_idx = ctx.sample_idx
    if firing_field_raw.ndim == 2:
        # Single step: shape (B, N)
        firing_field_2d = firing_field_raw
        selected_step = 0
        selection_reason: Literal["halt", "final"] = "final"
    else:
        # Multi-step: shape (T, B, N)
        T, B, N = firing_field_raw.shape
        if sample_idx >= B:
            sample_idx = 0
        # Try halting signal from trace
        halt_signal = None
        if trace.has("act/halted"):
            halt_arr = np.asarray(_to_np(trace.get("act/halted")))
            if halt_arr.ndim >= 2:
                while halt_arr.ndim > 2:
                    halt_arr = halt_arr.squeeze(0)
                if halt_arr.ndim == 2:
                    halted_t = halt_arr[:, sample_idx]
                    halt_indices = np.where(halted_t)[0]
                    if len(halt_indices) > 0:
                        selected_step = int(halt_indices[0])
                        selection_reason = "halt"
                    else:
                        selected_step = T - 1
                        selection_reason = "final"
                else:
                    selected_step = T - 1
                    selection_reason = "final"
            else:
                selected_step = T - 1
                selection_reason = "final"
        else:
            selected_step = T - 1
            selection_reason = "final"
        firing_field_2d = firing_field_raw[selected_step]

    # Extract per-sample prediction
    if firing_field_2d.ndim == 2:
        if sample_idx >= firing_field_2d.shape[0]:
            sample_idx = 0
        pred_values = np.asarray(firing_field_2d[sample_idx])
    else:
        pred_values = np.asarray(firing_field_2d)

    # Compact to valid-node space
    valid_indices = np.where(arrays["node_mask"])[0]
    pred_compact = pred_values[valid_indices]

    # Compute diagnostics
    target_compact = compact["target_field"]
    valid_mask_compact = np.ones(len(valid_indices), dtype=bool)
    diff = pred_compact - target_compact
    sq_error = diff**2
    field_mse = float(np.mean(sq_error[valid_mask_compact]))

    current_compact = geometry.current_node_index
    goal_compact = geometry.goal_node_index

    diagnostics = GoaltracePredictionDiagnostics(
        selected_trace_step=selected_step,
        selection_reason=selection_reason,
        field_mse=field_mse,
        current_observation_id=int(geometry.observation_ids[current_compact]),
        goal_observation_id=int(geometry.observation_ids[goal_compact]),
        current_target=float(target_compact[current_compact]),
        current_prediction=float(pred_compact[current_compact]),
        goal_target=float(target_compact[goal_compact]),
        goal_prediction=float(pred_compact[goal_compact]),
        largest_error_observation_id=int(
            geometry.observation_ids[np.argmax(sq_error)]
        ),
        largest_absolute_error=float(np.max(np.abs(diff))),
    )

    return GoaltracePredictionExampleFigureData(
        geometry=geometry,
        weight=compact["weight"],
        target_field=target_compact,
        predicted_field=pred_compact,
        diagnostics=diagnostics,
    )


# =============================================================================
__all__ = [
    "GoaltraceGraphGeometry",
    "GoaltracePredictionDiagnostics",
    "GoaltracePredictionExampleFigureData",
    "GoaltraceTaskOverviewFigureData",
    "select_goaltrace_prediction_example",
    "select_goaltrace_task_overview",
]
