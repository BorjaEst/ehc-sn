"""Routebind producer-side trace supplement API.

This module owns:

- :class:`RoutebindEvaluationSourceContext` — typed, frozen provider context for
  routebind evaluation cases.
- :class:`RoutebindTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_routebind_trace_supplements` — constructs supplements from source context.
- :func:`apply_routebind_trace_supplements` — attaches supplement data to a
  :class:`~ehc_sn.traces.trace_tree.TraceTree`.

Supplements are populated from the evaluation batch (not from disk).
Spatial topology tensors (cell_type, observation_id, start/goal flags) are
captured at trace-construction time when the evaluated sample is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ehc_sn.traces.keys import (
    ROUTEBIND_META_KEY_CANVAS_HEIGHT,
    ROUTEBIND_META_KEY_CANVAS_WIDTH,
    ROUTEBIND_META_KEY_CELL_MASK,
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
@dataclass(frozen=True)
class RoutebindEvaluationSourceContext:
    """Typed, frozen provider-side context for a routebind evaluation case batch.

    Attributes:
        dataset_path: Absolute path to the processed routebind task corpus root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError(
                "RoutebindEvaluationSourceContext.sample_ids must not be empty"
            )


# =============================================================================
@dataclass(frozen=True)
class RoutebindTraceSupplements:
    """Canonical supplement content for routebind traces.

    Populated from the evaluation batch at trace-construction time.
    No tensors are read from disk during figure rendering.
    """

    cell_type: NDArray
    observation_id: NDArray
    start_flag: NDArray
    goal_flag: NDArray
    target_trajectory: NDArray
    target_waypoint: NDArray


# =============================================================================
def build_routebind_trace_supplements(
    context: RoutebindEvaluationSourceContext,
    batch: dict[str, np.ndarray],
) -> RoutebindTraceSupplements:
    """Build trace supplements from an evaluation batch.

    Args:
        context: Provider context with dataset info, split, and sample ids.
        batch: Dict of channel arrays for one evaluation batch sample.

    Returns:
        Filled :class:`RoutebindTraceSupplements` instance.

    Raises:
        KeyError: When any required channel is absent from *batch*.
    """
    _ = context  # reserved for future use (e.g. dataset metadata lookup)
    required = (
        "cell_type",
        "observation_id",
        "start_flag",
        "goal_flag",
        "target_trajectory",
        "target_waypoint",
    )
    for key in required:
        if key not in batch:
            raise KeyError(
                f"build_routebind_trace_supplements: '{key}' missing from batch."
            )
    return RoutebindTraceSupplements(
        cell_type=np.asarray(batch["cell_type"]),
        observation_id=np.asarray(batch["observation_id"]),
        start_flag=np.asarray(batch["start_flag"]),
        goal_flag=np.asarray(batch["goal_flag"]),
        target_trajectory=np.asarray(batch["target_trajectory"]),
        target_waypoint=np.asarray(batch["target_waypoint"]),
    )


# =============================================================================
def apply_routebind_trace_supplements(
    trace: TraceTree,
    supplements: RoutebindTraceSupplements,
    *,
    n_observations: int | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> None:
    """Attach routebind supplement data to a trace tree.

    Args:
        trace: Mutable trace tree to attach data to.
        supplements: Pre-built supplement content.
        n_observations: Corpus-wide observation vocabulary size.
            When provided, attached as ``routebind/n_observations``
            so figure selectors can use the authoritative count instead
            of inferring from array data.
        canvas_width: Grid width in cells.
        canvas_height: Grid height in cells.
    """
    trace.attached_meta[ROUTEBIND_META_KEY_CELL_TYPE] = supplements.cell_type
    trace.attached_meta[ROUTEBIND_META_KEY_OBSERVATION_ID] = (
        supplements.observation_id
    )
    trace.attached_meta[ROUTEBIND_META_KEY_START_FLAG] = supplements.start_flag
    trace.attached_meta[ROUTEBIND_META_KEY_GOAL_FLAG] = supplements.goal_flag
    trace.attached_meta[ROUTEBIND_META_KEY_TARGET_TRAJECTORY] = (
        supplements.target_trajectory
    )
    trace.attached_meta[ROUTEBIND_META_KEY_TARGET_WAYPOINT] = (
        supplements.target_waypoint
    )
    if n_observations is not None:
        trace.attached_meta[ROUTEBIND_META_KEY_N_OBSERVATIONS] = n_observations
    if canvas_width is not None:
        trace.attached_meta[ROUTEBIND_META_KEY_CANVAS_WIDTH] = canvas_width
    if canvas_height is not None:
        trace.attached_meta[ROUTEBIND_META_KEY_CANVAS_HEIGHT] = canvas_height


# =============================================================================
__all__ = [
    "RoutebindEvaluationSourceContext",
    "RoutebindTraceSupplements",
    "build_routebind_trace_supplements",
    "apply_routebind_trace_supplements",
]
