"""Goaltrace producer-side trace supplement API.

This module owns:

- :class:`GoaltraceEvaluationSourceContext` — typed, frozen provider context for
  Goaltrace evaluation cases.
- :class:`GoaltraceTraceSupplements` — canonical supplement content ready to attach.
- :func:`build_goaltrace_trace_supplements` — constructs supplements from source context.
- :func:`apply_goaltrace_trace_supplements` — attaches supplement data to a
  :class:`~ehp_sn.traces.trace_tree.TraceTree`.

Supplements are populated from the evaluation batch (not from disk).
Topology tensors (successor_indices, successor_mask) are captured at
trace-construction time when the evaluated sample is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ehc_sn.traces.keys import (
    GOALTRACE_META_KEY_CURRENT_FLAG,
    GOALTRACE_META_KEY_GOAL_FLAG,
    GOALTRACE_META_KEY_NODE_MASK,
    GOALTRACE_META_KEY_OBSERVATION_ID,
    GOALTRACE_META_KEY_SUCCESSOR_INDICES,
    GOALTRACE_META_KEY_SUCCESSOR_MASK,
    GOALTRACE_META_KEY_TARGET_FIELD,
    GOALTRACE_META_KEY_WEIGHT,
)
from ehc_sn.traces.trace_tree import TraceTree


# =============================================================================
@dataclass(frozen=True)
class GoaltraceEvaluationSourceContext:
    """Typed, frozen provider-side context for a Goaltrace evaluation case batch.

    Attributes:
        dataset_path: Absolute path to the processed Goaltrace task corpus root.
        split: Dataset split the samples belong to (e.g. ``"val"``).
        sample_ids: Ordered sample ids in this batch.
    """

    dataset_path: Path
    split: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError(
                "GoaltraceEvaluationSourceContext.sample_ids must not be empty"
            )


# =============================================================================
@dataclass(frozen=True)
class GoaltraceTraceSupplements:
    """Canonical supplement content for Goaltrace traces.

    Populated from the evaluation batch at trace-construction time.
    No tensors are read from disk during figure rendering.
    """

    observation_id: NDArray
    weight: NDArray
    current_flag: NDArray
    goal_flag: NDArray
    node_mask: NDArray
    target_field: NDArray
    successor_indices: NDArray
    successor_mask: NDArray


# =============================================================================
def build_goaltrace_trace_supplements(
    source_context: GoaltraceEvaluationSourceContext,
    trace_length: int,
) -> GoaltraceTraceSupplements:
    """Build Goaltrace trace supplements from a typed source context.

    Args:
        source_context: Typed Goaltrace evaluation source context.
        trace_length: Number of time steps to cover.

    Returns:
        :class:`GoaltraceTraceSupplements` — must be populated by the caller
        after the batch is available by setting the returned dataclass fields.
    """
    # NOTE: The actual tensor values are filled by the Lightning trace builder
    # after the batch is available.  This function returns a stub that
    # apply_goaltrace_trace_supplements expects.
    _ = source_context, trace_length
    return GoaltraceTraceSupplements(
        observation_id=np.array([], dtype=np.int32),
        weight=np.array([], dtype=np.float32),
        current_flag=np.array([], dtype=bool),
        goal_flag=np.array([], dtype=bool),
        node_mask=np.array([], dtype=bool),
        target_field=np.array([], dtype=np.float32),
        successor_indices=np.array([], dtype=np.int32),
        successor_mask=np.array([], dtype=bool),
    )


# =============================================================================
def apply_goaltrace_trace_supplements(
    trace: TraceTree,
    supplements: GoaltraceTraceSupplements,
) -> None:
    """Attach Goaltrace supplement content to *trace* in-place.

    All supplement tensors are attached as sample metadata
    (``TraceTree.attached_meta``) under the canonical key paths defined
    in ``traces/keys.py``.

    Args:
        trace: The :class:`~ehp_sn.traces.trace_tree.TraceTree` to modify in-place.
        supplements: The supplement content to attach.
    """
    trace.attached_meta[GOALTRACE_META_KEY_OBSERVATION_ID] = (
        supplements.observation_id
    )
    trace.attached_meta[GOALTRACE_META_KEY_WEIGHT] = supplements.weight
    trace.attached_meta[GOALTRACE_META_KEY_CURRENT_FLAG] = (
        supplements.current_flag
    )
    trace.attached_meta[GOALTRACE_META_KEY_GOAL_FLAG] = supplements.goal_flag
    trace.attached_meta[GOALTRACE_META_KEY_NODE_MASK] = supplements.node_mask
    trace.attached_meta[GOALTRACE_META_KEY_TARGET_FIELD] = (
        supplements.target_field
    )
    trace.attached_meta[GOALTRACE_META_KEY_SUCCESSOR_INDICES] = (
        supplements.successor_indices
    )
    trace.attached_meta[GOALTRACE_META_KEY_SUCCESSOR_MASK] = (
        supplements.successor_mask
    )


# =============================================================================
__all__ = [
    "GoaltraceEvaluationSourceContext",
    "GoaltraceTraceSupplements",
    "build_goaltrace_trace_supplements",
    "apply_goaltrace_trace_supplements",
]
