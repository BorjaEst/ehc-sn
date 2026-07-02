"""Goaltrace prediction-reasoning selector.

Produces ``GoaltracePredictionReasoningData`` from a semantic
:class:`~ehp_sn.figures.selectors.goaltrace.GoaltraceEvaluationSample`.

The adapter from ``TraceTree`` to ``GoaltraceEvaluationSample`` is called at
the entry point — this module never touches raw trace keys or tree access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors._reasoning_steps import select_reasoning_snapshots
from ehc_sn.figures.selectors.goaltrace import (
    GoaltraceEvaluationSample,
    GoaltraceGraphGeometry,
    GoaltraceTaskSample,
    _build_geometry,
    build_goaltrace_evaluation_sample,
)


@dataclass(frozen=True)
class GoaltracePredictionReasoningData:
    """Prepared data for ``prediction_reasoning_goaltrace``.

    All arrays are compacted to valid-node space.  The template never
    touches raw traces.
    """

    geometry: GoaltraceGraphGeometry
    weight: NDArray  # (N,) float32 — oracle input weights (for GT context)
    target_field: NDArray  # (N,) float32 — oracle target firing field
    snapshots: tuple[np.ndarray, ...]  # each (N,) float32 — per-step pred
    selected_indices: tuple[int, ...]  # real deliberation step numbers
    halt_step: int | None
    truncated: bool
    field_mses: tuple[float, ...] = field(default_factory=tuple)
    """Per-snapshot masked MSE against target (task metric)."""


# ── Default cap ────────────────────────────────────────────────────────────

_MAX_SNAPSHOTS: int = 16


# ── Geometry helper ────────────────────────────────────────────────────────


def _build_geometry_from_task(
    task: GoaltraceTaskSample,
) -> GoaltraceGraphGeometry:
    """Build graph geometry from a semantic goaltrace task sample."""
    return _build_geometry(
        task.observation_id,
        task.node_mask,
        task.successor_indices,
        task.successor_mask,
        task.current_flag,
        task.goal_flag,
    )


def _compact_field(field: NDArray, node_mask: NDArray) -> NDArray:
    """Compact a full-size field to valid-node space."""
    valid = np.where(node_mask)[0]
    return field[valid]


# ── Public selector ────────────────────────────────────────────────────────


def select_goaltrace_prediction_reasoning(
    trace: object,
    ctx: FigureContext,
    max_snapshots: int = _MAX_SNAPSHOTS,
) -> GoaltracePredictionReasoningData:
    """Extract prediction-reasoning data from a Goaltrace evaluation trace.

    Accepts either a ``TraceTree`` (raw, calls adapter internally) or a
    pre-built ``GoaltraceEvaluationSample``.

    Args:
        trace: Trace tree with goaltrace keys **or** a
            :class:`GoaltraceEvaluationSample`.
        ctx: Figure context (``sample_idx`` selects batch element).
        max_snapshots: Maximum number of snapshot slots.

    Returns:
        Prepared figure data.
    """
    if isinstance(trace, GoaltraceEvaluationSample):
        sample = trace
    else:
        from ehc_sn.traces.trace_tree import TraceTree

        assert isinstance(trace, TraceTree), type(trace)
        sample = build_goaltrace_evaluation_sample(
            trace,
            sample_idx=ctx.sample_idx,
        )

    return _select_from_sample(sample, ctx, max_snapshots=max_snapshots)


def _select_from_sample(
    sample: GoaltraceEvaluationSample,
    ctx: FigureContext,
    *,
    max_snapshots: int = _MAX_SNAPSHOTS,
) -> GoaltracePredictionReasoningData:
    """Build prediction-reasoning data from a semantic evaluation sample."""
    task = sample.task
    geometry = _build_geometry_from_task(task)
    node_mask = task.node_mask
    valid = np.where(node_mask)[0]

    target_compact = _compact_field(task.target_field, node_mask)
    weight_compact = _compact_field(task.weight, node_mask)

    pred = sample.prediction_steps  # (T, N)
    halted = sample.halted
    T = len(halted)
    t_halt = sample.halt_step if sample.halt_step is not None else T - 1
    truncated = sample.truncated

    indices = select_reasoning_snapshots(t_halt, max_snapshots)

    snapshots: list[np.ndarray] = []
    mses: list[float] = []
    for t in indices:
        pred_t = pred[t] if t < len(pred) else np.zeros_like(task.target_field)
        pred_compact = _compact_field(pred_t, node_mask)
        snapshots.append(pred_compact)
        diff = pred_compact - target_compact
        mse = float(np.mean(diff**2))
        mses.append(mse)

    return GoaltracePredictionReasoningData(
        geometry=geometry,
        weight=weight_compact,
        target_field=target_compact,
        snapshots=tuple(snapshots),
        selected_indices=tuple(indices),
        halt_step=t_halt if (t_halt < T and halted[t_halt]) else None,
        truncated=truncated,
        field_mses=tuple(mses),
    )


__all__ = [
    "GoaltracePredictionReasoningData",
    "select_goaltrace_prediction_reasoning",
]
