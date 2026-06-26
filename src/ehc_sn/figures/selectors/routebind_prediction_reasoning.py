"""Routebind prediction-reasoning selector.

Produces ``RoutebindPredictionReasoningData`` from trace keys for the
``prediction_reasoning_routebind`` figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors._reasoning_steps import select_reasoning_snapshots
from ehc_sn.figures.selectors.routebind import (
    RoutebindTaskSample,
    build_routebind_task_sample,
)
from ehc_sn.figures.utils.grids import first_halt_index
from ehc_sn.traces.keys import (
    ROUTEBIND_META_KEY_CANVAS_HEIGHT,
    ROUTEBIND_META_KEY_CANVAS_WIDTH,
    ROUTEBIND_META_KEY_TARGET_TRAJECTORY,
)
from ehc_sn.traces.trace_tree import TraceTree

# Trace key for the per-step trajectory field prediction.
_ROUTEBIND_TRACE_TRAJECTORY_FIELD = "routebind/trajectory_field"


@dataclass(frozen=True)
class RoutebindPredictionReasoningData:
    """Prepared data for ``prediction_reasoning_routebind``.

    All arrays are already decoded — the template never touches raw traces.
    """

    task_sample: RoutebindTaskSample
    """Task metadata (cell_type, observation_id, start/goal, targets)."""
    target_trajectory: NDArray  # (S,) float32 — oracle target field (1-D)
    snapshots: tuple[np.ndarray, ...]  # each (S,) float32 — per-step pred
    selected_indices: tuple[int, ...]  # real deliberation step numbers
    halt_step: int | None
    truncated: bool
    field_mses: tuple[float, ...] = field(default_factory=tuple)
    """Per-snapshot MSE against target trajectory field (task metric)."""


# ── Default cap ────────────────────────────────────────────────────────────

_MAX_SNAPSHOTS: int = 16


# ── Public selector ────────────────────────────────────────────────────────


def select_routebind_prediction_reasoning(
    trace: TraceTree,
    ctx: FigureContext,
    max_snapshots: int = _MAX_SNAPSHOTS,
) -> RoutebindPredictionReasoningData:
    """Extract prediction-reasoning data from a Routebind evaluation trace.

    Args:
        trace: Trace tree with routebind keys (meta + ``routebind/trajectory_field``).
        ctx: Figure context (``sample_idx`` selects batch element).
        max_snapshots: Maximum number of snapshot slots.

    Returns:
        Prepared figure data.

    Raises:
        ValueError: If any required key is absent or has unexpected shape.
    """
    task_sample = build_routebind_task_sample(trace, sample_idx=ctx.sample_idx)
    target_trajectory = np.asarray(
        _to_cpu(trace.get_meta_path(ROUTEBIND_META_KEY_TARGET_TRAJECTORY))
    )
    # Index into batch dim if needed
    if target_trajectory.ndim >= 2:
        sample_idx = min(ctx.sample_idx, target_trajectory.shape[0] - 1)
        target_trajectory = _to_cpu(target_trajectory[sample_idx])

    # Read the per-step trajectory field trace
    pred_field = np.asarray(
        _to_cpu(trace.get(_ROUTEBIND_TRACE_TRAJECTORY_FIELD))
    )
    # Normalise to (T, B, S) or (T, S)
    while pred_field.ndim > 3:
        pred_field = pred_field.squeeze(0)

    halted = np.asarray(_to_cpu(trace.get("act/halted")))

    batch_size = halted.shape[1] if halted.ndim == 2 else 1
    sample_idx = ctx.sample_idx if ctx.sample_idx < batch_size else 0

    T = halted.shape[0]
    t_halt = first_halt_index(halted[:, sample_idx])
    truncated = bool(t_halt == T - 1 and not halted[t_halt, sample_idx])

    indices = select_reasoning_snapshots(t_halt, max_snapshots)

    # Extract per-sample, per-step predictions
    if pred_field.ndim == 2:
        # (T, S)
        preds_by_step = pred_field
    else:
        # (T, B, S)
        preds_by_step = pred_field[:, sample_idx]

    snapshots: list[np.ndarray] = []
    mses: list[float] = []
    for t in indices:
        pred = np.asarray(preds_by_step[t])
        snapshots.append(pred)
        diff = pred - target_trajectory
        mse = float(np.mean(diff**2))
        mses.append(mse)

    return RoutebindPredictionReasoningData(
        task_sample=task_sample,
        target_trajectory=target_trajectory,
        snapshots=tuple(snapshots),
        selected_indices=tuple(indices),
        halt_step=t_halt if halted[t_halt, sample_idx] else None,
        truncated=truncated,
        field_mses=tuple(mses),
    )


# ── Internal helper ────────────────────────────────────────────────────────


def _to_cpu(value: object) -> np.ndarray:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else np.asarray(value)  # type: ignore[union-attr]


__all__ = [
    "RoutebindPredictionReasoningData",
    "select_routebind_prediction_reasoning",
]
