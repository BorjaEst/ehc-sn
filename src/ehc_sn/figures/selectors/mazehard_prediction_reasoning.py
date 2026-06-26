"""MazeHard prediction-reasoning selector.

Produces ``MazehardPredictionReasoningData`` from a semantic
:class:`~ehc_sn.figures.selectors.mazehard.MazehardEvaluationSample`.

The adapter from ``TraceTree`` to ``MazehardEvaluationSample`` is called at
the entry point — this module never touches raw trace keys or tree access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors._reasoning_steps import select_reasoning_snapshots
from ehc_sn.figures.selectors.mazehard import (
    MazehardEvaluationSample,
    build_mazehard_evaluation_sample,
)


@dataclass(frozen=True)
class MazehardPredictionReasoningData:
    """Prepared data for ``prediction_reasoning_mazehard``.

    All arrays are already decoded — the template never touches raw traces.
    """

    input_ids: NDArray  # (H, W) int — input grid for the selected sample
    gt_overlay: NDArray  # (H, W) bool — ground truth path overlay
    snapshots: tuple[np.ndarray, ...]  # each (H, W) bool — per-step pred
    selected_indices: tuple[int, ...]  # real deliberation step numbers
    halt_step: int | None  # step at which the model halted, or None
    truncated: bool  # True if the trace reached max steps without halting
    path_ious: tuple[float, ...] = field(default_factory=tuple)
    """Per-snapshot path IoU against ground truth (task metric)."""


# ── Default cap ────────────────────────────────────────────────────────────

_MAX_SNAPSHOTS: int = 16


# ── Public selector ────────────────────────────────────────────────────────


def select_mazehard_prediction_reasoning(
    trace: object,
    ctx: FigureContext,
    max_snapshots: int = _MAX_SNAPSHOTS,
) -> MazehardPredictionReasoningData:
    """Extract prediction-reasoning data from a MazeHard evaluation trace.

    Accepts either a ``TraceTree`` (raw, calls adapter internally) or a
    pre-built ``MazehardEvaluationSample``.  The adapter call is the only
    point of storage access — all downstream logic consumes semantic arrays.

    Args:
        trace: Trace tree with MazeHard keys **or** a
            :class:`MazehardEvaluationSample`.
        ctx: Figure context (``sample_idx`` selects batch element).
        max_snapshots: Maximum number of snapshot slots.

    Returns:
        Prepared figure data.
    """
    # ── Resolve input ───────────────────────────────────────────────────
    if isinstance(trace, MazehardEvaluationSample):
        sample = trace
    else:
        # Raw TraceTree — build semantic sample (single storage access point).
        from ehc_sn.traces.trace_tree import TraceTree

        assert isinstance(trace, TraceTree), type(trace)
        sample = build_mazehard_evaluation_sample(
            trace,
            sample_idx=ctx.sample_idx,
        )

    return _select_from_sample(sample, ctx, max_snapshots=max_snapshots)


def _select_from_sample(
    sample: MazehardEvaluationSample,
    ctx: FigureContext,
    *,
    max_snapshots: int = _MAX_SNAPSHOTS,
) -> MazehardPredictionReasoningData:
    """Build prediction-reasoning data from a semantic evaluation sample."""
    pred = sample.prediction_steps  # (T, N)
    halted = sample.halted  # (T,)
    gt = sample.gt_overlay  # (N,) bool
    inp = sample.input_grid  # (H, W) or (N,)

    T = len(halted)
    t_halt = sample.halt_step if sample.halt_step is not None else T - 1
    truncated = sample.truncated

    indices = select_reasoning_snapshots(t_halt, max_snapshots)

    snapshots: list[np.ndarray] = []
    ious: list[float] = []
    for t in indices:
        pred_t = pred[t].astype(bool) if t < len(pred) else np.zeros_like(gt)
        snapshots.append(pred_t)
        intersection = (gt & pred_t).sum()
        union = (gt | pred_t).sum()
        iou = float(intersection / max(union, 1))
        ious.append(iou)

    return MazehardPredictionReasoningData(
        input_ids=inp,
        gt_overlay=gt,
        snapshots=tuple(snapshots),
        selected_indices=tuple(indices),
        halt_step=t_halt if halted[t_halt] else None,
        truncated=truncated,
        path_ious=tuple(ious),
    )


__all__ = [
    "MazehardPredictionReasoningData",
    "select_mazehard_prediction_reasoning",
]
