"""Selectors for MazeHard figures.

Reads ``target/solution_overlay`` from trace metadata — no adapter imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.grids import first_halt_index
from ehc_sn.traces.keys import (
    MAZEHARD_META_KEY_GT_OVERLAY,
    MAZEHARD_META_KEY_INPUT_IDS,
    MAZEHARD_TRACE_KEY_HALTED,
    MAZEHARD_TRACE_KEY_PRED_OVERLAY,
)
from ehc_sn.traces.trace_tree import TraceTree

# Default cap on mazehard_solution_overlay items when ctx.max_items is not set.
_DEFAULT_MAX_MAZES = 10


def _to_cpu(value: object) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value  # type: ignore[union-attr]


@dataclass
class MazehardSolutionOverlayFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.mazehard_solution_overlay.MazehardSolutionOverlayFigure`."""

    input_ids: NDArray  # (n, N)
    gt_overlays: NDArray  # (n, N) bool
    model_overlays: NDArray  # (n, N)


@dataclass
class MazehardPredictionEvolutionFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.mazehard_prediction_evolution.MazehardPredictionEvolutionFigure`."""

    input_ids: NDArray  # (B, N)
    gt_overlay: NDArray  # (N,) bool — single sample
    pred_is_o: NDArray  # (T, B, N)
    halted: NDArray  # (T, B) bool
    sample_idx: int
    t_halt: int
    t_indices: list[int]


def select_overlay(
    trace: TraceTree, ctx: FigureContext
) -> MazehardSolutionOverlayFigureData:
    """Extract mazehard_solution_overlay data from the trace, respecting ctx selection policy."""
    input_ids = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_INPUT_IDS))
    )
    gt_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_GT_OVERLAY))
    )
    halted = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_HALTED)))
    pred_is_o = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_PRED_OVERLAY)))

    if halted.ndim != 2:
        raise ValueError(f"{MAZEHARD_TRACE_KEY_HALTED} must have shape [T, B]")
    if pred_is_o.ndim != 3:
        raise ValueError(
            f"{MAZEHARD_TRACE_KEY_PRED_OVERLAY} must have shape [T, B, N]"
        )

    batch_size = halted.shape[1]
    start = ctx.sample_idx
    cap = ctx.max_items if ctx.max_items is not None else _DEFAULT_MAX_MAZES
    n = min(cap, batch_size - start)
    if n <= 0:
        n = min(_DEFAULT_MAX_MAZES, batch_size)
        start = 0

    end = start + n
    model_overlays = np.stack(
        [
            pred_is_o[first_halt_index(halted[:, b]), b]
            for b in range(start, end)
        ],
        axis=0,
    )
    return MazehardSolutionOverlayFigureData(
        input_ids=input_ids[start:end],
        gt_overlays=gt_raw[start:end].astype(bool),
        model_overlays=model_overlays,
    )


def select_evolution(
    trace: TraceTree, ctx: FigureContext, k_max: int = 16
) -> MazehardPredictionEvolutionFigureData:
    """Extract evolution data from the trace for the sample chosen by ctx."""
    input_ids = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_INPUT_IDS))
    )
    gt_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_GT_OVERLAY))
    )
    pred_is_o = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_PRED_OVERLAY)))
    halted = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_HALTED)))

    batch_size = halted.shape[1] if halted.ndim == 2 else 1
    sample_idx = ctx.sample_idx if ctx.sample_idx < batch_size else 0
    t_halt = first_halt_index(halted[:, sample_idx])
    t_indices = _select_timesteps(t_halt, k_max)

    return MazehardPredictionEvolutionFigureData(
        input_ids=input_ids,
        gt_overlay=gt_raw[sample_idx].astype(bool),
        pred_is_o=pred_is_o,
        halted=halted,
        sample_idx=sample_idx,
        t_halt=t_halt,
        t_indices=t_indices,
    )


@dataclass
class MazehardPredictionAccuracyFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.prediction_accuracy_over_steps.MazehardPredictionAccuracyFigure`."""

    token_accuracy: NDArray  # (T, B) — per-step token accuracy
    path_recall: NDArray  # (T, B) — per-step target-path recall
    sample_idx: int
    best_step: int  # step index of best token accuracy
    best_step_recall: int  # step index of best path recall
    final_step: int  # last step index (= T-1)

    # Future-proofing: computed but not plotted yet.
    path_precision: NDArray  # (T, B)
    path_iou: NDArray  # (T, B)
    incorrect_tokens: NDArray  # (T, B)


@dataclass
class MazehardTaskLayoutFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.mazehard_task_layout.MazehardTaskLayoutFigure`."""

    input_ids: NDArray  # (side, side) — categorical input grid
    target_overlay: NDArray  # (side, side) — binary target path
    target_path_cells: int
    case_id: str
    grid_shape: tuple[int, int]
    rollout_steps: int


def select_prediction_accuracy(
    trace: TraceTree,
    ctx: FigureContext,
) -> MazehardPredictionAccuracyFigureData:
    """Extract per-step accuracy and recall data from the trace.

    Computes token accuracy and target-path recall for every (step, batch)
    position, then picks the sample from ``ctx`` and identifies best-step
    indices.
    """
    gt_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_GT_OVERLAY))
    )  # (B, N)
    pred_is_o = np.asarray(
        _to_cpu(trace.get(MAZEHARD_TRACE_KEY_PRED_OVERLAY))
    )  # (T, B, N)

    if pred_is_o.ndim != 3:
        raise ValueError(
            f"{MAZEHARD_TRACE_KEY_PRED_OVERLAY} must have shape [T, B, N], "
            f"got {pred_is_o.shape}"
        )

    T, B, N = pred_is_o.shape

    # Binary ground-truth overlay — expand to (T, B, N) for broadcasting.
    target_bool = gt_raw.astype(bool)  # (B, N)
    pred_bool = pred_is_o > 0  # (T, B, N) — binarise at 0 threshold

    # Token accuracy: fraction of cells matching ground-truth class.
    correct = pred_bool == target_bool[None, :, :]  # (T, B, N)
    token_accuracy = correct.mean(axis=-1)  # (T, B)

    # Target-path recall: fraction of true-path cells correctly predicted.
    target_foreground = target_bool[None, :, :]  # (T, B, N) for broadcasting
    tp = pred_bool & target_foreground  # (T, B, N)
    n_target = target_foreground.sum(axis=-1)  # (T, B)
    # Avoid division by zero: if target has no foreground, recall = 1.0.
    path_recall = np.where(
        n_target > 0,
        tp.sum(axis=-1) / n_target,
        1.0,
    )  # (T, B)

    # Precision (future): fraction of predicted foreground cells that are correct.
    n_pred = pred_bool.sum(axis=-1)  # (T, B)
    path_precision = np.where(
        n_pred > 0,
        tp.sum(axis=-1) / n_pred,
        0.0,
    )

    # IoU (future): intersection over union.
    union = pred_bool | target_foreground
    n_union = union.sum(axis=-1)
    path_iou = np.where(
        n_union > 0,
        tp.sum(axis=-1) / n_union,
        0.0,
    )

    # Incorrect tokens (future): count of mismatched cells.
    incorrect_tokens = (N - correct.sum(axis=-1)).astype(np.float64)  # (T, B)

    # Select batch element from ctx.
    batch_size = B
    sample_idx = ctx.sample_idx if ctx.sample_idx < batch_size else 0

    # Best-step indices per batch element (accuracy).
    best_step = int(token_accuracy[:, sample_idx].argmax())

    # Best-step indices per batch element (recall).
    best_step_recall = int(path_recall[:, sample_idx].argmax())

    final_step = T - 1

    return MazehardPredictionAccuracyFigureData(
        token_accuracy=token_accuracy,
        path_recall=path_recall,
        sample_idx=sample_idx,
        best_step=best_step,
        best_step_recall=best_step_recall,
        final_step=final_step,
        path_precision=path_precision,
        path_iou=path_iou,
        incorrect_tokens=incorrect_tokens,
    )


def select_task_layout(
    trace: TraceTree, ctx: FigureContext
) -> MazehardTaskLayoutFigureData:
    """Extract task-layout data from the trace (meta keys only, no dense traces).

    Returns the input grid, binary target-path overlay, and case metadata.
    """
    input_raw = np.asarray(_to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_INPUT_IDS)))
    target_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_GT_OVERLAY))
    )

    # Squeeze leading batch dimension if present.
    if input_raw.ndim == 2 and input_raw.shape[0] == 1:
        input_raw = input_raw[0]
    if target_raw.ndim == 2 and target_raw.shape[0] == 1:
        target_raw = target_raw[0]

    n_cells = input_raw.size
    side = int(np.sqrt(n_cells))
    if side * side != n_cells:
        raise ValueError(
            f"input_ids has {n_cells} cells, which is not a perfect square; "
            f"cannot infer grid shape."
        )

    input_grid = input_raw.reshape(side, side)

    # Infer background value as the mode of the target overlay.
    values, counts = np.unique(target_raw, return_counts=True)
    background = values[np.argmax(counts)]
    target_fg = (target_raw != background).astype(bool).reshape(side, side)

    # Count target-path cells.
    n_fg = int(target_fg.sum())

    return MazehardTaskLayoutFigureData(
        input_ids=input_grid,
        target_overlay=target_fg,
        target_path_cells=n_fg,
        case_id="mazehard-test-0000",
        grid_shape=(side, side),
        rollout_steps=16,
    )


def _select_timesteps(t_halt: int, k_max: int) -> list[int]:
    if t_halt < 0:
        return [0]
    if t_halt + 1 <= k_max:
        return list(range(t_halt + 1))
    indices = np.linspace(0, t_halt, num=k_max, dtype=int)
    t_indices = sorted({int(idx) for idx in indices})
    if t_halt not in t_indices:
        t_indices.append(t_halt)
        t_indices.sort()
    return t_indices
