"""Selectors for MazeHard figures.

All selectors consume semantic dataclasses (``MazehardTaskSample``,
``MazehardEvaluationSample``), not raw ``TraceTree`` or batch dicts.
Storage-level access (trace keys, CPU conversion, axis squeezing) is
handled by the ``build_*`` adapter functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.grids import first_halt_index
from ehc_sn.traces.keys import (
    MAZEHARD_META_KEY_CASE_ID,
    MAZEHARD_META_KEY_GT_OVERLAY,
    MAZEHARD_META_KEY_INPUT_IDS,
    MAZEHARD_TRACE_KEY_HALTED,
    MAZEHARD_TRACE_KEY_PRED_OVERLAY,
)
from ehc_sn.traces.trace_tree import TraceTree

# Default cap on items when ctx.max_items is not set.
_DEFAULT_MAX_MAZES = 10


# ── Semantic source views ───────────────────────────────────────────────


@dataclass(frozen=True)
class MazehardTaskSample:
    """One MazeHard dataset sample — no model involved.

    Attributes:
        input_grid: ``(N,)`` int — token IDs for one maze.
        target_solution: ``(N,)`` bool — oracle solution path.
        sample_id: Unique sample identifier.
    """

    input_grid: NDArray  # (N,)
    target_solution: NDArray  # (N,) bool
    sample_id: str


@dataclass(frozen=True)
class MazehardEvaluationSample:
    """One MazeHard evaluated sample — batch-dim already indexed out.

    Attributes:
        input_grid: ``(N,)`` int — token IDs.
        prediction_steps: ``(T, N)`` uint8 — predicted overlay per step.
        halted: ``(T,)`` bool — halt signal per reasoning step.
        steps: ``(T,)`` int — step indices.
        gt_overlay: ``(N,)`` bool — oracle solution overlay.
        halt_step: ``int | None`` — index of first halt, or None.
        truncated: ``bool`` — whether the run was truncated.
        metrics: ``dict[str, float]`` — per-sample metrics.
    """

    input_grid: NDArray  # (N,)
    prediction_steps: NDArray  # (T, N) uint8
    halted: NDArray  # (T,) bool
    steps: NDArray  # (T,)
    gt_overlay: NDArray  # (N,) bool
    halt_step: int | None = None
    truncated: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


# ── Adapter functions (storage → semantic) ──────────────────────────────


def _to_cpu(value: object) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value  # type: ignore[union-attr]


def build_mazehard_task_sample(batch: dict) -> MazehardTaskSample:
    """Build a task sample from a raw dataset batch row.

    Parameters
    ----------
    batch:
        A single sample dict from a MazeHard ``ProcessedDataset`` or
        ``DataLoader``.  Expected keys: ``"input_ids"``, ``"labels"``
        or ``"solution"``.

    Returns
    -------
    MazehardTaskSample

    Raises
    ------
    KeyError
        If any required key is missing.
    """
    import numpy as np

    input_ids = np.asarray(_to_cpu(batch["input_ids"]))
    labels = np.asarray(_to_cpu(batch.get("labels", batch.get("solution"))))
    return MazehardTaskSample(
        input_grid=input_ids,
        target_solution=labels.astype(bool),
        sample_id=str(batch.get("sample_id", "")),
    )


def build_mazehard_evaluation_sample(
    trace: TraceTree,
    *,
    sample_idx: int = 0,
) -> MazehardEvaluationSample:
    """Build an evaluation sample from a persisted ``TraceTree``.

    Parameters
    ----------
    trace:
        Trace tree with mazehard keys populated.
    sample_idx:
        Which sample within the batch to extract (default 0).

    Returns
    -------
    MazehardEvaluationSample

    Raises
    ------
    KeyError
        If any required trace key is missing.
    """
    input_ids = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_INPUT_IDS))
    )
    gt_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_GT_OVERLAY))
    )
    halted_raw = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_HALTED)))
    pred_raw = np.asarray(_to_cpu(trace.get(MAZEHARD_TRACE_KEY_PRED_OVERLAY)))

    # Normalise shapes — squeeze trailing single dims
    while halted_raw.ndim > 2:
        halted_raw = halted_raw.squeeze(-1)
    while pred_raw.ndim > 3:
        pred_raw = pred_raw.squeeze(-1)

    # Index sample from batch dim
    halted = halted_raw[:, sample_idx] if halted_raw.ndim == 2 else halted_raw
    pred = pred_raw[:, sample_idx] if pred_raw.ndim == 3 else pred_raw
    grid = input_ids[sample_idx] if input_ids.ndim >= 2 else input_ids
    gt = (
        gt_raw[sample_idx].astype(bool)
        if gt_raw.ndim >= 2
        else gt_raw.astype(bool)
    )

    # Reshape flat spatial arrays to square grid for imshow rendering.
    # MazeHard canvases are always square (30×30 = 900 cells).
    if grid.ndim == 1:
        side = int(np.sqrt(grid.size))
        if side * side == grid.size:
            grid = grid.reshape(side, side)
    if gt.ndim == 1:
        side = int(np.sqrt(gt.size))
        if side * side == gt.size:
            gt = gt.reshape(side, side)
    if pred.ndim == 2 and grid.ndim == 2:
        side = grid.shape[0]
        pred = pred.reshape(pred.shape[0], side, side)

    # Steps
    steps = np.arange(len(halted), dtype=np.int64)

    # Halt detection
    halt_indices = np.where(halted)[0]
    halt_step = int(halt_indices[0]) if len(halt_indices) > 0 else None
    truncated = halt_step is None

    return MazehardEvaluationSample(
        input_grid=grid,
        prediction_steps=pred,
        halted=halted,
        steps=steps,
        gt_overlay=gt,
        halt_step=halt_step,
        truncated=truncated,
    )


@dataclass
class MazehardSolutionOverlayFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.
    mazehard_prediction_overlay.MazehardSolutionOverlayFigure`.
    """

    input_ids: NDArray  # (n, N)
    gt_overlays: NDArray  # (n, N) bool
    model_overlays: NDArray  # (n, N)


@dataclass
class MazehardPredictionEvolutionFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.
    prediction_reasoning_mazehard.MazehardPredictionEvolutionFigure`.
    """

    input_ids: NDArray  # (B, N)
    gt_overlay: NDArray  # (N,) bool — single sample
    pred_is_o: NDArray  # (T, B, N)
    halted: NDArray  # (T, B) bool
    sample_idx: int
    t_halt: int
    t_indices: list[int]


def select_overlay(
    sample: MazehardEvaluationSample, ctx: FigureContext
) -> MazehardSolutionOverlayFigureData:
    """Extract mazehard prediction overlay data from a single evaluated sample."""
    grid = sample.input_grid  # (N,)
    gt = sample.gt_overlay  # (N,) bool
    pred = sample.prediction_steps  # (T, N)

    # The overlay figure shows final predictions for comparison.
    halt_idx = sample.halt_step
    final_step = pred[-1] if halt_idx is None else pred[halt_idx]

    # Expand back to (1, N) for the FigureData contract (single-sample batch).
    return MazehardSolutionOverlayFigureData(
        input_ids=grid[np.newaxis, :],
        gt_overlays=gt[np.newaxis, :],
        model_overlays=final_step[np.newaxis, :],
    )


def select_evolution(
    sample: MazehardEvaluationSample, ctx: FigureContext, k_max: int = 16
) -> MazehardPredictionEvolutionFigureData:
    """Extract evolution data from an evaluated sample.

    Uses the sample's halted/reasoning steps and expands batch-dim arrays
    back to the existing ``MazehardPredictionEvolutionFigureData`` contract.
    """
    # The sample is already single-sample — replicate to (B, ...) shape.
    B = 1
    N = sample.input_grid.shape[-1]
    T = len(sample.halted)

    pred_is_o = sample.prediction_steps[np.newaxis, :, :]  # (1, T, N)
    halted = sample.halted[np.newaxis, :]  # (1, T)

    # Reshape input_grid to (B, N) for FigureData contract.
    input_grid = sample.input_grid.reshape(B, N)

    t_halt = int(sample.halt_step) if sample.halt_step is not None else T - 1
    t_indices = _select_timesteps(t_halt, k_max)

    return MazehardPredictionEvolutionFigureData(
        input_ids=input_grid,
        gt_overlay=sample.gt_overlay.astype(bool),
        pred_is_o=pred_is_o,
        halted=halted,
        sample_idx=0,
        t_halt=t_halt,
        t_indices=t_indices,
    )


@dataclass
class MazehardPredictionAccuracyFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.
    prediction_accuracy_over_steps.MazehardPredictionAccuracyFigure`.
    """

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
    """Prepared data for :class:`~ehc_sn.figures.templates.
    task_overview_mazehard.MazehardTaskLayoutFigure`.
    """

    input_ids: NDArray  # (side, side) — categorical input grid
    target_overlay: NDArray  # (side, side) — float32 target path
    target_path_cells: int
    case_id: str
    grid_shape: tuple[int, int]


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
    input_raw = np.asarray(
        _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_INPUT_IDS))
    )
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
    target_fg = (
        (target_raw != background).astype(np.float32).reshape(side, side)
    )

    # Count target-path cells.
    n_fg = int(target_fg.sum())

    # Read case_id from meta, fall back to hardcoded default.
    try:
        case_id = str(
            np.asarray(
                _to_cpu(trace.get_meta_path(MAZEHARD_META_KEY_CASE_ID))
            ).flat[0]
        )
    except (KeyError, IndexError):
        case_id = "mazehard-test-0000"

    return MazehardTaskLayoutFigureData(
        input_ids=input_grid,
        target_overlay=target_fg,
        target_path_cells=n_fg,
        case_id=case_id,
        grid_shape=(side, side),
    )


def _select_timesteps(t_halt: int, k_max: int) -> list[int]:
    """Thin wrapper around :func:`select_reasoning_snapshots` for backward compat.

    The hybrid-sampling logic has moved to
    ``ehc_sn.figures.selectors._reasoning_steps`` — this function delegates
    to the shared implementation.
    """
    from ehc_sn.figures.selectors._reasoning_steps import (
        select_reasoning_snapshots,
    )

    return select_reasoning_snapshots(t_halt, max_snapshots=k_max)
