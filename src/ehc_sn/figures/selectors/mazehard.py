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
