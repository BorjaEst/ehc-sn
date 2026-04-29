"""Selectors for MazeHard figures.

Reads ``target/solution_overlay`` from trace metadata — no adapter imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.grids import first_halt_index
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace / meta path constants ────────────────────────────────────
TRACE_KEY_HALTED = "act/halted"
TRACE_KEY_PRED_OVERLAY = "pred/solution_overlay"
META_KEY_INPUT_IDS = "input_ids"
META_KEY_GT_OVERLAY = "target/solution_overlay"

# Default cap on overlay items when ctx.max_items is not set.
_DEFAULT_MAX_MAZES = 10


@dataclass
class OverlayFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.overlay.OverlayFigure`."""

    input_ids: NDArray   # (n, N)
    gt_overlays: NDArray  # (n, N) bool
    model_overlays: NDArray  # (n, N)


@dataclass
class EvolutionFigureData:
    """Prepared data for :class:`~ehc_sn.figures.templates.evolution.PredictionEvolutionFigure`."""

    input_ids: NDArray     # (B, N)
    gt_overlay: NDArray    # (N,) bool — single sample
    pred_is_o: NDArray     # (T, B, N)
    halted: NDArray        # (T, B) bool
    sample_idx: int
    t_halt: int
    t_indices: list[int]


def select_overlay(trace: TraceTree, ctx: FigureContext) -> OverlayFigureData:
    """Extract overlay data from the trace, respecting ctx selection policy."""
    input_ids = np.asarray(trace.get_meta_path(META_KEY_INPUT_IDS))
    gt_raw = np.asarray(trace.get_meta_path(META_KEY_GT_OVERLAY))
    halted = np.asarray(trace.get(TRACE_KEY_HALTED))
    pred_is_o = np.asarray(trace.get(TRACE_KEY_PRED_OVERLAY))

    if halted.ndim != 2:
        raise ValueError(f"{TRACE_KEY_HALTED} must have shape [T, B]")
    if pred_is_o.ndim != 3:
        raise ValueError(f"{TRACE_KEY_PRED_OVERLAY} must have shape [T, B, N]")

    batch_size = halted.shape[1]
    start = ctx.sample_idx
    cap = ctx.max_items if ctx.max_items is not None else _DEFAULT_MAX_MAZES
    n = min(cap, batch_size - start)
    if n <= 0:
        n = min(_DEFAULT_MAX_MAZES, batch_size)
        start = 0

    end = start + n
    model_overlays = np.stack(
        [pred_is_o[first_halt_index(halted[:, b]), b] for b in range(start, end)], axis=0
    )
    return OverlayFigureData(
        input_ids=input_ids[start:end],
        gt_overlays=gt_raw[start:end].astype(bool),
        model_overlays=model_overlays,
    )


def select_evolution(trace: TraceTree, ctx: FigureContext, k_max: int = 16) -> EvolutionFigureData:
    """Extract evolution data from the trace for the sample chosen by ctx."""
    input_ids = np.asarray(trace.get_meta_path(META_KEY_INPUT_IDS))
    gt_raw = np.asarray(trace.get_meta_path(META_KEY_GT_OVERLAY))
    pred_is_o = np.asarray(trace.get(TRACE_KEY_PRED_OVERLAY))
    halted = np.asarray(trace.get(TRACE_KEY_HALTED))

    batch_size = halted.shape[1] if halted.ndim == 2 else 1
    sample_idx = ctx.sample_idx if ctx.sample_idx < batch_size else 0
    t_halt = first_halt_index(halted[:, sample_idx])
    t_indices = _select_timesteps(t_halt, k_max)

    return EvolutionFigureData(
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
