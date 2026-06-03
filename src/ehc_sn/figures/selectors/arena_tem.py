"""Selectors for Arena TEM prediction-mazehard_solution_overlay figures.

Reads ``target/observation_id`` from trace metadata — no adapter imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree
from ehc_sn.utils import to_cpu

# ── Canonical trace / meta path constants ────────────────────────────────────
TEM_TRACE_KEY_PRED_INFERENCE = "pred/observation_id/inference"
TEM_TRACE_KEY_PRED_RETRIEVED = "pred/observation_id/retrieved"
TEM_TRACE_KEY_PRED_ANCESTRAL = "pred/observation_id/ancestral"
TEM_META_KEY_TARGET_OBS_ID = "target/observation_id"

# Default cap on mazehard_solution_overlay items when ctx.max_items is not set.
_DEFAULT_MAX_SAMPLES = 10


# =============================================================================
@dataclass
class TEMOverlayData:
    """Prepared data for the per-step argmax prediction mazehard_solution_overlay figure.

    All fields have shape ``(n_cases, T)`` where *T* is the aligned sequence
    length (min of trace steps and GT steps).
    """

    gt_obs_ids: NDArray  # (n_cases, T) int
    pred_inference: NDArray  # (n_cases, T) int
    pred_retrieved: NDArray  # (n_cases, T) int
    pred_ancestral: NDArray  # (n_cases, T) int


# =============================================================================
def select_tem_prediction_overlay(
    trace: TraceTree, ctx: FigureContext
) -> TEMOverlayData:
    """Extract per-step argmax prediction-mazehard_solution_overlay data from the trace.

    Selects N samples (bounded by ``ctx.max_items``) and reads per-step
    predictions for all three pathways (inference, retrieved, ancestral),
    then time-aligns them with the ground-truth observation trajectory.

    Args:
        trace: Rollout trace with ``pred/observation_id/*`` (T, B) and
            ``target/observation_id`` meta (B, T_max).
        ctx: Figure context controlling ``sample_idx`` and ``max_items``.

    Returns:
        :class:`TEMOverlayData` with per-step fields.

    Raises:
        ValueError: If any prediction key has unexpected dimensionality.
    """
    # Ground-truth trajectory from metadata: (B, T_max) int.
    gt_raw = np.asarray(to_cpu(trace.get_meta_path(TEM_META_KEY_TARGET_OBS_ID)))
    if gt_raw.ndim != 2:
        raise ValueError(
            f"{TEM_META_KEY_TARGET_OBS_ID} must have shape (B, T_max), "
            f"got ndim={gt_raw.ndim}"
        )

    # Predictions from trace: (T, B) int each.
    pred_inf = np.asarray(to_cpu(trace.get(TEM_TRACE_KEY_PRED_INFERENCE)))
    pred_ret = np.asarray(to_cpu(trace.get(TEM_TRACE_KEY_PRED_RETRIEVED)))
    pred_anc = np.asarray(to_cpu(trace.get(TEM_TRACE_KEY_PRED_ANCESTRAL)))

    for name, arr in [
        (TEM_TRACE_KEY_PRED_INFERENCE, pred_inf),
        (TEM_TRACE_KEY_PRED_RETRIEVED, pred_ret),
        (TEM_TRACE_KEY_PRED_ANCESTRAL, pred_anc),
    ]:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must have shape (T, B), got ndim={arr.ndim}"
            )

    T_pred, B = pred_inf.shape
    batch_size = B

    # Select samples.
    start = ctx.sample_idx
    cap = ctx.max_items if ctx.max_items is not None else _DEFAULT_MAX_SAMPLES
    n = min(cap, batch_size - start)
    if n <= 0:
        n = min(_DEFAULT_MAX_SAMPLES, batch_size)
        start = 0
    end = start + n

    # Align trace steps with GT steps.
    T_gt = gt_raw.shape[1]
    T = min(T_pred, T_gt)
    if T_pred != T_gt:
        import warnings

        warnings.warn(
            f"Prediction trace steps ({T_pred}) != GT steps ({T_gt}). "
            f"Truncating to {T} overlapping steps."
        )

    return TEMOverlayData(
        gt_obs_ids=gt_raw[start:end, :T],
        pred_inference=pred_inf[:T, start:end].T,
        pred_retrieved=pred_ret[:T, start:end].T,
        pred_ancestral=pred_anc[:T, start:end].T,
    )


# =============================================================================
__all__ = [
    "TEMOverlayData",
    "TEM_TRACE_KEY_PRED_INFERENCE",
    "TEM_TRACE_KEY_PRED_RETRIEVED",
    "TEM_TRACE_KEY_PRED_ANCESTRAL",
    "TEM_META_KEY_TARGET_OBS_ID",
    "select_tem_prediction_overlay",
]
