"""Selectors for Arena TEM prediction-overlay figures.

Reads ``target/observation_id`` from trace metadata — no adapter imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace / meta path constants ────────────────────────────────────
TRACE_KEY_PRED_INFERENCE = "pred/observation_id/inference"
TRACE_KEY_PRED_RETRIEVED = "pred/observation_id/retrieved"
TRACE_KEY_PRED_ANCESTRAL = "pred/observation_id/ancestral"
META_KEY_TARGET_OBS_ID = "target/observation_id"

# Default cap on overlay items when ctx.max_items is not set.
_DEFAULT_MAX_SAMPLES = 10


def _to_cpu(value: object) -> object:
    """Move a tensor to CPU if needed; return other values unchanged."""
    return value.cpu() if hasattr(value, "cpu") else value  # type: ignore[union-attr]


@dataclass
class TEMOverlayData:
    """Prepared data for :class:`~ehc_sn.figures.templates.tem_prediction_overlay.TEMPredictionOverlayFigure`."""

    gt_obs_ids: NDArray  # (n, T_max) int — ground-truth trajectories
    pred_inference: NDArray  # (n,) int — inference prediction at final step
    pred_retrieved: NDArray  # (n,) int — retrieved prediction at final step
    pred_ancestral: NDArray  # (n,) int — ancestral prediction at final step
    gt_at_final: NDArray  # (n,) int — ground truth at final step index


def select_tem_prediction_overlay(
    trace: TraceTree, ctx: FigureContext
) -> TEMOverlayData:
    """Extract TEM prediction-overlay data from the trace.

    Selects N samples (bounded by ``ctx.max_items``) and reads predictions
    at the final rollout step ``T-1``, comparing against ground-truth
    ``observation_id`` at the corresponding trajectory position.

    Args:
        trace: Rollout trace with both ``pred/observation_id/*`` and
            ``target/observation_id`` metadata.
        ctx: Figure context controlling ``sample_idx`` and ``max_items``.

    Returns:
        :class:`TEMOverlayData` with selected samples.

    Raises:
        ValueError: If any prediction key has unexpected dimensionality.
    """
    # Ground-truth trajectory from metadata: (B, T_max) int.
    gt_raw = np.asarray(_to_cpu(trace.get_meta_path(META_KEY_TARGET_OBS_ID)))
    if gt_raw.ndim != 2:
        raise ValueError(
            f"{META_KEY_TARGET_OBS_ID} must have shape (B, T_max), "
            f"got ndim={gt_raw.ndim}"
        )

    # Predictions from trace: (T, B) int each.
    pred_inf = np.asarray(_to_cpu(trace.get(TRACE_KEY_PRED_INFERENCE)))
    pred_ret = np.asarray(_to_cpu(trace.get(TRACE_KEY_PRED_RETRIEVED)))
    pred_anc = np.asarray(_to_cpu(trace.get(TRACE_KEY_PRED_ANCESTRAL)))

    for name, arr in [
        (TRACE_KEY_PRED_INFERENCE, pred_inf),
        (TRACE_KEY_PRED_RETRIEVED, pred_ret),
        (TRACE_KEY_PRED_ANCESTRAL, pred_anc),
    ]:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must have shape (T, B), got ndim={arr.ndim}"
            )

    T, B = pred_inf.shape
    batch_size = B

    # Select samples.
    start = ctx.sample_idx
    cap = ctx.max_items if ctx.max_items is not None else _DEFAULT_MAX_SAMPLES
    n = min(cap, batch_size - start)
    if n <= 0:
        n = min(_DEFAULT_MAX_SAMPLES, batch_size)
        start = 0
    end = start + n

    # Final-step index: last available rollout step.
    t_final = T - 1

    return TEMOverlayData(
        gt_obs_ids=gt_raw[start:end],
        pred_inference=pred_inf[t_final, start:end],
        pred_retrieved=pred_ret[t_final, start:end],
        pred_ancestral=pred_anc[t_final, start:end],
        gt_at_final=gt_raw[start:end, t_final] if gt_raw.shape[1] > t_final else np.zeros(n, dtype=gt_raw.dtype),
    )


# =============================================================================
__all__ = [
    "TEMOverlayData",
    "TRACE_KEY_PRED_INFERENCE",
    "TRACE_KEY_PRED_RETRIEVED",
    "TRACE_KEY_PRED_ANCESTRAL",
    "META_KEY_TARGET_OBS_ID",
    "select_tem_prediction_overlay",
]
