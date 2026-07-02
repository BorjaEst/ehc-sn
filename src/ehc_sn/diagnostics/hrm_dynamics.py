"""HRM latent-state dynamics diagnostics for MazeHard traces.

Computes per-trace scalar summaries from pfc/z_H and pfc/z_L arrays,
characterising high-level vs low-level state dynamics during deliberation.

All functions accept raw numpy arrays of shape ``(T, B, S, D)`` where:
    T = number of time steps
    B = batch size
    S = number of slots (including controller)
    D = hidden dimension
"""

from __future__ import annotations

import numpy as np

from ehc_sn.traces import TraceTree
from ehc_sn.traces.keys import PFC_TRACE_KEY_Z_H, PFC_TRACE_KEY_Z_L

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_h_state_norm_mean(z_H: np.ndarray) -> float:
    """Mean L2 norm of the high-level state across time, batch, and slots.

    Args:
        z_H: High-level state array of shape ``(T, B, S, D)``.

    Returns:
        Scalar mean of ``||z_H||`` over all dimensions except the last.
    """
    return float(np.mean(np.linalg.norm(z_H, axis=-1)))


def compute_l_state_norm_mean(z_L: np.ndarray) -> float:
    """Mean L2 norm of the low-level state across time, batch, and slots.

    Args:
        z_L: Low-level state array of shape ``(T, B, S, D)``.

    Returns:
        Scalar mean of ``||z_L||`` over all dimensions except the last.
    """
    return float(np.mean(np.linalg.norm(z_L, axis=-1)))


def compute_h_state_delta_mean(z_H: np.ndarray) -> float:
    """Mean temporal delta magnitude of the high-level state.

    Computes ``||z_H[t] - z_H[t-1]||`` for ``t >= 1`` and returns the
    mean across all time deltas, batches, and slots.

    Args:
        z_H: High-level state array of shape ``(T, B, S, D)``.

    Returns:
        Scalar mean temporal delta, or 0.0 when ``T < 2``.
    """
    if z_H.shape[0] < 2:
        return 0.0
    deltas = np.linalg.norm(z_H[1:] - z_H[:-1], axis=-1)
    return float(np.mean(deltas))


def compute_l_state_delta_mean(z_L: np.ndarray) -> float:
    """Mean temporal delta magnitude of the low-level state.

    Args:
        z_L: Low-level state array of shape ``(T, B, S, D)``.

    Returns:
        Scalar mean temporal delta, or 0.0 when ``T < 2``.
    """
    if z_L.shape[0] < 2:
        return 0.0
    deltas = np.linalg.norm(z_L[1:] - z_L[:-1], axis=-1)
    return float(np.mean(deltas))


def compute_h_l_delta_ratio(z_H: np.ndarray, z_L: np.ndarray) -> float:
    """Ratio of mean H delta to mean L delta.

    A ratio ``< 1.0`` indicates the high-level state changes more slowly
    than the low-level state (stable H, fast L).  A ratio ``> 1.0``
    indicates the opposite.

    Returns ``0.0`` when both delta means are zero (no change in either
    state), or ``float('inf')`` when L delta is zero but H delta is not.

    Args:
        z_H: High-level state array of shape ``(T, B, S, D)``.
        z_L: Low-level state array of shape ``(T, B, S, D)``.

    Returns:
        Ratio of H delta mean to L delta mean.
    """
    h_delta = compute_h_state_delta_mean(z_H)
    l_delta = compute_l_state_delta_mean(z_L)

    if l_delta == 0.0:
        if h_delta == 0.0:
            return 0.0  # both static — ratio is undefined, return 0
        return float("inf")  # H changes, L does not

    return h_delta / l_delta


def compute_hrm_dynamics_metrics(
    z_H: np.ndarray,
    z_L: np.ndarray,
) -> dict[str, float]:
    """Compute all HRM latent-dynamics metrics from H/L state arrays.

    Args:
        z_H: High-level state array of shape ``(T, B, S, D)``.
        z_L: Low-level state array of shape ``(T, B, S, D)``.

    Returns:
        Dict with keys ``h_state_norm_mean``, ``l_state_norm_mean``,
        ``h_state_delta_mean``, ``l_state_delta_mean``,
        ``h_l_delta_ratio``.
    """
    return {
        "h_state_norm_mean": compute_h_state_norm_mean(z_H),
        "l_state_norm_mean": compute_l_state_norm_mean(z_L),
        "h_state_delta_mean": compute_h_state_delta_mean(z_H),
        "l_state_delta_mean": compute_l_state_delta_mean(z_L),
        "h_l_delta_ratio": compute_h_l_delta_ratio(z_H, z_L),
    }


# ---------------------------------------------------------------------------
# Trace-tree bridge
# ---------------------------------------------------------------------------


def compute_hrm_dynamics_metrics_from_trace(
    trace: TraceTree,
) -> dict[str, float]:
    """Extract z_H/z_L from a trace tree and compute dynamics metrics.

    Uses ``trace.get(PFC_TRACE_KEY_Z_H)`` and ``trace.get(PFC_TRACE_KEY_Z_L)`` to
    retrieve the dense arrays, validates their rank, then delegates to
    :func:`compute_hrm_dynamics_metrics`.

    Args:
        trace: A :class:`~ehp_sn.traces.TraceTree` containing dense
            leaves ``pfc/z_H`` and ``pfc/z_L``.

    Returns:
        Dict with keys ``h_state_norm_mean``, ``l_state_norm_mean``,
        ``h_state_delta_mean``, ``l_state_delta_mean``,
        ``h_l_delta_ratio``.

    Raises:
        ValueError: If either dense leaf is missing from the trace or
            does not have rank 4 ``(T, B, S, D)``.
    """

    def _extract(key: str) -> np.ndarray:
        arr = trace.get(key)
        if arr.ndim != 4:
            raise ValueError(
                f"Expected {key!r} with rank 4 (T, B, S, D), "
                f"got shape {arr.shape}"
            )
        return arr

    z_H = _extract(PFC_TRACE_KEY_Z_H)
    z_L = _extract(PFC_TRACE_KEY_Z_L)

    return compute_hrm_dynamics_metrics(z_H, z_L)
