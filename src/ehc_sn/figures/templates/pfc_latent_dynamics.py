"""HRM H/L latent dynamics figure — bounded online / offline diagnostic.

Three-panel figure:
    1. H and L state L2 norm over rollout time (mean over batch × slots).
    2. H and L state temporal delta magnitude over rollout time.
    3. Summary annotation with scalar metrics.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace path constants ───────────────────────────────────────────

TRACE_KEY_Z_H = "pfc/z_H"
TRACE_KEY_Z_L = "pfc/z_L"


# =============================================================================
def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a three-panel HRM H/L latent-dynamics figure.

    Args:
        trace: Trace containing ``pfc/z_H`` and ``pfc/z_L``, both of
            shape ``(T, B, S, D)`` (time × batch × slots × hidden dim).
        ctx: Figure context (uses ``max_items`` to cap batch samples).

    Returns:
        Matplotlib ``Figure`` with three vertically stacked panels.
    """
    z_H = _validate_extract(trace, TRACE_KEY_Z_H)
    z_L = _validate_extract(trace, TRACE_KEY_Z_L)

    T, B, S, D = z_H.shape

    # Mean over batch and slots for per-step norm and delta.
    h_norm = np.linalg.norm(z_H, axis=-1).mean(axis=(1, 2))  # (T,)
    l_norm = np.linalg.norm(z_L, axis=-1).mean(axis=(1, 2))  # (T,)

    h_delta = np.linalg.norm(z_H[1:] - z_H[:-1], axis=-1).mean(
        axis=(1, 2)
    )  # (T-1,)
    l_delta = np.linalg.norm(z_L[1:] - z_L[:-1], axis=-1).mean(
        axis=(1, 2)
    )  # (T-1,)

    # Summary scalars.
    h_norm_mean = float(h_norm.mean())
    l_norm_mean = float(l_norm.mean())
    h_delta_mean = float(h_delta.mean()) if len(h_delta) > 0 else 0.0
    l_delta_mean = float(l_delta.mean()) if len(l_delta) > 0 else 0.0
    ratio = (
        h_delta_mean / l_delta_mean
        if l_delta_mean > 0
        else (0.0 if h_delta_mean == 0.0 else float("inf"))
    )

    fig = Figure(figsize=(6, 7), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)

    # ── Panel 1: State norm over time ────────────────────────────────────
    ax: Axes = axes[0]
    ax.plot(h_norm, label="H-state", color="tab:blue", linewidth=1.5)
    ax.plot(l_norm, label="L-state", color="tab:orange", linewidth=1.5)
    ax.set_ylabel("L2 norm")
    ax.set_title("HRM H/L latent dynamics")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: State delta over time ───────────────────────────────────
    ax = axes[1]
    if len(h_delta) > 0:
        t_delta = np.arange(1, T)
        ax.plot(
            t_delta, h_delta, label="H-state", color="tab:blue", linewidth=1.5
        )
        ax.plot(
            t_delta, l_delta, label="L-state", color="tab:orange", linewidth=1.5
        )
    ax.set_ylabel("Delta magnitude")
    ax.set_xlabel("Step")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Summary annotation ──────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    summary_lines = [
        f"H-state norm mean: {h_norm_mean:.3f}",
        f"L-state norm mean: {l_norm_mean:.3f}",
        f"H-state delta mean: {h_delta_mean:.4f}",
        f"L-state delta mean: {l_delta_mean:.4f}",
        (
            f"H/L delta ratio: {ratio:.4f}"
            if isinstance(ratio, float) and np.isfinite(ratio)
            else f"H/L delta ratio: {ratio}"
        ),
        f"Steps: {T}",
        f"Batch: {B}, Slots: {S}, Hidden dim: {D}",
    ]
    text = "\n".join(summary_lines)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
    )

    return fig


# =============================================================================
def _validate_extract(trace: TraceTree, key: str) -> np.ndarray:
    """Extract and validate a rank-4 dense array from the trace."""
    arr = trace.get(key)
    if arr.ndim != 4:
        raise ValueError(
            f"Expected {key!r} with rank 4 (T, B, S, D), "
            f"got shape {arr.shape}"
        )
    return arr
