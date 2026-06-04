"""HRM H/L residual dynamics figure — detailed fixed-budget companion to pfc_latent_dynamics.

Three-panel figure:
    1. Forward residuals ||z[t] - z[t-1]|| over rollout steps.
    2. Consecutive-step cosine similarity.
    3. H/L residual separation ratio with mean annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

# ── Canonical trace path constants ───────────────────────────────────────────

TRACE_KEY_Z_H = "pfc/z_H"
TRACE_KEY_Z_L = "pfc/z_L"

# Colour scheme consistent with pfc_latent_dynamics.
H_COLOR = "tab:blue"
L_COLOR = "tab:orange"


# =============================================================================
def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a three‑panel H/L residual dynamics figure.

    Args:
        trace: Trace containing ``pfc/z_H`` and ``pfc/z_L``, both of
            shape ``(T, B, S, D)`` (time × batch × slots × hidden dim).
        ctx: Figure context (uses ``max_items`` to cap batch samples).

    Returns:
        Matplotlib ``Figure`` with three horizontally arranged panels.
    """
    return H_L_ResidualsOverStepsFigure(_extract_data(trace), ctx).plot()


# =============================================================================
class H_L_ResidualsOverStepsFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.25
    MOSAIC = [["residuals", "cosine_similarity", "separation"]]
    MOSAIC_KWARGS = {
        "width_ratios": [1.5, 1.5, 1.0],
    }

    def __init__(self, data: "_H_L_ResidualsData", ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # ── Panel A: forward residuals ───────────────────────────────────────────
    @panel()
    def residuals(self, ax: Axes) -> None:
        T = self.data.h_residual.shape[0]
        steps = np.arange(1, T + 1)  # residuals are indexed from step 1
        ax.plot(
            steps,
            self.data.h_residual,
            color=H_COLOR,
            label="H-state",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            self.data.l_residual,
            color=L_COLOR,
            label="L-state",
            linewidth=1.5,
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel(r"$||z_t - z_{t-1}||$")
        ax.set_title("Forward residuals")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)

    # ── Panel B: cosine similarity ───────────────────────────────────────────
    @panel()
    def cosine_similarity(self, ax: Axes) -> None:
        T = self.data.h_cosine.shape[0]
        steps = np.arange(1, T + 1)
        ax.plot(
            steps,
            self.data.h_cosine,
            color=H_COLOR,
            label="H-state",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            self.data.l_cosine,
            color=L_COLOR,
            label="L-state",
            linewidth=1.5,
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel(r"$\cos(z_t, z_{t-1})$")
        ax.set_title("Cosine similarity")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)

    # ── Panel C: H/L separation ratio ───────────────────────────────────────
    @panel()
    def separation(self, ax: Axes) -> None:
        T = self.data.separation.shape[0]
        steps = np.arange(1, T + 1)
        ax.plot(
            steps,
            self.data.separation,
            color="tab:purple",
            linewidth=1.5,
            label=r"$\Delta L \;/\; \Delta H$",
        )
        ax.axhline(
            y=1.0, color="gray", linestyle="--", linewidth=0.6, label="equal"
        )
        mean_ratio = float(np.mean(self.data.separation))
        ax.axhline(
            y=mean_ratio,
            color="tab:red",
            linestyle=":",
            linewidth=0.8,
            label=f"mean = {mean_ratio:.3f}",
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Ratio")
        ax.set_title("H/L separation")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)


# =============================================================================
@dataclass(frozen=True)
class _H_L_ResidualsData:
    """Internal data container for the residuals figure."""

    h_residual: np.ndarray  # (T-1,)
    l_residual: np.ndarray  # (T-1,)
    h_cosine: np.ndarray  # (T-1,)
    l_cosine: np.ndarray  # (T-1,)
    separation: np.ndarray  # (T-1,)
    n_steps: int
    n_batch: int
    n_slots: int
    n_hidden: int


# =============================================================================
def _extract_data(trace: TraceTree) -> _H_L_ResidualsData:
    """Extract and compute residual metrics from the trace."""
    z_H = _validate_extract(trace, TRACE_KEY_Z_H)
    z_L = _validate_extract(trace, TRACE_KEY_Z_L)

    T, B, S, D = z_H.shape

    # Forward residuals: mean over batch and slots.
    h_res = np.linalg.norm(z_H[1:] - z_H[:-1], axis=-1).mean(axis=(1, 2))
    l_res = np.linalg.norm(z_L[1:] - z_L[:-1], axis=-1).mean(axis=(1, 2))

    # Cosine similarity: mean over batch and slots.
    h_normed = z_H / (np.linalg.norm(z_H, axis=-1, keepdims=True) + 1e-12)
    l_normed = z_L / (np.linalg.norm(z_L, axis=-1, keepdims=True) + 1e-12)
    h_cos = (h_normed[1:] * h_normed[:-1]).sum(axis=-1).mean(axis=(1, 2))
    l_cos = (l_normed[1:] * l_normed[:-1]).sum(axis=-1).mean(axis=(1, 2))

    # H/L residual separation ratio.
    eps = 1e-12
    separation = l_res / (h_res + eps)

    return _H_L_ResidualsData(
        h_residual=h_res,
        l_residual=l_res,
        h_cosine=h_cos,
        l_cosine=l_cos,
        separation=separation,
        n_steps=T,
        n_batch=B,
        n_slots=S,
        n_hidden=D,
    )


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
