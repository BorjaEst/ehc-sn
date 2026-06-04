"""HRM H/L latent dynamics figure — broad overview diagnostic.

Three-panel horizontal figure:
    1. State magnitude (L2 norm) over rollout steps.
    2. Update magnitude (L2 delta) over rollout steps.
    3. Summary annotation with scalar metrics.
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

# Colour scheme — consistent with h_l_residuals_over_steps.
H_COLOR = "tab:blue"
L_COLOR = "tab:orange"


# =============================================================================
def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a three‑panel H/L latent-dynamics overview figure.

    Args:
        trace: Trace containing ``pfc/z_H`` and ``pfc/z_L``, both of
            shape ``(T, B, S, D)`` (time × batch × slots × hidden dim).
        ctx: Figure context.

    Returns:
        Matplotlib ``Figure`` with three horizontally arranged panels.
    """
    data = _extract_data(trace)
    return PfcLatentDynamicsFigure(data, ctx).plot()


# =============================================================================
class PfcLatentDynamicsFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["norm", "delta", "summary"]]
    MOSAIC_KWARGS = {
        "width_ratios": [1.5, 1.5, 1.0],
    }

    def __init__(
        self, data: "PFCLatentDynamicsData", ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    # ── Panel A: state magnitude ─────────────────────────────────────────
    @panel(slots=["norm"])
    def norm_panel(self, ax: Axes) -> None:
        steps = self.data.steps
        ax.plot(
            steps,
            self.data.h_norm,
            color=H_COLOR,
            label=r"$z_H$",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            self.data.l_norm,
            color=L_COLOR,
            label=r"$z_L$",
            linewidth=1.5,
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel(r"Mean $||z||$")
        ax.set_title("State magnitude")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)

    # ── Panel B: update magnitude ────────────────────────────────────────
    @panel(slots=["delta"])
    def delta_panel(self, ax: Axes) -> None:
        steps = self.data.delta_steps
        ax.plot(
            steps,
            self.data.h_delta,
            color=H_COLOR,
            label=r"$\Delta z_H$",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            self.data.l_delta,
            color=L_COLOR,
            label=r"$\Delta z_L$",
            linewidth=1.5,
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel(r"Mean $||\Delta z||$")
        ax.set_title("Update magnitude")
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.3)

    # ── Panel C: summary ─────────────────────────────────────────────────
    @panel(slots=["summary"])
    def summary_panel(self, ax: Axes) -> None:
        data: PFCLatentDynamicsData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ratio = data.delta_ratio_l_over_h
        ratio_str = f"{ratio:.4f}" if np.isfinite(ratio) else str(ratio)

        lines = [
            r"Mean $||z_H||$" + f" = {data.h_norm_mean:.3f}",
            r"Mean $||z_L||$" + f" = {data.l_norm_mean:.3f}",
            r"Mean $||\Delta z_H||$" + f" = {data.h_delta_mean:.4f}",
            r"Mean $||\Delta z_L||$" + f" = {data.l_delta_mean:.4f}",
            r"$\Delta L \;/\; \Delta H$" + f" = {ratio_str}",
            f"Steps: {data.n_steps}, "
            f"Batch: {data.n_batch}, "
            f"Slots: {data.n_slots}",
        ]
        ax.text(
            0.08,
            0.92,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=5.5,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_title("H/L overview")


# =============================================================================
@dataclass(frozen=True)
class PFCLatentDynamicsData:
    """Prepared data for :class:`PfcLatentDynamicsFigure`."""

    steps: np.ndarray  # (T,) — 0 .. T-1
    delta_steps: np.ndarray  # (T-1,) — 1 .. T-1
    h_norm: np.ndarray  # (T,)
    l_norm: np.ndarray  # (T,)
    h_delta: np.ndarray  # (T-1,)
    l_delta: np.ndarray  # (T-1,)
    h_norm_mean: float
    l_norm_mean: float
    h_delta_mean: float
    l_delta_mean: float
    delta_ratio_l_over_h: float
    n_steps: int
    n_batch: int
    n_slots: int


# =============================================================================
def _extract_data(trace: TraceTree) -> PFCLatentDynamicsData:
    """Extract and compute norm/delta metrics from the trace."""
    z_H = _validate_extract(trace, TRACE_KEY_Z_H)
    z_L = _validate_extract(trace, TRACE_KEY_Z_L)

    T, B, S, D = z_H.shape

    # Mean over batch and slots for per-step norm.
    h_norm = np.linalg.norm(z_H, axis=-1).mean(axis=(1, 2))  # (T,)
    l_norm = np.linalg.norm(z_L, axis=-1).mean(axis=(1, 2))  # (T,)

    # Mean over batch and slots for per-step delta.
    h_delta = np.linalg.norm(z_H[1:] - z_H[:-1], axis=-1).mean(
        axis=(1, 2)
    )  # (T-1,)
    l_delta = np.linalg.norm(z_L[1:] - z_L[:-1], axis=-1).mean(
        axis=(1, 2)
    )  # (T-1,)

    h_norm_mean = float(h_norm.mean())
    l_norm_mean = float(l_norm.mean())
    h_delta_mean = float(h_delta.mean()) if len(h_delta) > 0 else 0.0
    l_delta_mean = float(l_delta.mean()) if len(l_delta) > 0 else 0.0
    ratio = (
        l_delta_mean / h_delta_mean
        if h_delta_mean > 0
        else (0.0 if l_delta_mean == 0.0 else float("inf"))
    )

    return PFCLatentDynamicsData(
        steps=np.arange(T),
        delta_steps=np.arange(1, T),
        h_norm=h_norm,
        l_norm=l_norm,
        h_delta=h_delta,
        l_delta=l_delta,
        h_norm_mean=h_norm_mean,
        l_norm_mean=l_norm_mean,
        h_delta_mean=h_delta_mean,
        l_delta_mean=l_delta_mean,
        delta_ratio_l_over_h=ratio,
        n_steps=T,
        n_batch=B,
        n_slots=S,
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
