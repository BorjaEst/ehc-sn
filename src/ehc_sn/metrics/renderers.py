"""Pure renderer functions for reducer-backed diagnostic figures.

Each function consumes a computed Metric summary and returns a
``matplotlib.figure.Figure``.  They are stateless, do not depend on
``ehc_sn.figures`` or the ``FigureSpec`` registry, and must not save
figures to disk or log them — those are the caller's responsibilities.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor


# =============================================================================
def render_occupancy_histogram(  # --------------------------------------------
    grid: Tensor,
    *,
    title: str = "Occupancy",
    cmap: str = "Blues",
) -> Figure:
    """Render a 1-D occupancy probability distribution as a bar chart.

    Args:
        grid: normalized occupancy distribution, shape ``(n_locations,)``,
            ``float32``, sums to 1.0.
        title: Plot title.
        cmap: Matplotlib colormap name for bar colour.

    Returns:
        A ``Figure`` with one ``Axes``.  Caller owns persistence and
        ``plt.close(fig)``.
    """
    fig, ax = plt.subplots(figsize=(6, 3), layout="tight")
    n = grid.shape[0]
    colors = plt.get_cmap(cmap)(_bar_colour_values(n))
    ax.bar(range(n), grid.cpu().numpy(), color=colors, width=1.0)
    ax.set_xlabel("Location ID")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, max(float(grid.max()) * 1.15, 0.01))
    return fig


# =============================================================================
def render_hidden_norm_histogram(  # ------------------------------------------
    centers: Tensor,
    density: Tensor,
    *,
    title: str = "Hidden-state L2 norms",
    xlabel: str = "||h||\u2082",
) -> Figure:
    """Render a norm-density histogram as a bar chart.

    Args:
        centres: Bin-centre positions, shape ``(n_bins,)``, ``float32``.
        density: normalized per-bin density, shape ``(n_bins,)``, ``float32``,
            sums to 1.0.
        title: Plot title.
        xlabel: X-axis label.

    Returns:
        A ``Figure`` with one ``Axes``.  Caller owns persistence and
        ``plt.close(fig)``.
    """
    fig, ax = plt.subplots(figsize=(6, 3), layout="tight")
    width = float(centers[1] - centers[0]) if centers.shape[0] > 1 else 1.0
    ax.bar(
        centers.cpu().numpy(),
        density.cpu().numpy(),
        width=width,
        color="darkorange",
        align="center",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    return fig


# =============================================================================
def log_reducer_figure(  # ----------------------------------------------------
    logger: Any,
    tag: str,
    fig: Figure,
    *,
    global_step: int | None = None,
) -> None:
    """Log a reducer-backed figure and close it.

    Callers never import ``matplotlib`` or touch logger internals.
    ``None`` loggers result in a close-only no-op.

    Args:
        logger: A Lightning logger (``TensorBoardLogger``, etc.) or ``None``.
        tag: TensorBoard tag string.
        fig: Matplotlib ``Figure``.
        global_step: Optional global step for the x-axis.
    """
    if logger is not None:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=global_step)
    plt.close(fig)


# =============================================================================
def _bar_colour_values(  # ----------------------------------------------------
    n: int,
) -> list[float]:
    """Return evenly spaced values in ``[0.3, 0.9]`` for colour mapping."""
    if n <= 1:
        return [0.6]
    return [0.3 + 0.6 * i / (n - 1) for i in range(n)]


# =============================================================================
__all__ = [
    "log_reducer_figure",
    "render_hidden_norm_histogram",
    "render_occupancy_histogram",
]
