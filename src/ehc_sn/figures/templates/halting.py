"""Halting timeline figure — bounded online diagnostic for ACT/RL paradigms.

Shows a binary heatmap of ``act/halted`` over time for each batch sample.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a binary halting heatmap from a bounded diagnostic trace.

    Args:
        trace: Trace containing ``act/halted`` numeric leaf shaped ``(T, B)``.
        ctx: Figure context (uses ``max_items``).

    Returns:
        Matplotlib ``Figure`` with one colour-mapped axis.
    """
    halted = trace.get("act/halted")  # (T, B)
    if halted.ndim != 2:
        raise ValueError(
            f"act/halted must be 2-D (T, B), got shape {halted.shape}."
        )

    T, B = halted.shape
    max_samples = ctx.max_items or B
    n = min(B, max_samples)
    data = halted[:, :n].T  # (n, T) for imshow

    fig = Figure(figsize=(max(3, n * 1.2), 2.5), layout="constrained")
    ax: Axes = fig.subplots()

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Sample")
    ax.set_title("Halting signal over time")

    if n > 1:
        ax.set_yticks(range(n))
        ax.set_yticklabels([str(i) for i in range(n)])

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Halted")

    return fig
