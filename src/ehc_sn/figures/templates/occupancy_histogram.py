"""Occupancy histogram figure — bounded online summary for TEM/EHP spatial.

Wraps the existing ``render_occupancy_histogram`` as a ``bounded_trace``
figure that consumes a ``diagnostic/occupancy`` trace leaf.
"""

from __future__ import annotations

import torch
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext
from ehc_sn.metrics.renderers import render_occupancy_histogram
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render an occupancy histogram from a synthetic summary trace leaf.

    Args:
        trace: Trace containing ``diagnostic/occupancy`` numeric leaf
            shaped ``(N,)`` (normalized occupancy probabilities).
        ctx: Figure context (unused).

    Returns:
        Matplotlib ``Figure`` with one bar-chart axis.
    """
    grid = trace.get("diagnostic/occupancy")  # (N,)
    grid_t = torch.as_tensor(grid)
    return render_occupancy_histogram(
        grid_t, title="Occupancy (bounded summary)"
    )
