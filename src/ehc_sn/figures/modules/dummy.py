"""A minimal example/dummy figure module.

Used as a placeholder template for figure wiring and rendering.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the dummy figure."""
    return DummyFigure(trace, ctx).plot()


class DummyFigure(BaseFigureTemplate):
    """Single-panel dummy figure."""

    HEIGHT_FRAC: float = 0.25
    MOSAIC = [["fake_panel"]]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)

    # @panel("fake_panel")
    # def plot_panel(self, ax: Axes) -> None:
    @panel()
    def fake_panel(self, ax: Axes) -> None:
        """Render a single placeholder panel."""
        ax.text(0.5, 0.5, "Dummy figure", ha="center", va="center", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
