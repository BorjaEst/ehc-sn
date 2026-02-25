""" """

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hrm_sn.figures.figures.base import BaseFigureTemplate
from hrm_sn.figures.figures.panels import colorbar, panel
from hrm_sn.figures.registry import FigureContext
from hrm_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """ """
    return DummyFigure(trace, ctx).plot()


class DummyFigure(BaseFigureTemplate):
    """ """

    HEIGHT_FRAC: float = 0.25
    MOSAIC = [["fake_panel"]]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)

    # @panel("fake_panel")
    # def plot_panel(self, ax: Axes) -> None:
    @panel()
    def fake_panel(self, ax: Axes) -> None:
        """ """
        ax.text(0.5, 0.5, "Dummy figure", ha="center", va="center", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
