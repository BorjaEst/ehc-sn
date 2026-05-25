"""Dummy figure template — placeholder for wiring and rendering tests."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return DummyFigure(None, ctx).plot()


class DummyFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.25
    MOSAIC = [["fake_panel"]]

    @panel()
    def fake_panel(self, ax: Axes) -> None:
        ax.text(0.5, 0.5, "Dummy figure", ha="center", va="center", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
