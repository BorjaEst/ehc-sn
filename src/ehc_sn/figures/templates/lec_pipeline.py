"""LEC pipeline figure template."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.plots.raster import plot_activation, plot_observations
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.lec import LECPipelineFigureData, select_lec_pipeline
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return LECPipelineFigure(select_lec_pipeline(trace, ctx), ctx).plot()


class LECPipelineFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.40
    MOSAIC = [["raster_1"], ["raster_2"], ["raster_3"], ["raster_4"]]
    SHAREX: bool = True

    def __init__(self, data: LECPipelineFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def raster_1(self, ax: Axes) -> None:
        plot_observations(ax, self.data.observations)
        ax.set_title("Observations")

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def raster_2(self, ax: Axes) -> None:
        plot_activation(ax, self.data.cell_series, vmin=0.0, vmax=1.0, cmap="GnBu")
        ax.set_title(f"LEC cells - Freq {self.data.freq_idx}")

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def raster_3(self, ax: Axes) -> None:
        plot_activation(ax, self.data.cell_series)
        ax.set_title("LEC activations")

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def raster_4(self, ax: Axes) -> None:
        plot_activation(ax, self.data.filtered_series)
        ax.set_title("Filtered activations")
