from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.plots.rasterplot import plot_activation, plot_observations
from ehc_sn.figures.registry import FigureContext
from ehc_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a single-frequency LEC detail view from semantic rollout diagnostics."""
    return FeatCellsTimeseries(trace, ctx).plot()


class FeatCellsTimeseries(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the LEC overview."""

    HEIGHT_FRAC: float = 0.40
    MOSAIC = [["raster_1"], ["raster_2"], ["raster_3"], ["raster_4"]]
    SHAREX: bool = True

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("diagnostic/lec/cells", self.ctx.freq_idx)

        self.observations = self.trace.get("world_step/observation")[:, self.env_idx]
        self.cell_series = self.trace.get(f"diagnostic/lec/cells/{self.freq_idx}")[:, self.env_idx, :]
        self.mean_activity = np.mean(self.cell_series, axis=-1)
        self.peak_activity = np.max(self.cell_series, axis=-1)

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_1(self, ax: Axes) -> None:
        """Plot observation tokens over time."""
        plot_observations(ax, self.observations)
        ax.set_title("Observations")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_2(self, ax: Axes) -> None:
        """Plot semantic LEC cell activations over time."""
        plot_activation(ax, self.cell_series, vmin=0.0, vmax=1.0, cmap="GnBu")
        ax.set_title(f"LEC cells - Freq {self.freq_idx}")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_3(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        plot_activation(ax, self.cell_series)
        ax.set_title("Filtered activations (before ponderation)")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_4(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        plot_activation(ax, self.filtered_series)
        ax.set_title("Filtered activations (before ponderation)")
