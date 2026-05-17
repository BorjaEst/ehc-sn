"""HPC cells figure template."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.plots.autocorr import plot_radial_autocorrelogram_profile, plot_spatial_autocorrelogram_mosaic
from ehc_sn.figures.plots.ratemap import plot_rate_map_mosaic
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.hpc import HPCCellFigureData, select_hpc_cell
from ehc_sn.figures.utils.axes import mosaic_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return HPCCellsFigure(select_hpc_cell(trace, ctx), ctx).plot()


class HPCCellsFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [1.0, 2.5]}
    MOSAIC = [
        ["map_labels", "spatial_maps"],
        ["matrices_labels", "spatial_matrices"],
    ]

    def __init__(self, data: HPCCellFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @panel()
    def map_labels(self, ax: Axes) -> None:
        plot_time_colored_trajectory(ax, self.data.world, self.data.location_ids.tolist(), background_shape="square")
        ax.set_title("Trajectory colored by time")

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()
    def spatial_maps(self, ax: Axes) -> None:
        axes = mosaic_axes(ax, self.data.cells.shape[-1], wspace=0.01, hspace=0.01)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_rate_map_mosaic(axes_list, self.data.prepared_rate_maps, vmin=0.0, vmax=0.10)
        ax.set_title(f"HPC f{self.data.freq_idx} rate map")

    @panel()
    def matrices_labels(self, ax: Axes) -> None:
        plot_radial_autocorrelogram_profile(ax, self.data.prepared_rate_maps)
        ax.set_title("Mean radial autocorrelogram (±1 std)")
        ax.yaxis.set_label_position("right")

    @colorbar(group="autocorr", label="Spatial autocorr")
    @panel()
    def spatial_matrices(self, ax: Axes) -> None:
        axes = mosaic_axes(ax, self.data.cells.shape[-1], wspace=0.01, hspace=0.01)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_spatial_autocorrelogram_mosaic(axes_list, self.data.prepared_rate_maps, vmin=-1.0, vmax=1.0)
        ax.set_title(f"HPC f{self.data.freq_idx} spatial autocorr")
