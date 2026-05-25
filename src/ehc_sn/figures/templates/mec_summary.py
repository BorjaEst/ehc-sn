"""MEC summary figure template."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.plots.autocorr import plot_radial_autocorrelogram_profile, plot_spatial_autocorrelogram_mosaic
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mec import MECSummaryFigureData, select_mec_summary
from ehc_sn.figures.utils.axes import mosaic_axes, subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MECSummaryFigure(select_mec_summary(trace, ctx), ctx).plot()


class MECSummaryFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.65
    MAX_SPATIAL_CELLS: int = 36
    MOSAIC_KWARGS = {"width_ratios": [2.2, 6.0], "height_ratios": [1.0, 1.5, 0.5]}
    MOSAIC = [
        ["map_labels", "spatial_matrices"],
        ["matrices_labels", "spatial_matrices"],
        ["none", "spatial_matrices"],
    ]

    def __init__(self, data: MECSummaryFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @panel()
    def map_labels(self, ax: Axes) -> None:
        plot_time_colored_trajectory(ax, self.data.world, self.data.location_ids.tolist(), background_shape="square")
        ax.set_title("Trajectory colored by time", loc="left")

    @panel()
    def matrices_labels(self, ax: Axes) -> None:
        for rate_maps in self.data.prepared_rate_maps:
            plot_radial_autocorrelogram_profile(ax, rate_maps)
        ax.set_title("Radial autocorrelogram (±1 std)")
        ax.yaxis.tick_right()

    @panel()
    def spatial_matrices(self, ax: Axes) -> None:
        nrows = len(self.data.freq_idxs)
        shared_n_items = min(max(int(c.shape[-1]) for c in self.data.cells), self.MAX_SPATIAL_CELLS)
        axes = subdivide_axes(ax, nrows, 1, hspace=0.05, squeeze=True)
        freq_axes = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        for freq_idx, freq_ax in enumerate(freq_axes):
            rate_maps = self.data.prepared_rate_maps[freq_idx][:shared_n_items]
            axes_mosaic = mosaic_axes(freq_ax, shared_n_items, wspace=0.01, hspace=0.01)
            axes_list = list(np.ravel(axes_mosaic)) if isinstance(axes_mosaic, np.ndarray) else [axes_mosaic]
            plot_spatial_autocorrelogram_mosaic(axes_list, rate_maps, vmin=-1.0, vmax=1.0)
            freq_ax.set_title(f"Spatial autocorr - Freq {freq_idx}", fontsize=7)

    @panel()
    def none(self, ax: Axes) -> None:
        ax.axis("off")
