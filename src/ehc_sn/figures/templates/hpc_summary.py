"""HPC summary figure template."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.plots.ratemap import plot_rate_map_mosaic
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.hpc import HPCSummaryFigureData, select_hpc_summary
from ehc_sn.figures.utils.axes import mosaic_axes, subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return HPCSummaryFigure(select_hpc_summary(trace, ctx), ctx).plot()


class HPCSummaryFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.65
    MAX_SPATIAL_CELLS: int = 36
    MOSAIC_KWARGS = {"width_ratios": [2.2, 6.0], "height_ratios": [1.0, 1.0, 1.0]}
    MOSAIC = [
        ["map_labels", "spatial_matrices"],
        ["memory_panel_a", "spatial_matrices"],
        ["memory_panel_b", "spatial_matrices"],
    ]

    def __init__(self, data: HPCSummaryFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @panel()
    def map_labels(self, ax: Axes) -> None:
        plot_time_colored_trajectory(ax, self.data.world, self.data.location_ids.tolist(), background_shape="square")
        ax.set_title("Trajectory colored by time", loc="left")

    @colorbar(group="memory", label=None, tick_labelsize=6)
    @panel()
    def memory_panel_a(self, ax: Axes) -> None:
        ax.matshow(self.data.memory_g_cued, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_title("G-cued memory")
        ax.set_xlabel("Retrieved feature index")
        ax.set_ylabel("Cue feature index")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    @colorbar(group="memory", label=None, tick_labelsize=6)
    @panel()
    def memory_panel_b(self, ax: Axes) -> None:
        ax.matshow(self.data.memory_x_cued, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_title("X-cued memory")
        ax.set_xlabel("Retrieved feature index")
        ax.set_ylabel("Cue feature index")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    @panel()
    def spatial_matrices(self, ax: Axes) -> None:
        nrows = len(self.data.freq_idxs)
        shared_n_items = min(max(int(c.shape[-1]) for c in self.data.cells), self.MAX_SPATIAL_CELLS)
        axes = subdivide_axes(ax, nrows, 1, hspace=0.05, squeeze=True)
        freq_axes = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        for freq_idx, freq_ax in enumerate(freq_axes):
            sub_axes = mosaic_axes(freq_ax, shared_n_items, wspace=0.01, hspace=0.01)
            axes_list = list(np.ravel(sub_axes)) if isinstance(sub_axes, np.ndarray) else [sub_axes]
            rate_maps = self.data.prepared_rate_maps[freq_idx][:shared_n_items]
            plot_rate_map_mosaic(axes_list, rate_maps)
            freq_ax.set_title(f"Spatial rate maps - Freq {freq_idx}", fontsize=7)
