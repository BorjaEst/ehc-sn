"""Place-cell diagnostic figure with geometry-aware spatial diagnostics."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.modules._spatial import spatial_rate_smooth_sigma
from ehc_sn.figures.plots.autocorr import plot_radial_autocorrelogram_profile, plot_spatial_autocorrelogram_mosaic
from ehc_sn.figures.plots.ratemap import plot_rate_map_mosaic, prepare_rate_maps
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import mosaic_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a 2x5 HPC place-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return PlaceCellsAutocorr(trace, ctx).plot()


class PlaceCellsAutocorr(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the place-cell overview."""

    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [1.0, 2.5]}
    MOSAIC = [
        ["map_labels", "spatial_maps"],
        ["matrices_labels", "spatial_matrices"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("diagnostic/hpc/location_mean", self.ctx.freq_idx)
        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = self.trace.get(f"diagnostic/hpc/location_mean/{self.freq_idx}")[:, self.env_idx, :]
        self._prepared_rate_maps = prepare_rate_maps(
            self.world,
            self.cells,
            self.location_ids,
            smooth_sigma=spatial_rate_smooth_sigma(self.world),
        )

    @panel()
    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist(), background_shape="square")
        ax.set_title("Trajectory colored by time")

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()
    def spatial_maps(self, ax: Axes) -> None:
        """Plot a HPC rate map for all cells.

        Args:
            ax: Axes to draw into.
        """
        axes = mosaic_axes(ax, self.cells.shape[-1], wspace=0.01, hspace=0.01)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_rate_map_mosaic(axes_list, self._prepared_rate_maps, vmin=0.0, vmax=0.10)
        ax.set_title(f"HPC f{self.freq_idx} rate map")

    @panel()
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial spatial-diagnostic summaries for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_radial_autocorrelogram_profile(ax, self._prepared_rate_maps)
        ax.set_title("Mean radial autocorrelogram (±1 std)")
        ax.yaxis.set_label_position("right")

    @colorbar(group="autocorr", label="Spatial autocorr")
    @panel()
    def spatial_matrices(self, ax: Axes) -> None:
        """Plot spatial autocorrelation for all cells.

        Args:
            ax: Axes to draw into.
        """
        axes = mosaic_axes(ax, self.cells.shape[-1], wspace=0.01, hspace=0.01)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_spatial_autocorrelogram_mosaic(axes_list, self._prepared_rate_maps, vmin=-1.0, vmax=1.0)
        ax.set_title(f"HPC f{self.freq_idx} spatial autocorr")
