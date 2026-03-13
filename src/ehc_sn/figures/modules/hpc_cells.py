"""Place-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.plots.autocorr import plot_autocorr_mosaic, plot_radial_autocorr_cells
from ehc_sn.figures.plots.ratemap import plot_ratematx_mosaic
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import mosaic_axes
from ehc_sn.rollouts.trace_tree import TraceTree


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
        self.freq_idx = self.trace.validate_freq_idx("state/hpc/location/mean", self.ctx.freq_idx)
        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = self.trace.get(f"state/hpc/location/mean/{self.freq_idx}")[:, self.env_idx, :]

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_maps(self, ax: Axes) -> None:
        """Plot a HPC rate map for all cells.

        Args:
            ax: Axes to draw into.
        """
        axes = mosaic_axes(ax, self.cells.shape[-1], wspace=0.04, hspace=0.04)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_ratematx_mosaic(axes_list, self.world, self.cells, self.location_ids.tolist(), vmin=0.0, vmax=0.10)
        ax.set_title(f"HPC f{self.freq_idx} rate map")

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_radial_autocorr_cells(ax, self.world, self.cells, self.location_ids)
        ax.set_title("Mean radial autocorr (±1 std)")

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_matrices(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for all cells.

        Args:
            ax: Axes to draw into.
        """
        axes = mosaic_axes(ax, self.cells.shape[-1], wspace=0.04, hspace=0.04)
        axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        plot_autocorr_mosaic(axes_list, self.world, self.cells, self.location_ids, vmin=-0.10, vmax=0.10)
        ax.set_title(f"HPC f{self.freq_idx} spatial autocorr")
