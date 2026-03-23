"""Grid-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import panel
from ehc_sn.figures.plots.autocorr import plot_autocorr_mosaic, plot_radial_autocorr_cells
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import mosaic_axes, subdivide_axes
from ehc_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot a 2x5 MEC grid-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return GridCellsAutocorr(trace, ctx).plot()


class GridCellsAutocorr(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the grid-cell overview."""

    HEIGHT_FRAC: float = 0.65
    MAX_SPATIAL_CELLS: int = 36
    MOSAIC_KWARGS = {"width_ratios": [2.2, 6.0], "height_ratios": [1.0, 1.5, 0.5]}
    MOSAIC = [
        ["map_labels", "spatial_matrices"],
        ["matrices_labels", "spatial_matrices"],
        ["none", "spatial_matrices"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("diagnostic/mec/location_mean")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("diagnostic/mec/location_mean", f) for f in range(self.n_freq)]  # fmt: skip

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = [self.trace.get(f"diagnostic/mec/location_mean/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]  # fmt: skip

    @panel()
    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist(), background_shape="square")
        ax.set_title("Trajectory colored by time", loc="left")

    @panel()
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        for cells in self.cells:
            plot_radial_autocorr_cells(ax, self.world, cells, self.location_ids)
        ax.set_title("Radial autocorr (±1 std)")
        ax.yaxis.tick_right()

    @panel()
    def spatial_matrices(self, ax: Axes) -> None:
        nrows = len(self.freq_idxs)
        shared_n_items = min(max(int(cells.shape[-1]) for cells in self.cells), self.MAX_SPATIAL_CELLS)
        for freq_idx, freq_ax in enumerate(subdivide_axes(ax, nrows, 1, hspace=0.05, squeeze=True)):
            cells = self.cells[freq_idx]
            cell_indices = list(range(min(int(cells.shape[-1]), shared_n_items)))
            axes = mosaic_axes(freq_ax, shared_n_items, wspace=0.01, hspace=0.01)
            axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
            plot_autocorr_mosaic(axes_list, self.world, cells, self.location_ids, cell_indices=cell_indices)
            freq_ax.set_title(f"Spatial autocorr - Freq {freq_idx}", fontsize=7)

    @panel()
    def none(self, ax: Axes) -> None:
        """Empty panel to balance the mosaic."""
        ax.axis("off")
