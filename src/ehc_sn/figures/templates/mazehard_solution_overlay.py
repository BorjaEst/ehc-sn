"""MazeHard mazehard_solution_overlay figure template: ground-truth vs model overlays."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.plots.mazehard import plot_maze_with_overlay
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mazehard import (
    MazehardSolutionOverlayFigureData,
    select_overlay,
)
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.figures.utils.grids import reshape_grid
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MazehardSolutionOverlayFigure(select_overlay(trace, ctx), ctx).plot()


class MazehardSolutionOverlayFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["labels_maps"], ["solutions_maps"]]
    N_PANELS: int = 5

    def __init__(
        self, data: MazehardSolutionOverlayFigureData, ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    @panel()
    def labels_maps(self, ax: Axes) -> None:
        axs = subdivide_axes(ax, nrows=1, ncols=self.N_PANELS)
        for ax_i, input_grid, gt_grid in zip(
            axs[0], self.data.input_ids, self.data.gt_overlays
        ):
            plot_maze_with_overlay(
                ax_i, reshape_grid(input_grid), reshape_grid(gt_grid)
            )

    @panel()
    def solutions_maps(self, ax: Axes) -> None:
        axs = subdivide_axes(ax, nrows=1, ncols=self.N_PANELS)
        for ax_i, input_grid, model_grid in zip(
            axs[0], self.data.input_ids, self.data.model_overlays
        ):
            plot_maze_with_overlay(
                ax_i, reshape_grid(input_grid), reshape_grid(model_grid)
            )
