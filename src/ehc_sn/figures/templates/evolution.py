"""MazeHard prediction evolution figure template."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.plots.mazehard import plot_maze_with_overlay
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mazehard import EvolutionFigureData, select_evolution
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.figures.utils.grids import reshape_grid
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return PredictionEvolutionFigure(select_evolution(trace, ctx), ctx).plot()


class PredictionEvolutionFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.20
    MOSAIC = [["label", "evolution"]]
    MOSAIC_KWARGS = {"width_ratios": [1.0, 5.0], "gridspec_kw": {"wspace": 0.05}}

    def __init__(self, data: EvolutionFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @panel()
    def label(self, ax: Axes) -> None:
        input_grid = reshape_grid(self.data.input_ids[self.data.sample_idx])
        gt_overlay = reshape_grid(self.data.gt_overlay)
        plot_maze_with_overlay(ax, input_grid, gt_overlay, title="GT")

    @panel()
    def evolution(self, ax: Axes) -> None:
        axs = subdivide_axes(ax, nrows=2, ncols=8, wspace=0.02)
        input_grid = reshape_grid(self.data.input_ids[self.data.sample_idx])
        for ax_i, t in zip(axs.ravel(), self.data.t_indices):
            overlay = reshape_grid(self.data.pred_is_o[t, self.data.sample_idx])
            title = f"t={t}" + (" (halt)" if t == self.data.t_halt else "")
            plot_maze_with_overlay(ax_i, input_grid, overlay, title=title)
        for ax_i in axs.ravel()[len(self.data.t_indices):]:
            ax_i.axis("off")
