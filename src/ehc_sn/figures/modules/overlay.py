"""MazeHard figure: ground-truth vs model overlays.

This module renders a two-row panel where each column corresponds to one maze
example:

- Row 1 (**Ground truth**): input maze with a mask derived from labels.
- Row 2 (**Model**): input maze with a mask taken from the model prediction at
    the sample's first halt step.

Data requirements
-----------------

The rollout `TraceTree` metadata must provide:

- ``input_ids``: array-like of shape ``[B, N]`` (flattened grid tokens)
- ``labels``: array-like of shape ``[B, N]`` (flattened target tokens)

The rollout `TraceTree` must provide:

- ``act/halted``: array-like of shape ``[T, B]`` (boolean-like)
- ``pred/is_o``: array-like of shape ``[T, B, N]`` (model overlay score / probability)

Notes
-----

- The displayed batch size is controlled by :attr:`OverlayFigure.N_MAZES`
    (default: 6 columns).
- For each sample ``b``, the model overlay is taken at
    ``t_halt = first_halt_index(halted[:, b])``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.data.schema import O_ID
from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import panel
from ehc_sn.figures.plots.mazehard import plot_maze_with_overlay
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.figures.utils.grids import first_halt_index, reshape_grid
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Entry point used by the figure registry.

    Args:
        trace: Rollout trace containing predictions and halting signals.
        ctx: Figure context carrying render-time selection options.

    Returns:
        A matplotlib `Figure`.
    """
    return OverlayFigure(trace, ctx).plot()


class OverlayFigure(BaseFigureTemplate):
    """Figure template that plots GT overlays above model overlays.

    The layout is controlled by `MOSAIC`, with `labels_maps` on the first row and
    `solutions_maps` on the second row.
    """

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["labels_maps"], ["solutions_maps"]]
    N_PANELS: int = 5  # fixed panel count (developer-configurable)

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.input_ids, self.labels = _get_required_metadata(trace)
        self.gt_overlays = _select_gt_overlays(self.labels)
        self.model_overlays = _select_model_overlays(trace)

    @panel()
    def labels_maps(self, ax: Axes) -> None:
        """Plot the top row: ground-truth overlay masks."""
        axs = subdivide_axes(ax, nrows=1, ncols=self.N_PANELS)
        for ax_i, input_grid, gt_grid in zip(axs[0], self.input_ids, self.gt_overlays):
            input_grid, gt_grid = reshape_grid(input_grid), reshape_grid(gt_grid)
            plot_maze_with_overlay(ax_i, input_grid, gt_grid)

    @panel()
    def solutions_maps(self, ax: Axes) -> None:
        """Plot the bottom row: model overlay masks at the halt step."""
        axs = subdivide_axes(ax, nrows=1, ncols=self.N_PANELS)
        for ax_i, input_grid, model_grid in zip(axs[0], self.input_ids, self.model_overlays):
            input_grid, model_grid = reshape_grid(input_grid), reshape_grid(model_grid)
            plot_maze_with_overlay(ax_i, input_grid, model_grid)


def _get_required_metadata(trace: TraceTree) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate `input_ids` and `labels` from trace metadata.

    The figure only needs a small batch for visualization. We cap the returned
    arrays to 10 items to avoid accidentally rendering very wide figures.
    """
    input_ids_arr = np.asarray(trace.get_meta_path("input_ids"))
    labels_arr = np.asarray(trace.get_meta_path("labels"))
    return input_ids_arr, labels_arr


def _select_model_overlays(trace: TraceTree) -> np.ndarray:
    """Select per-sample model overlay masks at the first halt time.

    The trace is expected to provide:

    - `act/halted`: boolean-like array of shape `[T, B]`
    - `pred/is_o`: array of shape `[T, B, N]`, where `N` is the flattened grid size

    For each batch item `b`, we pick `t_halt = first_halt_index(halted[:, b])` and
    return `pred/is_o[t_halt, b]`.

    Max number of batch items is capped to 10 to limit figure width.
    """
    halted = np.asarray(trace.get("act/halted"))
    pred_is_o = np.asarray(trace.get("pred/solution_overlay"))
    if halted.ndim != 2:
        raise ValueError("act/halted must have shape [T, B]")
    if pred_is_o.ndim != 3:
        raise ValueError("pred/is_o must have shape [T, B, N]")

    overlays = []
    for b in range(min(10, halted.shape[1])):
        t_halt = first_halt_index(halted[:, b])
        overlays.append(pred_is_o[t_halt, b])
    return np.stack(overlays, axis=0)


def _select_gt_overlays(labels: np.ndarray) -> np.ndarray:
    """Return a boolean mask where labels equal the overlay token (`O_ID`)."""
    return labels == O_ID
