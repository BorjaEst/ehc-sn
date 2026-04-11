"""MazeHard figure: prediction evolution over rollout steps.

This figure renders a single MazeHard puzzle and shows how the model's binary
argmax overlay prediction (`pred/solution_overlay`) evolves over time. The first column
shows ground-truth overlay (`labels == O_ID`), followed by per-step predictions
up to the halt step.
"""

from __future__ import annotations

from typing import Iterable

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


# =================================================================================================
def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Entry point used by the figure registry."""
    return PredictionEvolutionFigure(trace, ctx).plot()


# =================================================================================================
class PredictionEvolutionFigure(BaseFigureTemplate):
    """Render GT + per-step argmax overlays for one MazeHard puzzle."""

    HEIGHT_FRAC: float = 0.20
    MOSAIC = [["label", "evolution"]]
    MOSAIC_KWARGS = {"width_ratios": [1.0, 5.0], "gridspec_kw": {"wspace": 0.05}}
    K_MAX: int = 16

    def __init__(  # ------------------------------------------------------------------------------
        self, trace: TraceTree, ctx: FigureContext,
    ) -> None:  # fmt: skip
        super().__init__(trace, ctx)
        self.input_ids, self.labels = _get_required_metadata(trace)
        self.pred_is_o, self.halted = _get_required_trace(trace)
        self.sample_idx = 0
        self.t_halt = first_halt_index(self.halted[:, self.sample_idx])
        self.t_indices = _select_timesteps(self.t_halt, self.K_MAX)

    @panel()
    def label(  # ---------------------------------------------------------------------------------
        self, ax: Axes,
    ) -> None:  # fmt: skip
        """Plot GT overlay for the selected sample."""
        input_grid = reshape_grid(self.input_ids[self.sample_idx])
        gt_overlay = reshape_grid(self.labels[self.sample_idx] == O_ID)
        plot_maze_with_overlay(ax, input_grid, gt_overlay, title="GT")

    @panel()
    def evolution(  # -----------------------------------------------------------------------------
        self, ax: Axes,
    ) -> None:  # fmt: skip
        """Plot GT + model overlays across selected rollout timesteps."""
        axs = subdivide_axes(ax, nrows=2, ncols=8, wspace=0.02)
        input_grid = reshape_grid(self.input_ids[self.sample_idx])

        for ax_i, t in zip(axs.ravel(), self.t_indices):
            overlay = reshape_grid(self.pred_is_o[t, self.sample_idx])
            title = f"t={t}"
            if t == self.t_halt:
                title = f"{title} (halt)"
            plot_maze_with_overlay(ax_i, input_grid, overlay, title=title)

        for ax_i in axs.ravel()[len(self.t_indices) :]:
            ax_i.axis("off")


# =================================================================================================
def _get_required_metadata(  # --------------------------------------------------------------------
    trace: TraceTree,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    return np.asarray(trace.get_meta_path("input_ids")), np.asarray(trace.get_meta_path("labels"))


# =================================================================================================
def _get_required_trace(  # -----------------------------------------------------------------------
    trace: TraceTree,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    halted = np.asarray(trace.get("act/halted"))
    pred_is_o = np.asarray(trace.get("pred/solution_overlay"))
    return pred_is_o, halted


# =================================================================================================
def _select_timesteps(t_halt: int, k_max: int) -> list[int]:
    if t_halt < 0:
        return [0]
    if t_halt + 1 <= k_max:
        return list(range(t_halt + 1))
    indices = np.linspace(0, t_halt, num=k_max, dtype=int)
    t_indices = sorted({int(idx) for idx in indices})
    if t_halt not in t_indices:
        t_indices.append(t_halt)
        t_indices.sort()
    return t_indices


# =================================================================================================
__all__ = ["plot", "PredictionEvolutionFigure"]
