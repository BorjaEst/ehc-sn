"""MazeHard prediction-reasoning figure.

Two-row figure:
    Row 0: Ground-truth path overlay | 2×8 snapshot mosaic of predicted paths.
    Row 1: (hidden)                  | Categorical legend.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import CategoricalLegend
from ehc_sn.figures.core.prediction_reasoning import PredictionReasoningTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.mazehard import render_maze_path_field
from ehc_sn.figures.selectors.mazehard_prediction_reasoning import (
    MazehardPredictionReasoningData,
    select_mazehard_prediction_reasoning,
)
from ehc_sn.traces.trace_tree import TraceTree

# Title font sizes — GT is full-width, snapshots are miniature cells.
_GT_TITLE_FONTSIZE: float = 7.0
_SNAPSHOT_TITLE_FONTSIZE: float = 5.5


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the MazeHard prediction-reasoning figure."""
    data = select_mazehard_prediction_reasoning(trace, ctx)
    return MazehardPredictionReasoningFigure(data, ctx).plot()


class MazehardPredictionReasoningFigure(PredictionReasoningTemplate):
    """MazeHard prediction-reasoning: GT path → per-step prediction mosaic."""

    def render_ground_truth(self, ax: Axes) -> None:
        render_maze_path_field(
            ax,
            self.data.gt_overlay,
            title="Target — Solution path",
            title_fontsize=_GT_TITLE_FONTSIZE,
            input_ids=self.data.input_ids,
        )

    def render_snapshot(
        self,
        ax: Axes,
        snapshot_data: object,
        *,
        step_label: str,
        metric_label: str | None,
    ) -> None:
        render_maze_path_field(
            ax,
            snapshot_data,
            title=step_label,
            title_fontsize=_SNAPSHOT_TITLE_FONTSIZE,
        )

    def annotation_for_gt(self) -> CategoricalLegend:
        return CategoricalLegend(
            entries=[
                ("Background", "#f0f0f0"),
                ("Predicted path", "#e53e3e"),
            ],
            ncol=2,
        )

    def _metric_for_snapshot(self, index: int) -> str | None:
        if index < len(self.data.path_ious):
            return f"IoU={self.data.path_ious[index]:.3f}"
        return None


__all__ = ["MazehardPredictionReasoningFigure", "plot"]
