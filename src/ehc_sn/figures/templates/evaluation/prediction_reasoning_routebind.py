"""Routebind prediction-reasoning figure.

Two-row figure:
    Row 0: Target trajectory field | 2×8 snapshot mosaic of predicted fields.
    Row 1: Continuous colorbar | Continuous colorbar.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import ContinuousScale
from ehc_sn.figures.core.prediction_reasoning import PredictionReasoningTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.routebind import render_routebind_trajectory
from ehc_sn.figures.selectors.routebind_prediction_reasoning import (
    RoutebindPredictionReasoningData,
    select_routebind_prediction_reasoning,
)
from ehc_sn.traces.trace_tree import TraceTree

# Shared normalisation — same scale for GT and all snapshots.
_ROUTEBIND_NORM = Normalize(vmin=0.0, vmax=1.0)
_ROUTEBIND_CMAP = "Blues"

# Title font sizes — GT is full-width, snapshots are miniature cells.
_GT_TITLE_FONTSIZE: float = 7.0
_SNAPSHOT_TITLE_FONTSIZE: float = 5.5


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the Routebind prediction-reasoning figure."""
    data = select_routebind_prediction_reasoning(trace, ctx)
    return RoutebindPredictionReasoningFigure(data, ctx).plot()


class RoutebindPredictionReasoningFigure(PredictionReasoningTemplate):
    """Routebind prediction-reasoning: target trajectory field → per-step mosaic."""

    def render_ground_truth(self, ax: Axes) -> ScalarMappable:
        return render_routebind_trajectory(
            ax,
            self.data.target_trajectory,
            self.data.task_sample.grid_width,
            title="Target — Oracle trajectory",
            title_fontsize=_GT_TITLE_FONTSIZE,
            cmap=_ROUTEBIND_CMAP,
            norm=_ROUTEBIND_NORM,
            cell_type=self.data.task_sample.cell_type,
            start_flag=self.data.task_sample.start_flag,
            goal_flag=self.data.task_sample.goal_flag,
        )

    def render_snapshot(
        self,
        ax: Axes,
        snapshot_data: object,
        *,
        step_label: str,
        metric_label: str | None,
    ) -> ScalarMappable:
        return render_routebind_trajectory(
            ax,
            np.asarray(snapshot_data),
            self.data.task_sample.grid_width,
            title=step_label,
            title_fontsize=_SNAPSHOT_TITLE_FONTSIZE,
            cmap=_ROUTEBIND_CMAP,
            norm=_ROUTEBIND_NORM,
        )

    def annotation_for_gt(self) -> ContinuousScale:
        return ContinuousScale(
            label="Activation",
            vmin=0.0,
            vmax=1.0,
        )

    def _metric_for_snapshot(self, index: int) -> str | None:
        if index < len(self.data.field_mses):
            return f"MSE={self.data.field_mses[index]:.3f}"
        return None


__all__ = ["RoutebindPredictionReasoningFigure", "plot"]
