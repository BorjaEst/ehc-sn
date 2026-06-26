"""Goaltrace prediction-reasoning figure.

Two-row figure:
    Row 0: Target firing field | 2×8 snapshot mosaic of predicted fields.
    Row 1: Continuous colorbar | Continuous colorbar.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from matplotlib.pyplot import cm

from ehc_sn.figures._contracts import ContinuousScale
from ehc_sn.figures.core.prediction_reasoning import PredictionReasoningTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.goaltrace import render_goaltrace_field
from ehc_sn.figures.selectors.goaltrace_prediction_reasoning import (
    GoaltracePredictionReasoningData,
    select_goaltrace_prediction_reasoning,
)
from ehc_sn.traces.trace_tree import TraceTree

# Shared normalisation — same scale for GT and all snapshots.
_GOALTRACE_NORM = PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0)
_GOALTRACE_CMAP = cm.Blues

# Title font sizes — GT is full-width, snapshots are miniature cells.
_GT_TITLE_FONTSIZE: float = 7.0
_SNAPSHOT_TITLE_FONTSIZE: float = 5.5


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the Goaltrace prediction-reasoning figure."""
    data = select_goaltrace_prediction_reasoning(trace, ctx)
    return GoaltracePredictionReasoningFigure(data, ctx).plot()


class GoaltracePredictionReasoningFigure(PredictionReasoningTemplate):
    """Goaltrace prediction-reasoning: target field → per-step field mosaic."""

    def render_ground_truth(self, ax: Axes) -> ScalarMappable:
        return render_goaltrace_field(
            ax,
            self.data.geometry,
            np.asarray(self.data.target_field),
            cmap=_GOALTRACE_CMAP,
            norm=_GOALTRACE_NORM,
            title="Target — Prospective field",
            title_fontsize=_GT_TITLE_FONTSIZE,
            node_size=320.0,
            label_fontsize=5.5,
            mark_current=True,
            mark_goal=True,
        )

    def render_snapshot(
        self,
        ax: Axes,
        snapshot_data: object,
        *,
        step_label: str,
        metric_label: str | None,
    ) -> ScalarMappable:
        return render_goaltrace_field(
            ax,
            self.data.geometry,
            np.asarray(snapshot_data),
            cmap=_GOALTRACE_CMAP,
            norm=_GOALTRACE_NORM,
            title=step_label,
            title_fontsize=_SNAPSHOT_TITLE_FONTSIZE,
            node_size=220.0,
            label_fontsize=4.5,
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


__all__ = ["GoaltracePredictionReasoningFigure", "plot"]
