"""MazeHard prediction accuracy over steps figure template.

Three-panel horizontal figure:
    1. Token accuracy over recurrent rollout step.
    2. Target-path recall over recurrent rollout step.
    3. Summary annotation with initial / best / final values.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mazehard import (
    MazehardPredictionAccuracyFigureData,
    select_prediction_accuracy,
)
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MazehardPredictionAccuracyFigure(
        select_prediction_accuracy(trace, ctx), ctx
    ).plot()


class MazehardPredictionAccuracyFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.25
    MOSAIC = [
        ["token_accuracy", "path_recall", "summary"],
    ]
    MOSAIC_KWARGS = {
        "width_ratios": [1.5, 1.5, 1.0],
    }

    def __init__(
        self, data: MazehardPredictionAccuracyFigureData, ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    @panel()
    def token_accuracy(self, ax: Axes) -> None:
        """Panel A: token accuracy over steps for the selected batch element."""
        accuracy = self.data.token_accuracy  # (T, B)
        b = self.data.sample_idx

        line = accuracy[:, b]
        steps = np.arange(len(line))
        ax.plot(steps, line, color="tab:blue", linewidth=1.5)
        ax.axvline(
            x=self.data.best_step,
            color="tab:green",
            linestyle="--",
            linewidth=0.8,
            label=f"best (step {self.data.best_step})",
        )
        ax.axvline(
            x=self.data.final_step,
            color="tab:red",
            linestyle=":",
            linewidth=0.8,
            label=f"final (step {self.data.final_step})",
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Token accuracy")
        ax.set_title("Token accuracy over steps")
        ax.legend(fontsize="x-small", loc="lower right")
        ax.grid(True, alpha=0.3)

    @panel()
    def path_recall(self, ax: Axes) -> None:
        """Panel B: target-path recall over steps for the selected batch element."""
        recall = self.data.path_recall  # (T, B)
        b = self.data.sample_idx

        line = recall[:, b]
        steps = np.arange(len(line))
        ax.plot(steps, line, color="tab:orange", linewidth=1.5)
        ax.axvline(
            x=self.data.best_step_recall,
            color="tab:green",
            linestyle="--",
            linewidth=0.8,
            label=f"best (step {self.data.best_step_recall})",
        )
        ax.axvline(
            x=self.data.final_step,
            color="tab:red",
            linestyle=":",
            linewidth=0.8,
        )
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Path recall")
        ax.set_title("Target-path recall over steps")
        ax.legend(fontsize="x-small", loc="lower right")
        ax.grid(True, alpha=0.3)

    @panel()
    def summary(self, ax: Axes) -> None:
        """Panel C: compact annotation table with initial / best / final values."""
        data: MazehardPredictionAccuracyFigureData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        b = self.data.sample_idx
        acc = self.data.token_accuracy[:, b]
        rec = self.data.path_recall[:, b]
        best_step = self.data.best_step
        best_step_recall = self.data.best_step_recall

        initial_acc = acc[0]
        best_acc = acc[best_step]
        final_acc = acc[-1]
        initial_rec = rec[0]
        best_rec = rec[best_step_recall]
        final_rec = rec[-1]

        lines = [
            "Token accuracy",
            f"  initial: {initial_acc:.4f}",
            f"  best:    {best_acc:.4f} (step {best_step})",
            f"  final:   {final_acc:.4f}",
            "",
            "Path recall",
            f"  initial: {initial_rec:.4f}",
            f"  best:    {best_rec:.4f} (step {best_step_recall})",
            f"  final:   {final_rec:.4f}",
            "",
            f"Samples: {acc.shape[0]} steps",
        ]
        ax.text(
            0.08,
            0.92,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=5.5,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_title("Acc overview")
