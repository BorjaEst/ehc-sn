"""MazeHard task layout figure — task-context orientation for §7.

Three-panel horizontal figure (meta keys only, no dense traces):
    1. Input token grid (categorical).
    2. Target solution path (binary foreground).
    3. Task summary text.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mazehard import (
    MazehardTaskLayoutFigureData,
    select_task_layout,
)
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MazehardTaskLayoutFigure(select_task_layout(trace, ctx), ctx).plot()


class MazehardTaskLayoutFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["input", "target", "summary_text"]]
    MOSAIC_KWARGS = {"gridspec_kw": {"wspace": 0.08}}

    def __init__(
        self,
        data: MazehardTaskLayoutFigureData,
        ctx: FigureContext,
    ) -> None:
        super().__init__(data, ctx)

    # ── Panel A: input token grid ────────────────────────────────────────────
    @panel(slots=["input"])
    def input_panel(self, ax: Axes) -> None:
        grid = self.data.input_ids
        ax.imshow(
            grid,
            cmap="Set2",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(a) Input — Maze grid", fontsize=7)

    # ── Panel B: target path ─────────────────────────────────────────────────
    @panel(slots=["target"])
    def target_panel(self, ax: Axes) -> None:
        target = self.data.target_overlay.astype(float)
        ax.imshow(
            target,
            cmap=ListedColormap(["#f0f0f0", "#e53e3e"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(b) Target path", fontsize=7)

    # ── Panel C: task summary ────────────────────────────────────────────────
    @panel(slots=["summary_text"])
    def summary_panel(self, ax: Axes) -> None:
        data: MazehardTaskLayoutFigureData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        lines = [
            "Task objective",
            "",
            "Predict the solution-path overlay",
            "for the MazeHard input grid.",
            "",
            f"Case: {data.case_id}",
            f"Grid: {data.grid_shape[0]} x {data.grid_shape[1]}",
            f"Output cells: {data.grid_shape[0] * data.grid_shape[1]}",
            f"Target path cells: {data.target_path_cells}",
            f"Rollout steps: {data.rollout_steps}",
            "",
            "Prediction figures below compare",
            "model outputs to this target.",
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
