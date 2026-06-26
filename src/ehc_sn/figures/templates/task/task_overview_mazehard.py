"""MazeHard task layout figure — task-context orientation for §7.

Two-row figure (meta keys only, no dense traces):
    Row 0: Input token grid (categorical) | Target solution path (binary) | Summary.
    Row 1: Categorical legend              | (empty)                       | Summary.

Refactored to inherit ``TaskOverviewTemplate`` — owns layout, annotation
slots, and three-zone summary.  This template owns only the task-specific
visual encoding and summary text.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import CategoricalLegend
from ehc_sn.figures.core.task_overview import TaskOverviewTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.mazehard import render_maze_path_field
from ehc_sn.figures.selectors.mazehard import (
    MazehardTaskLayoutFigureData,
    select_task_layout,
)
from ehc_sn.traces.trace_tree import TraceTree

# ── Public entry point ─────────────────────────────────────────────────────


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the MazeHard task overview figure from a persisted eval trace."""
    return MazehardTaskLayoutFigure(select_task_layout(trace, ctx), ctx).plot()


# ── Figure template ────────────────────────────────────────────────────────


class MazehardTaskLayoutFigure(TaskOverviewTemplate):
    """MazeHard task-overview: input grid → target path → summary."""

    # ── Visual encoding (task-owned) ────────────────────────────────────

    def render_input(self, ax: Axes) -> None:
        """Categorical input token grid — wall/free/start/goal."""
        ax.imshow(
            self.data.input_ids,
            cmap="Set2",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(a) Input — Maze grid", fontsize=7)

    def render_target(self, ax: Axes) -> None:
        """Binary target solution path — red foreground (delegates to renderer)."""
        render_maze_path_field(
            ax,
            self.data.target_overlay,
            title="(b) Target — Solution path",
        )

    # ── Annotation slots (task-owned) ───────────────────────────────────

    def annotation_for_input(self) -> CategoricalLegend:
        return CategoricalLegend(
            entries=[
                ("Wall", "#1b1f24"),
                ("Free", "#f7f4ef"),
                ("Start", "#2b6cb0"),
                ("Goal", "#d69e2e"),
            ],
            ncol=4,
        )

    def annotation_for_target(self) -> None:
        return None  # binary path is self-explanatory

    def summary_title(self) -> str:
        return "MazeHard Task"

    # ── Summary content (task-owned) ────────────────────────────────────

    def objective_text(self) -> list[str]:
        return [
            "Predict shortest path",
            "overlay for the input",
            "maze grid.",
        ]

    def sample_rows(self) -> list[tuple[str, str]]:
        data: MazehardTaskLayoutFigureData = self.data
        return [
            ("Case:", data.case_id),
            (
                "Grid:",
                f"{data.grid_shape[0]}\u2009\u00d7\u2009{data.grid_shape[1]}",
            ),
            (
                "Output cells:",
                str(data.grid_shape[0] * data.grid_shape[1]),
            ),
            ("Target path cells:", str(data.target_path_cells)),
        ]

    def contract_notation(self) -> str:
        return (
            r"$(\mathrm{grid}, \mathrm{start}, \mathrm{goal})"
            r"\;\longrightarrow\; \mathrm{path}$"
        )


__all__ = ["MazehardTaskLayoutFigure", "plot"]
