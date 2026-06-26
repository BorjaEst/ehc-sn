"""Goaltrace task overview figure.

Two-row figure (meta keys only, no dense traces):
    Row 0: Oracle input weights (DAG) | Target prospective field (DAG) | Summary.
    Row 1: Continuous colorbar          | Continuous colorbar            | Summary.

Refactored to inherit ``TaskOverviewTemplate`` — owns layout, annotation
slots, and three-zone summary.  This template owns only the task-specific
visual encoding and summary text.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from matplotlib.pyplot import cm

from ehc_sn.figures._contracts import ContinuousScale
from ehc_sn.figures.core.task_overview import TaskOverviewTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.goaltrace import render_goaltrace_field
from ehc_sn.figures.selectors.goaltrace import (
    GoaltraceTaskOverviewFigureData,
    select_task_overview_goaltrace,
)
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return GoaltraceTaskOverviewFigure(
        select_task_overview_goaltrace(trace, ctx), ctx
    ).plot()


class GoaltraceTaskOverviewFigure(TaskOverviewTemplate):
    """Goaltrace task-overview: input weights → target field → summary."""

    # ── Visual encoding (task-owned) ────────────────────────────────────

    def render_input(self, ax: Axes) -> ScalarMappable:
        """Horizontal bar chart of oracle input weights, sorted desc."""
        values = np.asarray(self.data.weight)
        obs_ids = self.data.geometry.observation_ids
        norm = PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0)

        order = np.argsort(values)[::-1]
        sorted_vals = values[order]
        sorted_ids = obs_ids[order]
        colors = cm.YlOrBr(norm(sorted_vals))

        n = len(values)
        ax.barh(
            range(n), sorted_vals, height=0.65, color=colors, edgecolor="none"
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [str(int(i)) for i in sorted_ids],
            fontsize=4.5,
        )
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title("(a) Input — Oracle weights", fontsize=7)
        ax.tick_params(left=False)

        return ScalarMappable(norm=norm, cmap=cm.YlOrBr)

    def render_target(self, ax: Axes) -> ScalarMappable:
        """Horizontal bar chart of target prospective field, sorted desc."""
        return render_goaltrace_field(
            ax,
            self.data.geometry,
            np.asarray(self.data.target_field),
            cmap=cm.Blues,
            norm=PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0),
            title="(b) Target — Prospective field",
            node_size=320.0,
            label_fontsize=5.5,
        )

    # ── Annotation slots ────────────────────────────────────────────────

    def annotation_for_input(self) -> ContinuousScale:
        return ContinuousScale(
            label="Input weight",
            vmin=0.0,
            vmax=1.0,
        )

    def annotation_for_target(self) -> ContinuousScale:
        return ContinuousScale(
            label="Target activation",
            vmin=0.0,
            vmax=1.0,
        )

    def summary_title(self) -> str:
        return "Goaltrace Task"

    # ── Summary content ─────────────────────────────────────────────────

    def objective_text(self) -> list[str]:
        return [
            "Given graph, location,",
            "goal \u2192 firing field over",
            "all nodes. Distributed",
            "representation, not path.",
        ]

    def sample_rows(self) -> list[tuple[str, str]]:
        data: GoaltraceTaskOverviewFigureData = self.data
        cur_id = int(
            data.geometry.observation_ids[data.geometry.current_node_index]
        )
        goal_id = int(
            data.geometry.observation_ids[data.geometry.goal_node_index]
        )
        return [
            ("Valid nodes:", str(data.n_valid)),
            ("Current obs:", f"obs_{cur_id:02d}"),
            ("Goal obs:", f"obs_{goal_id:02d}"),
        ]

    def contract_notation(self) -> str:
        return (
            r"$(\mathbf{w}_t, i_t, i_{\text{goal}}, G)"
            r"\;\longrightarrow\; \mathbf{f}_t^*$"
        )


__all__ = ["GoaltraceTaskOverviewFigure", "plot"]
