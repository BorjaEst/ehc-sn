"""Routebind task overview figure template.

Two-row figure (meta keys only, no dense traces):
    Row 0: Input layout | Oracle trajectory field | Summary.
    Row 1: Categorical legend | Continuous colorbar | Summary.

Refactored to inherit ''TaskOverviewTemplate'' — owns layout, annotation
slots, and three-zone summary.  This template owns only the task-specific
visual encoding and summary text.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as mpl_cmap
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import CategoricalLegend, ContinuousScale
from ehc_sn.figures.core.task_overview import TaskOverviewTemplate
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.renders.routebind import render_routebind_trajectory
from ehc_sn.figures.selectors.routebind import (
    RoutebindTaskOverviewData,
    select_task_overview_routebind,
)
from ehc_sn.traces.trace_tree import TraceTree

_WALL_COLOR = (0.25, 0.25, 0.25)
_FREE_COLOR = (0.92, 0.92, 0.92)
_OBS_CMAP = "tab20"


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    data = select_task_overview_routebind(trace, ctx)
    return RoutebindTaskOverviewFigure(data, ctx).plot()


class RoutebindTaskOverviewFigure(TaskOverviewTemplate):

    def render_input(self, ax: Axes) -> None:
        data: RoutebindTaskOverviewData = self.data
        W = data.grid_width
        H = W
        ct = data.cell_type.reshape(H, W)
        oid = data.observation_id.reshape(H, W)
        rgb = np.ones((H, W, 3), dtype=np.float32)
        rgb[ct == 0] = _WALL_COLOR
        rgb[ct == 1] = _FREE_COLOR
        obs_colors = _build_obs_colors(data.n_observations)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == 2:
                    o = int(oid[r, c])
                    if 0 <= o < len(obs_colors):
                        rgb[r, c] = obs_colors[o]
        ax.imshow(rgb, interpolation="nearest", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        sp = np.argmax(data.start_flag)
        sr, sc = divmod(int(sp), W)
        ax.plot(
            sc,
            sr,
            marker="*",
            color="#00cc44",
            markersize=14,
            markeredgewidth=0.8,
            markeredgecolor="#003300",
            zorder=5,
        )
        gp = np.where(data.goal_flag)[0]
        for g in gp:
            gr, gc = divmod(int(g), W)
            ax.plot(
                gc,
                gr,
                marker="o",
                color="#ffcc00",
                markersize=8,
                markeredgewidth=0.8,
                markeredgecolor="#664400",
                zorder=4,
                alpha=0.85,
            )
        ax.set_title("(a) Input — Spatial layout", fontsize=7)

    def render_target(self, ax: Axes) -> ScalarMappable:
        data: RoutebindTaskOverviewData = self.data
        return render_routebind_trajectory(
            ax,
            data.target_trajectory,
            data.grid_width,
            title="(b) Target — Oracle trajectory field",
        )

    def annotation_for_input(self) -> CategoricalLegend:
        return CategoricalLegend(
            entries=[
                ("Wall", "#404040"),
                ("Free", "#ebebeb"),
                ("Observation", "#888888"),
                ("Start", "#00cc44"),
                ("Goal", "#ffcc00"),
            ],
            ncol=3,
        )

    def annotation_for_target(self) -> ContinuousScale:
        return ContinuousScale(
            label="Trajectory activation", vmin=0.0, vmax=1.0
        )

    def summary_title(self) -> str:
        return "Routebind Task"

    def objective_text(self) -> list[str]:
        return [
            "Spatial layout + semantic",
            "goal observation \u2192 predict:",
            "  spatial trajectory field",
            "  semantic waypoint field",
        ]

    def sample_rows(self) -> list[tuple[str, str]]:
        data: RoutebindTaskOverviewData = self.data
        W = data.grid_width
        sp_pos = int(np.argmax(data.start_flag))
        start_r, start_c = divmod(sp_pos, W)
        start_obs = int(data.observation_id[sp_pos])
        goal_positions = np.where(data.goal_flag)[0]
        goal_obs = (
            int(data.observation_id[goal_positions[0]])
            if len(goal_positions) > 0
            else -1
        )
        waypoints = data.waypoint_obs_sequence
        ids = [str(o) for o in waypoints]
        if len(ids) > 5:
            waypoint_str = f"[{', '.join(ids[:5])}, …({len(ids)})]"
        else:
            waypoint_str = f"[{', '.join(ids)}]"
        return [
            ("Grid:", f"{W}\u2009\u00d7\u2009{W}"),
            (
                "Cells:",
                f"{int((data.cell_type==0).sum())}W "
                f"{int((data.cell_type==1).sum())}F "
                f"{int((data.cell_type==2).sum())}O",
            ),
            ("Obs. vocab:", str(data.n_observations)),
            ("Start obs:", f"id {start_obs} ({start_r},{start_c})"),
            ("Goal obs:", f"id {goal_obs}"),
            ("Waypoints:", waypoint_str),
        ]

    def contract_notation(self) -> str:
        return (
            r"$(\mathbf{X}_{\mathrm{space}},"
            r"\mathbf{G}_{\mathrm{semantic}}, s, o_g)"
            r"\;\longrightarrow\; \mathbf{f}_{\mathrm{trajectory}}$"
        )


def _build_obs_colors(n: int) -> list:
    if n == 0:
        return []
    cmap = mpl_cmap[_OBS_CMAP]
    return [
        (
            float(cmap(i % cmap.N)[0]),
            float(cmap(i % cmap.N)[1]),
            float(cmap(i % cmap.N)[2]),
        )
        for i in range(n)
    ]


__all__ = ["RoutebindTaskOverviewFigure", "plot"]
