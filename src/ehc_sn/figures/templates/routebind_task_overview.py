"""Routebind task overview figure template.

Three-panel horizontal layout:
    A. Input layout — cell types + observation IDs + start/goal markers.
    B. Oracle trajectory field — discounted spatial heatmap.
    C. Task summary text — stable prose + sample facts.

This is a **task-context figure**, not a model-diagnostic figure.
It renders from meta keys only (no dense traces) and answers
"what task is being evaluated?"
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.routebind import (
    RoutebindTaskOverviewData,
    select_routebind_task_overview,
)
from ehc_sn.traces.trace_tree import TraceTree

# ── Cell-type colours ────────────────────────────────────────────────────────
_WALL_COLOR = (0.25, 0.25, 0.25)
_FREE_COLOR = (0.92, 0.92, 0.92)
_OBS_CMAP = "tab20"

# ── Public entry point ───────────────────────────────────────────────────────


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the Routebind task overview figure from a persisted eval trace.

    Args:
        trace: ``TraceTree`` containing ``routebind/*`` keys.
        ctx: Figure context for styling.

    Returns:
        Matplotlib ``Figure`` with three panels.
    """
    data = select_routebind_task_overview(trace, ctx)
    return RoutebindTaskOverviewFigure(data, ctx).plot()


# ── Figure template ──────────────────────────────────────────────────────────


class RoutebindTaskOverviewFigure(BaseFigureTemplate):
    """Three-panel task-overview figure for routebind."""

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["input", "field", "summary"]]
    MOSAIC_KWARGS = {"gridspec_kw": {"wspace": 0.08}}

    def __init__(
        self,
        data: RoutebindTaskOverviewData,
        ctx: FigureContext,
    ) -> None:
        super().__init__(data, ctx)

    # ── Panel A: input layout ───────────────────────────────────────────────
    @panel(order=0)
    def input(self, ax: Axes) -> None:
        """Render the visible input grid."""
        data: RoutebindTaskOverviewData = self.data
        W = data.grid_width
        H = W
        ct = data.cell_type.reshape(H, W)
        oid = data.observation_id.reshape(H, W)

        rgb = np.ones((H, W, 3), dtype=np.float32)
        rgb[ct == 0] = _WALL_COLOR  # WALL
        rgb[ct == 1] = _FREE_COLOR  # FREE

        obs_colors = _build_obs_colors(data.n_observations)
        for r in range(H):
            for c in range(W):
                if ct[r, c] == 2:  # OBSERVATION
                    o = int(oid[r, c])
                    if 0 <= o < len(obs_colors):
                        rgb[r, c] = obs_colors[o]

        ax.imshow(rgb, interpolation="nearest", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        # Start marker — green star
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

        # Goal markers — gold circles
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

        # Legend
        legend_handles = [
            mpatches.Patch(color=_WALL_COLOR, label="Wall"),
            mpatches.Patch(color=_FREE_COLOR, label="Free"),
            mpatches.Patch(color="#888888", label="Observation"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=4.5,
            framealpha=0.85,
            borderpad=0.2,
        )
        ax.set_title("(a) Input layout", fontsize=7)

    # ── Panel B: oracle trajectory field ────────────────────────────────────
    @panel(order=1)
    @colorbar(group="field", label="Trajectory activation")
    def field(self, ax: Axes) -> None:
        """Render the oracle trajectory field heatmap."""
        data: RoutebindTaskOverviewData = self.data
        W = data.grid_width
        H = W
        tf = data.target_trajectory.reshape(H, W)

        im = ax.imshow(
            tf,
            cmap="Blues",
            norm=Normalize(vmin=0.0, vmax=1.0),
            interpolation="nearest",
            origin="upper",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(b) Oracle trajectory field", fontsize=7)
        return im

    # ── Panel C: summary text ───────────────────────────────────────────────
    @panel(order=2)
    def summary(self, ax: Axes) -> None:
        """Task description and sample statistics."""
        data: RoutebindTaskOverviewData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        W = data.grid_width
        n_walls = int((data.cell_type == 0).sum())
        n_free = int((data.cell_type == 1).sum())
        n_obs_cells = int((data.cell_type == 2).sum())
        n_goal_occurrences = int(data.goal_flag.sum())

        sp_pos = int(np.argmax(data.start_flag))
        start_r, start_c = divmod(sp_pos, W)
        start_obs = int(data.observation_id[sp_pos])

        tf_active = int((data.target_trajectory > 0.01).sum())

        lines = [
            "Routebind Task",
            "",
            f"Grid: {W}\u2009\u00d7\u2009{W}",
            f"Walls: {n_walls}  Free: {n_free}",
            f"Observation cells: {n_obs_cells}",
            f"Distinct obs IDs: {data.n_observations}",
            f"Goal occurrences: {n_goal_occurrences}",
            f"Start: ({start_r},{start_c}) obs={start_obs}",
            f"Traj field active: {tf_active} cells",
            "",
            "Given a spatial layout and a",
            "semantic goal, the model must",
            "predict two discounted fields:",
            "  (1) the spatial trajectory",
            "  (2) the semantic waypoints",
            "",
            "Spatial reachability is visible;",
            "observation transitions are hidden",
            "and must be learned.",
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


# ── Internal helpers ────────────────────────────────────────────────────────


def _build_obs_colors(n: int) -> list[tuple[float, float, float]]:
    """Build *n* distinguishable observation colours from qualitative colormap."""
    if n == 0:
        return []
    from matplotlib import colormaps as mpl_cmap

    cmap = mpl_cmap[_OBS_CMAP]
    colors: list[tuple[float, float, float]] = []
    for i in range(n):
        rgba = cmap(i % cmap.N)
        colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2])))
    return colors


__all__ = [
    "RoutebindTaskOverviewFigure",
    "plot",
]
