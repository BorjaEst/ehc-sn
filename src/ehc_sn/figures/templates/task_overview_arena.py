"""Arena task overview figure template.

Two-row figure (meta keys only, no dense traces):
    Row 0: Environment map (obs IDs) | Agent trajectory | Summary.
    Row 1: (empty)                    | Start/end/revisit legend | Summary.

Refactored to inherit ``TaskOverviewTemplate`` — owns layout, annotation
slots, and three-zone summary.  This template owns only the task-specific
visual encoding and summary text.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import AnyWorld, CategoricalLegend, ColorStrip
from ehc_sn.figures.core.task_overview import TaskOverviewTemplate
from ehc_sn.figures.plots.map import plot_map
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.arena_task import (
    ArenaTaskOverviewData,
    select_arena_task_overview,
)
from ehc_sn.figures.utils.axes import _environment_locations
from ehc_sn.traces.trace_tree import TraceTree

# ── Public entry point ──────────────────────────────────────────────────────


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render the Arena task overview figure from a persisted eval trace.

    Args:
        trace: ``TraceTree`` containing ``arena/*`` keys.
        ctx: Figure context for styling.

    Returns:
        Matplotlib ``Figure`` with three panels.
    """
    data = select_arena_task_overview(trace, ctx)
    return ArenaTaskLayoutFigure(data, ctx).plot()


# ── Figure template ─────────────────────────────────────────────────────────


class ArenaTaskLayoutFigure(TaskOverviewTemplate):
    """Arena task-overview: env map → agent trajectory → summary."""

    _TRAJECTORY_LW: float = 2.0

    # ── Visual encoding (task-owned) ────────────────────────────────────

    def render_input(self, ax: Axes) -> None:
        """Panel (a): Dungeon topology with per-cell observation IDs."""
        data: ArenaTaskOverviewData = self.data
        world: AnyWorld = data.world

        values = data.observation_ids.ravel().astype(float)
        values[values <= 0] = np.nan  # wall cells → background

        n_obs = int(data.observation_ids.max())
        obs_cmap = _build_observation_cmap(max(n_obs, 2))

        plot_map(
            world,
            values,
            ax=ax,
            shape="square",
            location_cm=obs_cmap,
            num_cols=n_obs,
            vmin=1,
            vmax=n_obs,
            radius=0.75,
        )
        ax.set_title("(a) Input — Observation IDs", fontsize=7)
        ax.axis("off")

    def render_target(self, ax: Axes) -> None:
        """Panel (b): Agent trajectory with start, end, and revisit rings."""
        data: ArenaTaskOverviewData = self.data
        world: AnyWorld = data.world
        location_ids = data.trajectory_locations.tolist()

        plot_time_colored_trajectory(
            ax,
            world,
            location_ids,
            cmap="plasma",
            show_endpoints=False,
            background_shape="square",
            line_width=self._TRAJECTORY_LW,
        )

        _overlay_endpoints(ax, world, location_ids)
        _overlay_revisit_rings(
            ax, world, data.trajectory_locations, data.revisit_mask
        )

        ax.set_title("(b) Target — Agent Trajectory", fontsize=7)

    # ── Annotation slots ────────────────────────────────────────────────

    def annotation_for_input(self) -> ColorStrip:
        """Observation-ID colour strip — compact bar row to fit annotation slot."""
        data: ArenaTaskOverviewData = self.data
        n_obs = int(data.observation_ids.max())
        cmap = _build_observation_cmap(max(n_obs, 2))
        entries: list[tuple[str, str]] = []
        for obs_id in range(1, n_obs + 1):
            rgba = cmap(obs_id)
            hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            entries.append((str(obs_id), hex_color))
        return ColorStrip(entries=entries, rows=2)

    def annotation_for_target(self) -> CategoricalLegend:
        return CategoricalLegend(
            entries=[
                ("start", "forestgreen"),
                ("end", "firebrick"),
                ("revisit", "gold"),
            ],
            ncol=3,
        )

    def summary_title(self) -> str:
        return "Arena Replay Task"

    # ── Summary content ─────────────────────────────────────────────────

    def objective_text(self) -> list[str]:
        return [
            "Navigates a dungeon;",
            "predicts next obs at each step.",
            "Revisit accuracy is key memory",
            "diagnostic.",
        ]

    def sample_rows(self) -> list[tuple[str, str]]:
        data: ArenaTaskOverviewData = self.data
        n_steps = len(data.trajectory_locations)
        n_revisits = int(data.revisit_mask.sum())
        n_obs = int(data.observation_ids.max())
        H, W = data.wall_mask.shape
        n_passable = int(data.wall_mask.sum())
        revisit_pct = 100.0 * n_revisits / max(n_steps, 1)
        return [
            ("Grid:", f"{H}\u2009\u00d7\u2009{W}, {n_passable} passable"),
            ("Unique obs:", str(n_obs)),
            ("Steps:", str(n_steps)),
            ("Revisits:", f"{n_revisits} ({revisit_pct:.0f}%)"),
        ]

    def contract_notation(self) -> str:
        return r"$(o_t, a_t)_{1:T}\;\longrightarrow\;" r"\text{learned memory}$"


# ── Internal helpers ────────────────────────────────────────────────────────


def _build_observation_cmap(n_obs: int) -> ListedColormap:
    """Build a qualitative colormap with *n_obs* distinguishable colours.

    Stacks ``tab20``, ``tab20b``, and ``tab20c`` (60 colours total) and
    returns the first *n_obs* entries so every observation ID maps to a
    visually distinct category.
    """
    colours: list = []
    for base in ("tab20", "tab20b", "tab20c"):
        cmap = mpl_colormaps[base]
        for i in range(cmap.N):
            colours.append(cmap(i))
    return ListedColormap(colours[:n_obs], name=f"arena_obs_{n_obs}")


def _trajectory_coords(world: AnyWorld, location_ids: list[int]) -> np.ndarray:
    """Return ``(T, 2)`` trajectory coordinates in ``(o, y)`` order."""
    from ehc_sn.figures.utils.axes import _environment_locations

    locations = _environment_locations(world)
    coords = []
    for loc_id in location_ids:
        if 0 <= loc_id < len(locations):
            loc = locations[loc_id]
            coords.append([loc["o"], loc["y"]])
    return np.asarray(coords, dtype=float)


def _overlay_endpoints(
    ax: Axes,
    world: AnyWorld,
    location_ids: list[int],
) -> None:
    """Large, legible start (green ▲) and end (red ■) markers.

    Placed at the first and last positions of *location_ids* with a white
    stroke so they remain visible against the trajectory line and background
    map.
    """
    coords = _trajectory_coords(world, location_ids)
    if coords.shape[0] == 0:
        return

    # Start — green triangle.
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        marker="^",
        s=30,
        color="forestgreen",
        edgecolor="white",
        linewidths=0.0,
        zorder=10,
        label="start",
    )

    # End — red square.
    if coords.shape[0] > 1:
        ax.scatter(
            coords[-1, 0],
            coords[-1, 1],
            marker="s",
            s=20,
            color="firebrick",
            edgecolor="white",
            linewidths=0.0,
            zorder=10,
            label="end",
        )


def _overlay_revisit_rings(
    ax: Axes,
    world: AnyWorld,
    trajectory_locations: np.ndarray,
    revisit_mask: np.ndarray,
) -> None:
    """Hollow gold rings at revisit steps.

    Drawn as unfilled circles with a visible gold stroke above the
    trajectory line but below endpoint markers (zorder=9).  A hollow
    ring communicates "this location was visited before" without
    occluding the time-coloured trajectory segment at that position.
    """
    locations = _environment_locations(world)
    revisit_coords: list[list[float]] = []
    for i, loc_id in enumerate(trajectory_locations.tolist()):
        if not revisit_mask[i]:
            continue
        if 0 <= loc_id < len(locations):
            loc = locations[loc_id]
            revisit_coords.append([loc["o"], loc["y"]])

    if revisit_coords:
        coords = np.asarray(revisit_coords, dtype=float)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            marker="o",
            s=25,
            facecolor="none",
            edgecolor="gold",
            linewidths=0.4,
            zorder=9,
            label="revisit",
        )


__all__ = ["ArenaTaskLayoutFigure", "plot"]
