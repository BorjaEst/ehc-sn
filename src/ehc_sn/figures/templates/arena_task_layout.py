"""Arena task overview figure template.

A three-panel paradigm figure that explains the Arena evaluation task:

- **Panel (a) — Environment**: Renders the dungeon topology with per-cell
  observation IDs using ``plot_map()``.  Wall cells are shown in a neutral
  background colour, passable cells are coloured by categorical observation ID.

- **Panel (b) — Trajectory**: Renders the agent's trajectory through the
  environment using ``plot_time_colored_trajectory()`` with large start/end
  markers and hollow gold revisit rings.

- **Panel (c) — Summary**: Compact text panel with task statistics and
  paradigm description.

This is a **paradigm figure**, not a results figure.  It is rendered once
per regime (first valid case) and answers "what task is being evaluated?"
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ehc_sn.figures._contracts import AnyWorld
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.plots.map import plot_map
from ehc_sn.figures.plots.trajectory import plot_time_colored_trajectory
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.arena_task import (
    ArenaTaskOverviewData,
    select_arena_task_overview,
)
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


class ArenaTaskLayoutFigure(BaseFigureTemplate):
    """Three-panel paradigm figure for the Arena evaluation task."""

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["environment", "trajectory", "summary_text"]]
    MOSAIC_KWARGS = {"gridspec_kw": {"wspace": 0.08}}

    # Rendering parameters scoped to this figure — no global changes.
    _TRAJECTORY_LW: float = 2.0

    def __init__(
        self,
        data: ArenaTaskOverviewData,
        ctx: FigureContext,
    ) -> None:
        super().__init__(data, ctx)

    # ── Panel (a): Environment ──────────────────────────────────────────
    @panel(order=0)
    def environment(self, ax: Axes) -> None:
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
        ax.set_title("(a) Environment — Observation IDs", fontsize=7)
        ax.axis("off")

    # ── Panel (b): Trajectory ───────────────────────────────────────────
    @panel(order=1)
    def trajectory(self, ax: Axes) -> None:
        """Panel (b): Agent trajectory with start, end, and revisit rings."""
        data: ArenaTaskOverviewData = self.data
        world: AnyWorld = data.world
        location_ids = data.trajectory_locations.tolist()

        # Base map + time-coloured trajectory (no default endpoint markers).
        plot_time_colored_trajectory(
            ax,
            world,
            location_ids,
            cmap="plasma",
            show_endpoints=False,
            background_shape="square",
            line_width=self._TRAJECTORY_LW,
        )

        # Custom large markers.
        _overlay_endpoints(ax, world, location_ids)
        _overlay_revisit_rings(
            ax, world, data.trajectory_locations, data.revisit_mask
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=3,
            fontsize=5,
            framealpha=0.85,
            borderpad=0.2,
            handlelength=0.8,
            handletextpad=0.3,
        )

        ax.set_title("(b) Agent Trajectory (time-coloured)", fontsize=7)

    # ── Panel (c): Summary ──────────────────────────────────────────────
    @panel(order=2)
    def summary_text(self, ax: Axes) -> None:
        """Text summary panel: task description and statistics."""
        data: ArenaTaskOverviewData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n_steps = len(data.trajectory_locations)
        n_revisits = int(data.revisit_mask.sum())
        n_obs = int(data.observation_ids.max())
        H, W = data.wall_mask.shape
        n_passable = int(data.wall_mask.sum())

        revisit_pct = 100.0 * n_revisits / max(n_steps, 1)

        lines = [
            "Arena Replay Task",
            "",
            f"Grid: {H}\u2009\u00d7\u2009{W}",
            f"Passable cells: {n_passable}",
            f"Unique observations: {n_obs}",
            f"Trajectory steps: {n_steps}",
            f"Revisits: {n_revisits} ({revisit_pct:.0f}%)",
            "",
            "Agents navigate a dungeon",
            "topology. At each step the",
            "model predicts the observation",
            "at the next position.",
            "",
            "Revisit accuracy is the",
            "primary structural-memory",
            "diagnostic.",
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
    from ehc_sn.figures.utils.axes import _environment_locations

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
