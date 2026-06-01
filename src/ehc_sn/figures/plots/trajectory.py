"""Trajectory-related plotting panels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ehc_sn.figures._contracts import AnyWorld
from ehc_sn.figures.plots.map import plot_map
from ehc_sn.figures.utils.axes import _environment_locations


# =============================================================================
def plot_time_colored_trajectory(
    ax: plt.Axes,
    world: AnyWorld,
    location_ids: list[int],
    *,
    cmap: str = "plasma",
    show_endpoints: bool = True,
    background_shape: str = "square",
    line_width: float = 1.5,
) -> plt.Axes:
    """Plot a trajectory colored by time on an existing axes.

    The trajectory uses the same image/grid coordinate convention as
    :func:`ehc_sn.figures.plots.map.plot_map`: ``o`` is the column index,
    ``y`` is the row index, and row 0 is displayed at the top.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        location_ids: Ordered list of visited location indices.
        cmap: Colormap name for time coloring.
        show_endpoints: Whether to mark start/end points.
        background_shape: Shape for the background map markers.
        line_width: Width of the trajectory line.

    Returns:
        The axes with the trajectory rendered.
    """
    if not location_ids:
        ax.text(
            0.5, 0.5, "No trajectory", ha="center", va="center", fontsize=10
        )
        ax.axis("off")
        return ax

    n_locations = len(_environment_locations(world))
    values = np.full(n_locations, np.nan, dtype=float)
    plot_map(world, values, ax=ax, shape=background_shape, radius=0.25)

    coords = _trajectory_coords(world, location_ids)
    if coords.shape[0] == 0:
        ax.text(
            0.5,
            0.5,
            "No valid locations",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        return ax
    if coords.shape[0] < 2:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, color="black")
        return ax

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    colors = np.linspace(0, 1, segments.shape[0])

    lc = LineCollection(
        segments,
        cmap=cmap,
        array=colors,
        linewidths=line_width,
    )
    ax.add_collection(lc)
    if show_endpoints:
        ax.scatter(coords[0, 0], coords[0, 1], s=20, color="black", zorder=3)
        ax.scatter(
            coords[-1, 0],
            coords[-1, 1],
            s=20,
            color="white",
            edgecolor="black",
            zorder=3,
        )

    ax.set_aspect(1)
    ax.axis("off")
    return ax


# =============================================================================
def _trajectory_coords(world: AnyWorld, location_ids: list[int]) -> np.ndarray:
    """Return trajectory coordinates in canonical world order ``(o, y)``."""
    coords = []
    locations = _environment_locations(world)
    for loc_id in location_ids:
        if 0 <= loc_id < len(locations):
            loc = locations[loc_id]
            coords.append([loc["o"], loc["y"]])
    return np.asarray(coords, dtype=float)


# =============================================================================
__all__ = ["plot_time_colored_trajectory"]
