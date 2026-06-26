"""MazeHard rendering primitives.

Extracted from ``task_overview_mazehard.py`` so both ``task_overview_*``
and ``prediction_reasoning_*`` can reuse the same visual encoding without
calling one template from another.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


def render_maze_path_field(
    ax: Axes,
    overlay: np.ndarray,
    *,
    title: str | None = None,
    cmap: ListedColormap | None = None,
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Render a binary path overlay on *ax*.

    The overlay is displayed as a semi-transparent red mask on a light-grey
    background so the maze structure remains legible underneath.
    """
    _cmap = cmap if cmap is not None else ListedColormap(["#f0f0f0", "#e53e3e"])
    ax.imshow(
        np.asarray(overlay),
        cmap=_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


__all__ = ["render_maze_path_field"]
