"""MazeHard plot helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage

from ehc_sn.figures.utils.colors import maze_cmap


# =============================================================================
def plot_maze_with_overlay(  # ------------------------------------------------
    ax: Axes,
    inputs_grid: np.ndarray,
    overlay_mask: np.ndarray,
    *,
    title: Optional[str] = None,
) -> tuple[AxesImage, AxesImage]:
    """Render maze inputs with a semi-transparent mazehard_solution_overlay mask.

    The maze grid uses image/grid coordinates with row 0 displayed at the top,
    matching the processed-dataset and environment-map figure conventions.

    Returns the base and mazehard_solution_overlay images for downstream composition.
    """
    base = np.asarray(inputs_grid)
    mazehard_solution_overlay = np.asarray(overlay_mask).astype(float)

    base_img = ax.imshow(
        base,
        cmap=maze_cmap(),
        vmin=0,
        vmax=5,
        interpolation="nearest",
        origin="upper",
    )
    overlay_img = ax.imshow(
        mazehard_solution_overlay,
        cmap=ListedColormap(["none", "#e53e3e"]),
        alpha=0.6,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return base_img, overlay_img


# =============================================================================
__all__ = ["plot_maze_with_overlay"]
