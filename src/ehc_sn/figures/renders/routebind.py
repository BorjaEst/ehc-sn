"""Routebind rendering primitives.

Extracted from ``task_overview_routebind.py`` so both ``task_overview_*``
and ``prediction_reasoning_*`` can reuse the same visual encoding without
calling one template from another.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def render_routebind_trajectory(
    ax: Axes,
    field: np.ndarray,
    grid_width: int,
    *,
    title: str | None = None,
    cmap: str = "Blues",
    norm: Normalize | None = None,
) -> ScalarMappable:
    """Render a routebind spatial trajectory field as a W×W heatmap on *ax*.

    Args:
        ax: Matplotlib axes to draw on.
        field: 1-D array of shape ``(W*W,)`` with trajectory activation values.
        grid_width: Number of cells per side (W).
        title: Optional panel title.
        cmap: Matplotlib colormap name.
        norm: Normalization.  Defaults to ``Normalize(vmin=0.0, vmax=1.0)``.

    Returns:
        ``ScalarMappable`` for attaching a colorbar.
    """
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=1.0)

    H = grid_width
    reshaped = np.asarray(field).reshape(H, H)
    mappable = ax.imshow(
        reshaped,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)

    return ScalarMappable(norm=norm, cmap=cmap)


__all__ = ["render_routebind_trajectory"]
