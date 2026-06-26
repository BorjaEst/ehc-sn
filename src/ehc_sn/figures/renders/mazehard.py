"""MazeHard rendering primitives.

Extracted from ``task_overview_mazehard.py`` so both ``task_overview_*``
and ``prediction_reasoning_*`` can reuse the same visual encoding without
calling one template from another.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

# ── Semantic colour palette (matches task_overview_mazehard) ────────────

_WALL_COLOR = np.array([0.106, 0.122, 0.141])  # #1b1f24
_EMPTY_COLOR = np.array([0.969, 0.957, 0.937])  # #f7f4ef
_START_COLOR = np.array([0.169, 0.424, 0.690])  # #2b6cb0
_GOAL_COLOR = np.array([0.839, 0.620, 0.180])  # #d69e2e
_PATH_COLOR = np.array([0.898, 0.243, 0.243])  # #e53e3e
_FALLBACK_COLOR = np.array([0.85, 0.85, 0.85])  # unknown token

# ── Vocab IDs (matching tasks/mazehard/runtime.py) ──────────────────────

_WALL_ID: int = 1
_EMPTY_ID: int = 2
_START_ID: int = 3
_GOAL_ID: int = 4


def render_maze_path_field(
    ax: Axes,
    overlay: np.ndarray,
    *,
    title: str | None = None,
    cmap: ListedColormap | None = None,
    vmin: float = 0,
    vmax: float = 1,
    title_fontsize: float = 7.0,
    input_ids: np.ndarray | None = None,
) -> None:
    """Render a binary path overlay on *ax*.

    When *input_ids* is provided, renders the full maze topology (walls,
    free cells, start, goal) with the path overlaid in red.  When
    *input_ids* is ``None``, renders the path as a red/grey binary
    colormap (backward-compatible with categorical-only usage).

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    overlay:
        ``(H, W)`` boolean — ``True`` = path cell.
    title:
        Optional panel title.
    cmap:
        Optional binary colormap (only used when *input_ids* is ``None``).
    vmin, vmax:
        Colour limits for binary path colormap (only used without *input_ids*).
    title_fontsize:
        Font size for the panel title.
    input_ids:
        ``(H, W)`` integer token IDs.  If provided, walls, start, goal, and
        free cells are drawn as a contextual substrate and the path is
        overlaid on top.
    """
    overlay_arr = np.asarray(overlay, dtype=bool)

    if input_ids is not None:
        # ── Rich substrate with path overlay ────────────────────────────
        ids = np.asarray(input_ids, dtype=int)
        H, W = ids.shape
        rgb = np.ones((H, W, 3), dtype=np.float32) * _FALLBACK_COLOR

        rgb[ids == _WALL_ID] = _WALL_COLOR
        rgb[ids == _EMPTY_ID] = _EMPTY_COLOR
        rgb[ids == _START_ID] = _START_COLOR
        rgb[ids == _GOAL_ID] = _GOAL_COLOR
        # Path overrides whatever cell type is underneath.
        rgb[overlay_arr] = _PATH_COLOR

        ax.imshow(rgb, interpolation="nearest", origin="upper")
    else:
        # ── Binary path-only colormap (backward-compatible) ─────────────
        _cmap = (
            cmap if cmap is not None else ListedColormap(["#f0f0f0", "#e53e3e"])
        )
        ax.imshow(
            overlay_arr,
            cmap=_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="upper",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=title_fontsize)


__all__ = ["render_maze_path_field"]
