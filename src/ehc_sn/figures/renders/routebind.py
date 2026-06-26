"""Routebind rendering primitives.

Extracted from ``task_overview_routebind.py`` so both ``task_overview_*``
and ``prediction_reasoning_*`` can reuse the same visual encoding without
calling one template from another.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as mpl_cmap
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ── Semantic substrate colours (no legend needed) ──────────────────────

_WALL_COLOR = (0.25, 0.25, 0.25)  # dark — wall
_FREE_COLOR = (0.94, 0.93, 0.90)  # light cream — empty traversable
_OBS_COLOR = (0.78, 0.75, 0.70)  # warmer grey — observation cell
_PAD_COLOR = (0.97, 0.97, 0.96)  # near-white — padding / unknown


def render_routebind_trajectory(
    ax: Axes,
    field: np.ndarray,
    grid_width: int,
    *,
    title: str | None = None,
    title_fontsize: float = 7.0,
    cmap: str = "Blues",
    norm: Normalize | None = None,
    cell_type: np.ndarray | None = None,
    start_flag: np.ndarray | None = None,
    goal_flag: np.ndarray | None = None,
) -> ScalarMappable:
    """Render a routebind spatial trajectory field as a W×W heatmap on *ax*.

    When *cell_type* is provided, renders walls and free cells as a
    contextual substrate behind the trajectory field.  When *start_flag*
    and/or *goal_flag* are provided, overlays start (★ green) and goal
    (● gold) markers with an inset legend inside the panel.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    field:
        1-D array of shape ``(W*W,)`` with trajectory activation values.
    grid_width:
        Number of cells per side (W).
    title:
        Optional panel title.
    title_fontsize:
        Font size for the panel title.
    cmap:
        Matplotlib colormap name (or object) for the trajectory field.
    norm:
        Normalization.  Defaults to ``Normalize(vmin=0.0, vmax=1.0)``.
    cell_type:
        ``(W*W,)`` int — 0=wall, 1=free, 2=observation.  When provided, the
        spatial layout is rendered as a contextual substrate with distinct
        colours for walls, free cells, observation cells, and padding.
    start_flag:
        ``(W*W,)`` bool — cells containing the start position.
    goal_flag:
        ``(W*W,)`` bool — cells containing goal positions.

    Returns:
        ``ScalarMappable`` for attaching a colorbar.
    """
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=1.0)

    H = grid_width
    reshaped = np.asarray(field).reshape(H, H)

    # ── Spatial layout substrate (optional) ─────────────────────────────
    if cell_type is not None:
        ct = np.asarray(cell_type, dtype=int).reshape(H, H)

        bg = np.ones((H, H, 3), dtype=np.float32) * _PAD_COLOR
        bg[ct == 0] = _WALL_COLOR  # wall
        bg[ct == 1] = _FREE_COLOR  # free (empty traversable)
        bg[ct == 2] = _OBS_COLOR  # observation cell

        ax.imshow(bg, interpolation="nearest", origin="upper")

        # Alpha-blend trajectory field on top
        field_cmap = mpl_cmap[cmap] if isinstance(cmap, str) else cmap
        field_rgba = field_cmap(norm(reshaped))  # (H, W, 4)
        field_rgba[..., 3] = reshaped  # alpha = activation strength
        ax.imshow(field_rgba, interpolation="nearest", origin="upper")

        # ── Start / goal markers ────────────────────────────────────────
        legend_handles: list[Line2D] = []

        if start_flag is not None:
            sf = np.asarray(start_flag, dtype=bool).reshape(H, H)
            sr, sc = np.where(sf)
            ax.scatter(
                sc,
                sr,
                marker="*",
                color="#00cc44",
                s=80,
                edgecolor="#003300",
                linewidth=0.6,
                zorder=10,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="#00cc44",
                    markeredgecolor="#003300",
                    markeredgewidth=0.6,
                    markersize=8,
                    linestyle="none",
                    label="Start",
                )
            )

        if goal_flag is not None:
            gf = np.asarray(goal_flag, dtype=bool).reshape(H, H)
            gr, gc = np.where(gf)
            ax.scatter(
                gc,
                gr,
                marker="o",
                color="#ffcc00",
                s=40,
                edgecolor="#664400",
                linewidth=0.6,
                zorder=10,
                alpha=0.85,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#ffcc00",
                    markeredgecolor="#664400",
                    markeredgewidth=0.6,
                    markersize=6,
                    linestyle="none",
                    label="Goal",
                )
            )

        # Inset legend within the GT panel (top-left corner)
        if legend_handles:
            leg = ax.legend(
                handles=legend_handles,
                loc="upper left",
                fontsize=5.0,
                borderpad=0.3,
                handlelength=0.6,
                handletextpad=0.4,
            )
            # Style the frame imperatively — scienceplots sets
            # legend.frameon=False, so kwargs are ignored.
            leg.set_frame_on(True)
            frame = leg.get_frame()
            frame.set_visible(True)
            frame.set_facecolor("white")
            frame.set_edgecolor("#333333")
            frame.set_linewidth(0.8)
            frame.set_alpha(1.0)
    else:
        ax.imshow(
            reshaped,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    return ScalarMappable(
        norm=norm, cmap=cmap if isinstance(cmap, str) else cmap
    )


__all__ = ["render_routebind_trajectory"]


__all__ = ["render_routebind_trajectory"]
