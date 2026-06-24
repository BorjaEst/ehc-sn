"""Goaltrace graph renderer — scalar-agnostic DAG visualisation.

Owns node circles, directed arrows, observation-ID labels, current/goal
markers, and local legend glyphs.  Does not own layout computation,
semantic field selection, or panel composition.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from numpy.typing import NDArray

from ehc_sn.figures.selectors.goaltrace import GoaltraceGraphGeometry


# =============================================================================
def plot_goaltrace_graph(
    ax: Axes,
    geometry: GoaltraceGraphGeometry,
    values: NDArray,
    *,
    cmap: Colormap,
    norm: Normalize,
    title: str,
    node_size: float = 400.0,
    arrow_width: float = 1.2,
    label_fontsize: float = 6.0,
) -> ScalarMappable:
    """Render one goaltrace DAG panel on *ax*.

    Args:
        ax: Matplotlib axes to draw on.
        geometry: Precomputed graph geometry (positions, edges, markers).
        values: Per-node scalar values for fill colour, shape ``(N,)``.
        cmap: Continuous colormap for node fill.
        norm: Normalisation applied to *values* before mapping.
        title: Panel title (shown above the axes).
        node_size: Diameter of each node circle in points.
        arrow_width: Line width for directed edges.
        label_fontsize: Font size for observation-ID labels.

    Returns:
        ScalarMappable that can be used to attach a figure-level colorbar.

    Raises:
        ValueError: If ``len(values) != len(geometry.observation_ids)``.
    """
    if len(values) != len(geometry.observation_ids):
        raise ValueError(
            f"values length ({len(values)}) must match "
            f"number of nodes ({len(geometry.observation_ids)})."
        )

    N = len(geometry.observation_ids)
    pos = geometry.positions
    colors = cmap(norm(values))

    # ── Directed edges ───────────────────────────────────────────────────
    for u, v in geometry.edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        dx = xv - xu
        dy = yv - yu
        ax.arrow(
            xu,
            yu,
            dx * 0.75,
            dy * 0.75,
            head_width=0.08,
            head_length=0.08,
            fc="gray",
            ec="gray",
            lw=arrow_width * 0.5,
            length_includes_head=True,
            zorder=1,
        )

    # ── Node fill circles ────────────────────────────────────────────────
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_size,
        c=colors,
        edgecolors="none",
        zorder=2,
    )

    # ── Current-node marker: thick dark ring ─────────────────────────────
    ci = geometry.current_node_index
    ax.scatter(
        [pos[ci, 0]],
        [pos[ci, 1]],
        s=node_size * 1.5,
        facecolors="none",
        edgecolors="black",
        linewidths=2.5,
        zorder=3,
    )

    # ── Goal-node marker: double ring ────────────────────────────────────
    gi = geometry.goal_node_index
    ax.scatter(
        [pos[gi, 0]],
        [pos[gi, 1]],
        s=node_size * 1.8,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )

    # ── Current == goal: combined marker ─────────────────────────────────
    if ci == gi:
        ax.scatter(
            [pos[ci, 0]],
            [pos[ci, 1]],
            s=node_size * 2.2,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    # ── Observation-ID labels ────────────────────────────────────────────
    for i in range(N):
        ax.text(
            pos[i, 0],
            pos[i, 1] - 0.22,
            str(int(geometry.observation_ids[i])),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            zorder=6,
        )

    # ── Legend glyphs (upper-right corner) ───────────────────────────────
    legend_x = ax.get_xlim()[1]
    legend_y = ax.get_ylim()[1]
    leg_dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.065

    # Current marker legend
    ax.scatter(
        [legend_x - 0.5],
        [legend_y - leg_dy],
        s=node_size * 0.5,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        zorder=7,
    )
    ax.text(
        legend_x - 0.3,
        legend_y - leg_dy,
        "current",
        fontsize=label_fontsize,
        va="center",
        zorder=7,
    )

    # Goal marker legend
    ax.scatter(
        [legend_x - 0.5],
        [legend_y - 2.5 * leg_dy],
        s=node_size * 0.5,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
    )
    ax.text(
        legend_x - 0.3,
        legend_y - 2.5 * leg_dy,
        "goal",
        fontsize=label_fontsize,
        va="center",
        zorder=7,
    )

    # ── Title and axis styling ───────────────────────────────────────────
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")

    return ScalarMappable(norm=norm, cmap=cmap)


# =============================================================================
__all__ = ["plot_goaltrace_graph"]
