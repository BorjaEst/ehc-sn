"""Goaltrace rendering primitives.

Extracted from ``task_overview_goaltrace.py`` so both ``task_overview_*``
and ``prediction_reasoning_*`` can reuse the same visual encoding without
calling one template from another.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, PowerNorm
from numpy.typing import NDArray

from ehc_sn.figures.plots.goaltrace_graph import plot_goaltrace_graph
from ehc_sn.figures.selectors.goaltrace import GoaltraceGraphGeometry


def render_goaltrace_field(
    ax: Axes,
    geometry: GoaltraceGraphGeometry,
    values: NDArray,
    *,
    cmap: Colormap,
    norm: Normalize | None = None,
    title: str | None = None,
    title_fontsize: float = 7.0,
    node_size: float = 320.0,
    label_fontsize: float = 5.5,
    mark_current: bool = False,
    mark_goal: bool = False,
) -> ScalarMappable:
    """Render a goaltrace DAG with per-node scalar values on *ax*.

    Uses the standard sorted-horizontal-bar encoding from the task-overview
    for consistency.

    When *mark_current* is ``True``, the bar corresponding to
    ``geometry.current_node_index`` is given an accent border and a ``★``
    marker.  When *mark_goal* is ``True``, the goal node bar is marked
    with ``●``.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    geometry:
        Graph geometry (observation IDs, node indices, positions).
    values:
        ``(N,)`` float — per-node scalar values.
    cmap:
        Matplotlib colormap for value-to-colour mapping.
    norm:
        Normalization.  Defaults to ``PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0)``.
    title:
        Optional panel title.
    title_fontsize:
        Font size for the title.
    node_size:
        Unused; kept for API compatibility.
    label_fontsize:
        Font size for y-tick (observation ID) labels.
    mark_current:
        If ``True``, highlight the current-node bar.
    mark_goal:
        If ``True``, highlight the goal-node bar.
    """
    if norm is None:
        norm = PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0)

    order = np.argsort(values)[::-1]
    sorted_vals = values[order]
    sorted_ids = geometry.observation_ids[order]
    colors = cmap(norm(sorted_vals))

    n = len(values)
    bars = ax.barh(
        range(n), sorted_vals, height=0.65, color=colors, edgecolor="none"
    )

    # ── Current / goal markers ──────────────────────────────────────────
    if mark_current:
        cur_pos = int(np.where(order == geometry.current_node_index)[0][0])
        bars[cur_pos].set_edgecolor("#2b6cb0")
        bars[cur_pos].set_linewidth(1.5)
        ax.annotate(
            "★",
            xy=(sorted_vals[cur_pos] + 0.02, cur_pos),
            fontsize=5.5,
            color="#2b6cb0",
            va="center",
            ha="left",
        )
    if mark_goal:
        goal_pos = int(np.where(order == geometry.goal_node_index)[0][0])
        bars[goal_pos].set_edgecolor("#d69e2e")
        bars[goal_pos].set_linewidth(1.5)
        ax.annotate(
            "●",
            xy=(sorted_vals[goal_pos] + 0.02, goal_pos),
            fontsize=5.5,
            color="#d69e2e",
            va="center",
            ha="left",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [str(int(i)) for i in sorted_ids],
        fontsize=label_fontsize,
    )
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(left=False)

    return ScalarMappable(norm=norm, cmap=cmap)


__all__ = ["render_goaltrace_field"]
