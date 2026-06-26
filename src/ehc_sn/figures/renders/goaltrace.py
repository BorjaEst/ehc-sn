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
    node_size: float = 320.0,
    label_fontsize: float = 5.5,
) -> ScalarMappable:
    """Render a goaltrace DAG with per-node scalar values on *ax*.

    Delegates to ``plot_goaltrace_graph`` with sorted-by-value visual
    encoding (bars) for compact presentation, or node-colour encoding
    for the DAG view.
    """
    # Use the standard sorted-horizontal-bar encoding from the task-overview
    # for consistency.
    if norm is None:
        norm = PowerNorm(gamma=0.2, vmin=0.0, vmax=1.0)

    order = np.argsort(values)[::-1]
    sorted_vals = values[order]
    sorted_ids = geometry.observation_ids[order]
    colors = cmap(norm(sorted_vals))

    n = len(values)
    ax.barh(range(n), sorted_vals, height=0.65, color=colors, edgecolor="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [str(int(i)) for i in sorted_ids],
        fontsize=label_fontsize,
    )
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=7)
    ax.tick_params(left=False)

    return ScalarMappable(norm=norm, cmap=cmap)


__all__ = ["render_goaltrace_field"]
