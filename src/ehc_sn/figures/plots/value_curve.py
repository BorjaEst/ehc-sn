"""Shared private helper for Q-value and halt-logit evolution figures.

Provides ``plot_value_curve`` used by both ``q_value_evolution`` and
``halt_logit_evolution``.  Not a public figure template.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext


def plot_value_curve(
    values: np.ndarray,
    ctx: FigureContext,
    *,
    ylabel: str = "Value",
) -> Figure:
    """Render a line-plot of value/logit curves over rollout steps.

    Args:
        values: Numeric array shaped ``(T, B, A)``.
        ctx: Figure context (uses ``max_items``).
        ylabel: Y-axis label (e.g. ``"Q-value"`` or ``"Halt/continue logit"``).

    Returns:
        Matplotlib ``Figure`` with one line-plot axis.
    """
    T, B, A = values.shape
    max_samples = ctx.max_items or B
    n = min(B, max_samples)

    fig = Figure(figsize=(5, 3), layout="constrained")
    ax = fig.subplots()

    steps = np.arange(T)
    for i in range(n):
        ax.plot(
            steps,
            values[:, i, 0],
            label=f"Sample {i}" if n <= 6 else None,
            alpha=0.8,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)

    if n <= 6:
        ax.legend(fontsize="small")

    return fig
