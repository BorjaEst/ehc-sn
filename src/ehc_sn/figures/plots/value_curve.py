"""Value-curve plot helper functions."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def plot_value_curve(
    ax: Axes,
    values: np.ndarray,
    *,
    max_items: int | None = None,
    ylabel: str = "Value",
) -> Axes:
    """Draw value/logit curves over rollout steps onto *ax*.

    Args:
        ax: Axes to draw into (mutated in place).
        values: Numeric array shaped ``(T, B, A)``.
        max_items: Maximum number of batch samples to plot
            (if ``None``, all samples are shown).
        ylabel: Y-axis label (e.g. ``"Q-value"`` or
            ``"Halt/continue logit"``).

    Returns:
        The same *ax* instance.
    """
    T, B, A = values.shape
    n = min(B, max_items) if max_items is not None else B

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

    return ax
