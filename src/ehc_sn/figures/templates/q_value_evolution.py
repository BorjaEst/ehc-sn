"""Q-value evolution figure — bounded online diagnostic for RL paradigms.

Plots Q-values over rollout steps for the first N batch samples.
This module handles only ``value/q_values`` (RL-style).  For ACT-style
halt/continue logits see :mod:`halt_logit_evolution`.
"""

from __future__ import annotations

from ehc_sn.figures.plots.value_curve import plot_value_curve
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.keys import TRACE_KEY_Q_VALUES
from ehc_sn.traces.trace_tree import TraceTree

_TRACE_KEY = TRACE_KEY_Q_VALUES


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a Q-value evolution trace.

    Args:
        trace: Trace containing ``value/q_values`` shaped ``(T, B, A)``.
        ctx: Figure context (uses ``max_items``).

    Returns:
        Matplotlib ``Figure`` with one line-plot axis.

    Raises:
        ValueError: If ``value/q_values`` is absent or has wrong rank.
    """
    try:
        values = trace.get(_TRACE_KEY)
    except (ValueError, KeyError):
        raise ValueError(
            f"q_value_evolution requires '{_TRACE_KEY}' in the trace, "
            "but it was not found."
        ) from None

    if values.ndim != 3:
        raise ValueError(
            f"{_TRACE_KEY} must be 3-D (T, B, A), got shape {values.shape}."
        )

    fig = plot_value_curve(values, ctx, ylabel="Q-value")
    fig.axes[0].set_title("Q-value evolution")
    return fig
