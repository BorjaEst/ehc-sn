"""Halt-logit evolution figure — bounded online diagnostic for ACT paradigms.

Plots halt/continue logits over rollout steps for the first N batch samples.
This module handles only ``value/q_logits`` (ACT-style).  For RL-style
Q-values see :mod:`q_value_evolution`.
"""

from __future__ import annotations

from ehc_sn.figures.plots.value_curve import plot_value_curve
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

_TRACE_KEY = "value/q_logits"


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a halt/continue logit evolution trace.

    Args:
        trace: Trace containing ``value/q_logits`` shaped ``(T, B, A)``.
        ctx: Figure context (uses ``max_items``).

    Returns:
        Matplotlib ``Figure`` with one line-plot axis.

    Raises:
        ValueError: If ``value/q_logits`` is absent or has wrong rank.
    """
    try:
        values = trace.get(_TRACE_KEY)
    except (ValueError, KeyError):
        raise ValueError(
            f"halt_logit_evolution requires '{_TRACE_KEY}' in the trace, "
            "but it was not found."
        ) from None

    if values.ndim != 3:
        raise ValueError(
            f"{_TRACE_KEY} must be 3-D (T, B, A), got shape {values.shape}."
        )

    fig = plot_value_curve(values, ctx, ylabel="Halt/continue logit")
    fig.axes[0].set_title("Halt-logit evolution")
    return fig
