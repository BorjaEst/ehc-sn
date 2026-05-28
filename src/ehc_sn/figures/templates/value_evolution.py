"""Value evolution figure — bounded online diagnostic for ACT/RL paradigms.

Plots Q-values (or Q-logits) over rollout steps for the first N batch samples.
Tolerates either ``value/q_values`` (RL) or ``value/q_logits`` (ACT) trace key.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a value-evolution trace from either Q-values or Q-logits.

    Args:
        trace: Trace containing ``value/q_values`` or ``value/q_logits``
            shaped ``(T, B, A)``.  Optionally ``act/halted`` for markers.
        ctx: Figure context (uses ``max_items``).

    Returns:
        Matplotlib ``Figure`` with one line-plot axis.

    Raises:
        ValueError: If neither ``value/q_values`` nor ``value/q_logits``
            exists in the trace.
    """
    source_key: str | None = None
    values = _try_get(trace, "value/q_values")
    if values is not None:
        source_key = "value/q_values"
    else:
        values = _try_get(trace, "value/q_logits")
        if values is not None:
            source_key = "value/q_logits"

    if source_key is None or values is None:
        raise ValueError(
            "value_evolution requires either 'value/q_values' or "
            "'value/q_logits' in the trace, but neither was found."
        )

    if values.ndim != 3:
        raise ValueError(
            f"{source_key} must be 3-D (T, B, A), got shape {values.shape}."
        )

    T, B, A = values.shape
    max_samples = ctx.max_items or B
    n = min(B, max_samples)

    fig = Figure(figsize=(5, 3), layout="constrained")
    ax = fig.subplots()

    steps = np.arange(T)
    for i in range(n):
        # Plot mean across actions for readability; overlay first action
        ax.plot(
            steps,
            values[:, i, 0],
            label=f"Sample {i}" if n <= 6 else None,
            alpha=0.8,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(f"value_evolution \u2014 source: {source_key}")

    if n <= 6:
        ax.legend(fontsize="small")

    return fig


def _try_get(trace: TraceTree, key: str) -> np.ndarray | None:
    """Return a trace leaf or ``None`` if the key is absent."""
    try:
        return trace.get(key)
    except (ValueError, KeyError):
        return None
