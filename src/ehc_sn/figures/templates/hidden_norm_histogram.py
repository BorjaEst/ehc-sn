"""Hidden-state norm histogram figure — bounded online summary for TEM/EHC.

Wraps the existing ``render_hidden_norm_histogram`` as a ``bounded_trace``
figure that consumes ``diagnostic/hidden_norms`` (bin centers) and
``diagnostic/hidden_norms_density`` trace leaves.
"""

from __future__ import annotations

import torch
from matplotlib.figure import Figure

from ehc_sn.figures.registry import FigureContext
from ehc_sn.metrics.renderers import render_hidden_norm_histogram
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a hidden-norm histogram from synthetic summary trace leaves.

    Args:
        trace: Trace containing ``diagnostic/hidden_norms`` (bin centers)
            and ``diagnostic/hidden_norms_density`` (density values),
            both shaped ``(N_bins,)``.
        ctx: Figure context (unused).

    Returns:
        Matplotlib ``Figure`` with one bar-chart axis.
    """
    centers = torch.as_tensor(trace.get("diagnostic/hidden_norms"))
    density = torch.as_tensor(trace.get("diagnostic/hidden_norms_density"))
    return render_hidden_norm_histogram(
        centers,
        density,
        title="Hidden-state norms (bounded summary)",
    )
