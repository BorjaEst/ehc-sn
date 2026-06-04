"""Halting timeline figure — fixed-budget termination diagnostic.

Two-panel horizontal figure:
    1. Binary termination heatmap over rollout steps.
    2. Summary annotation with termination policy and statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a two-panel halting termination diagnostic.

    Args:
        trace: Trace containing ``act/halted`` numeric leaf shaped ``(T, B)``.
        ctx: Figure context.

    Returns:
        Matplotlib ``Figure`` with two horizontally arranged panels.
    """
    data = _extract_data(trace, ctx)
    return HaltingTimelineFigure(data, ctx).plot()


# =============================================================================
class HaltingTimelineFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.16
    MOSAIC = [["timeline", "summary"]]
    MOSAIC_KWARGS = {
        "width_ratios": [2.0, 1.0],
    }

    def __init__(self, data: "HaltingTimelineData", ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # ── Panel A: termination signal ──────────────────────────────────────
    @panel(slots=["timeline"])
    def timeline_panel(self, ax: Axes) -> None:
        data = self.data.halted  # (n, T)
        n, T = data.shape

        ax.imshow(
            data,
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Case")
        ax.set_title("Termination signal")

        if n == 1:
            ax.set_yticks([0])
            ax.set_yticklabels(["Case 0"])
        else:
            ax.set_yticks(range(n))
            ax.set_yticklabels([str(i) for i in range(n)])

        # Compact colour bar with active/terminated labels.
        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="RdYlBu_r")
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["active", "terminated"])

    # ── Panel B: termination summary ─────────────────────────────────────
    @panel(slots=["summary"])
    def summary_panel(self, ax: Axes) -> None:
        data: HaltingTimelineData = self.data
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        lines = [
            f"policy: {data.policy_label}",
            f"max steps: {data.max_steps}",
            f"completed: {data.completed_count} / {data.n_cases}",
        ]
        if data.observed_halt_steps and len(set(data.observed_halt_steps)) == 1:
            lines.append(f"halt step: {data.observed_halt_steps[0]}")

        ax.text(
            0.08,
            0.92,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=5.5,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_title("Termination summary")


# =============================================================================
@dataclass(frozen=True)
class HaltingTimelineData:
    """Prepared data for :class:`HaltingTimelineFigure`."""

    halted: np.ndarray  # (n, T) — binary strip(s)
    observed_halt_steps: list[int | None]  # first halt step per case
    completed_count: int
    n_cases: int
    max_steps: int
    policy_label: str


# =============================================================================
def _extract_data(trace: TraceTree, ctx: FigureContext) -> HaltingTimelineData:
    """Extract halting data and classify termination policy."""
    halted = trace.get("act/halted")  # (T, B)
    if halted.ndim != 2:
        raise ValueError(
            f"act/halted must be 2-D (T, B), got shape {halted.shape}"
        )

    T, B = halted.shape
    max_cases = ctx.max_items or B
    n = min(B, max_cases)
    data = halted[:, :n].T  # (n, T)

    # Find first halt step per case.
    observed: list[int | None] = []
    completed = 0
    for i in range(n):
        row = data[i]
        halt_idx = int(np.argmax(row)) if row.any() else -1
        if halt_idx >= 0 and row[halt_idx] > 0:
            observed.append(halt_idx)
            completed += 1
        else:
            observed.append(None)

    # Classify policy from data.
    if completed == n and all(h == T - 1 for h in observed if h is not None):
        policy = "fixed-budget (all terminate at max step)"
    elif completed == n:
        policy = "variable termination"
    else:
        policy = f"incomplete ({completed} / {n} completed)"

    return HaltingTimelineData(
        halted=data,
        observed_halt_steps=observed,
        completed_count=completed,
        n_cases=n,
        max_steps=T - 1,
        policy_label=policy,
    )
