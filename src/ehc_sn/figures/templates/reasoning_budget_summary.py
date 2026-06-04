"""Reasoning budget summary — termination policy diagnostic for notebook §10.

Three-panel figure (no heatmap, no colorbar):
    1. Recurrent budget — horizontal step ruler showing active rollout.
    2. Termination event — lollipop marker at halt step.
    3. Computation policy — compact text summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.traces.trace_tree import TraceTree

BAR_COLOR = "steelblue"
MARKER_COLOR = "#e53e3e"


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    data = _extract_data(trace, ctx)
    return ReasoningBudgetSummaryFigure(data, ctx).plot()


class ReasoningBudgetSummaryFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.18
    MOSAIC = [["budget", "termination", "policy"]]
    MOSAIC_KWARGS = {"width_ratios": [2.0, 1.0, 2.0]}

    def __init__(
        self, data: "_ReasoningBudgetData", ctx: FigureContext
    ) -> None:
        super().__init__(data, ctx)

    # ── Panel A: recurrent budget ────────────────────────────────────────────
    @panel(slots=["budget"])
    def budget_panel(self, ax: Axes) -> None:
        d = self.data
        T = d.total_steps

        # Horizontal bar from step 0 to last step index.
        for step in range(T):
            color = BAR_COLOR if step < d.halt_step else "lightgray"
            ax.barh(0, 1, left=step, height=0.4, color=color, edgecolor="none")

        # Vertical line at halt step.
        ax.axvline(
            x=d.halt_step,
            color=MARKER_COLOR,
            linewidth=2,
            linestyle="--",
            label=f"halt (step {d.halt_step})",
        )

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Recurrent step")
        ax.set_yticks([])
        ax.set_title("Recurrent budget")
        ax.legend(fontsize="x-small", loc="upper right")
        ax.grid(False)

    # ── Panel B: termination event ───────────────────────────────────────────
    @panel(slots=["termination"])
    def termination_panel(self, ax: Axes) -> None:
        d = self.data
        ax.plot(d.halt_step, 1, "o", color=MARKER_COLOR, markersize=8)
        ax.vlines(d.halt_step, 0, 1, color=MARKER_COLOR, linewidth=2)
        ax.set_xlim(-0.5, d.total_steps - 0.5)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_title("Termination event")
        ax.text(
            d.halt_step,
            1.08,
            "terminated",
            ha="center",
            va="bottom",
            fontsize=7,
            color=MARKER_COLOR,
        )
        ax.grid(False)

    # ── Panel C: computation policy ──────────────────────────────────────────
    @panel(slots=["policy"])
    def policy_panel(self, ax: Axes) -> None:
        d = self.data
        lines = [
            f"policy: {d.policy_label}",
            f"learned early stop: not evaluated",
            f"used steps: {d.used_steps} / {d.total_steps}",
            f"termination step: {d.halt_step}",
            f"completed: {d.completed} / 1",
            "prediction step: final",
        ]
        text = "\n".join(lines)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3
            ),
        )
        ax.set_title("Computation policy")


# =============================================================================
@dataclass(frozen=True)
class _ReasoningBudgetData:
    total_steps: int
    halt_step: int
    used_steps: int
    completed: bool
    policy_label: str


# =============================================================================
def _extract_data(trace: TraceTree, ctx: FigureContext) -> _ReasoningBudgetData:
    """Extract budget data from the act/halted trace."""
    halted = trace.get("act/halted")  # (T, B)
    if halted.ndim != 2:
        raise ValueError(
            f"act/halted must be 2-D (T, B), got shape {halted.shape}"
        )
    T, B = halted.shape

    # Use first batch element.
    row = halted[:, 0]
    halt_idx = int(np.argmax(row)) if row.any() else -1
    if halt_idx >= 0 and row[halt_idx] > 0:
        completed = True
        used_steps = halt_idx + 1
    else:
        completed = False
        halt_idx = T - 1
        used_steps = T

    if completed and halt_idx == T - 1:
        policy = "fixed-budget rollout"
    elif completed:
        policy = "variable termination"
    else:
        policy = "incomplete"

    return _ReasoningBudgetData(
        total_steps=T,
        halt_step=halt_idx,
        used_steps=used_steps,
        completed=completed,
        policy_label=policy,
    )
