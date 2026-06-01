"""MEC gridness metrics by frequency — report-facing figure template."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mec import (
    MECGridMetricsData,
    select_mec_grid_metrics,
)
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MECGridMetricsFigure(select_mec_grid_metrics(trace, ctx), ctx).plot()


class MECGridMetricsFigure(BaseFigureTemplate):
    """Gridness and spacing by frequency band.

    Panel A: gridness scores for all cells per frequency.
    Panel B: grid spacing for cells with valid spacing estimates.
    """

    HEIGHT_FRAC: float = 0.35
    MOSAIC = [["gridness", "spacing"]]
    MOSAIC_KWARGS = {"width_ratios": [1.0, 1.0]}

    def __init__(self, data: MECGridMetricsData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _freq_label(idx: int) -> str:
        return f"f{idx}"

    # ── Panel A: gridness by frequency ──────────────────────────────────────

    @panel()
    def gridness(self, ax: Axes) -> None:
        data = self.data
        n_freqs = len(data.freq_indices)
        positions = np.arange(n_freqs)

        for f in range(n_freqs):
            scores = data.gridness[f, :]
            finite = scores[np.isfinite(scores)]
            if finite.size == 0:
                continue
            # Jittered strip plot
            jitter = np.random.default_rng(seed=f).uniform(
                -0.15, 0.15, size=finite.size
            )
            ax.scatter(
                positions[f] + jitter,
                finite,
                s=12,
                alpha=0.6,
                edgecolors="none",
                zorder=2,
            )
            # Median line
            median = float(np.median(finite))
            ax.plot(
                [positions[f] - 0.3, positions[f] + 0.3],
                [median, median],
                color="C1",
                linewidth=1.5,
                zorder=3,
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7, zorder=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [self._freq_label(int(i)) for i in data.freq_indices], fontsize=8
        )
        ax.set_xlabel("Frequency band", fontsize=9)
        ax.set_ylabel("Gridness score", fontsize=9)
        ax.set_title("MEC gridness by frequency", fontsize=10)

    # ── Panel B: spacing by frequency ───────────────────────────────────────

    @panel()
    def spacing(self, ax: Axes) -> None:
        data = self.data
        n_freqs = len(data.freq_indices)
        positions = np.arange(n_freqs)

        for f in range(n_freqs):
            vals = data.spacing[f, :]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            jitter = np.random.default_rng(seed=f + 100).uniform(
                -0.15, 0.15, size=finite.size
            )
            ax.scatter(
                positions[f] + jitter,
                finite,
                s=12,
                alpha=0.6,
                edgecolors="none",
                zorder=2,
            )
            median = float(np.median(finite))
            ax.plot(
                [positions[f] - 0.3, positions[f] + 0.3],
                [median, median],
                color="C1",
                linewidth=1.5,
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [self._freq_label(int(i)) for i in data.freq_indices], fontsize=8
        )
        ax.set_xlabel("Frequency band", fontsize=9)
        ax.set_ylabel("Grid spacing λ (world units)", fontsize=9)
        ax.set_title("MEC spacing by frequency", fontsize=10)
