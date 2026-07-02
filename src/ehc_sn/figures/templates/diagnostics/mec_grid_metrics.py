"""MEC gridness metrics by frequency -- report-facing figure template."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.adapters.mec import load_mec_grid_figure_data
from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mec import (
    MECGridMetricsData,
    select_mec_grid_metrics,
)
from ehc_sn.figures.utils.axes import subdivide_axes

if TYPE_CHECKING:
    from ehc_sn.evaluation.contracts import ProducedArtifact
    from ehc_sn.traces.trace_tree import TraceTree


def plot(
    trace: object = None,
    *,
    ctx: FigureContext | None = None,
    artifact: ProducedArtifact | None = None,
    artifact_root: Path | None = None,
) -> Figure:
    """Render MEC grid metrics from artifact data or legacy trace.

    Prefers artifact-based loading when ``artifact`` and ``artifact_root``
    are both provided.  Falls back to trace-based selector when only
    ``trace`` is given.

    Args:
        trace: Legacy trace tree (used when artifact is not available).
        ctx: Figure selection and rendering context.
        artifact: Produced artifact descriptor from the MEC analysis runner.
        artifact_root: Root directory containing the artifact.

    Returns:
        Matplotlib figure with gridness, spacing, and mosaic panels.

    Raises:
        ValueError: If neither trace nor artifact is provided, or if
            artifact is provided without artifact_root.
    """
    if ctx is None:
        ctx = FigureContext()

    if artifact is not None and artifact_root is not None:
        data = load_mec_grid_figure_data(artifact, artifact_root)
    elif trace is not None:
        from ehc_sn.traces.trace_tree import TraceTree

        assert isinstance(trace, TraceTree)
        data = select_mec_grid_metrics(trace, ctx)
    else:
        raise ValueError(
            "mec_grid_metrics requires either a trace or an "
            "analysis artifact.  Use plot(trace=trace, ctx=ctx) "
            "or plot(artifact=artifact, artifact_root=root, ctx=ctx)."
        )

    return MECGridMetricsFigure(data, ctx).plot()


class MECGridMetricsFigure(BaseFigureTemplate):
    """Gridness and spacing by frequency band with representative examples.

    Fixed 1x3 report layout:
        gridness scatter | spacing scatter | 2x2 top-autocorrelogram mosaic
    """

    HEIGHT_FRAC: float = 0.22
    MOSAIC = [["gridness", "spacing", "top_mosaic"]]
    MOSAIC_KWARGS = {"width_ratios": [1.0, 1.0, 0.8]}

    _N_MOSAIC_ROWS: int = 2
    _N_MOSAIC_COLS: int = 2
    _MAX_EXAMPLES: int = 4

    def __init__(self, data: MECGridMetricsData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @staticmethod
    def _freq_label(idx: int) -> str:
        return f"f{idx}"

    # -- Panel: gridness by frequency --

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

    # -- Panel: spacing by frequency --

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
        ax.set_ylabel("Grid spacing", fontsize=9)
        ax.set_title("MEC spacing by frequency", fontsize=10)

    # -- Panel: top examples (fixed 2x2 autocorrelogram mosaic) --

    # Shared colormap with NaN color = light gray.
    _CMAP = plt.get_cmap("coolwarm").copy()
    _CMAP.set_bad("0.85")

    @panel()
    def top_mosaic(self, ax: Axes) -> None:
        data = self.data
        n_examples = min(len(data.top_autocorrs), self._MAX_EXAMPLES)

        if n_examples == 0:
            ax.text(
                0.5,
                0.5,
                "No examples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return

        cell_axes = subdivide_axes(
            ax,
            self._N_MOSAIC_ROWS,
            self._N_MOSAIC_COLS,
            wspace=0.004,
            hspace=0.004,
        )
        ax_list = list(np.atleast_1d(cell_axes).ravel())

        for i, cell_ax in enumerate(ax_list):
            if i >= n_examples:
                cell_ax.axis("off")
                continue

            autocorr = data.top_autocorrs[i]
            if autocorr.size == 0 or not np.isfinite(autocorr).any():
                cell_ax.axis("off")
                continue

            cell_ax.imshow(
                np.ma.masked_invalid(autocorr),
                origin="lower",
                cmap=self._CMAP,
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
            )
            cell_ax.set_aspect("equal")
            cell_ax.set_xticks([])
            cell_ax.set_yticks([])

            for spine in cell_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.25)
                spine.set_edgecolor("0.35")

            freq = int(data.top_freq_indices[i])
            cell = int(data.top_cell_indices[i])
            score = float(data.top_gridness[i])
            cell_ax.text(
                0.03,
                0.97,
                f"f{freq}.c{cell}\nG={score:.2f}",
                transform=cell_ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.65,
                    "edgecolor": "none",
                    "pad": 0.4,
                },
            )

        ax.axis("off")
        ax.set_title("Top 4 samples", fontsize=10, pad=6)


# =============================================================================
__all__ = [
    "MECGridMetricsFigure",
    "plot",
]
