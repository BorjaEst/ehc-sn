"""MEC grid example figure — selected rate-map/autocorrelogram pairs."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.plots.autocorr import plot_spatial_autocorrelogram
from ehc_sn.figures.plots.ratemap import plot_rate_map
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mec import (
    MECGridExamplesData,
    select_mec_grid_examples,
)
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MECGridExamplesFigure(
        select_mec_grid_examples(trace, ctx), ctx
    ).plot()


class MECGridExamplesFigure(BaseFigureTemplate):
    """Selected top-gridness MEC cells with paired rate maps and autocorrelograms.

    Each row shows one cell: rate map (left) and spatial autocorrelogram (right).
    The row title includes frequency band, cell index, and gridness score.
    """

    HEIGHT_FRAC: float = 0.55
    MOSAIC_KWARGS = {"width_ratios": [1.0, 1.0]}

    def __init__(self, data: MECGridExamplesData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)
        self._n_rows = max(data.n_examples, 1)
        self.MOSAIC = [["rmap", "acorr"]] * self._n_rows

    # ── Panel: rate map ─────────────────────────────────────────────────────

    @panel()
    def rmap(self, ax: Axes) -> None:
        data = self.data
        if data.n_examples == 0:
            ax.text(0.5, 0.5, "No examples", ha="center", va="center")
            ax.axis("off")
            return

        # This panel owns all rows of the rate-map column via subdivide.
        axes = subdivide_axes(ax, self._n_rows, 1, hspace=0.05, squeeze=True)
        ax_list = (
            list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        )

        for row_idx, rm in enumerate(data.rate_maps):
            cell_ax = ax_list[row_idx]
            plot_rate_map(cell_ax, rm, vmin=0.0, vmax=0.10)
            freq = int(data.freq_indices[row_idx])
            cell = int(data.cell_indices[row_idx])
            g = data.gridness[row_idx]
            cell_ax.set_title(
                f"f{freq} · c{cell} · G={g:.2f}",
                fontsize=7,
                loc="left",
                pad=1,
            )

        for row_idx in range(data.n_examples, len(ax_list)):
            ax_list[row_idx].axis("off")

    # ── Panel: autocorrelogram ──────────────────────────────────────────────

    @panel()
    def acorr(self, ax: Axes) -> None:
        data = self.data
        if data.n_examples == 0:
            ax.text(0.5, 0.5, "No examples", ha="center", va="center")
            ax.axis("off")
            return

        axes = subdivide_axes(ax, self._n_rows, 1, hspace=0.05, squeeze=True)
        ax_list = (
            list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        )

        for row_idx, autocorr in enumerate(data.autocorrs):
            cell_ax = ax_list[row_idx]
            if autocorr.size == 0 or not np.isfinite(autocorr).any():
                cell_ax.text(0.5, 0.5, "No autocorr", ha="center", va="center")
                cell_ax.axis("off")
                continue
            cell_ax.imshow(
                autocorr,
                origin="lower",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            cell_ax.set_aspect("equal")
            cell_ax.axis("off")

        for row_idx in range(data.n_examples, len(ax_list)):
            ax_list[row_idx].axis("off")
