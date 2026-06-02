"""HPC rate-map population mosaic — flat wrapped small-multiple layout.

Shows the population-level distribution of HPC place-like rate maps,
ordered by spatial information descending.

This is the supplement/report-detail counterpart to ``hpc_place_metrics``:
``hpc_place_metrics`` answers "do some HPC units show spatially informative
place-like fields?" while this figure answers "is this structure present
across the population, not just the selected examples?"

References
----------
Whittington et al. (2020). "The Tolman-Eichenbaum Machine".
    Cell 183(5):1248–1262 e23.
Whittington et al. (2022). "Relating transformers to models and neural
    representations of the hippocampal formation".  arXiv:2112.04035.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.hpc import (
    HPCRateMapMosaicData,
    select_hpc_rate_map_mosaic,
)
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return HPCRateMapMosaicFigure(
        select_hpc_rate_map_mosaic(trace, ctx), ctx
    ).plot()


class HPCRateMapMosaicFigure(BaseFigureTemplate):
    """Flat wrapped rate-map population mosaic for HPC cells.

    Each tile is one cell's occupancy-normalised rate map.
    Tiles are ordered by descending finite spatial information.
    All tiles have the same physical size.
    """

    HEIGHT_FRAC: float = 0.65
    MOSAIC = [["mosaic"]]

    _N_COLS: int = 8
    _MAX_CELLS: int = 64

    # Sequential colormap for firing rates (nonnegative values).
    _CMAP = plt.get_cmap("viridis").copy()
    _CMAP.set_bad("0.85")

    def __init__(self, data: HPCRateMapMosaicData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # ------------------------------------------------------------------
    # Shared vmax across all tiles
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_vmax(tiles: list, percentile: float = 99.0) -> float:
        """Compute the shared vmax from the 99th percentile of all tile values."""
        all_vals = np.concatenate(
            [tile.rate_map.ravel() for tile in tiles if tile.rate_map.size > 0]
        )
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            return 1.0
        vmax = float(np.nanpercentile(finite, percentile))
        if not np.isfinite(vmax) or vmax <= 0:
            return 1.0
        return vmax

    # ------------------------------------------------------------------
    # Panel: mosaic
    # ------------------------------------------------------------------

    @panel()
    def mosaic(self, ax: Axes) -> None:
        tiles = list(self.data.tiles)

        if not tiles:
            ax.text(
                0.5,
                0.5,
                "No finite HPC spatial-information values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.axis("off")
            return

        n_tiles = len(tiles)
        n_cols = self._N_COLS
        n_rows = int(math.ceil(n_tiles / n_cols))

        cell_axes = subdivide_axes(
            ax,
            n_rows,
            n_cols,
            wspace=0.005,
            hspace=0.005,
        )

        vmax = self._shared_vmax(tiles)

        axes_flat = list(np.atleast_1d(cell_axes).ravel())

        for i, cell_ax in enumerate(axes_flat):
            if i >= n_tiles:
                cell_ax.axis("off")
                continue

            tile = tiles[i]
            rate_map = tile.rate_map
            if rate_map.size == 0 or not np.isfinite(rate_map).any():
                cell_ax.axis("off")
                continue

            cell_ax.imshow(
                np.ma.masked_invalid(rate_map),
                origin="lower",
                cmap=self._CMAP,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            cell_ax.set_aspect("equal")
            cell_ax.set_xticks([])
            cell_ax.set_yticks([])

            for spine in cell_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.25)
                spine.set_edgecolor("0.35")

            cell_ax.text(
                0.03,
                0.97,
                f"c{tile.cell}\nI={tile.spatial_information:.2f}",
                transform=cell_ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.5,
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.65,
                    "edgecolor": "none",
                    "pad": 0.3,
                },
            )

        ax.axis("off")
