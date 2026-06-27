"""MEC autocorrelogram population mosaic -- flat wrapped small-multiple layout."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.mec import (
    MECAutocorrMosaicData,
    select_mec_autocorr_mosaic,
)
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return MECAutocorrMosaicFigure(
        select_mec_autocorr_mosaic(trace, ctx), ctx
    ).plot()


@dataclass(frozen=True)
class _MosaicTile:
    """One autocorrelogram tile in the flat mosaic."""

    freq: int
    cell: int
    gridness: float
    autocorr: NDArray[np.floating]


class MECAutocorrMosaicFigure(BaseFigureTemplate):
    """Flat wrapped autocorrelogram mosaic.

    Each tile is one cell's spatial autocorrelogram.
    Tiles are ordered by (frequency ascending, gridness descending).
    All tiles have the same physical size.
    """

    HEIGHT_FRAC: float = 0.58
    MOSAIC = [["mosaic"]]

    _N_COLS: int = 8
    _MAX_CELLS: int = 56

    # Shared colormap with NaN color = light gray.
    _CMAP = plt.get_cmap("coolwarm").copy()
    _CMAP.set_bad("0.85")

    def __init__(self, data: MECAutocorrMosaicData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # ------------------------------------------------------------------
    # Build flat tile list
    # ------------------------------------------------------------------

    def _build_tiles(self) -> list[_MosaicTile]:
        """Flatten the selector's already-filtered/sorted per-frequency output."""
        data = self.data
        tiles: list[_MosaicTile] = []

        for freq, gridness, autocorrs, cell_indices in zip(
            data.freq_indices,
            data.gridness_by_freq,
            data.autocorrs_by_freq,
            data.cell_indices_by_freq,
            strict=True,
        ):
            for i in range(len(gridness)):
                tiles.append(
                    _MosaicTile(
                        freq=int(freq),
                        cell=int(cell_indices[i]),
                        gridness=float(gridness[i]),
                        autocorr=autocorrs[i],
                    )
                )

        # Global cap.  The selector already sorts and caps per frequency.
        return tiles[: self._MAX_CELLS]

    # ------------------------------------------------------------------
    # Panel: mosaic
    # ------------------------------------------------------------------

    @panel()
    def mosaic(self, ax: Axes) -> None:
        tiles = self._build_tiles()

        if not tiles:
            ax.text(
                0.5,
                0.5,
                "No finite MEC gridness values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return

        n_tiles = len(tiles)
        n_cols = self._N_COLS
        n_rows = int(np.ceil(n_tiles / n_cols))

        cell_axes = subdivide_axes(
            ax,
            n_rows,
            n_cols,
            wspace=0.004,
            hspace=0.004,
        )
        ax_list = list(np.atleast_1d(cell_axes).ravel())

        for i, cell_ax in enumerate(ax_list):
            if i >= n_tiles:
                cell_ax.axis("off")
                continue

            self._plot_tile(cell_ax, tiles[i])

        ax.axis("off")

    # ------------------------------------------------------------------
    # Render one tile
    # ------------------------------------------------------------------

    def _plot_tile(self, ax: Axes, tile: _MosaicTile) -> None:
        autocorr = tile.autocorr

        if autocorr.size == 0 or not np.isfinite(autocorr).any():
            ax.axis("off")
            return

        ax.imshow(
            np.ma.masked_invalid(autocorr),
            origin="lower",
            cmap=self._CMAP,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Thin tile border.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.25)
            spine.set_edgecolor("0.35")

        # In-panel label at top-left.
        ax.text(
            0.03,
            0.97,
            f"f{tile.freq} c{tile.cell}\nG={tile.gridness:.2f}",
            transform=ax.transAxes,
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
