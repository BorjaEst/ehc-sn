"""HPC place-cell metrics — report-facing figure template.

Consumes ``HPCPlaceMetricsData`` to produce a fixed 1×3 layout:

    spatial-information strip | field-center scatter | top rate-map 2×2 mosaic

This is the canonical report/paper figure for HPC place-like spatial
selectivity.  It replaces the diagnostic autocorrelogram-centric
``hpc_cells`` layout with the correct visual grammar for place cells:
localised rate maps, spatial-information distributions, and field-center
coverage.

References
----------
Whittington et al. (2020). "The Tolman-Eichenbaum Machine".
    Cell 183(5):1248–1262 e23.
Whittington et al. (2022). "Relating transformers to models and neural
    representations of the hippocampal formation".  arXiv:2112.04035.
Hafting et al. (2005). "Microstructure of a spatial map in the entorhinal
    cortex".  Nature 436:801–806.
Skaggs et al. (1993). "An Information-Theoretic Approach to Deciphering
    the Hippocampal Code".  NeurIPS 5.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.hpc import (
    HPCPlaceMetricsData,
    select_hpc_place_metrics,
)
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return HPCPlaceMetricsFigure(
        select_hpc_place_metrics(trace, ctx), ctx
    ).plot()


class HPCPlaceMetricsFigure(BaseFigureTemplate):
    """Place-cell metrics by spatial information — report-facing figure.

    Fixed 1×3 report layout:

        spatial information | field centers | top rate-map 2×2 mosaic
    """

    HEIGHT_FRAC: float = 0.35
    MOSAIC = [["spatial_information", "field_centers", "top_maps"]]
    MOSAIC_KWARGS = {
        "width_ratios": [1.0, 1.0, 1.2],
        "gridspec_kw": {"wspace": 0.08},
    }

    _N_MOSAIC_ROWS: int = 2
    _N_MOSAIC_COLS: int = 2
    _MAX_EXAMPLES: int = 4

    # Sequential colormap for firing-rate maps (nonnegative values).
    _CMAP = plt.get_cmap("viridis").copy()
    _CMAP.set_bad("0.85")

    def __init__(self, data: HPCPlaceMetricsData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    # -- Panel: spatial information strip plot ---------------------------------

    @panel()
    def spatial_information(self, ax: Axes) -> None:
        """Strip plot of spatial information per cell with median marker."""
        data = self.data
        si = np.asarray(data.spatial_information, dtype=float)
        finite = si[np.isfinite(si)]

        if finite.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite SI values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.axis("off")
            return

        # Jittered strip at x=0.
        rng = np.random.default_rng(seed=0)
        jitter = rng.uniform(-0.25, 0.25, size=finite.size)
        ax.scatter(
            jitter,
            finite,
            s=12,
            alpha=0.5,
            edgecolors="none",
            zorder=2,
        )

        median = float(np.median(finite))
        ax.plot(
            [-0.35, 0.35],
            [median, median],
            color="C1",
            linewidth=1.5,
            zorder=3,
        )

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel("Spatial information (bits)", fontsize=9)
        ax.set_title("HPC spatial information", fontsize=10)

    # -- Panel: field centers over arena ---------------------------------------

    @panel()
    def field_centers(self, ax: Axes) -> None:
        """Scatter of place-field centers over the arena extent."""
        data = self.data
        fx = np.asarray(data.field_x, dtype=float)
        fy = np.asarray(data.field_y, dtype=float)
        si = np.asarray(data.spatial_information, dtype=float)
        fa = np.asarray(data.field_area, dtype=float)

        valid = np.isfinite(fx) & np.isfinite(fy)

        if not valid.any():
            ax.text(
                0.5,
                0.5,
                "No valid field centers",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.axis("off")
            return

        # Point size proportional to field area, clipped to a readable range.
        areas = fa[valid]
        if areas.size > 0 and np.nanmax(areas) > 0:
            sizes = 10.0 + 60.0 * (areas - np.nanmin(areas)) / (
                np.nanmax(areas) - np.nanmin(areas)
            )
        else:
            sizes = np.full(areas.shape, 20.0)

        sc = ax.scatter(
            fx[valid],
            fy[valid],
            c=si[valid],
            s=sizes,
            cmap=plt.get_cmap("viridis"),
            alpha=0.7,
            edgecolors="none",
            zorder=3,
        )

        xmin, xmax, ymin, ymax = data.extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_title("Place-field centers", fontsize=10)

        # Colour bar for SI.
        cb = self.fig.colorbar(sc, ax=ax, fraction=0.08, pad=0.04)
        cb.set_label("SI (bits)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # -- Panel: top rate-map examples (fixed 2×2) ------------------------------

    @panel()
    def top_maps(self, ax: Axes) -> None:
        """Fixed 2×2 mosaic of the top-N rate maps by spatial information."""
        data = self.data
        n_examples = min(len(data.top_rate_maps), self._MAX_EXAMPLES)

        if n_examples == 0:
            ax.text(
                0.5,
                0.5,
                "No examples",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.axis("off")
            return

        # Determine shared vmax across all top examples.
        all_values = np.concatenate(
            [
                rm.ravel()
                for rm in data.top_rate_maps[:n_examples]
                if rm.size > 0
            ]
        )
        finite_vals = all_values[np.isfinite(all_values)]
        if finite_vals.size > 0:
            vmax = float(np.nanpercentile(finite_vals, 99))
        else:
            vmax = 1.0

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

            rate_map = data.top_rate_maps[i]
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

            cell = int(data.top_cell_indices[i])
            si_val = float(data.top_spatial_information[i])
            cell_ax.text(
                0.03,
                0.97,
                f"c{cell}\nI={si_val:.2f}",
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
        ax.set_title("Top 4 place cells", fontsize=10, pad=6)
