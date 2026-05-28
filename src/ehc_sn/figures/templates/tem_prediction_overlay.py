"""Arena TEM prediction-overlay figure template.

Ground-truth vs predicted observation IDs for N samples at the final rollout
step.  Each sample shows a horizontal bar: GT on the left, then the three
TEM prediction variants (inference, retrieved, ancestral), colour-mapped by
observation-id integer value.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.arena_tem import (
    TEMOverlayData,
    select_tem_prediction_overlay,
)
from ehc_sn.traces.trace_tree import TraceTree

# ── Colour map constants ────────────────────────────────────────────────────
_CMAP_NAME = "tab10"
_LABELS = ("GT", "Inference", "Retrieved", "Ancestral")
_LABEL_COLORS = ("black", "steelblue", "firebrick", "darkgreen")


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return TEMPredictionOverlayFigure(
        select_tem_prediction_overlay(trace, ctx), ctx
    ).plot()


class TEMPredictionOverlayFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.18
    MOSAIC = [["overlay_grid", "legend_panel"]]
    MOSAIC_KWARGS = {
        "width_ratios": [8.0, 1.5],
        "gridspec_kw": {"wspace": 0.02},
    }

    def __init__(self, data: TEMOverlayData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)
        self._cmap = mcolors.Colormap(_CMAP_NAME)

    @panel()
    def overlay_grid(self, ax: Axes) -> None:
        """Render the GT-vs-prediction colour grid."""
        data = self.data
        n = len(data.gt_obs_ids)
        rows = 4  # GT + 3 predictions
        cols = data.gt_obs_ids.shape[1]  # T_max

        # Build the colour grid: (rows, n, cols).
        grid = np.stack(
            [
                data.gt_obs_ids,  # (n, T_max)
                # Broadcast per-sample scalar predictions to full T_max length
                # for panel consistency.
                np.full_like(data.gt_obs_ids, data.pred_inference[:, None]),
                np.full_like(data.gt_obs_ids, data.pred_retrieved[:, None]),
                np.full_like(data.gt_obs_ids, data.pred_ancestral[:, None]),
            ],
            axis=0,
        )  # (4, n, T_max)

        # Normalise to [0, 1] for colormap lookup.
        vmin = grid.min()
        vmax = grid.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1))
        coloured = self._cmap(norm(grid))  # (4, n, T_max, 4)

        # Render each sample as a horizontal stack.
        for row_idx in range(rows):
            for sample_idx in range(n):
                y0 = (rows - 1 - row_idx) * n + sample_idx
                cell_colours = coloured[row_idx, sample_idx]
                for t in range(cols):
                    ax.add_patch(
                        mcolors.Polygon(
                            [
                                (t, y0),
                                (t + 1, y0),
                                (t + 1, y0 + 1),
                                (t, y0 + 1),
                            ],
                            facecolor=cell_colours[t],
                            edgecolor="none",
                        )
                    )

        # Axis labels.
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Sample")
        ax.set_yticks(
            np.arange(n // 2, rows * n, n),
            [_LABELS[r] for r in range(rows)],
            fontsize="small",
        )
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows * n)
        ax.tick_params(axis="y", labelsize="small")
        ax.invert_yaxis()

    @panel()
    def legend_panel(self, ax: Axes) -> None:
        """Render a colourbar and per-row label legend."""
        ax.axis("off")

        # Colourbar for observation-id values.
        norm = mcolors.Normalize(
            vmin=self.data.gt_obs_ids.min(),
            vmax=max(
                self.data.gt_obs_ids.max(),
                self.data.gt_obs_ids.min() + 1,
            ),
        )
        mappable = mcolors.ScalarMappable(cmap=self._cmap, norm=norm)
        mappable.set_array([])
        cbar = self.fig.colorbar(mappable, ax=ax, shrink=0.6)
        cbar.set_label("Obs ID", fontsize="small")

        # Per-row label legend.
        patches = [
            Patch(facecolor="none", edgecolor=c, label=l)
            for l, c in zip(_LABELS, _LABEL_COLORS)
        ]
        ax.legend(
            handles=patches,
            loc="lower left",
            fontsize="x-small",
            frameon=False,
        )


# =============================================================================
__all__ = [
    "TEMPredictionOverlayFigure",
    "plot",
]
