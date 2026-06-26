"""Arena TEM prediction-mazehard_solution_overlay figure template.

Ground-truth vs predicted observation IDs for N samples at the final rollout
step.  Each sample shows a horizontal bar: GT on the left, then the three
TEM prediction variants (inference, retrieved, ancestral), colour-mapped by
observation-id integer value.
"""

from __future__ import annotations

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.arena_tem import (
    TEMOverlayData,
    select_tem_prediction_overlay,
)
from ehc_sn.figures.utils.colors import categorical_id_colormap
from ehc_sn.traces.trace_tree import TraceTree

# ── Colour map constants ────────────────────────────────────────────────────
_LABELS = ("GT", "Inference", "Retrieved", "Ancestral")


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return ArenaObservationOverlayFigure(
        select_tem_prediction_overlay(trace, ctx), ctx
    ).plot()


class ArenaObservationOverlayFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.18
    MOSAIC = [["overlay_grid", "legend_panel"]]
    MOSAIC_KWARGS = {
        "width_ratios": [8.0, 1.4],
        "gridspec_kw": {"wspace": 0.05},
    }

    def __init__(self, data: TEMOverlayData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

        # Pre-compute colormap for the full observation-ID range.
        max_id = max(
            int(data.gt_obs_ids.max()),
            int(data.pred_post.max()),
            int(data.pred_recall.max()),
            int(data.pred_path.max()),
        )
        n_obs = max_id + 1
        cmap, norm = categorical_id_colormap(n_obs)
        self._cmap = cmap
        self._norm = norm

    @panel()
    def overlay_grid(self, ax: Axes) -> None:
        """Render the GT-vs-prediction colour grid."""
        data = self.data
        n = data.gt_obs_ids.shape[0]
        rows = 4
        cols = data.gt_obs_ids.shape[1]

        grid = np.stack(
            [
                data.gt_obs_ids,
                data.pred_post,
                data.pred_recall,
                data.pred_path,
            ],
            axis=0,
        )  # (4, n, T)

        coloured = self._cmap(self._norm(grid))  # (4, n, T, 4)

        for row_idx in range(rows):
            for sample_idx in range(n):
                y0 = row_idx * n + sample_idx
                cell_colours = coloured[row_idx, sample_idx]

                for t in range(cols):
                    ax.add_patch(
                        mpatches.Polygon(
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

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Prediction source")

        ax.set_yticks(
            np.arange(n / 2, rows * n, n),
            _LABELS,
            fontsize="small",
        )

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows * n)
        ax.invert_yaxis()

        ax.tick_params(axis="y", labelsize="small")

    @panel()
    def legend_panel(self, ax: Axes) -> None:
        """Render a compact categorical legend for observation IDs."""
        ax.axis("off")

        n_obs = self._cmap.N

        # For many IDs, use multiple columns and sparse labels.
        n_cols = 3 if n_obs > 24 else 2
        n_rows = int(np.ceil(n_obs / n_cols))

        swatch_w = 0.8
        swatch_h = 0.8
        x_gap = 1.0
        y_gap = 0.15

        ax.set_title("Obs ID", fontsize="small", pad=4)

        for obs_id in range(n_obs):
            col = obs_id // n_rows
            row = obs_id % n_rows

            # top-to-bottom layout
            y = n_rows - 1 - row
            x = col * (swatch_w + x_gap + 0.9)

            colour = self._cmap(self._norm(obs_id))

            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    swatch_w,
                    swatch_h,
                    facecolor=colour,
                    edgecolor="none",
                )
            )

            # For many categories, do not label every single one.
            if n_obs <= 20 or obs_id % 5 == 0:
                ax.text(
                    x + swatch_w + 0.15,
                    y + swatch_h / 2,
                    str(obs_id),
                    va="center",
                    ha="left",
                    fontsize="x-small",
                )

        # Small explanatory note.
        if n_obs > 20:
            ax.text(
                0,
                -1.1,
                "Distinct colours denote distinct\nobservation IDs\n(labels shown every 5 IDs)",
                fontsize="xx-small",
                ha="left",
                va="top",
            )

        total_w = n_cols * (swatch_w + x_gap + 0.9)
        ax.set_xlim(0, total_w)
        ax.set_ylim(-1.5, n_rows + 0.5)


# =============================================================================
__all__ = [
    "ArenaObservationOverlayFigure",
    "plot",
]
