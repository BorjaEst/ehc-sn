"""LEC summary figure template."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import colorbar, panel
from ehc_sn.figures.plots.raster import plot_activation, plot_observations
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.selectors.lec import LECSummaryFigureData, select_lec_summary
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.traces.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    return LECSummaryFigure(select_lec_summary(trace, ctx), ctx).plot()


class LECSummaryFigure(BaseFigureTemplate):
    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [1.0, 2.5], "height_ratios": [1.0, 5.0]}
    MOSAIC = [["params", "observations"], ["params", "activations"]]

    def __init__(self, data: LECSummaryFigureData, ctx: FigureContext) -> None:
        super().__init__(data, ctx)

    @panel()
    def params(self, ax: Axes) -> None:
        n_freq = min(len(self.data.alpha), len(self.data.w_f))
        freq_ids = np.arange(n_freq)
        ax.plot(freq_ids, self.data.alpha[:n_freq], marker="o", label="sigmoid(alpha)")
        ax.plot(freq_ids, self.data.w_f[:n_freq], marker="s", label="sigmoid(w_f)")
        ax.set_title("LEC parameters by frequency")
        ax.set_xlabel("Frequency index")
        ax.set_ylabel("Value")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize="small")

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def observations(self, ax: Axes) -> None:
        plot_observations(ax, self.data.obs_values)
        ax.set_title("Observations timeseries (one-hot encoded)")

    @colorbar(group="lec_activity", label="Activation")
    @panel()
    def activations(self, ax: Axes) -> None:
        options = {"vmin": 0.0, "vmax": 1.0, "cmap": "GnBu"}
        axes = subdivide_axes(ax, len(self.data.freq_idxs), 1, hspace=0.1, squeeze=True)
        freq_axes = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else [axes]
        for freq_idx, freq_ax in enumerate(freq_axes):
            plot_activation(freq_ax, self.data.cells[freq_idx], **options)
            freq_ax.set_title(f"Activation timeseries - Freq {freq_idx}", fontsize=7)
            freq_ax.set_yticks([]); freq_ax.set_xticks([])  # fmt: skip
