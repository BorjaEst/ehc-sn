from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
import torch
from matplotlib.axes import Axes

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.plots.rasterplot import plot_activation, plot_observations
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.modules.lec import LECModel
from ehc_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a per-frequency LEC overview with observations and parameters."""
    return LECOverview(trace, ctx).plot()


class LECOverview(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the LEC overview."""

    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [1.0, 2.5], "height_ratios": [1.0, 5.0]}
    MOSAIC = [
        ["params", "observations"],
        ["params", "activations"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("state/lec/cells")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("state/lec/cells", f) for f in range(self.n_freq)]

        self.obs_values = self.trace.get("world_step/observation")[:, self.env_idx]
        self.cells = [self.trace.get(f"state/lec/cells/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]

        lec: LECModel = self.ctx.extras.get("lec")
        self.alpha = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.filter.alpha]
        self.w_f = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.w_f]

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def params(self, ax: Axes) -> None:
        """Plot per-frequency LEC parameters if available."""
        n_freq = min(len(self.alpha), len(self.w_f))
        freq_ids = np.arange(n_freq)
        ax.plot(freq_ids, self.alpha[:n_freq], marker="o", label="sigmoid(alpha)")
        ax.plot(freq_ids, self.w_f[:n_freq], marker="s", label="sigmoid(w_f)")
        ax.set_title("LEC parameters by frequency")
        ax.set_xlabel("Frequency index")
        ax.set_ylabel("Value")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize="small")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def observations(self, ax: Axes) -> None:
        """Plot LEC activations over time for all frequencies."""
        plot_observations(ax, self.obs_values)
        ax.set_title("Observations timeseries (one-hot encoded)")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def activations(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        nrows = len(self.freq_idxs)
        options = {"vmin": 0.0, "vmax": 1.0, "cmap": "GnBu"}
        for freq_idx, freq_ax in enumerate(subdivide_axes(ax, nrows, 1, hspace=0.1)):
            plot_activation(freq_ax, self.cells[freq_idx], **options)
            freq_ax.set_title(f"Activation timeseries - Freq {freq_idx}", fontsize=7)
            freq_ax.set_yticks([]); freq_ax.set_xticks([])  # fmt: skip
