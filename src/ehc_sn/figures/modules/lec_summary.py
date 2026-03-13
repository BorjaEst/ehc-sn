from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import colorbar, panel
from ehc_sn.figures.plots.rasterplot import plot_activation, plot_observations
from ehc_sn.figures.registry import FigureContext
from ehc_sn.figures.utils.axes import subdivide_axes
from ehc_sn.rollouts.trace_tree import TraceTree


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a per-frequency LEC overview from semantic rollout diagnostics."""
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
        self.n_freq = trace.n_freq("diagnostic/lec/cells")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("diagnostic/lec/cells", f) for f in range(self.n_freq)]

        self.obs_values = self.trace.get("world_step/observation")[:, self.env_idx]
        self.cells = [
            self.trace.get(f"diagnostic/lec/cells/{f}")[:, self.env_idx, :] 
            for f in range(self.n_freq)
        ]  # fmt: skip

        self.mean_activity = np.asarray([float(np.mean(cells)) for cells in self.cells])
        self.peak_activity = np.asarray([float(np.max(cells)) for cells in self.cells])

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def params(self, ax: Axes) -> None:
        """Plot per-frequency LEC activation summaries derived from the rollout trace."""
        freq_ids = np.arange(self.n_freq)
        ax.plot(freq_ids, self.mean_activity, marker="o", label="mean activation")
        ax.plot(freq_ids, self.peak_activity, marker="s", label="peak activation")
        ax.set_title("LEC activity summary by frequency")
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
        child_axes = np.ravel(subdivide_axes(ax, nrows, 1, hspace=0.1))
        for freq_idx, freq_ax in enumerate(child_axes):
            plot_activation(freq_ax, self.cells[freq_idx], **options)
            freq_ax.set_title(f"Activation timeseries - Freq {freq_idx}", fontsize=7)
            freq_ax.set_yticks([]); freq_ax.set_xticks([])  # fmt: skip
