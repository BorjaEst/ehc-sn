"""Multi-panel visualization of a processed maze sample.

Renders a single sample's channels grouped into semantically coherent panels:

- **Navigation** — ``topology`` + ``start``, ``goals``, ``solution``
- **Structure** — ``topology`` + ``regions``, ``landmarks``
- **Perception** — ``topology`` + ``observations``, ``mask_valid``

Panels whose overlay channels are all absent are omitted automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ehc_sn.data.schema import (  # fmt: skip
    CHANNEL_GOALS,
    CHANNEL_LANDMARKS,
    CHANNEL_MASK_VALID,
    CHANNEL_OBSERVATIONS,
    CHANNEL_REGIONS,
    CHANNEL_SOLUTION,
    CHANNEL_START,
    CHANNEL_TOPOLOGY,
)
from ehc_sn.figures.figures.base import BaseFigureTemplate
from ehc_sn.figures.figures.panels import panel
from ehc_sn.figures.registry import FigureContext


# =================================================================================================
def plot(  # --------------------------------------------------------------------------------------
    channels: dict[str, Any], *, ctx: FigureContext | None = None,
) -> Figure:  # fmt: skip
    """Render a multi-panel visualization of a processed maze sample.

    Args:
        channels: Dict of channel name → ``ndarray`` or ``Tensor`` with shape
            ``(H, W)``. Must contain at least ``"topology"``.
        ctx: Optional figure context for styling overrides.

    Returns:
        Matplotlib ``Figure`` with one panel per active channel group.

    Example::

        from ehc_sn.data.datasets import MazeDataset
        from ehc_sn.data.index import read_index
        from ehc_sn.figures.modules.processed import plot

        entries = read_index(Path("data/processed/dungeons/index.jsonl"))
        ds = MazeDataset(entries, Path("data/processed/dungeons"), transform=None)
        fig = plot(ds[0])
        fig.savefig("sample.pdf")
    """
    np_channels = {k: _to_numpy(v) for k, v in channels.items()}

    if CHANNEL_TOPOLOGY not in np_channels:
        raise ValueError(f"channels must contain the mandatory '{CHANNEL_TOPOLOGY}' channel")

    if ctx is None:
        ctx = FigureContext()

    return ProcessedSampleFigure(np_channels, ctx).plot()


# =================================================================================================
class ProcessedSampleFigure(BaseFigureTemplate):
    """Multi-panel figure for a single processed-data sample.

    Dynamically builds its MOSAIC from whichever overlay channels are present.
    """

    HEIGHT_FRAC: float = 0.22
    MOSAIC: list[list[str]] = [["navigation", "structure", "perception"]]

    def __init__(  # ------------------------------------------------------------------------------
        self, channels: dict[str, np.ndarray], ctx: FigureContext,
    ) -> None:  # fmt: skip
        """Create the figure template for a single processed sample.

        Args:
            channels: Channel name → numpy array, typically produced by a dataset transform.
            ctx: Figure context for styling and output controls.
        """
        self.channels = channels
        super().__init__(None, ctx)  # trace=None — not used

    @panel(order=0)
    def navigation(  # ----------------------------------------------------------------------------
        self, ax: Axes,
    ) -> None:  # fmt: skip
        """Topology + start (blue), goals (gold), solution (sequential)."""
        _plot_topology_base(ax, self.channels[CHANNEL_TOPOLOGY])
        if CHANNEL_SOLUTION in self.channels:
            _overlay_sequential(ax, self.channels[CHANNEL_SOLUTION], cmap="YlOrRd")
        if CHANNEL_START in self.channels:
            _overlay_bool(ax, self.channels[CHANNEL_START], "#2b6cb0")
        if CHANNEL_GOALS in self.channels:
            _overlay_bool(ax, self.channels[CHANNEL_GOALS], "#d69e2e")
        ax.set_title("Navigation", fontsize=6)

    @panel(order=1)
    def structure(  # -----------------------------------------------------------------------------
        self, ax: Axes,
    ) -> None:  # fmt: skip
        """Topology + regions (discrete cmap), landmarks (markers)."""
        _plot_topology_base(ax, self.channels[CHANNEL_TOPOLOGY])
        if CHANNEL_REGIONS in self.channels:
            _overlay_categorical(ax, self.channels[CHANNEL_REGIONS], cmap="Set3")
        if CHANNEL_LANDMARKS in self.channels:
            _overlay_categorical(ax, self.channels[CHANNEL_LANDMARKS], cmap="Accent")
        ax.set_title("Structure", fontsize=6)

    @panel(order=2)
    def perception(  # ----------------------------------------------------------------------------
        self, ax: Axes,
    ) -> None:  # fmt: skip
        """Topology + observations (discrete cmap), mask_valid (green/red)."""
        _plot_topology_base(ax, self.channels[CHANNEL_TOPOLOGY])
        if CHANNEL_OBSERVATIONS in self.channels:
            _overlay_categorical(ax, self.channels[CHANNEL_OBSERVATIONS], cmap="tab20")
        if CHANNEL_MASK_VALID in self.channels:
            _overlay_bool(ax, self.channels[CHANNEL_MASK_VALID], "#38a169", alpha=0.3)
        ax.set_title("Perception", fontsize=6)


# =================================================================================================
def _to_numpy(  #----------------------------------------------------------------------------------
    v: Any,
) -> np.ndarray:  # fmt: skip
    """Convert a value to a numpy array (no-op if already ndarray)."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


# =================================================================================================
def _plot_topology_base(  # -----------------------------------------------------------------------
    ax: Axes, topology: np.ndarray,
) -> None:  # fmt: skip
    """Render the topology channel as a black/white grid (base layer).

    Walls are black, passable cells are white.
    """
    cmap = ListedColormap(["#1b1f24", "#f7f4ef"])  # False=wall, True=passable
    ax.imshow(topology.astype(float), cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# =================================================================================================
def _overlay_bool(  # -----------------------------------------------------------------------------
    ax: Axes, mask: np.ndarray, color: str, *, alpha: float = 0.7,
) -> None:  # fmt: skip
    """Overlay a boolean mask as coloured semi-transparent cells."""
    rgba = np.zeros((*mask.shape, 4))
    from matplotlib.colors import to_rgba

    r, g, b, _ = to_rgba(color)
    rgba[mask, :] = [r, g, b, alpha]
    ax.imshow(rgba, origin="upper", interpolation="nearest")


# =================================================================================================
def _overlay_categorical(  # ----------------------------------------------------------------------
    ax: Axes, arr: np.ndarray, *, cmap: str = "tab20", alpha: float = 0.7,
) -> None:  # fmt: skip
    """Overlay an int32 categorical channel with a discrete colormap."""
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(masked, cmap=cmap, origin="upper", interpolation="nearest", alpha=alpha)


# =================================================================================================
def _overlay_sequential(  # -----------------------------------------------------------------------
    ax: Axes, arr: np.ndarray, *, cmap: str = "viridis", alpha: float = 0.6,
) -> None:  # fmt: skip
    """Overlay an int32 ordinal channel with a sequential colormap."""
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(masked, cmap=cmap, origin="upper", interpolation="nearest", alpha=alpha)


# =================================================================================================
__all__ = ["plot"]
    """Overlay an int32 ordinal channel with a sequential colormap."""
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(masked, cmap=cmap, origin="upper", interpolation="nearest", alpha=alpha)


# =================================================================================================
__all__ = ["plot"]
