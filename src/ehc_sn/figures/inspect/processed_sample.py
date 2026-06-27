"""Multi-panel visualization of a processed maze sample.

Renders a single sample's channels grouped into semantically coherent panels:

- **Navigation** — ``topology`` + ``start``, ``goals``, ``solution``
- **Structure** — ``topology`` + ``regions``, ``landmarks``
- **Perception** — ``topology`` + ``observations``, ``mask_valid``

"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# Grid2d channel name constants (string-only, no build-layer dependency).
_CHANNEL_TOPOLOGY: str = "topology"
_CHANNEL_OBSERVATIONS: str = "observations"
_CHANNEL_MASK_VALID: str = "mask_valid"
_CHANNEL_REGIONS: str = "regions"
_CHANNEL_LANDMARKS: str = "landmarks"

from ehc_sn.figures.core.base import BaseFigureTemplate
from ehc_sn.figures.core.panels import panel
from ehc_sn.figures.registry import FigureContext


def plot_processed_sample(
    channels: dict[str, Any],
    *,
    ctx: FigureContext | None = None,
) -> Figure:
    """Render a multi-panel visualization of a processed maze sample.

    Args:
        channels: Dict of channel name → ``ndarray`` or ``Tensor`` with shape
            ``(H, W)``. Must contain at least ``"topology"``.
        ctx: Optional figure context for styling overrides.

    Returns:
        Matplotlib ``Figure`` with one panel per active channel group.
    """
    np_channels = {k: _to_numpy(v) for k, v in channels.items()}

    if _CHANNEL_TOPOLOGY not in np_channels:
        raise ValueError(
            f"channels must contain the mandatory '{_CHANNEL_TOPOLOGY}' channel"
        )

    if ctx is None:
        ctx = FigureContext()

    return ProcessedSampleFigure(np_channels, ctx).plot()


class ProcessedSampleFigure(BaseFigureTemplate):
    """Multi-panel figure for a single processed-data sample."""

    HEIGHT_FRAC: float = 0.22
    MOSAIC: list[list[str]] = [["navigation", "structure", "perception"]]

    def __init__(
        self, channels: dict[str, np.ndarray], ctx: FigureContext
    ) -> None:
        super().__init__(channels, ctx)

    @panel(order=0)
    def navigation(self, ax: Axes) -> None:
        _plot_topology_base(ax, self.data[_CHANNEL_TOPOLOGY])
        if "solution" in self.data:
            _overlay_sequential(ax, self.data["solution"], cmap="YlOrRd")
        if "start" in self.data:
            _overlay_bool(ax, self.data["start"], "#2b6cb0")
        if "goals" in self.data:
            _overlay_bool(ax, self.data["goals"], "#d69e2e")
        ax.set_title("Navigation", fontsize=6)

    @panel(order=1)
    def structure(self, ax: Axes) -> None:
        _plot_topology_base(ax, self.data[_CHANNEL_TOPOLOGY])
        if _CHANNEL_REGIONS in self.data:
            _overlay_categorical(ax, self.data[_CHANNEL_REGIONS], cmap="Set3")
        if _CHANNEL_LANDMARKS in self.data:
            _overlay_categorical(
                ax, self.data[_CHANNEL_LANDMARKS], cmap="Accent"
            )
        ax.set_title("Structure", fontsize=6)

    @panel(order=2)
    def perception(self, ax: Axes) -> None:
        _plot_topology_base(ax, self.data[_CHANNEL_TOPOLOGY])
        if _CHANNEL_OBSERVATIONS in self.data:
            _overlay_categorical(
                ax, self.data[_CHANNEL_OBSERVATIONS], cmap="tab20"
            )
        if _CHANNEL_MASK_VALID in self.data:
            _overlay_bool(
                ax, self.data[_CHANNEL_MASK_VALID], "#38a169", alpha=0.3
            )
        ax.set_title("Perception", fontsize=6)


def _to_numpy(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _plot_topology_base(ax: Axes, topology: np.ndarray) -> None:
    cmap = ListedColormap(["#1b1f24", "#f7f4ef"])
    ax.imshow(
        topology.astype(float),
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _overlay_bool(
    ax: Axes, mask: np.ndarray, color: str, *, alpha: float = 0.7
) -> None:
    from matplotlib.colors import to_rgba

    rgba = np.zeros((*mask.shape, 4))
    r, g, b, _ = to_rgba(color)
    rgba[mask, :] = [r, g, b, alpha]
    ax.imshow(rgba, origin="upper", interpolation="nearest")


def _overlay_categorical(
    ax: Axes, arr: np.ndarray, *, cmap: str = "tab20", alpha: float = 0.7
) -> None:
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(
        masked, cmap=cmap, origin="upper", interpolation="nearest", alpha=alpha
    )


def _overlay_sequential(
    ax: Axes, arr: np.ndarray, *, cmap: str = "viridis", alpha: float = 0.6
) -> None:
    masked = np.ma.masked_where(arr <= 0, arr)
    ax.imshow(
        masked, cmap=cmap, origin="upper", interpolation="nearest", alpha=alpha
    )
