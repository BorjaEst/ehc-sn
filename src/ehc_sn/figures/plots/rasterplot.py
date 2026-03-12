"""Rasterplot utilities for observations and activations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable


def plot_observations(
    ax: Axes,
    observations: np.ndarray,
    *,
    cmap: str = "GnBu",
    **options: Dict[str, Any],
) -> ScalarMappable:
    """Plot the observation raster panel."""
    return ax.imshow(
        observations.T,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        **options,
    )


def plot_activation(
    ax: Axes,
    activation: np.ndarray,
    *,
    cmap: str = "GnBu",
    **options: Dict[str, Any],
) -> ScalarMappable | None:
    """Plot a single activation heatmap panel."""
    return ax.imshow(
        activation.T,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        **options,
    )
