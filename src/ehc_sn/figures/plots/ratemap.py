"""Rate-map drawing functions.

``PreparedRateMap`` is defined in ``ehc_sn.figures._contracts`` and produced
by the spatial selector layer (``ehc_sn.figures.selectors.spatial``).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ehc_sn.figures._contracts import PreparedRateMap

__all__ = ["plot_rate_map", "plot_rate_map_mosaic"]



def plot_rate_map(
    ax: Axes,
    prepared_rate_map: PreparedRateMap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Axes:
    """Plot one prepared rate map as a raster image."""
    rate_map = np.asarray(prepared_rate_map.rate_map, dtype=float)

    if rate_map.size == 0 or not np.isfinite(rate_map).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    extent = _imshow_extent_from_center_extent(prepared_rate_map.extent, rate_map.shape)

    finite = np.isfinite(rate_map)
    if vmin is None:
        vmin = float(np.nanmin(rate_map[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(rate_map[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")
    ax.imshow(
        rate_map,
        origin="upper",
        interpolation="nearest",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    ax.axis("off")
    return ax


def plot_rate_map_mosaic(
    axes: Sequence[Axes] | Axes,
    prepared_rate_maps: Sequence[PreparedRateMap],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Sequence[Axes]:
    """Render multiple prepared rate maps into provided axes."""
    axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else ([axes] if isinstance(axes, Axes) else list(axes))
    if not axes_list:
        return axes_list

    if not prepared_rate_maps:
        axes_list[0].text(0.5, 0.5, "No data", ha="center", va="center")
        for empty_ax in axes_list:
            empty_ax.axis("off")
        return axes_list

    for ax, prepared_rate_map in zip(axes_list, prepared_rate_maps):
        plot_rate_map(ax, prepared_rate_map, vmin=vmin, vmax=vmax, cmap=cmap)

    for ax in axes_list[len(prepared_rate_maps) :]:
        ax.axis("off")

    return axes_list


def _imshow_extent_from_center_extent(
    extent: tuple[float, float, float, float],
    shape: tuple[int, int],
    *,
    min_span: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return an ``imshow`` edge extent from center-based raster coordinates."""
    n_rows, n_cols = shape
    xmin, xmax, ymin, ymax = extent
    left, right = _pixel_edge_bounds(xmin, xmax, n_cols, min_span=min_span)
    bottom, top = _pixel_edge_bounds(ymin, ymax, n_rows, min_span=min_span)
    return left, right, top, bottom


def _pixel_edge_bounds(
    min_coord: float,
    max_coord: float,
    n_pixels: int,
    *,
    min_span: float = 1.0,
) -> tuple[float, float]:
    """Return edge bounds for a 1D center-based raster coordinate span."""
    span = float(max_coord) - float(min_coord)
    if n_pixels <= 1 or not np.isfinite(span) or span <= 0:
        center = 0.5 * (float(min_coord) + float(max_coord))
        half = min_span / 2.0
        return center - half, center + half

    step = span / float(n_pixels - 1)
    return float(min_coord) - step / 2.0, float(max_coord) + step / 2.0
