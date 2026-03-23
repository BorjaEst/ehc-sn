from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ehc_sn.figures.plots.map import plot_map
from ehc_sn.figures.utils.axes import _environment_n_locations
from ehc_sn.figures.utils.rasterize import rasterize_locations
from ehc_sn.rollouts.analysis import aggregate_rate_map


def plot_ratemap_cell(
    ax: Axes,
    world: object,
    cells: np.ndarray,
    location_ids: list[int],
    cell_idx: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    shape: str = "square",
    cmap: str = "copper_r",
) -> Axes:
    """Plot a single cell's rate map on an existing axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_idx: Index of the cell to plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        shape: Shape for the background map markers.
        cmap: Colormap name for the rate map.

    Returns:
        The axes with the rate map rendered.
    """
    n_locations = _environment_n_locations(world)
    rate_map, _ = aggregate_rate_map(cells, location_ids, n_locations)
    if rate_map.size == 0 or cell_idx < 0 or cell_idx >= rate_map.shape[0]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    values = rate_map[cell_idx]

    finite_mask = np.isfinite(values)
    if vmin is None:
        vmin = float(np.min(values[finite_mask])) if finite_mask.any() else 0.0
    if vmax is None:
        vmax = float(np.max(values[finite_mask])) if finite_mask.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    plot_map(world, values, ax=ax, vmin=vmin, vmax=vmax, shape=shape, location_cm=cmap)
    return ax


def plot_ratematx_cell(
    ax: Axes,
    world: object,
    cells: np.ndarray,
    location_ids: list[int],
    cell_idx: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Axes:
    """Plot a single cell's rate matrix on an existing axes.

    The rasterized matrix uses the same image/grid convention as the vector
    map renderers: ``o`` is the column index, ``y`` is the row index, and row
    0 is displayed at the top.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_idx: Index of the cell to plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        cmap: Colormap name for the rate map.

    Returns:
        The axes with the rate map matrix form rendered.
    """
    n_locations = _environment_n_locations(world)
    rate_map, _ = aggregate_rate_map(cells, location_ids, n_locations)
    if rate_map.size == 0 or cell_idx < 0 or cell_idx >= rate_map.shape[0]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    values = rate_map[cell_idx]
    grid, _, extent = rasterize_locations(world, values)
    extent = _expand_degenerate_extent(extent)

    if grid.size == 0 or not np.isfinite(grid).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    finite = np.isfinite(grid)
    if vmin is None:
        vmin = float(np.nanmin(grid[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(grid[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")  # NaNs show as light gray

    ax.imshow(
        grid,
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


def _expand_degenerate_extent(
    extent: tuple[float, float, float, float],
    *,
    min_span: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return an ``imshow`` extent with non-zero span on both axes."""
    xmin, xmax, ymin, ymax = extent
    if xmin == xmax:
        half = min_span / 2
        xmin -= half
        xmax += half
    if ymin == ymax:
        half = min_span / 2
        ymin -= half
        ymax += half
    return xmin, xmax, ymin, ymax


def plot_ratematx_mosaic(
    axes: Sequence[Axes],
    world: object,
    cells_trace: NDArray,
    location_ids: list[int],
    *,
    cell_indices: Optional[Sequence[int]] = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> Sequence[Axes]:
    """Render multiple cell rate maps into provided axes.

    Args:
        axes: Axes to draw into.
        world: Environment world with location coordinates.
        cells_trace: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_indices: Optional cell indices to include.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        cmap: Colormap name for the rate map.

    Returns:
        Sequence of axes that were provided.
    """
    axes_list = [axes] if isinstance(axes, Axes) else list(axes)
    if not axes_list:
        return axes_list

    cell_array = np.asarray(cells_trace)
    if cell_array.ndim < 2 or cell_array.shape[-1] == 0:
        axes_list[0].text(0.5, 0.5, "No data", ha="center", va="center")
        for empty_ax in axes_list:
            empty_ax.axis("off")
        return axes_list

    n_cells_total = int(cell_array.shape[-1])
    if cell_indices is None:
        indices = list(range(n_cells_total))
    else:
        indices = []
        for idx in cell_indices:
            idx_int = int(idx)
            if 0 <= idx_int < n_cells_total:
                indices.append(idx_int)

    if not indices:
        axes_list[0].text(0.5, 0.5, "No data", ha="center", va="center")
        for empty_ax in axes_list:
            empty_ax.axis("off")
        return axes_list

    options = {"vmin": vmin, "vmax": vmax, "cmap": cmap}
    for ax, cell_idx in zip(axes_list, indices):
        plot_ratematx_cell(ax, world, cells_trace, location_ids, cell_idx, **options)

    for ax in axes_list[len(indices) :]:
        ax.axis("off")

    return axes_list
