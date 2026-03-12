"""Spatial autocorrelogram utilities for grid-cell diagnostics."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ehc_sn.figures.utils.rasterize import rasterize_locations
from ehc_sn.rollouts.analysis import aggregate_rate_map


def plot_spatial_autocorrelogram(
    ax: Axes,
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    cell_idx: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    grid_res: float | None = None,
    cmap: str = "coolwarm",
) -> Axes:
    """Plot a 2D spatial autocorrelogram for a selected cell.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells_trace: Cell activations (T, B, C) or (T, C).
        location_ids: Ordered list of visited location indices.
        cell_idx: Cell index to render.
        vmin: Optional min value for color scaling.
        vmax: Optional max value for color scaling.
        grid_res: Optional grid resolution for rasterization.
        cmap: Colormap name.

    Returns:
        The axes with the autocorrelogram rendered.
    """
    values = _rate_map_cell_values(cells_trace, location_ids, world, cell_idx)
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    grid, mask, _ = rasterize_locations(world, values, grid_res=grid_res)
    autocorr = spatial_autocorr_2d(grid, mask)
    if autocorr.size == 0 or not np.isfinite(autocorr).any():
        ax.text(0.5, 0.5, "No autocorr", ha="center", va="center")
        ax.axis("off")
        return ax

    finite = np.isfinite(autocorr)
    if vmin is None:
        vmin = float(np.nanmin(autocorr[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(autocorr[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    ax.imshow(autocorr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_radial_autocorr_cells(
    ax: Axes,
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    *,
    cell_indices: Optional[Sequence[int]] = None,
    grid_res: float | None = None,
    n_bins: int = 32,
    color: Optional[str] = None,
) -> Axes:
    """Plot mean radial autocorrelation profile with std across cells.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells_trace: Cell activations (T, B, C) or (T, C).
        location_ids: Ordered list of visited location indices.
        cell_indices: Optional cell indices to include. If None, use all cells.
        grid_res: Optional grid resolution for rasterization.
        n_bins: Number of radial bins.
        cmap: Colormap name for curve colors.

    Returns:
        The axes with the mean profile rendered.
    """
    options = {"cell_indices": cell_indices, "grid_res": grid_res, "n_bins": n_bins}
    radii, mean, std, n_profiles = _radial_autocorr_summary(world, cells_trace, location_ids, **options)
    if radii.size == 0 or not np.isfinite(mean).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    ax.plot(radii, mean, color=color, label=f"N={n_profiles}")
    color = ax.get_lines()[-1].get_color() if color is None else color
    ax.fill_between(radii, mean - std, mean + std, color=color, alpha=0.25)
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Radial autocorrelation (±1 std)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left", handlelength=1.0)
    return ax


def _radial_autocorr_summary(
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    *,
    cell_indices: Optional[Sequence[int]] = None,
    grid_res: float | None = None,
    n_bins: int = 32,
) -> tuple[NDArray, NDArray, NDArray, int]:
    location_ids = np.asarray(location_ids, dtype=int)
    n_locations = len(getattr(world, "locations", []))
    rate_map, _ = aggregate_rate_map(cells_trace, location_ids, n_locations)
    if rate_map.size == 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, 0

    if cell_indices is None:
        indices = range(rate_map.shape[0])
    else:
        indices = [int(idx) for idx in cell_indices]

    profiles: list[NDArray] = []
    ref_radii: NDArray | None = None
    for idx in indices:
        if idx < 0 or idx >= rate_map.shape[0]:
            continue
        values = rate_map[idx]
        if values.size == 0:
            continue
        grid, mask, _ = rasterize_locations(world, values, grid_res=grid_res)
        autocorr = spatial_autocorr_2d(grid, mask)
        if autocorr.size == 0 or not np.isfinite(autocorr).any():
            continue
        radii, profile = radial_profile(autocorr, n_bins=n_bins)
        if radii.size == 0 or profile.size == 0:
            continue
        if ref_radii is None:
            ref_radii = radii
        elif not np.allclose(radii, ref_radii, equal_nan=True):
            finite = np.isfinite(profile)
            if finite.sum() < 2:
                continue
            profile = np.interp(ref_radii, radii[finite], profile[finite], left=np.nan, right=np.nan)
        profiles.append(profile)

    if not profiles or ref_radii is None:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, 0

    stack = np.vstack(profiles)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    return ref_radii, mean, std, stack.shape[0]


def build_shared_autocorr_range(
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    cell_indices: Sequence[int],
    *,
    grid_res: float | None = None,
) -> tuple[float, float]:
    """Compute shared color scaling for autocorrelograms.

    Args:
        world: Environment world with location coordinates.
        cells_trace: Cell activations (T, B, C) or (T, C).
        location_ids: Ordered list of visited location indices.
        cell_indices: Cell indices to include.
        grid_res: Optional grid resolution for rasterization.

    Returns:
        Tuple of (vmin, vmax) over the selected autocorrelograms.
    """
    values: list[NDArray] = []
    for idx in cell_indices:
        rate_values = _rate_map_cell_values(cells_trace, location_ids, world, idx)
        if rate_values.size == 0:
            continue
        grid, mask, _ = rasterize_locations(world, rate_values, grid_res=grid_res)
        autocorr = spatial_autocorr_2d(grid, mask)
        if autocorr.size == 0:
            continue
        values.append(autocorr)

    if not values:
        return 0.0, 1.0
    flat = np.concatenate([v.ravel() for v in values])
    finite = np.isfinite(flat)
    if not finite.any():
        return 0.0, 1.0
    vmin = float(np.nanmin(flat[finite]))
    vmax = float(np.nanmax(flat[finite]))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def spatial_autocorr_2d(grid: NDArray, mask: NDArray) -> NDArray:
    """Compute a 2D spatial autocorrelogram via FFT.

    Args:
        grid: Rasterized values with NaNs for missing pixels.
        mask: Boolean mask indicating valid pixels.

    Returns:
        2D autocorrelogram with center at the grid midpoint.
    """
    if grid.size == 0:
        return np.zeros((0, 0), dtype=float)

    values = np.where(mask, grid, 0.0)
    mask_f = mask.astype(float)

    fft_vals = np.fft.fft2(values)
    fft_mask = np.fft.fft2(mask_f)
    corr = np.fft.ifft2(fft_vals * np.conj(fft_vals)).real
    norm = np.fft.ifft2(fft_mask * np.conj(fft_mask)).real

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(norm > 0, corr / norm, np.nan)

    return np.fft.fftshift(corr)


def radial_profile(autocorr: NDArray, *, n_bins: int = 32) -> tuple[NDArray, NDArray]:
    """Compute a radial profile from a 2D autocorrelogram.

    Args:
        autocorr: 2D autocorrelogram array.
        n_bins: Number of radial bins.

    Returns:
        Tuple of (radii, profile) for the autocorrelogram.
    """
    if autocorr.size == 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    yy, xx = np.indices(autocorr.shape)
    center_y = (autocorr.shape[0] - 1) / 2.0
    center_x = (autocorr.shape[1] - 1) / 2.0
    radii = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    max_radius = float(np.nanmax(radii)) if radii.size else 0.0
    if max_radius <= 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    bins = np.linspace(0.0, max_radius, n_bins + 1)
    profile = np.full((n_bins,), np.nan, dtype=float)
    for idx in range(n_bins):
        mask = (radii >= bins[idx]) & (radii < bins[idx + 1])
        mask &= np.isfinite(autocorr)
        if mask.any():
            profile[idx] = float(np.nanmean(autocorr[mask]))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, profile


def _rate_map_cell_values(
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    world: object,
    cell_idx: int,
) -> NDArray:
    location_ids = np.asarray(location_ids, dtype=int)
    n_locations = len(getattr(world, "locations", []))
    rate_map, _ = aggregate_rate_map(cells_trace, location_ids, n_locations)
    if rate_map.size == 0 or cell_idx >= rate_map.shape[0]:
        return np.zeros((0,), dtype=float)
    return rate_map[cell_idx]


def plot_autocorr_mosaic(
    axes: Sequence[Axes] | Axes,
    world: object,
    cells_trace: NDArray,
    location_ids: Sequence[int] | NDArray,
    *,
    cell_indices: Optional[Sequence[int]] = None,
    vmin: float | None = None,
    vmax: float | None = None,
    grid_res: float | None = None,
    cmap: str = "coolwarm",
) -> Sequence[Axes]:
    """Render a mosaic of spatial autocorrelograms into provided axes.

    Args:
        axes: Axes to draw into.
        world: Environment world with location coordinates.
        cells_trace: Cell activations (T, B, C) or (T, C).
        location_ids: Ordered list of visited location indices.
        cell_indices: Optional cell indices to include.
        vmin: Optional min value for shared color scaling.
        vmax: Optional max value for shared color scaling.
        grid_res: Optional grid resolution for rasterization.
        cmap: Colormap name.

    Returns:
        Sequence of axes that were provided.
    """
    axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else axes
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

    options = {"vmin": vmin, "vmax": vmax, "grid_res": grid_res, "cmap": cmap}
    for ax, cell_idx in zip(axes_list, indices):
        plot_spatial_autocorrelogram(ax, world, cells_trace, location_ids, cell_idx, **options)

    for ax in axes_list[len(indices) :]:
        ax.axis("off")

    return axes_list
