"""Spatial autocorrelogram utilities for diagnostic figures."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ehc_sn.figures.plots.ratemap import PreparedRateMap

DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP = 4


def plot_spatial_autocorrelogram(
    ax: Axes,
    prepared_rate_map: PreparedRateMap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    min_overlap: int = DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP,
) -> Axes:
    """Plot a 2D spatial autocorrelogram from a prepared rate map."""
    autocorrelogram = compute_spatial_autocorrelogram(
        prepared_rate_map.rate_map,
        prepared_rate_map.valid_mask,
        min_overlap=min_overlap,
    )
    if autocorrelogram.size == 0 or not np.isfinite(autocorrelogram).any():
        ax.text(0.5, 0.5, "No autocorr", ha="center", va="center")
        ax.axis("off")
        return ax

    if vmin is None:
        vmin = -1.0
    if vmax is None:
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    ax.imshow(autocorrelogram, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_radial_autocorrelogram_profile(
    ax: Axes,
    prepared_rate_maps: Sequence[PreparedRateMap],
    *,
    n_bins: int = 32,
    color: str | None = None,
    min_overlap: int = DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP,
) -> Axes:
    """Plot the mean radial autocorrelogram profile with std across inputs."""
    radii, mean, std, n_profiles = _summarize_radial_autocorrelogram_profiles(
        prepared_rate_maps,
        n_bins=n_bins,
        min_overlap=min_overlap,
    )
    if radii.size == 0 or not np.isfinite(mean).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    ax.plot(radii, mean, color=color, label=f"N={n_profiles}")
    color = ax.get_lines()[-1].get_color() if color is None else color
    ax.fill_between(radii, mean - std, mean + std, color=color, alpha=0.25)
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Radial autocorrelogram (±1 std)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left", handlelength=1.0)
    return ax


def _summarize_radial_autocorrelogram_profiles(
    prepared_rate_maps: Sequence[PreparedRateMap],
    *,
    n_bins: int = 32,
    min_overlap: int = DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP,
) -> tuple[NDArray, NDArray, NDArray, int]:
    autocorrelograms = [
        compute_spatial_autocorrelogram(prepared_rate_map.rate_map, prepared_rate_map.valid_mask, min_overlap=min_overlap)
        for prepared_rate_map in prepared_rate_maps
    ]
    return _summarize_radial_profiles(autocorrelograms, n_bins=n_bins)


def _summarize_radial_profiles(
    autocorrelograms: Sequence[NDArray],
    *,
    n_bins: int = 32,
) -> tuple[NDArray, NDArray, NDArray, int]:
    profiles: list[NDArray] = []
    reference_radii: NDArray | None = None
    for autocorrelogram in autocorrelograms:
        if autocorrelogram.size == 0 or not np.isfinite(autocorrelogram).any():
            continue
        radii, profile = radial_profile(autocorrelogram, n_bins=n_bins)
        if radii.size == 0 or profile.size == 0:
            continue
        if reference_radii is None:
            reference_radii = radii
        elif not np.allclose(radii, reference_radii, equal_nan=True):
            finite = np.isfinite(profile)
            if finite.sum() < 2:
                continue
            profile = np.interp(reference_radii, radii[finite], profile[finite], left=np.nan, right=np.nan)
        profiles.append(profile)

    if not profiles or reference_radii is None:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, 0

    stack = np.vstack(profiles)
    mean = np.full((stack.shape[1],), np.nan, dtype=float)
    std = np.full((stack.shape[1],), np.nan, dtype=float)
    for idx in range(stack.shape[1]):
        column = stack[:, idx]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        mean[idx] = float(np.mean(column[finite]))
        std[idx] = float(np.std(column[finite]))
    return reference_radii, mean, std, stack.shape[0]


def compute_spatial_autocorrelogram(
    rate_map: NDArray,
    valid_mask: NDArray,
    *,
    min_overlap: int = DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP,
) -> NDArray:
    """Compute a linear Pearson 2D spatial autocorrelogram.

    Args:
        rate_map: Rasterized rate-map values with NaNs for missing pixels.
        valid_mask: Boolean mask indicating valid pixels.

    Returns:
        2D autocorrelogram with shape ``(2H - 1, 2W - 1)`` and center at the
        array midpoint.
    """
    if rate_map.size == 0:
        return np.zeros((0, 0), dtype=float)
    if min_overlap < 1:
        raise ValueError(f"min_overlap must be >= 1, got {min_overlap}.")

    values = np.asarray(rate_map, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != valid_mask.shape:
        raise ValueError("rate_map and valid_mask must have the same shape")

    height, width = values.shape
    autocorrelogram = np.full((2 * height - 1, 2 * width - 1), np.nan, dtype=float)

    for lag_y in range(-(height - 1), height):
        src_y, dst_y = _overlap_slices(height, lag_y)
        for lag_x in range(-(width - 1), width):
            src_x, dst_x = _overlap_slices(width, lag_x)

            source = values[src_y, src_x]
            target = values[dst_y, dst_x]
            pair_mask = valid_mask[src_y, src_x] & valid_mask[dst_y, dst_x]
            if np.count_nonzero(pair_mask) < min_overlap:
                continue

            source_values = source[pair_mask]
            target_values = target[pair_mask]
            source_centered = source_values - float(np.mean(source_values))
            target_centered = target_values - float(np.mean(target_values))
            denom = float(np.linalg.norm(source_centered) * np.linalg.norm(target_centered))
            if denom <= 0.0:
                continue

            autocorrelogram[lag_y + height - 1, lag_x + width - 1] = float(np.dot(source_centered, target_centered) / denom)

    return autocorrelogram


def radial_profile(autocorrelogram: NDArray, *, n_bins: int = 32) -> tuple[NDArray, NDArray]:
    """Compute a radial profile from a 2D autocorrelogram.

    Args:
        autocorrelogram: 2D autocorrelogram array.
        n_bins: Number of radial bins.

    Returns:
        Tuple of (radii, profile) for the autocorrelogram.
    """
    if autocorrelogram.size == 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    yy, xx = np.indices(autocorrelogram.shape)
    center_y = (autocorrelogram.shape[0] - 1) / 2.0
    center_x = (autocorrelogram.shape[1] - 1) / 2.0
    radii = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    max_radius = float(np.nanmax(radii)) if radii.size else 0.0
    if max_radius <= 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty

    bins = np.linspace(0.0, max_radius, n_bins + 1)
    profile = np.full((n_bins,), np.nan, dtype=float)
    for idx in range(n_bins):
        mask = (radii >= bins[idx]) & (radii < bins[idx + 1])
        mask &= np.isfinite(autocorrelogram)
        if mask.any():
            profile[idx] = float(np.nanmean(autocorrelogram[mask]))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, profile


def _overlap_slices(size: int, lag: int) -> tuple[slice, slice]:
    """Return aligned source/target slices for a signed lag."""
    if lag >= 0:
        return slice(lag, size), slice(0, size - lag)
    return slice(0, size + lag), slice(-lag, size)


def plot_spatial_autocorrelogram_mosaic(
    axes: Sequence[Axes] | Axes,
    prepared_rate_maps: Sequence[PreparedRateMap],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    min_overlap: int = DEFAULT_SPATIAL_AUTOCORRELOGRAM_MIN_OVERLAP,
) -> Sequence[Axes]:
    """Render a mosaic of spatial autocorrelograms from prepared rate maps."""
    axes_list = list(np.ravel(axes)) if isinstance(axes, np.ndarray) else ([axes] if isinstance(axes, Axes) else list(axes))
    if not axes_list:
        return axes_list

    if not prepared_rate_maps:
        axes_list[0].text(0.5, 0.5, "No data", ha="center", va="center")
        for empty_ax in axes_list:
            empty_ax.axis("off")
        return axes_list

    for ax, prepared_rate_map in zip(axes_list, prepared_rate_maps):
        plot_spatial_autocorrelogram(
            ax,
            prepared_rate_map,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            min_overlap=min_overlap,
        )

    for ax in axes_list[len(prepared_rate_maps) :]:
        ax.axis("off")

    return axes_list
