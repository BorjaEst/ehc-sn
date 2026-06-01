"""Gridness scoring and spacing/orientation estimation for spatial autocorrelograms.

All functions require **full uncropped** autocorrelograms of odd shape
``(2H - 1, 2W - 1)`` with the centre at the array midpoint.  Display-cropped
or even-shaped arrays are rejected.

References
----------
Barry & Burgess (2017). "To be a Grid Cell: Shuffling procedures for
    determining gridness".  DOI: 10.1101/163592
Wills, Barry & Burgess (2012). "The abrupt development of adult-like grid
    cell firing in the medial entorhinal cortex".  Front. Neural Circuits 6:21.
Hafting, Fyhn, Molden, Moser & Moser (2005). "Microstructure of a spatial map
    in the entorhinal cortex".  Nature 436:801–806.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import rotate as ndimage_rotate

from ehc_sn.analysis.spatial.geometry import SpatialBinGeometry

# ── Default annular mask radii (fraction of max valid radius) ────────────────
_DEFAULT_INNER_FRACTION: float = 0.10
_DEFAULT_OUTER_FRACTION: float = 0.55

# ── Minimum valid pixel count for a meaningful correlation ───────────────────
_MIN_OVERLAP: int = 10

# ── Rotational-correlation angles (degrees) ──────────────────────────────────
_GRIDNESS_ANGLES: tuple[int, ...] = (30, 60, 90, 120, 150)

# ── Spacing / orientation peak-finding ───────────────────────────────────────
_SPACING_MIN_PEAK_PROMINENCE: float = 0.05
_SPACING_REQUIRED_PEAKS: int = 6


# =============================================================================
# Result data structures
# =============================================================================


@dataclass(frozen=True)
class GridnessResult:
    """Result of a gridness computation.

    Attributes
    ----------
    score : float
        Gridness score ``min(r60, r120) - max(r30, r90, r150)``.
        ``NaN`` when the annular mask has too few valid pixels.
    correlations : dict[int, float]
        Per-angle rotational correlations keyed by rotation in degrees.
    inner_radius : float
        Inner annular-mask radius in world units.
    outer_radius : float
        Outer annular-mask radius in world units.
    mask_strategy : str
        Description of how mask radii were chosen (e.g. ``"fractional"``).
    valid_pixel_count : int
        Number of pixels used in each rotational correlation.
    """

    score: float
    correlations: dict[int, float] = field(default_factory=dict)
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    mask_strategy: str = ""
    valid_pixel_count: int = 0


@dataclass(frozen=True)
class SpacingOrientationResult:
    """Result of grid spacing and orientation estimation.

    Attributes
    ----------
    spacing : float
        Grid spacing / wavelength in world units.  ``NaN`` if fewer than
        ``_SPACING_REQUIRED_PEAKS`` credible first-ring peaks were found.
    orientation_deg : float
        Grid orientation wrapped to ``[0°, 60°)``.  ``NaN`` if spacing is
        ``NaN``.
    orientation_raw_deg : float
        Unwrapped grid orientation in degrees (for debugging).
    peak_distances : np.ndarray
        Distances from centre of the identified first-ring peaks.
    peak_angles_deg : np.ndarray
        Angles from centre of the identified first-ring peaks, in degrees.
    n_peaks : int
        Number of credible first-ring peaks found.
    """

    spacing: float = float("nan")
    orientation_deg: float = float("nan")
    orientation_raw_deg: float = float("nan")
    peak_distances: NDArray = field(
        default_factory=lambda: np.zeros((0,), dtype=float)
    )
    peak_angles_deg: NDArray = field(
        default_factory=lambda: np.zeros((0,), dtype=float)
    )
    n_peaks: int = 0


# =============================================================================
# Input validation
# =============================================================================


def _validate_autocorr(autocorr: NDArray) -> None:
    """Check that *autocorr* is a valid full-size 2D autocorrelogram."""
    if autocorr.ndim != 2:
        raise ValueError(f"Autocorrelogram must be 2D, got {autocorr.ndim}D")
    h, w = autocorr.shape
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError(
            f"Autocorrelogram must have odd dimensions (full-size, not display-cropped), "
            f"got ({h}, {w})"
        )


def _validate_radii(inner_radius: float, outer_radius: float) -> None:
    """Check that annular radii are sensible."""
    if inner_radius >= outer_radius:
        raise ValueError(
            f"Inner radius ({inner_radius}) must be less than outer radius ({outer_radius})"
        )
    if inner_radius < 0:
        raise ValueError(
            f"Inner radius must be non-negative, got {inner_radius}"
        )


# =============================================================================
# Annular mask
# =============================================================================


def annular_mask(
    shape: tuple[int, int],
    inner_radius_px: float,
    outer_radius_px: float,
) -> NDArray:
    """Return a boolean mask for an annulus centred in a 2D array.

    Parameters
    ----------
    shape : tuple[int, int]
        Array shape ``(height, width)``.
    inner_radius_px : float
        Inner radius in pixel units (excluded).
    outer_radius_px : float
        Outer radius in pixel units (included).

    Returns
    -------
    NDArray
        Boolean mask with ``True`` inside the annulus.
    """
    h, w = shape
    yy, xx = np.indices(shape, dtype=float)
    centre_y = (h - 1) / 2.0
    centre_x = (w - 1) / 2.0
    radii = np.sqrt((xx - centre_x) ** 2 + (yy - centre_y) ** 2)
    return (radii >= inner_radius_px) & (radii <= outer_radius_px)


# =============================================================================
# Rotation
# =============================================================================


def rotate_autocorrelogram(autocorr: NDArray, angle_deg: float) -> NDArray:
    """Rotate a full autocorrelogram by *angle_deg* using bilinear interpolation.

    Parameters
    ----------
    autocorr : NDArray
        Full-size autocorrelogram ``(2H - 1, 2W - 1)`` with odd dimensions.
    angle_deg : float
        Counter-clockwise rotation in degrees.

    Returns
    -------
    NDArray
        Rotated array of the same shape.  Pixels that were not present in the
        original are filled with ``NaN``.
    """
    _validate_autocorr(autocorr)
    # Bilinear interpolation, no reshape, NaN fill for out-of-bounds.
    return ndimage_rotate(
        autocorr,
        angle_deg,
        order=1,
        reshape=False,
        mode="constant",
        cval=float("nan"),
    )


# =============================================================================
# Rotational correlation
# =============================================================================


def rotational_correlation(
    autocorr: NDArray,
    mask: NDArray,
    angle_deg: float,
    *,
    min_overlap: int = _MIN_OVERLAP,
) -> float:
    """Pearson correlation between *autocorr* and a rotated copy inside *mask*.

    Only pixels that are finite in **both** arrays inside the mask contribute.
    Returns ``NaN`` when fewer than *min_overlap* valid pixel pairs exist.

    Parameters
    ----------
    autocorr : NDArray
        Full-size autocorrelogram.
    mask : NDArray
        Boolean mask of the same shape (e.g. from :func:`annular_mask`).
    angle_deg : float
        Rotation angle in degrees.
    min_overlap : int
        Minimum number of valid pixel pairs required.

    Returns
    -------
    float
        Pearson correlation coefficient, or ``NaN``.
    """
    rotated = rotate_autocorrelogram(autocorr, angle_deg)
    valid = mask & np.isfinite(autocorr) & np.isfinite(rotated)
    if np.count_nonzero(valid) < min_overlap:
        return float("nan")

    a = autocorr[valid]
    b = rotated[valid]
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


# =============================================================================
# Gridness score
# =============================================================================


def compute_gridness(
    autocorr: NDArray,
    *,
    geometry: SpatialBinGeometry,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    mask_strategy: Literal["fractional"] = "fractional",
) -> GridnessResult:
    """Compute the gridness score from a full spatial autocorrelogram.

    Parameters
    ----------
    autocorr : NDArray
        Full-size autocorrelogram of odd shape ``(2H - 1, 2W - 1)``.
    geometry : SpatialBinGeometry
        Spatial bin geometry (must be square).
    inner_radius : float, optional
        Inner annular-mask radius in world units.  Defaults to 10 % of the
        maximum valid radius.
    outer_radius : float, optional
        Outer annular-mask radius in world units.  Defaults to 55 % of the
        maximum valid radius.
    mask_strategy : Literal["fractional"]
        How mask radii were selected.  Only ``"fractional"`` is supported in
        this version.

    Returns
    -------
    GridnessResult
        Gridness score with per-angle correlations and mask metadata.
    """
    _validate_autocorr(autocorr)
    bin_sz = geometry.square_bin_size()

    max_r = geometry.max_valid_radius(autocorr.shape)
    if inner_radius is None:
        inner_radius = _DEFAULT_INNER_FRACTION * max_r
    if outer_radius is None:
        outer_radius = _DEFAULT_OUTER_FRACTION * max_r
    _validate_radii(inner_radius, outer_radius)

    inner_px = inner_radius / bin_sz
    outer_px = outer_radius / bin_sz

    mask = annular_mask(autocorr.shape, inner_px, outer_px)

    correlations: dict[int, float] = {}
    for angle in _GRIDNESS_ANGLES:
        correlations[angle] = rotational_correlation(
            autocorr, mask, float(angle)
        )

    valid_pixels = int(np.count_nonzero(mask & np.isfinite(autocorr)))

    r60 = correlations.get(60, float("nan"))
    r120 = correlations.get(120, float("nan"))
    r30 = correlations.get(30, float("nan"))
    r90 = correlations.get(90, float("nan"))
    r150 = correlations.get(150, float("nan"))

    if any(np.isnan(v) for v in (r60, r120, r30, r90, r150)):
        score = float("nan")
    else:
        score = float(min(r60, r120) - max(r30, r90, r150))

    return GridnessResult(
        score=score,
        correlations=correlations,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        mask_strategy=mask_strategy,
        valid_pixel_count=valid_pixels,
    )


# =============================================================================
# Grid spacing and orientation
# =============================================================================


def _local_maxima(
    arr: NDArray,
    *,
    threshold: float | None = None,
    exclude_central_disk_px: float = 0.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Find local maxima in a 2D array using maximum filter.

    Parameters
    ----------
    arr : NDArray
        Input array.
    threshold : float, optional
        Minimum absolute value for a pixel to be considered a peak.
        If ``None``, uses 50 % of the finite maximum outside the excluded
        central disk.
    exclude_central_disk_px : float
        Exclude peaks within this radius (pixels) of the centre.

    Returns
    -------
    y_coords : NDArray
        Row indices of identified peaks.
    x_coords : NDArray
        Column indices of identified peaks.
    values : NDArray
        Peak values.
    """
    from scipy.ndimage import maximum_filter  # fmt: skip

    h, w = arr.shape
    centre_y = (h - 1) / 2.0
    centre_x = (w - 1) / 2.0

    # Compute local maxima via maximum filter
    neighborhood_size = 3
    max_filtered = maximum_filter(arr, size=neighborhood_size)
    maxima_mask = (arr == max_filtered) & np.isfinite(arr)

    # Remove pixels near the centre
    yy, xx = np.indices(arr.shape, dtype=float)
    dist = np.sqrt((xx - centre_x) ** 2 + (yy - centre_y) ** 2)
    maxima_mask = maxima_mask & (dist >= exclude_central_disk_px)

    if threshold is None:
        # Determine a relative threshold from the finite values outside the
        # excluded central disk.
        outer_finite = arr[(dist >= exclude_central_disk_px) & np.isfinite(arr)]
        threshold = (
            0.50 * float(np.max(outer_finite)) if outer_finite.size > 0 else 0.0
        )

    maxima_mask = maxima_mask & (arr >= threshold)

    peak_y, peak_x = np.where(maxima_mask)
    peak_vals = arr[peak_y, peak_x]

    # Sort by distance from centre
    dists = np.sqrt((peak_x - centre_x) ** 2 + (peak_y - centre_y) ** 2)
    order = np.argsort(dists)
    return peak_y[order], peak_x[order], peak_vals[order]


def estimate_grid_spacing_orientation(
    autocorr: NDArray,
    *,
    geometry: SpatialBinGeometry,
    central_exclusion_radius: float | None = None,
    min_peak_prominence: float | None = None,
) -> SpacingOrientationResult:
    """Estimate grid spacing and orientation from a full spatial autocorrelogram.

    Parameters
    ----------
    autocorr : NDArray
        Full-size autocorrelogram of odd shape ``(2H - 1, 2W - 1)``.
    geometry : SpatialBinGeometry
        Spatial bin geometry (must be square).
    central_exclusion_radius : float, optional
        Exclude peaks within this world-unit radius of centre.  Defaults to
        10 % of the maximum valid radius.
    min_peak_prominence : float, optional
        Minimum autocorrelation value for a pixel to be considered a peak.
        If ``None``, uses 25 % of the finite maximum outside the central
        exclusion disk.

    Returns
    -------
    SpacingOrientationResult
        Spacing and orientation estimate, or NaN fields when insufficient
        credible peaks are found.
    """
    _validate_autocorr(autocorr)
    bin_sz = geometry.square_bin_size()

    max_r = geometry.max_valid_radius(autocorr.shape)
    if central_exclusion_radius is None:
        central_exclusion_radius = _DEFAULT_INNER_FRACTION * max_r
    if min_peak_prominence is None:
        min_peak_prominence = _SPACING_MIN_PEAK_PROMINENCE

    exclude_px = max(central_exclusion_radius / bin_sz, 1.0)

    peak_y, peak_x, peak_vals = _local_maxima(
        autocorr,
        threshold=min_peak_prominence,
        exclude_central_disk_px=exclude_px,
    )

    if len(peak_y) < _SPACING_REQUIRED_PEAKS:
        return SpacingOrientationResult(n_peaks=len(peak_y))

    # Take the closest N peaks as the first ring
    peak_y = peak_y[:_SPACING_REQUIRED_PEAKS]
    peak_x = peak_x[:_SPACING_REQUIRED_PEAKS]
    peak_vals = peak_vals[:_SPACING_REQUIRED_PEAKS]

    h, w = autocorr.shape
    centre_y = (h - 1) / 2.0
    centre_x = (w - 1) / 2.0

    dists_px = np.sqrt((peak_x - centre_x) ** 2 + (peak_y - centre_y) ** 2)
    angles_rad = np.arctan2(peak_y - centre_y, peak_x - centre_x)
    angles_deg = np.rad2deg(angles_rad)

    spacing = float(np.mean(dists_px)) * bin_sz
    orientation_raw = float(np.mean(angles_deg))
    orientation_wrapped = orientation_raw % 60.0

    return SpacingOrientationResult(
        spacing=spacing,
        orientation_deg=orientation_wrapped,
        orientation_raw_deg=orientation_raw,
        peak_distances=dists_px * bin_sz,
        peak_angles_deg=angles_deg,
        n_peaks=_SPACING_REQUIRED_PEAKS,
    )
