"""Spatial bin geometry for converting between pixel and world coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _factorize_location_count(
    n_locations: int,
) -> tuple[int, int]:
    """Factor a location count into ``(rows, cols)`` for grid reshape.

    Prefers square grids; falls back to integer factorisation with the
    smaller dimension as the first factor.

    Args:
        n_locations: Total number of location bins.

    Returns:
        ``(rows, cols)`` such that ``rows * cols == n_locations``.

    Raises:
        ValueError: If ``n_locations`` has no integer factorisation.
    """
    sqrt = int(np.sqrt(n_locations))
    for r in range(sqrt, 0, -1):
        if n_locations % r == 0:
            return (r, n_locations // r)
    raise ValueError(
        f"n_locations={n_locations} cannot be factored into "
        f"integer (rows, cols).  Grid reshape requires a "
        f"rectangular topology."
    )


@dataclass(frozen=True)
class SpatialBinGeometry:
    """Explicit spatial bin dimensions for one rate-map grid.

    Parameters
    ----------
    bin_size_x : float
        World-unit width of one raster column (x-axis).
    bin_size_y : float
        World-unit height of one raster row (y-axis).
    """

    bin_size_x: float
    bin_size_y: float

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_extent_and_shape(
        cls,
        extent: tuple[float, float, float, float],
        shape: tuple[int, int],
    ) -> SpatialBinGeometry:
        """Derive bin geometry from a rate map's world extent and raster shape.

        Parameters
        ----------
        extent : tuple[float, float, float, float]
            ``(xmin, xmax, ymin, ymax)`` in world units (same convention as
            ``PreparedRateMap.extent``).
        shape : tuple[int, int]
            Raster shape ``(n_rows, n_cols)``.

        Returns
        -------
        SpatialBinGeometry
            Bin geometry with per-pixel world-unit spacing.
        """
        xmin, xmax, ymin, ymax = extent
        n_rows, n_cols = shape
        if n_cols <= 1 or n_rows <= 1:
            raise ValueError(
                f"Rate-map shape must have at least 2 rows and 2 columns, got {shape}"
            )
        span_x = float(xmax) - float(xmin)
        span_y = float(ymax) - float(ymin)
        if span_x <= 0 or span_y <= 0:
            raise ValueError(f"Extent must have positive spans, got {extent}")
        return cls(
            bin_size_x=span_x / float(n_cols - 1),
            bin_size_y=span_y / float(n_rows - 1),
        )

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def is_square(self) -> bool:
        """Whether bins are square within floating-point tolerance."""
        return bool(np.isclose(self.bin_size_x, self.bin_size_y))

    def square_bin_size(self) -> float:
        """Return the common bin size when bins are square.

        Returns
        -------
        float
            Shared bin size in world units.

        Raises
        ------
        ValueError
            If bins are not square (``bin_size_x != bin_size_y`` within
            floating-point tolerance).
        """
        if not self.is_square:
            msg = (
                f"Gridness analysis requires square spatial bins, "
                f"got bin_size_x={self.bin_size_x}, bin_size_y={self.bin_size_y}"
            )
            raise ValueError(msg)
        return float(self.bin_size_x)

    # ── pixel / world conversion helpers ────────────────────────────────────

    def world_to_pixel_radius(self, world_radius: float) -> int:
        """Convert a world-unit radius to pixel steps (uses square bin size).

        Raises ``ValueError`` if bins are not square.
        """
        return max(int(np.ceil(world_radius / self.square_bin_size())), 0)

    def max_valid_radius(self, autocorr_shape: tuple[int, int]) -> float:
        """Maximum valid world-unit radius from the center of an autocorr."""
        h, w = autocorr_shape
        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0
        bin_sz = self.square_bin_size()
        return float(min(center_y, center_x)) * bin_sz

    # ── flat / grid reshape helpers ─────────────────────────────────────────

    def flat_to_grid(self, arr: np.ndarray) -> np.ndarray:
        """Reshape a flat-location array to a spatial grid.

        The location axis (the axis at position ``-2 + ndim`` for
        2-D+ arrays, or ``-1`` for 1-D arrays) is split into
        ``(rows, cols)`` where ``rows * cols == L``.

        Args:
            arr: Array with a trailing location axis of length ``L``,
                e.g. shape ``(E, L)`` or ``(E, L, U)``.

        Returns:
            Array with ``(E, rows, cols)`` or ``(E, rows, cols, U)``.

        Raises:
            ValueError: If ``L`` cannot be factored.
        """
        arr = np.asarray(arr)
        loc_axis = arr.ndim - 1
        n_locations = arr.shape[loc_axis]
        rows, cols = _factorize_location_count(n_locations)
        new_shape = arr.shape[:loc_axis] + (rows, cols) + arr.shape[loc_axis + 1:]
        return arr.reshape(new_shape)

    def grid_to_flat(self, arr: np.ndarray) -> np.ndarray:
        """Inverse of ``flat_to_grid`` — merge ``(rows, cols)`` back to ``L``.

        Args:
            arr: Array with spatial axes preceding trailing dims,
                e.g. shape ``(E, rows, cols)`` or ``(E, rows, cols, U)``.

        Returns:
            Array with merged location axis.
        """
        arr = np.asarray(arr)
        grid_axis = arr.ndim - 2  # rows axis
        n_rows, n_cols = arr.shape[grid_axis], arr.shape[grid_axis + 1]
        new_shape = (
            arr.shape[:grid_axis]
            + (n_rows * n_cols,)
            + arr.shape[grid_axis + 2:]
        )
        return arr.reshape(new_shape)

    # ── repr ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"bin_size_x={self.bin_size_x}, bin_size_y={self.bin_size_y})"
        )
