"""Internal figure data contracts.

Shared data types consumed by both the selector layer and the plot layer.
This module has no dependencies on figure templates or the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, runtime_checkable

from numpy.typing import NDArray

# ── World protocol ────────────────────────────────────────────────────────────


# =============================================================================
@runtime_checkable
class WorldLike(Protocol):
    """Minimal structural protocol for environment world objects.

    Figures and selectors type ``world`` parameters as ``WorldLike | Mapping``
    so both object-style and dict-style world containers satisfy the contract
    without importing adapter or task types.

    Location entries are heterogeneous mappings: keys include ``"o"`` (column),
    ``"y"`` (row), ``"actions"``, ``"valid"``, and ``"shiny"``, with values
    that are not uniformly float.  The element type is therefore ``Any``.
    """

    locations: Sequence[Mapping[str, Any]]
    n_locations: int


# Convenience alias accepted wherever a world is needed.
AnyWorld: TypeAlias = Union[WorldLike, Mapping[str, Any]]


# ── PreparedRateMap ───────────────────────────────────────────────────────────


# =============================================================================
@dataclass(frozen=True)
class PreparedRateMap:
    """Prepared rate-map surfaces for one cell.

    Produced by the spatial selector layer and consumed by the rate-map and
    autocorrelogram plot functions.  This is the single canonical definition;
    do not import it from ``selectors.spatial`` or ``plots.ratemap``.
    """

    location_responses: NDArray
    location_counts: NDArray
    response_mass_grid: NDArray
    occupancy_grid: NDArray
    smoothed_response_mass_grid: NDArray
    smoothed_occupancy_grid: NDArray
    rate_map: NDArray
    valid_mask: NDArray
    extent: tuple[float, float, float, float]
    smooth_sigma: float
    min_bin_occupancy: float


# =============================================================================
__all__ = [
    "AnyWorld",
    "PreparedRateMap",
]
