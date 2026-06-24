"""Internal figure data contracts.

Shared data types consumed by both the selector layer and the plot layer.
This module has no dependencies on figure templates or the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, runtime_checkable

from matplotlib.cm import ScalarMappable
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


# ── Panel annotation types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoricalLegend:
    """Specification for a categorical legend in a panel annotation slot.

    The owning template creates ``Patch`` artists from *entries*; this
    dataclass carries only the semantic specification, not the artists.

    When *label_stride* > 1, only every *label_stride*-th entry renders a
    text label — all others show swaths only.  This is useful for large
    categorical vocabularies (e.g. 40+ observation IDs) where labeling
    every entry would overflow the annotation slot.
    """

    entries: list[tuple[str, str]]
    """Sequence of ``(label, hex_color)`` pairs."""
    ncol: int = 3
    """Number of legend columns."""
    title: str | None = None
    """Optional legend title."""
    label_stride: int = 1
    """Show text label only for every *label_stride*-th entry (1 = all)."""
    compact: bool = False
    """Render entries as thin ``Line2D`` strokes instead of filled
    ``Patch`` squares.  Compact mode uses substantially less vertical
    space and is appropriate for large categorical vocabularies
    (e.g. 40+ observation IDs) rendered in the annotation slot."""


@dataclass(frozen=True)
class ColorStrip:
    """Inline horizontal colour strip for large categorical palettes.

    Renders as a single row of thin coloured vertical bars with sparse
    numeric labels.  Designed for annotation slots where a full
    ``CategoricalLegend`` would overflow (e.g. 40–60+ observation IDs in
    a 12 %-height slot).

    *label_every* adapts automatically so roughly 8 labels are shown
    regardless of entry count — override it explicitly only when needed.
    """

    entries: list[tuple[str, str]]
    """``(label, hex_color)`` pairs, one per category."""
    label_every: int | None = None
    """Show a text label every N entries (``None`` = auto: ~8 labels)."""
    title: str | None = None
    """Optional title rendered above the strip."""
    rows: int = 1
    """Number of horizontal rows.  Set to 2 for a double-row strip that
    gives each colour bar twice the width without increasing total
    horizontal footprint."""


@dataclass
class ContinuousScale:
    """Specification for a continuous colorbar in a panel annotation slot.

    *mappable* is the ``ScalarMappable`` returned by the render function
    (e.g. the ``AxesImage`` from ``ax.imshow()``).  The annotation
    renderer calls ``fig.colorbar(mappable, cax=annotation_ax, ...)``.
    """

    mappable: ScalarMappable | None = None
    """ScalarMappable from the primary render call."""
    label: str = ""
    """Colorbar axis label."""
    vmin: float = 0.0
    vmax: float = 1.0
    ticks: list[float] | None = None
    """Optional explicit tick positions."""


PanelAnnotation: TypeAlias = Union[
    None, CategoricalLegend, ColorStrip, ContinuousScale
]
"""Content of one annotation slot.

``None``     — slot is hidden (``ax.set_visible(False)``).
``CategoricalLegend`` — render a categorical legend.
``ColorStrip`` — render an inline colour strip.
``ContinuousScale``   — render a horizontal colorbar.
"""


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
    "CategoricalLegend",
    "ColorStrip",
    "ContinuousScale",
    "PanelAnnotation",
    "PreparedRateMap",
]
