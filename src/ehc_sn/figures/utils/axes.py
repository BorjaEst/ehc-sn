from __future__ import annotations

"""Matplotlib axes utilities used by the figure system.

This module provides small, composable helpers to:

- Configure axes for plotting environment-like coordinate maps.
- Subdivide an existing axes' bounding box into a grid of inset child axes.
- Choose a roughly square mosaic layout for an arbitrary number of panels.

All layout units (`*_pad`, `wspace`, `hspace`) are expressed in *parent-axes
fraction* coordinates (0..1).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Sequence, Tuple, cast, overload

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


# =================================================================================================
class EnvironmentLike(Protocol):
    """Minimal protocol for environment coordinate metadata.

    The plotting utilities only rely on a set of named locations with numeric
    coordinates and (optionally) a location count.
    """

    locations: Sequence[Mapping[str, float]]
    n_locations: int


def _environment_locations(environment: object) -> Sequence[Mapping[str, Any]]:
    """Return environment locations from either an object or mapping contract."""
    if isinstance(environment, Mapping):
        locations = environment.get("locations", [])
    else:
        locations = getattr(environment, "locations", [])
    return locations if isinstance(locations, Sequence) else []


def _environment_n_locations(environment: object) -> int:
    """Return the declared location count or infer it from the locations list."""
    if isinstance(environment, Mapping):
        value = environment.get("n_locations")
    else:
        value = getattr(environment, "n_locations", None)
    if value is None:
        return len(_environment_locations(environment))
    return int(value)


# =================================================================================================
def configure_environment_axes(  # ----------------------------------------------------------------
    ax: Axes,
    *,
    environment: Optional[EnvironmentLike] = None, radius: Optional[float] = None,
    padding_scale: float = 2.0, invert_y: bool = False,
) -> Axes:  # fmt: skip
    """Configure `ax` for plotting a 2D environment/map.

    This helper standardizes axes defaults used across the figure system:

    - Sets x/y limits based on `environment.locations` when available.
    - Enforces equal aspect ratio.
    - Hides ticks/spines (the map content should define the visual frame).
    - Optionally inverts the y-axis (common for image-like coordinate systems).

    Coordinate convention
    ---------------------

    The expected coordinate keys are ``"o"`` (x-like) and ``"y"`` (y-like).
    The somewhat unusual ``"o"`` name is retained for compatibility with
    upstream metadata.

    Args:
        ax: Axes to configure.
        environment: Optional object providing `.locations` and `.n_locations`.
        radius: Optional marker radius used to pad axis limits.
        padding_scale: Multiplier applied to `radius` for padding.
        invert_y: If True, invert the y-axis after configuring limits.

    Returns:
        The same `ax` instance (mutated).
    """

    locations = [] if environment is None else _environment_locations(environment)
    if locations:
        coords = np.array([[loc.get("o"), loc.get("y")] for loc in locations], dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        coords = coords[valid]
        if coords.size > 0:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            if radius is None:
                radius = _default_radius(_environment_n_locations(environment))
            pad = (radius or 0.02) * padding_scale
            if x_min == x_max:
                x_min -= 1.0
                x_max += 1.0
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            ax.set_xlim((float(x_min - pad), float(x_max + pad)))
            ax.set_ylim((float(y_min - pad), float(y_max + pad)))
        else:
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))
    else:
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if invert_y:
        ax.invert_yaxis()

    return ax


def _default_radius(n_locations: int) -> float:
    """Return a heuristic marker radius for padding environment axes limits."""
    if n_locations <= 0:
        return 0.05
    return 2 * (0.01 + 1 / (10 * np.sqrt(n_locations)))


# =================================================================================================
@overload
def subdivide_axes(  # ----------------------------------------------------------------------------
    ax: Axes, nrows: int = 1, ncols: int = 1,
    *,
    wspace: float = 0.0, hspace: float = 0.0, left_pad: float = 0.0, right_pad: float = 0.0,
    top_pad: float = 0.0, bottom_pad: float = 0.0, hide_parent: bool = True,
    squeeze: Literal[False] = False,
) -> NDArray[np.object_]:  # fmt: skip
    ...  # fmt: skip


@overload
def subdivide_axes(  # ----------------------------------------------------------------------------
    ax: Axes, nrows: int = 1, ncols: int = 1,
    *,
    wspace: float = 0.0, hspace: float = 0.0, left_pad: float = 0.0, right_pad: float = 0.0,
    top_pad: float = 0.0, bottom_pad: float = 0.0, hide_parent: bool = True,
    squeeze: Literal[True] = True,
) -> Axes | NDArray[np.object_]:  # fmt: skip
    ...  # fmt: skip


def subdivide_axes(  # ----------------------------------------------------------------------------
    ax: Axes, nrows: int = 1, ncols: int = 1,
    *,
    wspace: float = 0.0, hspace: float = 0.0, left_pad: float = 0.0, right_pad: float = 0.0,
    top_pad: float = 0.0, bottom_pad: float = 0.0, hide_parent: bool = True,
    squeeze: bool = False,
) -> NDArray[np.object_] | Axes:  # fmt: skip
    """Subdivide `ax` into a regular grid of inset child axes.

    Child axes are created using :meth:`matplotlib.axes.Axes.inset_axes` with
    `transform=ax.transAxes`. This means their positions are expressed in
    *parent-axes fraction* coordinates and will track later layout adjustments
    to the parent (e.g., tight layout, constrained layout, adding colorbars).

    Spacing and padding
    -------------------

    All spacing/padding arguments (`left_pad`, `right_pad`, `top_pad`,
    `bottom_pad`, `wspace`, `hspace`) are expressed in parent-axes fraction
    units in the range [0, 1].

    Parent axes behavior
    --------------------

    By default (`hide_parent=True`) the parent `ax` is visually hidden but not
    removed, allowing it to still carry titles/annotations if needed.

    Args:
        ax: Parent axes whose bounding box is subdivided.
        nrows: Number of rows in the grid (must be > 0).
        ncols: Number of columns in the grid (must be > 0).
        wspace: Horizontal space between columns, in parent-axes fraction units.
        hspace: Vertical space between rows, in parent-axes fraction units.
        left_pad: Left padding inside the parent axes.
        right_pad: Right padding inside the parent axes.
        top_pad: Top padding inside the parent axes.
        bottom_pad: Bottom padding inside the parent axes.
        hide_parent: If True, hide ticks/spines/patch on the parent axes.
        squeeze: If True, squeeze singleton dimensions in the returned array.

    Returns:
        If `squeeze=False`, returns a 2D numpy array of shape `(nrows, ncols)`
        with `dtype=object` holding Axes instances.

        If `squeeze=True`, returns:
        - a scalar Axes for a 1×1 grid,
        - a 1D array of Axes for a 1×N or N×1 grid,
        - otherwise the same 2D array.
    """
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"nrows and ncols must be positive, got nrows={nrows}, ncols={ncols}")

    if hide_parent:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.patch.set_alpha(0.0)
        ax.set_navigate(False)

    # Usable area inside the parent axes, in parent axes coordinates.
    usable_w = 1.0 - left_pad - right_pad
    usable_h = 1.0 - top_pad - bottom_pad
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Padding leaves no usable space in the parent axes.")

    # Cell size in parent axes coords.
    cell_w = (usable_w - (ncols - 1) * wspace) / ncols
    cell_h = (usable_h - (nrows - 1) * hspace) / nrows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("Spacing leaves no usable cell size in the parent axes.")

    out: list[list[Axes]] = []
    for r in range(nrows):
        row: list[Axes] = []
        for c in range(ncols):
            x0 = left_pad + c * (cell_w + wspace)
            y0 = bottom_pad + (nrows - 1 - r) * (cell_h + hspace)
            child = ax.inset_axes((x0, y0, cell_w, cell_h), transform=ax.transAxes)
            row.append(child)
        out.append(row)

    arr: NDArray[np.object_] = np.array(out, dtype=object)

    if not squeeze:
        return arr
    if nrows == 1 and ncols == 1:
        return arr[0, 0]
    if nrows == 1:  # 1 × N
        return arr[0]
    if ncols == 1:  # N × 1
        return arr[:, 0]
    return arr


# =================================================================================================
@dataclass(frozen=True)
class _MosaicChoice:
    nrows: int
    ncols: int
    key: tuple[float, float, int]


# =================================================================================================
def mosaic_axes(  # -------------------------------------------------------------------------------
    ax: Axes, n_items: int,
    *,
    wspace: float = 0.0, hspace: float = 0.0, left_pad: float = 0.0, right_pad: float = 0.0,
    top_pad: float = 0.0, bottom_pad: float = 0.0, hide_parent: bool = True,
) -> NDArray[np.object_]:  # fmt: skip
    """Create a roughly square mosaic of axes sized to fit `n_items` plots.

    Args:
        ax: Slot axes whose bounding box defines the available space.
        n_items: Number of panels to layout.
        wspace: Horizontal spacing in parent-axes fraction units (0..1).
        hspace: Vertical spacing in parent-axes fraction units (0..1).
        left_pad: Left padding in parent-axes fraction units (0..1).
        right_pad: Right padding in parent-axes fraction units (0..1).
        top_pad: Top padding in parent-axes fraction units (0..1).
        bottom_pad: Bottom padding in parent-axes fraction units (0..1).

    Returns:
        A 2D numpy array of Axes instances laid out within the slot.
    """
    fig = ax.figure
    if fig is None:
        raise ValueError("Cannot create a mosaic for an Axes that is not attached to a Figure.")

    # Ensure layout has been computed before reading positions.
    if getattr(fig, "canvas", None) is not None:
        fig.canvas.draw()

    bbox = ax.get_position()

    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got n_items={n_items}")

    fig_w_in, fig_h_in = _get_figure_size_inches(fig)
    slot_w_in = bbox.width * fig_w_in
    slot_h_in = bbox.height * fig_h_in
    if slot_w_in <= 0 or slot_h_in <= 0:
        raise ValueError("Parent axes has no drawable area.")

    aspect = slot_w_in / slot_h_in
    usable_w = 1.0 - left_pad - right_pad
    usable_h = 1.0 - top_pad - bottom_pad
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Padding leaves no usable space in the parent axes.")

    def clamp_cols(value: int) -> int:
        return max(1, min(int(value), n_items))

    ncols0 = int(round(math.sqrt(n_items * aspect)))
    candidates = {clamp_cols(ncols0 - 1), clamp_cols(ncols0), clamp_cols(ncols0 + 1)}

    best: Optional[_MosaicChoice] = None
    for ncols in sorted(candidates):
        nrows = int(math.ceil(n_items / ncols))
        cell_w = (usable_w - (ncols - 1) * wspace) / ncols
        cell_h = (usable_h - (nrows - 1) * hspace) / nrows
        if cell_w <= 0 or cell_h <= 0:
            continue
        cell_w_in = cell_w * slot_w_in
        cell_h_in = cell_h * slot_h_in
        score = min(cell_w_in, cell_h_in)
        if score <= 0:
            continue

        squareness = 1.0 - abs(cell_w_in - cell_h_in) / max(cell_w_in, cell_h_in)
        empties = nrows * ncols - n_items
        key = (float(score), float(squareness), int(-empties))
        if best is None or key > best.key:
            best = _MosaicChoice(nrows=nrows, ncols=ncols, key=key)

    if best is None:
        raise ValueError("No valid mosaic layout found for given constraints.")

    return subdivide_axes(
        ax, int(best.nrows), int(best.ncols),
        wspace=wspace, hspace=hspace, left_pad=left_pad, right_pad=right_pad, bottom_pad=bottom_pad,
        top_pad=top_pad, hide_parent=hide_parent, squeeze=False,
    )  # fmt: skip


# =================================================================================================
def _get_figure_size_inches(  # -------------------------------------------------------------------
    fig: Any,
) -> tuple[float, float]:  # fmt: skip
    """Return `(width_in, height_in)` for a matplotlib Figure-like object."""
    size = getattr(fig, "get_size_inches", None)
    if callable(size):
        values = cast(Sequence[float], size())
        return (float(values[0]), float(values[1]))
    root = getattr(fig, "figure", None)
    if root is not None and callable(getattr(root, "get_size_inches", None)):
        values = cast(Sequence[float], root.get_size_inches())
        return (float(values[0]), float(values[1]))
    raise ValueError("Cannot determine figure size in inches.")


__all__ = ["configure_environment_axes", "mosaic_axes", "subdivide_axes"]
