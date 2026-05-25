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
from typing import Any, Literal, Optional, Sequence, TypeAlias, cast

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ehc_sn.figures._contracts import AnyWorld


# =============================================================================
@dataclass(frozen=True)
class _AxesRect:
    """Rectangle in container coordinates."""

    x0: float
    y0: float
    width: float
    height: float


# =============================================================================
@dataclass(frozen=True)
class _SolvedMosaic:
    """Aspect-aware mosaic solution in container coordinates.

    The rectangle list always describes a full rectangular lattice with
    ``nrows * ncols`` slots. If the caller renders fewer items than slots,
    the trailing slots remain part of the centered footprint and are expected
    to be hidden or left empty by the renderer.
    """

    nrows: int
    ncols: int
    panel_width: float
    panel_height: float
    occupied_width: float
    occupied_height: float
    rectangles: tuple[_AxesRect, ...]


# =============================================================================
PlacementMode: TypeAlias = Literal["center", "top-left"]


# =============================================================================
def _environment_locations(  # ------------------------------------------------
    environment: object,
) -> Sequence[Mapping[str, Any]]:
    """Return environment locations from either an object or mapping contract."""
    if isinstance(environment, Mapping):
        locations = environment.get("locations", [])
    else:
        locations = getattr(environment, "locations", [])
    return locations if isinstance(locations, Sequence) else []


# =============================================================================
def _environment_n_locations(  # ----------------------------------------------
    environment: object,
) -> int:
    """Return the declared location count or infer it from the locations list."""
    if isinstance(environment, Mapping):
        value = environment.get("n_locations")
    else:
        value = getattr(environment, "n_locations", None)
    if value is None:
        return len(_environment_locations(environment))
    return int(value)


# =============================================================================
def _environment_plot_coords(  # ----------------------------------------------
    environment: object,
) -> NDArray[np.float64]:
    """Return finite coordinates for the locations that should be rendered.

    When location metadata exposes a boolean ``valid`` flag, invalid locations
    are omitted so axis fitting matches the visible occupancy rather than the
    full latent lattice. If every location is marked invalid, fall back to all
    finite coordinates so callers still get deterministic bounds.
    """
    locations = _environment_locations(environment)
    if not locations:
        return np.zeros((0, 2), dtype=float)

    coords = np.asarray(
        [[loc.get("o"), loc.get("y")] for loc in locations], dtype=float
    )
    finite = np.isfinite(coords).all(axis=1)
    visible = np.asarray(
        [bool(loc.get("valid", True)) for loc in locations], dtype=bool
    )
    selected = finite & visible
    if not np.any(selected):
        selected = finite
    return coords[selected]


# =============================================================================
def configure_environment_axes(  # --------------------------------------------
    ax: Axes,
    *,
    environment: Optional[AnyWorld] = None,
    radius: Optional[float] = None,
    padding_scale: float = 2.0,
    invert_y: bool = False,
) -> Axes:
    """Configure `ax` for plotting a 2D environment/map.

    This helper standardizes axes defaults used across the figure system:

    - Sets x/y limits based on the visible environment occupancy when available.
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

    if environment is not None:
        coords = _environment_plot_coords(environment)
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


# =============================================================================
def _default_radius(  # -------------------------------------------------------
    n_locations: int,
) -> float:
    """Return a heuristic marker radius for padding environment axes limits."""
    if n_locations <= 0:
        return 0.05
    return 2 * (0.01 + 1 / (10 * np.sqrt(n_locations)))


# =============================================================================
def _hide_parent_axes(  # -----------------------------------------------------
    ax: Axes,
) -> None:
    """Hide ticks, spines, and patch for a parent axes hosting inset children."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_navigate(False)


# =============================================================================
def subdivide_axes(  # --------------------------------------------------------
    ax: Axes,
    nrows: int = 1,
    ncols: int = 1,
    *,
    wspace: float = 0.0,
    hspace: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    top_pad: float = 0.0,
    bottom_pad: float = 0.0,
    hide_parent: bool = True,
    squeeze: bool = False,
) -> NDArray[np.object_] | Axes:
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
        raise ValueError(
            f"nrows and ncols must be positive, got nrows={nrows}, ncols={ncols}"
        )

    if hide_parent:
        _hide_parent_axes(ax)

    # Usable area inside the parent axes, in parent axes coordinates.
    usable_w = 1.0 - left_pad - right_pad
    usable_h = 1.0 - top_pad - bottom_pad
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Padding leaves no usable space in the parent axes.")

    # Cell size in parent axes coords.
    cell_w = (usable_w - (ncols - 1) * wspace) / ncols
    cell_h = (usable_h - (nrows - 1) * hspace) / nrows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError(
            "Spacing leaves no usable cell size in the parent axes."
        )

    out: list[list[Axes]] = []
    for r in range(nrows):
        row: list[Axes] = []
        for c in range(ncols):
            x0 = left_pad + c * (cell_w + wspace)
            y0 = bottom_pad + (nrows - 1 - r) * (cell_h + hspace)
            child = ax.inset_axes(
                (x0, y0, cell_w, cell_h), transform=ax.transAxes
            )
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


# =============================================================================
def mosaic_axes(  # -----------------------------------------------------------
    ax: Axes,
    n_items: int,
    *,
    wspace: float = 0.0,
    hspace: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    top_pad: float = 0.0,
    bottom_pad: float = 0.0,
    hide_parent: bool = True,
) -> NDArray[np.object_]:
    """Create a mosaic of inset axes within `ax`.

    Args:
        ax: Slot axes whose bounding box defines the available space.
        n_items: Number of panels to layout.
        wspace: Horizontal spacing in parent-axes fraction units (0..1).
        hspace: Vertical spacing in parent-axes fraction units (0..1).
        left_pad: Left padding in parent-axes fraction units (0..1).
        right_pad: Right padding in parent-axes fraction units (0..1).
        top_pad: Top padding in parent-axes fraction units (0..1).
        bottom_pad: Bottom padding in parent-axes fraction units (0..1).

    Notes:
        This helper samples the parent axes bounding box after the current
        figure layout has been resolved and treats that geometry as fixed while
        creating inset axes.

    Returns:
        A 2D numpy array of Axes instances laid out within the slot.
    """
    fig = ax.figure
    if fig is None:
        raise ValueError(
            "Cannot create a mosaic for an Axes that is not attached to a Figure."
        )

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

    nrows, ncols = _choose_generic_mosaic_shape(
        container_width=slot_w_in,
        container_height=slot_h_in,
        n_items=n_items,
        wspace=wspace * slot_w_in,
        hspace=hspace * slot_h_in,
        left_pad=left_pad * slot_w_in,
        right_pad=right_pad * slot_w_in,
        top_pad=top_pad * slot_h_in,
        bottom_pad=bottom_pad * slot_h_in,
    )
    return subdivide_axes(
        ax,
        int(nrows),
        int(ncols),
        wspace=wspace,
        hspace=hspace,
        left_pad=left_pad,
        right_pad=right_pad,
        bottom_pad=bottom_pad,
        top_pad=top_pad,
        hide_parent=hide_parent,
        squeeze=False,
    )


# =============================================================================
@dataclass(frozen=True)
class _MosaicChoice:
    nrows: int
    ncols: int
    key: tuple[float, int, float]


# =============================================================================
def _choose_generic_mosaic_shape(  # ------------------------------------------
    *,
    container_width: float,
    container_height: float,
    n_items: int,
    wspace: float = 0.0,
    hspace: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    top_pad: float = 0.0,
    bottom_pad: float = 0.0,
) -> tuple[int, int]:
    """Choose a free-aspect mosaic shape using effective drawable cell sizes."""
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got n_items={n_items}")
    if not np.isfinite(container_width) or container_width <= 0:
        raise ValueError(
            f"container_width must be positive, got {container_width}"
        )
    if not np.isfinite(container_height) or container_height <= 0:
        raise ValueError(
            f"container_height must be positive, got {container_height}"
        )

    usable_w = container_width - left_pad - right_pad
    usable_h = container_height - top_pad - bottom_pad
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Padding leaves no usable space in the container.")

    best: Optional[_MosaicChoice] = None
    for ncols in range(1, n_items + 1):
        nrows = int(math.ceil(n_items / ncols))
        cell_w = (usable_w - (ncols - 1) * wspace) / ncols
        cell_h = (usable_h - (nrows - 1) * hspace) / nrows
        if cell_w <= 0 or cell_h <= 0:
            continue

        score = min(cell_w, cell_h)
        if score <= 0:
            continue

        squareness = 1.0 - abs(cell_w - cell_h) / max(cell_w, cell_h)
        empties = nrows * ncols - n_items
        key = (float(score), int(-empties), float(squareness))
        if best is None or key > best.key:
            best = _MosaicChoice(nrows=nrows, ncols=ncols, key=key)

    if best is None:
        raise ValueError("No valid mosaic layout found for given constraints.")
    return int(best.nrows), int(best.ncols)


# =============================================================================
def _get_figure_size_inches(  # -----------------------------------------------
    fig: Any,
) -> tuple[float, float]:
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


# =============================================================================
__all__ = ["configure_environment_axes", "mosaic_axes", "subdivide_axes"]
