from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from ehc_sn.figures.utils.actions import action_patch
from ehc_sn.figures.utils.axes import (
    _environment_locations,
    _environment_n_locations,
    configure_environment_axes,
)


def plot_map(
    environment,
    values: NDArray,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_cols: int = 100,
    location_cm: str = "viridis",
    action_cm: str = "Pastel1",
    do_plot_actions: bool = False,
    shape: str = "circle",
    radius: Optional[float] = None,
) -> plt.Axes:
    """Render an environment map with per-location scalar values.

    Draws locations as colored circles/squares where color represents the value.
    Optionally overlays action arrows. Shiny locations are highlighted with red outlines.
    When location metadata includes a boolean ``valid`` field, invalid cells are
    omitted entirely so the background reflects accessible occupancy only.

    Args:
        environment: Environment object with .locations list and .n_locations, .n_actions.
        values: Per-location scalar values (shape: n_locations,).
        ax: Axes to draw on. If None, initializes new axes.
        vmin: Minimum value for colormap normalization (None = auto from values).
        vmax: Maximum value for colormap normalization (None = auto from values).
        num_cols: Number of discrete colors in the colormap.
        location_cm: Colormap name for location coloring.
        action_cm: Colormap name for action arrows.
        do_plot_actions: Whether to draw action transition arrows.
        shape: Location marker shape ("circle" or "square").
        radius: Radius/half-width of location markers (None = auto-scale).

    Returns:
        The axes object with the environment map rendered.
    """
    values = np.asarray(values, dtype=float)
    locations = _environment_locations(environment)
    n_locations = _environment_n_locations(environment)
    if values.size != n_locations:
        raise ValueError("values length must match number of locations: " f"{values.size} != {n_locations}")
    has_finite = values.size > 0 and np.isfinite(values).any()

    # Handle NaN values by using nanmin/nanmax when possible
    vmin = (np.nanmin(values) if has_finite else 0.0) if vmin is None else vmin
    vmax = (np.nanmax(values) if has_finite else 1.0) if vmax is None else vmax

    location_cm = plt.get_cmap(location_cm, num_cols)
    action_cm = plt.get_cmap(action_cm, max(getattr(environment, "n_actions", 0), 1))

    # Track invalid values for dedicated styling
    invalid_mask = ~np.isfinite(values)
    valid_locations = _location_valid_mask(locations)

    # Auto-scale radius based on environment density
    if radius is None:
        radius = _default_radius(n_locations)

    if ax is None:
        _, ax = plt.subplots()
    ax = configure_environment_axes(ax, environment=environment, radius=radius)

    location_patches: List = []
    nan_patches: List = []
    action_patches: List = []
    outline_patches: List = []

    # Draw locations
    for i, location in enumerate(locations):
        if not valid_locations[i]:
            continue

        is_invalid = invalid_mask[i] if invalid_mask.size else False
        if shape == "square":
            patch = plt.Rectangle(
                (location["o"] - radius / 2, location["y"] - radius / 2),
                radius,
                radius,
            )
        else:  # circle
            patch = plt.Circle(
                (location["o"], location["y"]),
                radius,
            )

        if is_invalid:
            nan_patches.append(patch)
        else:
            location_patches.append(patch)

        # Draw action arrows if requested
        if do_plot_actions:
            for action in location["actions"]:
                if action["probability"] > 0:
                    transitions = np.array(action["transition"])
                    loc_indices = np.where((transitions > 0) & valid_locations)[0]
                    locations_to = [locations[loc_to] for loc_to in loc_indices]
                    for loc_to in locations_to:
                        action_patches.append(
                            action_patch(
                                location,
                                loc_to,
                                radius,
                                action_cm(action["id"]),
                            )
                        )

    # Highlight shiny locations with red outline
    for i, location in enumerate(locations):
        if not valid_locations[i]:
            continue
        if location.get("shiny", False):
            if shape == "square":
                outline = plt.Rectangle(
                    (location["o"] - radius / 2, location["y"] - radius / 2),
                    radius,
                    radius,
                    linewidth=1,
                    facecolor="none",
                    edgecolor=[1, 0, 0],
                )
            else:
                outline = plt.Circle(
                    (location["o"], location["y"]),
                    radius,
                    linewidth=1,
                    facecolor="none",
                    edgecolor=[1, 0, 0],
                )
            outline_patches.append(outline)

    if nan_patches:
        nan_collection = PatchCollection(
            nan_patches,
            facecolor="#d9d9d9",
            edgecolor="#444444",
            linewidth=0.6,
        )
        ax.add_collection(nan_collection)

    if location_patches:
        location_collection = PatchCollection(
            location_patches,
            cmap=location_cm,
            edgecolor="none",
            linewidth=0.0,
        )
        location_collection.set_norm(Normalize(vmin=vmin, vmax=vmax))
        location_collection.set_array(np.asarray(values[valid_locations & ~invalid_mask], dtype=float))  # fmt: skip
        ax.add_collection(location_collection)

    # Add action arrows and shiny outlines on top of the locations.
    for patch in action_patches + outline_patches:
        ax.add_patch(patch)

    return ax


def _default_radius(n_locations: int) -> float:
    if n_locations <= 0:
        return 0.05
    return 2 * (0.01 + 1 / (10 * np.sqrt(n_locations)))


def _location_valid_mask(locations: List[dict]) -> NDArray[np.bool_]:
    """Return the canonical occupancy mask for environment locations.

    When location metadata provides a ``valid`` field, invalid cells are treated
    as structurally inaccessible and omitted from background rendering. Missing
    ``valid`` fields default to ``True`` for backward compatibility.
    """
    if not locations:
        return np.zeros((0,), dtype=bool)
    return np.asarray([bool(location.get("valid", True)) for location in locations], dtype=bool)
