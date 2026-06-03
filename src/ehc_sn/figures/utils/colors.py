"""Shared colormaps for figures."""

from __future__ import annotations

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# =============================================================================
def colormap_with_nan_color(name: str, nan_color: str = "0.92"):
    """Return a copy of a named matplotlib colormap with NaN colour set.

    Args:
        name: Matplotlib colormap name (e.g. ``"GnBu"``).
        nan_color: Colour string for NaN/bad values.

    Returns:
        Copy of the colormap with ``set_bad(nan_color)`` applied.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=nan_color)
    return cmap


# =============================================================================
def maze_cmap() -> ListedColormap:
    """Return the base colormap for MazeHard grids."""
    return ListedColormap(
        [
            "#f7f4ef",  # 0 (unused)
            "#1b1f24",  # 1 wall '#'
            "#f7f4ef",  # 2 space ' '
            "#2b6cb0",  # 3 start 'S'
            "#d69e2e",  # 4 goal 'G'
            "#f7f4ef",  # 5 path 'o' (not used in base)
        ]
    )


# =============================================================================
def attach_aligned_colorbar_fmt(mappable, vmax: float) -> None:
    """Store a ``FormatStrFormatter`` on a ScalarMappable for aligned colourbar ticks.

    The formatter uses space-padded positive values (``% .{ndp}f``) so that
    decimal points align vertically regardless of minus signs.  The stored
    ``_tem_cbar_fmt`` attribute is read by custom ``_apply_colorbars``
    overrides.

    Args:
        mappable: The ScalarMappable (returned by ``imshow``) to attach to.
        vmax: Symmetric vmax value; determines decimal places.
    """
    import matplotlib.ticker as ticker

    ndp = max(1, -int(np.floor(np.log10(vmax))) + 1)
    ndp = min(ndp, 3)
    mappable._tem_cbar_fmt = ticker.FormatStrFormatter(f"% .{ndp}f")


def observation_id_colormap(
    n_obs: int,
) -> tuple[ListedColormap, BoundaryNorm]:
    """Stable categorical colormap for Arena observation IDs.

    Maps integer observation IDs ``0..(n_obs-1)`` to perceptually
    distinct, deterministic colours.  All Arena figures that render
    observation identity must use this function with the same *n_obs*
    to keep ID→colour mappings consistent across figures.

    Uses golden-ratio hue spacing with fixed saturation and value.
    No external dependencies.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")

    golden = 0.618033988749895
    hues = [(i * golden) % 1.0 for i in range(n_obs)]
    colours = [colorsys.hsv_to_rgb(h, 0.68, 0.92) for h in hues]

    cmap = ListedColormap(colours, name=f"arena_obs_id_{n_obs}")
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, n_obs + 0.5, 1),
        ncolors=n_obs,
        clip=True,
    )
    return cmap, norm


# =============================================================================
def categorical_id_colormap(
    n_categories: int,
) -> tuple[mcolors.ListedColormap, BoundaryNorm]:
    """Return a categorical colormap and norm for *n_categories* distinct IDs.

    Designed for nominal category identifiers (observation IDs, region
    labels, etc.) where adjacent integer values have **no intrinsic
    ordering**.  Uses Matplotlib qualitative tab20-family palettes for up to
    60 categories, then falls back to deterministic HSV-derived colors for
    larger category counts.

    Args:
        n_categories: Number of distinct category IDs (must be >= 1).

    Returns:
        ``(cmap, norm)`` tuple where *cmap* is a
        :class:`~matplotlib.colors.ListedColormap` with exactly
        *n_categories* entries and *norm* is a
        :class:`~matplotlib.colors.BoundaryNorm` that maps integer
        category IDs ``0..(n_categories-1)`` to the corresponding
        colour bin.

    Raises:
        ValueError: If *n_categories* < 1.
    """
    if n_categories < 1:
        raise ValueError(f"n_categories must be >= 1, got {n_categories}.")

    # Collect colours from Matplotlib qualitative palettes.
    colours: list = []
    sources: list[tuple[str, int]] = [
        ("tab20", 20),
        ("tab20b", 20),
        ("tab20c", 20),
    ]
    for name, limit in sources:
        cm = plt.get_cmap(name)  # noqa: PLC0415
        colours.extend(
            cm(i) for i in range(min(limit, n_categories - len(colours)))
        )
        if len(colours) >= n_categories:
            break

    # Deterministic HSV fallback for >60 categories.
    while len(colours) < n_categories:
        base = colours[len(colours) % len(colours)] if colours else (0.5, 0.5, 0.5, 1.0)  # fmt: skip
        r, g, b, a = base
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        h = (h + 0.15 * (len(colours) // 60 + 1)) % 1.0
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        colours.append((r2, g2, b2, a))

    cmap = ListedColormap(colours[:n_categories], name=f"id_cat_{n_categories}")
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, n_categories + 0.5, 1),
        ncolors=n_categories,
        clip=True,
    )
    return cmap, norm


# =============================================================================
__all__ = [
    "categorical_id_colormap",
    "observation_id_colormap",
    "maze_cmap",
]
