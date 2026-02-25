"""Shared colormaps for figures."""

from __future__ import annotations

from matplotlib.colors import ListedColormap


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


__all__ = ["maze_cmap"]
