from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def action_patch(location_from: dict, location_to: dict, radius: float, colour) -> plt.Polygon:
    """Create a triangular patch representing an action/transition arrow.

    Args:
        location_from: Source location dict with keys "o" (x), "y" (y), "id".
        location_to: Destination location dict with keys "o" (x), "y" (y), "id".
        radius: Radius of location circles (for arrow scaling).
        colour: Matplotlib color spec for the arrow.

    Returns:
        plt.Polygon representing an arrow from location_from toward location_to.
    """
    if location_to["id"] == location_from["id"]:
        # Self-transition: arrow points down (pi/2 in inverted y-axis)
        a_dir = np.pi / 2
        xdat = location_from["o"] + radius * np.array([2 * np.cos(a_dir - np.pi / 6), 2 * np.cos(a_dir + np.pi / 6), 3 * np.cos(a_dir)])
        ydat = location_from["y"] - radius * 3 + radius * np.array([2 * np.sin(a_dir - np.pi / 6), 2 * np.sin(a_dir + np.pi / 6), 3 * np.sin(a_dir)])
    else:
        # Directed transition: compute direction vector
        xvec = location_to["o"] - location_from["o"]
        yvec = location_from["y"] - location_to["y"]  # Inverted y
        a_dir = np.arctan2(-yvec, xvec)

        xdat = location_from["o"] + radius * np.array([2 * np.cos(a_dir - np.pi / 6), 2 * np.cos(a_dir + np.pi / 6), 3 * np.cos(a_dir)])
        ydat = location_from["y"] + radius * np.array([2 * np.sin(a_dir - np.pi / 6), 2 * np.sin(a_dir + np.pi / 6), 3 * np.sin(a_dir)])

    return plt.Polygon(np.stack([xdat, ydat], axis=1), color=colour)
