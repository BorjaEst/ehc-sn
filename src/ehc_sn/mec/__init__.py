"""
Medial Entorhinal Cortex (MEC) module.

This module provides classes and functions for simulating the medial entorhinal cortex's
role in spatial navigation, including grid cells and their interactions with
the hippocampus.

Classes
-------
GridCellsLayer
    Represents a layer of grid cells in the medial entorhinal cortex (MEC).
MECNetwork
    Represents a network of grid cells in the medial entorhinal cortex (MEC).

Examples
--------
>>> from ehc_sn.mec import GridCellsLayer, MECNetwork
>>> import numpy as np
>>> # Create a MEC network with two grid cell layers
>>> model = MECNetwork([
...     GridCellsLayer(width=10, height=10, spacing=0.2, orientation=0.0),
...     GridCellsLayer(width=10, height=10, spacing=0.4, orientation=np.pi/6)
... ])
>>> # Update grid activity based on a position
>>> model.set_position((1.0, 1.0))
"""

from ehc_sn.mec import view, core


class GridCellsLayer(view.GridLayerView, core.GridCellsBase):
    """
    Represents a layer of grid cells in the medial entorhinal cortex (MEC).

    This class combines the core grid cell functionality with visualization capabilities.
    Grid cells are arranged in a hexagonal pattern and respond to specific locations
    in the environment, creating a spatial map that aids in navigation.

    Parameters
    ----------
    width : int
        Number of grid cells in the horizontal direction.
    height : int
        Number of grid cells in the vertical direction.
    spacing : float
        Distance between adjacent grid cells, which determines the scale of the grid pattern.
    orientation : float
        Orientation angle of the grid cells in radians.

    Attributes
    ----------
    width : int
        Width of the grid cell layer.
    height : int
        Height of the grid cell layer.
    spacing : float
        Spacing between grid cells.
    orientation : float
        Orientation of the grid in radians.
    activity : ndarray
        2D array of shape (height, width) containing the activation level of each grid cell.
    positions : ndarray
        3D array of shape (height, width, 2) containing the (x, y) coordinates of each grid cell.

    Methods
    -------
    set_position(position)
        Update grid cell activities based on current position.
    get_neighbors(i, j)
        Get the indices of neighboring cells in the hexagonal grid.
    show(ax=None, cell_size=0.8, edgecolor="black", title=None)
        Visualize the grid cell layer with hexagonal cells colored by activity.
    """


class MECNetwork(view.MECView, core.MECBase):
    """
    Represents a network of grid cells in the medial entorhinal cortex (MEC).

    This class manages multiple layers of grid cells with different scales and orientations,
    allowing for complex spatial representations similar to those found in the mammalian brain.
    The MEC network integrates information across grid scales and projects to the hippocampus
    for place cell formation.

    Parameters
    ----------
    grid_cells : list of GridCellsLayer
        List of grid cell layers to be included in the network.
    noise_level : float, optional
        Level of noise to add to grid cell activations (default=0.1).

    Attributes
    ----------
    grid_cells : list
        List of GridCellsLayer objects.
    noise_level : float
        Amount of noise added to grid cell activities.
    current_position : ndarray
        Current (x, y) position in the environment.
    total_cells : int
        Total number of grid cells across all layers.
    velocity_weights : ndarray
        Weights for velocity inputs used in path integration.

    Methods
    -------
    set_position(position)
        Update the activity of all grid cells based on the current position.
    path_integrate(velocity, dt=0.1)
        Update grid cell activity based on velocity input (path integration).
    get_spatial_encoding()
        Get a concatenated vector of all grid cell activities.
    output_to_hippocampus()
        Generate output that would be sent to the hippocampus.
    get_population_vector()
        Calculate a population vector from grid cell activity.
    show(figsize=(16, 10))
        Visualize all grid cell layers in the network.
    """
