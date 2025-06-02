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

from ehc_sn.mec.grid_cells import GridCellsLayer as _GridCellsLayer
from ehc_sn.mec.network import MEC as _MECNetwork

__all__ = ["GridCellsLayer", "MECNetwork"]


class GridCellsLayer(_GridCellsLayer):
    """
    Represents a layer of grid cells in the medial entorhinal cortex (MEC).

    This class models a hexagonal grid cell arrangement found in the MEC that responds
    to specific locations in the environment, creating a periodic spatial map.
    These cells form the basis for spatial navigation and are thought to provide
    metric information to the hippocampus.

    Parameters
    ----------
    width : int
        Number of grid cells in the horizontal direction.
    height : int
        Number of grid cells in the vertical direction.
    spacing : float
        Distance between adjacent grid cells, determines the scale of the grid pattern.
        Biological grid cells show discrete scaling that increases along the dorsoventral axis.
    orientation : float, optional
        Orientation angle of the grid cells in radians, default is 0.0.
        In rodents, grid cell orientations are typically aligned to environmental boundaries.

    Attributes
    ----------
    width : int
        Width of the grid cell layer.
    height : int
        Height of the grid cell layer.
    spacing : float
        Spacing between grid cells, correlates with the scale of grid field.
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

    Notes
    -----
    Grid cells were discovered by May-Britt Moser and Edvard I. Moser, who shared
    the 2014 Nobel Prize in Physiology or Medicine with John O'Keefe.
    The grid pattern forms a hexagonal lattice that tessellates the environment,
    with multiple scales that may support hierarchical spatial representation.

    References
    ----------
    .. [1] Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
       Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
    .. [2] Moser, E. I., Kropff, E., & Moser, M. B. (2008).
       Place cells, grid cells, and the brain's spatial representation system.
       Annual review of neuroscience, 31, 69-89.
    """
    pass


class MECNetwork(_MECNetwork):
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
        Level of noise to add to grid cell activations, default is 0.1.
        Simulates biological variability in neural responses.

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

    Notes
    -----
    The MEC network is organized into modules of grid cells with similar spacing and
    orientation. These modules are arranged along the dorsoventral axis of the MEC,
    with spacing increasing from dorsal to ventral regions.

    Path integration in the MEC is thought to be mediated by velocity signals that
    update the grid cell representation, allowing for tracking of position even
    in the absence of external sensory cues.

    References
    ----------
    .. [1] Stensola, H., Stensola, T., Solstad, T., Frøland, K., Moser, M. B., & Moser, E. I. (2012).
       The entorhinal grid map is discretized. Nature, 492(7427), 72-78.
    .. [2] Bush, D., Barry, C., Manson, D., & Burgess, N. (2015).
       Using grid cells for navigation. Neuron, 87(3), 507-520.
    .. [3] McNaughton, B. L., Battaglia, F. P., Jensen, O., Moser, E. I., & Moser, M. B. (2006).
       Path integration and the neural basis of the 'cognitive map'.
       Nature Reviews Neuroscience, 7(8), 663-678.
    """
    pass
