"""
Hippocampus (HPC) module.

This module provides classes and functions for simulating the hippocampus's
role in spatial navigation, including place cells and their interactions with
the entorhinal cortex.

Classes
-------
PlaceCellsLayer
    Represents a layer of place cells in the hippocampus.
HPCNetwork
    Represents a network of place cells in the hippocampus.

Examples
--------
>>> from ehc_sn.hpc import PlaceCellsLayer, HPCNetwork
>>> import numpy as np
>>> # Create a HPC network
>>> model = HPCNetwork(
...     PlaceCellsLayer(n_cells=100, field_size=0.3)
... )
>>> # Update place cell activity based on a position
>>> model.set_position((1.0, 1.0))
"""

from ehc_sn.hpc.network import HPC as _HPCNetwork
from ehc_sn.hpc.place_cells import PlaceCellsLayer as _PlaceCellsLayer

__all__ = ["PlaceCellsLayer", "HPCNetwork"]


class PlaceCellsLayer(_PlaceCellsLayer):
    """
    Represents a layer of place cells in the hippocampus.

    This class models place cells in the hippocampus that respond to specific
    locations in the environment. Each place cell has a preferred spatial location
    (place field) where it fires maximally, with firing rate decreasing with
    distance from this location.

    Parameters
    ----------
    n_cells : int
        Number of place cells in the layer.
    field_size : float
        Size (standard deviation) of each place field, affecting the spatial
        selectivity of the place cells.
    environment_size : tuple of float, optional
        Size of the environment as (width, height), default is (10.0, 10.0).

    Attributes
    ----------
    n_cells : int
        Number of place cells in the layer.
    field_size : float
        Size of each place field (standard deviation).
    centers : ndarray
        2D array of shape (n_cells, 2) containing the (x, y) coordinates of each place field center.
    activity : ndarray
        1D array of shape (n_cells,) containing the activation level of each place cell.
    environment_size : tuple
        Size of the environment as (width, height).

    Methods
    -------
    set_position(position)
        Update place cell activities based on current position.
    calculate_activity(position)
        Calculate place cell activities for a given position without updating the internal state.
    show(ax=None, resolution=100, colormap='viridis', title=None)
        Visualize place cell activity across the environment.

    Notes
    -----
    Place cells were first described by John O'Keefe, who shared the 2014 Nobel Prize
    in Physiology or Medicine with May-Britt Moser and Edvard I. Moser.
    Place cells typically have a single preferred location in an environment, unlike
    grid cells which have multiple firing fields arranged in a hexagonal pattern.

    References
    ----------
    .. [1] O'Keefe, J., & Dostrovsky, J. (1971).
       The hippocampus as a spatial map. Preliminary evidence from unit activity in the
       freely-moving rat. Brain research, 34(1), 171-175.
    .. [2] O'Keefe, J. (1976).
       Place units in the hippocampus of the freely moving rat.
       Experimental neurology, 51(1), 78-109.
    """

    pass


class HPCNetwork(_HPCNetwork):
    """
    Represents a network of place cells in the hippocampus.

    This class manages place cells and their interactions with inputs from
    the medial entorhinal cortex, modeling how the hippocampus integrates
    spatial information to form a cognitive map of the environment.

    Parameters
    ----------
    place_cells : PlaceCellsLayer
        Layer of place cells to be included in the network.
    noise_level : float, optional
        Level of noise to add to place cell activations, default is 0.1.
        Simulates biological variability in neural responses.
    memory_decay : float, optional
        Rate of decay for stored place cell activity patterns, default is 0.05.
        Higher values lead to faster forgetting of previous locations.

    Attributes
    ----------
    place_cells : PlaceCellsLayer
        Layer of place cells.
    noise_level : float
        Amount of noise added to place cell activities.
    memory_decay : float
        Rate of decay for stored place cell activity patterns.
    current_position : ndarray or None
        Current (x, y) position in the environment, or None if not set.
    memory_trace : list
        History of place cell activity patterns, representing an episodic memory trace.

    Methods
    -------
    set_position(position)
        Update the activity of all place cells based on the current position.
    update_from_grid_cells(grid_activity)
        Update place cell activity based on input from grid cells.
    store_memory()
        Store current place cell activity pattern in episodic memory.
    recall_memory(pattern)
        Retrieve stored memory pattern most similar to the provided pattern.
    calculate_overlap(pattern1, pattern2)
        Calculate the overlap between two activity patterns.
    show(figsize=(10, 8))
        Visualize place cell activity across the environment.

    Notes
    -----
    The hippocampus is thought to perform pattern separation and pattern completion,
    which allow similar experiences to be stored as distinct memories and partial
    cues to trigger recall of complete memories, respectively.

    Episodic memory depends on the hippocampus, which binds together different
    aspects of an experience, including its spatial context.

    References
    ----------
    .. [1] Rolls, E. T. (2013).
       The mechanisms for pattern completion and pattern separation in the hippocampus.
       Frontiers in systems neuroscience, 7, 74.
    .. [2] Yassa, M. A., & Stark, C. E. (2011).
       Pattern separation in the hippocampus.
       Trends in neurosciences, 34(10), 515-525.
    .. [3] Hartley, T., Lever, C., Burgess, N., & O'Keefe, J. (2014).
       Space in the brain: how the hippocampal formation supports spatial cognition.
       Philosophical Transactions of the Royal Society B: Biological Sciences, 369(1635), 20120510.
    """

    pass
