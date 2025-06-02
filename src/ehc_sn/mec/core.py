"""
Medial Entorhinal Cortex (MEC) module.
Provides classes and functions for simulating the medial entorhinal cortex's
role in spatial navigation, including grid cells and their interactions with
the hippocampus.
"""

import numpy as np
from abc import ABC, abstractmethod


class GridCellsBase(ABC):
    """
    A class representing a network of grid cells in the medial entorhinal cortex,
    arranged in a hexagonal toroidal grid.

    This class simulates the behavior of grid cells, which are crucial for
    spatial navigation and mapping in the brain. The hexagonal arrangement
    matches the observed pattern in real grid cells, and the toroidal structure
    ensures continuous mapping.

    This class is the core component for modeling grid cells and contains the
    activity matrix, grid cell positions, and methods for updating
    activities based on spatial position. Coordinates are calculated based on
    hexagonal offset coordinates, and the grid can be oriented in any direction.

    Parameters
    ----------
    width : int
        Width of the grid (number of cells)
    height : int
        Height of the grid (number of cells)
    spacing : float
        Distance between grid cells (grid scale)
    orientation : float, optional
        Orientation of the grid in radians, default is 0
    """

    def __init__(self, width, height, spacing, orientation=0.0):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.orientation = orientation

        # Initialize grid activity matrix
        self.activity = np.zeros((height, width))

        # Initialize grid cell positions
        self.positions = self._initialize_grid_positions()

    def _initialize_grid_positions(self):
        """
        Initialize the positions of grid cells in hexagonal arrangement.

        Returns
        -------
        numpy.ndarray
            Array of shape (height, width, 2) containing the (x, y) coordinates
            of each grid cell
        """
        positions = np.zeros((self.height, self.width, 2))

        # Constants for hexagonal grid
        sqrt3 = np.sqrt(3)

        # Create rotation matrix for grid orientation
        rot_matrix = np.array(
            [
                [np.cos(self.orientation), -np.sin(self.orientation)],
                [np.sin(self.orientation), np.cos(self.orientation)],
            ]
        )

        for i in range(self.height):
            for j in range(self.width):
                # Calculate hexagonal grid positions
                # For even rows, shift x-coordinate
                x = j + (i % 2) * 0.5
                y = i * (sqrt3 / 2)

                # Apply rotation
                pos = np.dot(rot_matrix, np.array([x, y]))

                # Scale by spacing
                pos *= self.spacing

                positions[i, j] = pos

        return positions

    def get_neighbors(self, i, j):
        """
        Get the indices of neighboring cells in the hexagonal grid.

        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index

        Returns
        -------
        list
            List of (row, col) tuples for neighboring cells, accounting for
            toroidal boundaries
        """
        # Directions for hexagonal grid
        # For even rows: up, up-right, down-right, down, down-left, up-left
        # For odd rows, slightly different due to hexagonal staggering

        neighbors = []

        if i % 2 == 0:  # Even row
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)]
        else:  # Odd row
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (0, -1)]

        for di, dj in directions:
            ni = (i + di) % self.height  # Toroidal wrap
            nj = (j + dj) % self.width  # Toroidal wrap
            neighbors.append((ni, nj))

        return neighbors

    @abstractmethod
    def set_position(self, position):
        """
        Update grid cell activities based on current position.

        Parameters
        ----------
        position : tuple or array-like
            Current (x, y) position
        """
        pass

    def __repr__(self):
        """Return string representation of the GridCellsNetwork."""
        return (
            f"GridCellsNetwork(width={self.width}, height={self.height}, "
            f"spacing={self.spacing}, orientation={self.orientation})"
        )


class MECBase(ABC):
    """
    Medial Entorhinal Cortex (MEC) class for spatial navigation modeling.
    this class serves as a container for grid cells and their interactions
    with the hippocampus, simulating the neural mechanisms underlying
    spatial navigation and memory formation.

    Parameters
    ----------
    grid_cells : list of GridCellsNetwork
        List of grid cell networks to be managed by the MEC
    noise_level : float, optional
        Level of noise to add to grid cell activations, default is 0.1
    """

    def __init__(self, grid_cells, noise_level=0.1):
        """
        Initialize the MEC with a list of grid cell networks.

        Parameters
        ----------
        grid_cells : list of GridCellsNetwork
            List of grid cell networks to be managed by the MEC
        noise_level : float, optional
            Level of noise to add to grid cell activations, default is 0.1
        """
        self.grid_cells = grid_cells
        self.noise_level = noise_level
        self.current_position = np.zeros(2)

        # Track the total number of grid cells across all networks
        self.total_cells = sum(grid.width * grid.height for grid in grid_cells)

        # Initialize velocity input connections for path integration
        self.velocity_weights = np.random.randn(2, self.total_cells) * 0.01

    def set_position(self, position):
        """
        Update the activity of all grid cells based on the current position.

        Parameters
        ----------
        position : tuple or array-like
            Current (x, y) position in the environment
        """
        self.current_position = np.array(position)
        for grid in self.grid_cells:
            grid.set_position(position)

            # Add some noise to model biological variability
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, grid.activity.shape)
                grid.activity = np.clip(grid.activity + noise, 0, 1)

    def path_integrate(self, velocity, dt=0.1):
        """
        Update grid cell activity based on velocity input (path integration).

        This method simulates the path integration mechanism of grid cells,
        where activity patterns shift based on self-motion cues.

        Parameters
        ----------
        velocity : tuple or array-like
            Current (vx, vy) velocity vector
        dt : float, optional
            Time step for integration, default is 0.1

        Returns
        -------
        tuple
            Updated position after path integration

        References
        ----------
        .. [1] McNaughton, B. L., et al. (2006). Path integration and the
               neural basis of the 'cognitive map'. Nature Reviews Neuroscience.
        """
        velocity = np.array(velocity)

        # Update position based on velocity
        new_position = self.current_position + velocity * dt

        # Update grid cell activities for the new position
        self.set_position(new_position)

        return tuple(new_position)

    def get_spatial_encoding(self):
        """
        Get a concatenated vector of all grid cell activities, which forms
        a unique spatial encoding for the current position.

        Returns
        -------
        numpy.ndarray
            1D array containing all grid cell activations
        """
        # Flatten and concatenate all grid cell activities
        encoding = np.concatenate([grid.activity.flatten() for grid in self.grid_cells])

        return encoding

    def output_to_hippocampus(self):
        """
        Generate output that would typically be sent to the hippocampus.

        In biological systems, MEC grid cells project to the hippocampus
        to support place cell formation and spatial memory.

        Returns
        -------
        numpy.ndarray
            Processed output representation for hippocampal input
        """
        # Get the basic spatial encoding
        encoding = self.get_spatial_encoding()

        # Apply a nonlinear transformation to mimic biological processing
        # Threshold weak activations to create sparse coding
        threshold = 0.2
        encoding[encoding < threshold] = 0

        return encoding

    def get_population_vector(self):
        """
        Calculate a population vector from grid cell activity that represents
        the encoded position.

        Returns
        -------
        tuple
            (x, y) coordinates of the population vector
        """
        x_sum, y_sum, total_activity = 0, 0, 0

        for grid in self.grid_cells:
            for i in range(grid.height):
                for j in range(grid.width):
                    activity = grid.activity[i, j]
                    pos = grid.positions[i, j]

                    x_sum += pos[0] * activity
                    y_sum += pos[1] * activity
                    total_activity += activity

        # Avoid division by zero
        if total_activity > 0:
            x = x_sum / total_activity
            y = y_sum / total_activity
            return (x, y)
        else:
            return (0, 0)

    def __repr__(self):
        """Return string representation of the MEC."""
        return f"MEC(grid_cells={len(self.grid_cells)}, noise_level={self.noise_level})"
