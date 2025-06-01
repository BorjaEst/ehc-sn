import numpy as np

from ehc_sn.representations.grid_cells import GridCellModule
from ehc_sn.representations.place_cells import PlaceCellEnsemble


class SpatialScaffold:
    """
    A basic memory spatial scaffold representing a cognitive map of space.

    This is the foundational implementation without heteroassociations.
    """

    def __init__(self, dimensions=(100, 100), resolution=1.0):
        """
        Initialize a spatial scaffold.

        Parameters:
        -----------
        dimensions : tuple
            The dimensions of the space (width, height)
        resolution : float
            The resolution of the grid (size of each cell)
        """
        self.dimensions = dimensions
        self.resolution = resolution

        # Calculate the number of cells in each dimension
        self.grid_size = (
            int(dimensions[0] / resolution),
            int(dimensions[1] / resolution),
        )

        # Initialize an empty grid for the scaffold
        self.grid = np.zeros(self.grid_size)

        # Track the current position in the scaffold
        self.current_position = (0, 0)

    def set_activation(self, position, activation):
        """Set the activation value at a specific position."""
        cell_x = int(position[0] / self.resolution)
        cell_y = int(position[1] / self.resolution)

        # Ensure position is within bounds
        cell_x = max(0, min(cell_x, self.grid_size[0] - 1))
        cell_y = max(0, min(cell_y, self.grid_size[1] - 1))

        self.grid[cell_x, cell_y] = activation

    def get_activation(self, position):
        """Get the activation value at a specific position."""
        cell_x = int(position[0] / self.resolution)
        cell_y = int(position[1] / self.resolution)

        # Ensure position is within bounds
        cell_x = max(0, min(cell_x, self.grid_size[0] - 1))
        cell_y = max(0, min(cell_y, self.grid_size[1] - 1))

        return self.grid[cell_x, cell_y]

    def move(self, position):
        """Move to a new position in the spatial scaffold."""
        # Update current position
        self.current_position = position

        # Activate the current position
        self.set_activation(position, 1.0)

    def decay_activations(self, decay_factor=0.9):
        """Apply decay to all activations in the grid."""
        self.grid *= decay_factor


class MemorySpatialScaffold:
    """
    A comprehensive memory spatial scaffold integrating grid and place cell representations.

    This implementation provides the basic scaffold without heteroassociative connections.
    """

    def __init__(self, dimensions=(100, 100), resolution=1.0):
        """Initialize the memory spatial scaffold."""
        self.dimensions = dimensions
        self.resolution = resolution

        # Basic spatial scaffold for general activation tracking
        self.scaffold = SpatialScaffold(dimensions, resolution)

        # Grid cell modules at different scales
        self.grid_modules = [
            GridCellModule(
                scale=10.0,
                orientation=0.0,
                dimensions=dimensions,
                resolution=resolution,
            ),
            GridCellModule(
                scale=20.0,
                orientation=np.pi / 6,
                dimensions=dimensions,
                resolution=resolution,
            ),
            GridCellModule(
                scale=40.0,
                orientation=np.pi / 4,
                dimensions=dimensions,
                resolution=resolution,
            ),
        ]

        # Place cell ensemble
        self.place_cells = PlaceCellEnsemble(
            num_cells=200, dimensions=dimensions, field_width=15.0
        )

        # Current position and state
        self.current_position = (0, 0)
        self.grid_activations = [
            module.get_activation(self.current_position) for module in self.grid_modules
        ]
        self.place_activations = self.place_cells.get_activations(self.current_position)

    def update_position(self, position):
        """Update the current position and all related activations."""
        self.current_position = position

        # Update basic scaffold
        self.scaffold.move(position)

        # Update grid cell activations
        self.grid_activations = [
            module.get_activation(position) for module in self.grid_modules
        ]

        # Update place cell activations
        self.place_activations = self.place_cells.get_activations(position)

    def get_state_representation(self):
        """Get the current state representation from the spatial scaffold."""
        return {
            "position": self.current_position,
            "grid_activations": self.grid_activations,
            "place_activations": self.place_activations,
        }

    def decay_memory(self, decay_factor=0.9):
        """Apply decay to the memory scaffold."""
        self.scaffold.decay_activations(decay_factor)

    def get_position_encoding(self):
        """Get a combined position encoding vector from grid and place cells."""
        # Flatten grid activations
        grid_vector = np.array(self.grid_activations)

        # Get place cell population vector
        place_vector = self.place_cells.get_population_vector(self.current_position)

        # Combine the representations
        position_encoding = np.concatenate([grid_vector, place_vector])

        return position_encoding
