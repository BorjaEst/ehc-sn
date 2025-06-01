"""
Medial Entorhinal Cortex (MEC) module.
Provides classes and functions for simulating the medial entorhinal cortex's
role in spatial navigation, including grid cells and their interactions with
the hippocampus.
"""

import numpy as np
from ehc_sn.mec import view, core


class GridCellsLayer(view.GridLayerView, core.GridCellsBase):
    """
    Represents a layer of grid cells in the medial entorhinal cortex (MEC).

    This class extends the GridCellsBase for core functionality and GridLayerView
    for visualization capabilities.

    Parameters
    ----------
    width : int
        Number of grid cells in the horizontal direction.
    height : int
        Number of grid cells in the vertical direction.
    spacing : float
        Distance between adjacent grid cells.
    orientation : float
        Orientation angle of the grid cells in radians.
    """

    def __init__(self, width, height, spacing, orientation):
        super().__init__(width, height, spacing, orientation)

    def update_activity(self, position):
        """
        Update grid cell activities based on current position.

        Parameters
        ----------
        position : tuple or array-like
            Current (x, y) position
        """
        position = np.array(position)

        # Calculate activity for each grid cell
        for i in range(self.height):
            for j in range(self.width):
                # Calculate distance to current position, accounting for toroidal wrap
                cell_pos = self.positions[i, j]

                # For toroidal distance, need to consider wrapping in both dimensions
                dx = min(
                    abs(cell_pos[0] - position[0]),
                    self.width * self.spacing - abs(cell_pos[0] - position[0]),
                )
                dy = min(
                    abs(cell_pos[1] - position[1]),
                    self.height * self.spacing - abs(cell_pos[1] - position[1]),
                )

                distance = np.sqrt(dx**2 + dy**2)

                # Gaussian activation based on distance
                self.activity[i, j] = np.exp(-(distance**2) / (2 * self.spacing**2))


class MECNetwork(view.MECView, core.MECBase):
    """
    Represents a network of grid cells in the medial entorhinal cortex (MEC).

    This class manages multiple layers of grid cells, allowing for complex spatial
    representations.

    Parameters
    ----------
    grid_cells : list of GridCellsLayer
        List of grid cell layers to be included in the network.
    noise_level : float, optional
        Level of noise to add to grid cell activations, default is 0.1
    """

    def __init__(self, grid_cells, noise_level=0.1):
        super().__init__(grid_cells, noise_level)

    def set_position(self, position):
        """
        Update the activity of grid cells based on a given position.

        Parameters
        ----------
        position : tuple
            The (x, y) coordinates to update the grid cell activities.
        """
        for layer in self.grid_cells:
            layer.update_activity(position)


if __name__ == "__main__":
    # Example usage of the GridCellsNetwork class
    model = MECNetwork(
        [
            GridCellsLayer(width=10, height=10, spacing=0.2, orientation=0.0),
            GridCellsLayer(width=10, height=10, spacing=0.4, orientation=np.pi / 6),
        ]
    )

    print("MEC instance created:", model)

    # Update grid activity based on a position
    test_position = (1.0, 1.0)
    model.set_position(test_position)

    # Display the activity of the first grid cell network
    print("Grid cell activities after update at position", test_position)
    print(model.grid_cells[0].activity)
