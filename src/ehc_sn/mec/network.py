import matplotlib.pyplot as plt
import numpy as np


class MECBase:
    # Base class for MEC networks, providing core functionality for grid cell layers

    def __init__(self, grid_cells, noise_level=0.1):
        self.grid_cells = grid_cells
        self.noise_level = noise_level
        self.current_position = np.zeros(2)

        # Track the total number of grid cells across all networks
        self.total_cells = sum(grid.width * grid.height for grid in grid_cells)

        # Initialize velocity input connections for path integration
        self.velocity_weights = 0.01 * np.random.randn(2, self.total_cells)

    def set_position(self, position):

        self.current_position = np.array(position)
        for grid in self.grid_cells:
            grid.set_position(position)

            # Add some noise to model biological variability
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, grid.activity.shape)
                grid.activity = np.clip(grid.activity + noise, 0, 1)

    def path_integrate(self, velocity, dt=0.1):

        velocity = np.array(velocity)

        # Update position based on velocity
        new_position = self.current_position + velocity * dt

        # Update grid cell activities for the new position
        self.set_position(new_position)

        return tuple(new_position)

    def get_spatial_encoding(self):

        # Flatten and concatenate all grid cell activities
        encoding = np.concatenate([grid.activity.flatten() for grid in self.grid_cells])

        return encoding

    def output_to_hippocampus(self):

        # Get the basic spatial encoding
        encoding = self.get_spatial_encoding()

        # Apply a nonlinear transformation to mimic biological processing
        # Threshold weak activations to create sparse coding
        threshold = 0.2
        encoding[encoding < threshold] = 0

        return encoding

    def get_population_vector(self):

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


class MECView(MECBase):
    # A view class for MEC networks that inherits core functionality and adds visualization

    def show(self, figsize=(16, 10)):
        # Creates a visual representation of all grid cell layers in the network

        n_grids = len(self.grid_cells)

        # Create a figure with just the grid structures
        fig, axes = plt.subplots(1, n_grids, figsize=figsize)

        # Handle case with single grid
        if n_grids == 1:
            axes = [axes]

        # Plot each grid layer
        for i, grid in enumerate(self.grid_cells):
            grid.show(
                ax=axes[i],
                title=f"Grid Cells (spacing={grid.spacing:.2f}, orientation={grid.orientation:.2f})",
            )

        plt.tight_layout()
        return fig


class MEC(MECView, MECBase):
    """Represents a network of grid cells in the medial entorhinal cortex (MEC)."""

    def __init__(self, grid_cells, noise_level=0.1):
        MECBase.__init__(self, grid_cells, noise_level)

    def __repr__(self):
        return f"MEC(grid_cells={len(self.grid_cells)}, noise_level={self.noise_level})"
