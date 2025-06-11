import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class PlaceCellsLayer:
    # Implementation of PlaceCellsLayer.

    def __init__(self, n_cells, field_size, environment_size=(10.0, 10.0)):
        self.n_cells = n_cells
        self.field_size = field_size
        self.environment_size = environment_size

        # Generate random place field centers within the environment
        width, height = environment_size
        self.centers = np.random.uniform(low=[0, 0], high=[width, height], size=(n_cells, 2))

        # Initialize place cell activity to zeros
        self.activity = np.zeros(n_cells)

    def calculate_activity(self, position):
        # Calculate the activity of place cells based on the current position.
        position = np.asarray(position)

        # Calculate squared distances from position to all place field centers
        squared_dists = np.sum((self.centers - position) ** 2, axis=1)

        # Calculate Gaussian activation based on distance to centers
        # exp(-(d^2) / (2 * sigma^2)), where sigma is the field_size
        sigma_squared = 2 * (self.field_size**2)
        activities = np.exp(-squared_dists / sigma_squared)

        return activities

    def set_position(self, position):
        # Update the activity of place cells based on the current position.
        self.activity = self.calculate_activity(position)

        return self.activity

    def show(self, ax=None, resolution=100, colormap="viridis", title=None):
        # Visualize the spatial activity of place cells as a heatmap.
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        width, height = self.environment_size
        x = np.linspace(0, width, resolution)
        y = np.linspace(0, height, resolution)
        xx, yy = np.meshgrid(x, y)

        # Create a spatial activity map by computing activity at each grid point
        activity_map = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                position = (xx[i, j], yy[i, j])
                # Sum activity across all place cells for this position
                cell_activities = self.calculate_activity(position)
                activity_map[i, j] = np.sum(cell_activities)

        # Normalize for better visualization
        if np.max(activity_map) > 0:
            activity_map = activity_map / np.max(activity_map)

        # Plot spatial activity as a heatmap
        im = ax.imshow(
            activity_map,
            extent=(0, width, 0, height),
            origin="lower",
            cmap=colormap,
            norm=mcolors.Normalize(vmin=0, vmax=1),
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Normalized Activity")

        # Set labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(title or "Place Cell Activity Map")

        # Plot place field centers
        ax.scatter(
            self.centers[:, 0],
            self.centers[:, 1],
            color="red",
            s=10,
            alpha=0.5,
            label="Field Centers",
        )
        ax.legend()

        return ax
