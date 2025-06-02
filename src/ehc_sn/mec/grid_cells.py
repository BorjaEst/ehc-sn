import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


class GridCellsBase:
    # Base class for grid cells, providing core functionality for grid cell layers

    def __init__(self, width, height, spacing, orientation=0.0):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.orientation = orientation

        # Initialize activity matrix
        self.activity = np.zeros((height, width))

        # Initialize grid positions
        self.positions = self._initialize_grid_positions()

    def _initialize_grid_positions(self):

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

    def set_position(self, position):
        # Update grid cell activities based on current position
        position = np.array(position)

        # Calculate the environment dimensions
        env_width = self.width * self.spacing
        env_height = self.height * self.spacing

        # Normalize position to be within environment bounds
        # This handles multiple wraps by converting any position to its
        # equivalent position within the environment
        normalized_pos = np.array([position[0] % env_width, position[1] % env_height])

        # Calculate activity for each grid cell
        for i in range(self.height):
            for j in range(self.width):
                # Calculate distance to current position, accounting for toroidal wrap
                cell_pos = self.positions[i, j]

                # For toroidal distance, need to consider wrapping in both dimensions
                dx = min(
                    abs(cell_pos[0] - normalized_pos[0]),
                    env_width - abs(cell_pos[0] - normalized_pos[0]),
                )
                dy = min(
                    abs(cell_pos[1] - normalized_pos[1]),
                    env_height - abs(cell_pos[1] - normalized_pos[1]),
                )

                distance = np.sqrt(dx**2 + dy**2)

                # Gaussian activation based on distance
                self.activity[i, j] = np.exp(-(distance**2) / (2 * self.spacing**2))


class GridLayerView(GridCellsBase):
    # A view class for grid cell layers that inherits core functionality and adds visualization

    def show(self, ax=None, cell_size=0.8, edgecolor="black", title=None):
        # Creates a visual representation of the grid cell layer

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create proper colormap for activity coloring
        cmap = plt.get_cmap("viridis")
        # Normalize activity values to [0, 1] for colormap
        norm = colors.Normalize(vmin=0, vmax=np.max(self.activity) or 1.0)

        # Plot hexagons at each grid cell position
        for i in range(self.height):
            for j in range(self.width):
                pos = self.positions[i, j]

                # Create hexagon
                hex_cell = RegularPolygon(
                    pos,
                    numVertices=6,
                    radius=cell_size * self.spacing / 2,
                    orientation=np.pi / 6,  # Orientation for flat-topped hexagon
                    edgecolor=edgecolor,
                    facecolor=cmap(norm(self.activity[i, j])),
                    alpha=0.8,
                )
                ax.add_patch(hex_cell)

        # Set axis limits a bit larger than the grid
        max_pos = np.max(self.positions) + self.spacing
        min_pos = np.min(self.positions) - self.spacing
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(min_pos, max_pos)
        ax.set_aspect("equal")

        # Set title
        if title is None:
            title = f"Grid Cell Layer Structure (spacing={self.spacing})"
        ax.set_title(title)

        return ax


class GridCellsLayer(GridLayerView, GridCellsBase):
    """Represents a layer of grid cells in the medial entorhinal cortex (MEC)."""

    def __init__(self, width, height, spacing, orientation=0.0):
        GridCellsBase.__init__(self, width, height, spacing, orientation)

    def __repr__(self):
        return (
            f"GridCellsLayer(width={self.width}, height={self.height}, "
            f"spacing={self.spacing}, orientation={self.orientation})"
        )
