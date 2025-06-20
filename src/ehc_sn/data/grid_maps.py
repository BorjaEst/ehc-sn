import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, field_validator

from ehc_sn.constants import Direction, GridSize, ObstacleMap, Position
from ehc_sn.data import _base
from ehc_sn.data._base import DataModule, DataModuleParams


class GeneratorParams(_base.GeneratorParams):
    """Parameters for generating grid maps with obstacles and goal positions."""

    grid_size: GridSize = Field(default=(16, 16), description="Size of the grid as (height, width)")
    obstacle_density: float = Field(default=0.2, description="Probability of a cell being an obstacle (0.0-1.0)")
    min_obstacles: int = Field(default=5, description="Minimum number of obstacles to generate")
    max_obstacles: int = Field(default=15, description="Maximum number of obstacles to generate")
    obstacle_size_range: Tuple[int, int] = Field(default=(1, 3), description="Range of obstacle sizes (min, max)")

    @field_validator("obstacle_density")
    @classmethod
    def validate_obstacle_density(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("obstacle_density must be between 0.0 and 1.0")
        return v

    @field_validator("obstacle_size_range")
    @classmethod
    def validate_size_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("min size must be less than or equal to max size")
        return v


class Generator(_base.Generator):
    """Generates grid maps with random obstacles and a goal position."""

    def __init__(self, params: Optional[GeneratorParams] = None):
        params = params or GeneratorParams()
        super().__init__(params)
        self.grid_size = params.grid_size
        self.obstacle_density = params.obstacle_density
        self.min_obstacles = params.min_obstacles
        self.max_obstacles = params.max_obstacles
        self.obstacle_size_range = params.obstacle_size_range

    def _is_valid_position(self, grid: ObstacleMap, pos: Position) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)."""
        h, w = grid.shape
        i, j = pos
        return 0 <= i < h and 0 <= j < w and grid[i, j] == 0

    def _get_valid_position(self, grid: ObstacleMap) -> Position:
        """Get a random valid position (not occupied by an obstacle)."""
        h, w = grid.shape
        while True:
            i, j = random.randint(0, h - 1), random.randint(0, w - 1)
            if grid[i, j] == 0:
                return (i, j)

    def _add_random_obstacle_cluster(self, grid: ObstacleMap) -> ObstacleMap:
        """Add a cluster of obstacles with random size and shape."""
        h, w = grid.shape
        size = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])

        # Start position for obstacle cluster
        start_i, start_j = random.randint(0, h - 1), random.randint(0, w - 1)
        grid[start_i, start_j] = 1  # Mark as obstacle

        # Grow obstacle cluster
        positions: List[Position] = [(start_i, start_j)]
        for _ in range(size - 1):
            if not positions:
                break

            # Pick a random position from existing obstacle positions
            pos_idx = random.randint(0, len(positions) - 1)
            i, j = positions[pos_idx]

            # Try to add an adjacent cell
            directions: List[Direction] = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for di, dj in directions:
                new_i, new_j = i + di, j + dj
                if 0 <= new_i < h and 0 <= new_j < w and grid[new_i, new_j] == 0:
                    grid[new_i, new_j] = 1
                    positions.append((new_i, new_j))
                    break

        return grid

    def __next__(self) -> ObstacleMap:
        """
        Generate a grid map with obstacles and a goal position.

        Returns:
            A one-hot encoded grid map with obstacles (1s) and free spaces (0s).
            The goal position is represented as a single 1 in a separate tensor.
        """
        # Initialize empty grid
        grid = torch.zeros(self.grid_size, dtype=torch.float32)

        # Add random obstacle clusters
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        for _ in range(num_obstacles):
            grid = self._add_random_obstacle_cluster(grid)

        # Ensure there's at least one free space for goal
        if torch.all(grid == 1):
            # If somehow all cells became obstacles, clear one
            i, j = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            grid[i, j] = 0

        # Return a one-hot encoder for the obstacle map
        return grid


class PlotMapParams(BaseModel):
    """Parameters for plotting grid maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    cmap: str = Field(default="binary", description="Colormap for the grid map")
    annot: bool = Field(default=False, description="Whether to annotate cells")
    title: Optional[str] = Field(default=None, description="Title for the plot")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return params as a dictionary for seaborn heatmap."""
        return dict(self.model_dump(exclude={"title"}))


def plot(ax: Axes, grid: ObstacleMap, params: Optional[PlotMapParams] = None) -> None:
    """Adds a grid map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        grid: ObstacleMap containing obstacle map and goal position
        params: Grid map visualization params

    Returns:
        The matplotlib axis with the plot
    """
    params = params or PlotMapParams()

    # Extract obstacle map
    if isinstance(grid, torch.Tensor):
        obstacles_data = grid.detach().cpu().numpy()
    else:
        obstacles_data = np.array(grid)

    # Plot the heatmap
    sns.heatmap(obstacles_data, ax=ax, cbar=False, **params.kwargs)

    # Set title if provided
    if params.title:
        ax.set_title(params.title)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add visual boundary to differentiate between subplots
    for spine in ax.spines.values():
        spine.set_visible(True)


# Example usage of the DataModule and plotting function
if __name__ == "__main__":

    # Create a generator with custom parameters
    generator_params = GeneratorParams(grid_size=(10, 10), obstacle_density=0.3)
    generator = Generator(generator_params)

    # Create the parameters for the data loader
    datamodule_params = DataModuleParams(num_samples=100, batch_size=16, val_split=0.2, test_split=0.1)
    data_module = DataModule(generator, datamodule_params)
    data_module.setup()

    # Visualize the generated map
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for ax, sample in zip(axs.flatten(), data_module.train_dataset):
        # Ensure sample is a tensor
        plot(ax, sample)

    plt.show()
