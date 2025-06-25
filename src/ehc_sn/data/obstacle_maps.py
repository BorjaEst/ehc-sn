import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, field_validator
from torch import Tensor

from ehc_sn.constants import Direction, GridSize, ObstacleMap, Position
from ehc_sn.data import _base
from ehc_sn.data._base import DataModule, DataModuleParams


# -------------------------------------------------------------------------------------------
class ObstaclesParams(_base.GeneratorParams):
    """Parameters for obstacles map with block structures."""

    # -----------------------------------------------------------------------------------
    grid_size: GridSize = Field(
        default=(16, 16),
        description="Size of the grid as (height, width)",
    )

    # Control parameters
    # -----------------------------------------------------------------------------------
    obstacle_density: float = Field(
        default=0.3,
        description="Approximate percentage of grid to fill with obstacles",
    )
    # -----------------------------------------------------------------------------------
    min_obstacles: int = Field(
        default=1,
        description="Minimum number of obstacle structures to generate",
    )
    # -----------------------------------------------------------------------------------
    max_obstacles: int = Field(
        default=5,
        description="Maximum number of obstacle structures to generate",
    )

    # -----------------------------------------------------------------------------------
    @field_validator("obstacle_density")
    @classmethod
    def validate_obstacle_density(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("obstacle_density must be between 0.0 and 1.0")
        return v


# -------------------------------------------------------------------------------------------
class BaseObstacleGenerator(_base.Generator):
    """Base class for obstacle map generators with common utility methods."""

    def __init__(self, params: ObstaclesParams):
        super().__init__(params)
        self.grid_size = params.grid_size
        self.obstacle_density = params.obstacle_density
        self.min_obstacles = params.min_obstacles
        self.max_obstacles = params.max_obstacles

    # -----------------------------------------------------------------------------------
    def _is_valid_position(self, grid: ObstacleMap, pos: Position) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)."""
        h, w = grid.shape
        i, j = pos
        return 0 <= i < h and 0 <= j < w and grid[i, j] == 0

    # -----------------------------------------------------------------------------------
    def _get_valid_position(self, grid: ObstacleMap, min_distance: int = 0) -> Position:
        """Get a random valid position (not occupied by an obstacle) using vectorized operations."""
        h, w = grid.shape

        # Fast path for min_distance=0 case
        if min_distance == 0:
            # Find all valid positions (where grid == 0)
            valid_positions = torch.nonzero(grid == 0, as_tuple=False)

            if valid_positions.size(0) == 0:  # No valid positions
                # Create one valid position by clearing a random cell
                i, j = random.randint(0, h - 1), random.randint(0, w - 1)
                grid[i, j] = 0
                return (i, j)

            # Select a random valid position
            idx = random.randint(0, valid_positions.size(0) - 1)
            return tuple(valid_positions[idx].tolist())

        # For min_distance > 0, we need more complex logic
        # Create a mask of valid positions first (zeros in grid)
        valid_mask = grid == 0

        # Then for each valid position, check if it maintains min_distance from obstacles
        if min_distance > 0:
            # Create a dilated obstacle map using convolution
            kernel_size = 2 * min_distance + 1
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=grid.device)

            # Prepare grid for convolution (add batch and channel dimensions)
            grid_for_conv = grid.unsqueeze(0).unsqueeze(0)

            # Dilate obstacles using convolution
            # Any non-zero value in the result means it's too close to an obstacle
            dilated = torch.nn.functional.conv2d(grid_for_conv, kernel, padding=min_distance).squeeze() > 0

            # Update valid mask to exclude positions too close to obstacles
            valid_mask = valid_mask & ~dilated

        # Convert mask to position indices
        valid_positions = torch.nonzero(valid_mask, as_tuple=False)

        if valid_positions.size(0) == 0:
            # Fall back to any valid position without min_distance constraint
            return self._get_valid_position(grid, min_distance=0)

        # Select a random valid position
        idx = random.randint(0, valid_positions.size(0) - 1)
        return tuple(valid_positions[idx].tolist())

    # -----------------------------------------------------------------------------------
    def _ensure_free_space(self, grid: ObstacleMap) -> ObstacleMap:
        """Ensure there's at least one free space in the grid."""
        if torch.all(grid == 1):
            i, j = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            grid[i, j] = 0
        return grid

    # -----------------------------------------------------------------------------------
    @abstractmethod
    def _add_obstacles(self, grid: ObstacleMap) -> ObstacleMap:
        """Abstract method to add obstacles to the grid. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _add_obstacles")

    # -----------------------------------------------------------------------------------
    def __next__(self) -> Tuple[ObstacleMap]:
        """
        Generate a grid map with structured obstacles more efficiently.

        Returns:
            A tensor grid map with obstacles (1s) and free spaces (0s).
        """
        # Initialize empty grid
        grid = torch.zeros(self.grid_size, dtype=torch.float32)

        # Determine target obstacle count and density
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        target_density = self.obstacle_density
        max_attempts = num_obstacles * 2  # Adaptive attempt limit

        # Generate obstacles more efficiently by tracking the current density
        attempts = 0
        current_density = 0

        while current_density < target_density and attempts < max_attempts:
            original_density = current_density
            grid = self._add_obstacles(grid)
            current_density = torch.mean(grid).item()

            # Adaptive attempt limit - stop early if we're not making progress
            if current_density <= original_density:
                attempts += 1
            else:
                # Reset attempts when we make progress
                attempts = 0

        # Ensure there's at least one free space
        grid = self._ensure_free_space(grid)

        return (grid,)


# -------------------------------------------------------------------------------------------
class BlockMapParams(ObstaclesParams):
    """Parameters for obstacles map with block structures."""

    # Obstacles features
    # -----------------------------------------------------------------------------------
    min_block_size: Tuple[int, int] = Field(
        default=(2, 2),
        description="Minimum size of block obstacles (h, w)",
    )
    # -----------------------------------------------------------------------------------
    max_block_size: Tuple[int, int] = Field(
        default=(4, 4),
        description="Maximum size of block obstacles (h, w)",
    )


class BlockMapGen(BaseObstacleGenerator):
    """Generates grid maps with structured block obstacles."""

    def __init__(self, params: Optional[BlockMapParams] = None):
        params = params or BlockMapParams()
        super().__init__(params)
        self.min_block_size = params.min_block_size
        self.max_block_size = params.max_block_size

    # -----------------------------------------------------------------------------------
    def _add_obstacles(self, grid: ObstacleMap) -> ObstacleMap:
        """Add a rectangular block obstacle."""
        return self._add_block(grid)

    # -----------------------------------------------------------------------------------
    def _add_block(self, grid: ObstacleMap) -> ObstacleMap:
        """Add a rectangular block obstacle with vectorized operations."""
        h, w = grid.shape

        # Determine block size
        block_h = random.randint(self.min_block_size[0], min(self.max_block_size[0], h // 2))
        block_w = random.randint(self.min_block_size[1], min(self.max_block_size[1], w // 2))

        # Pre-compute maximum valid positions for block placement
        max_i = h - block_h
        max_j = w - block_w

        # Generate multiple candidate positions at once
        num_candidates = min(50, max_i * max_j // 4)  # Limit number of candidates

        if num_candidates == 0:  # Handle edge case of very small grid
            return grid

        # Generate candidate positions
        candidate_is = torch.randint(0, max_i + 1, (num_candidates,)).tolist()
        candidate_js = torch.randint(0, max_j + 1, (num_candidates,)).tolist()

        # Check each candidate position
        for start_i, start_j in zip(candidate_is, candidate_js):
            # Extract the region where we want to place the block
            region = grid[start_i : start_i + block_h, start_j : start_j + block_w]

            # If the region is completely free (all zeros), place the block
            if torch.all(region == 0):
                grid[start_i : start_i + block_h, start_j : start_j + block_w] = 1
                return grid

        # If we couldn't place a block after checking all candidates, try a smaller block
        if block_h > self.min_block_size[0] or block_w > self.min_block_size[1]:
            new_block_h = max(self.min_block_size[0], block_h - 1)
            new_block_w = max(self.min_block_size[1], block_w - 1)

            # Create a temporary block to place
            temp_block = torch.ones((new_block_h, new_block_w), dtype=grid.dtype)

            # Find random position for smaller block
            for _ in range(20):  # Limit attempts
                start_i = random.randint(0, h - new_block_h)
                start_j = random.randint(0, w - new_block_w)
                region = grid[start_i : start_i + new_block_h, start_j : start_j + new_block_w]

                if torch.all(region == 0):
                    grid[start_i : start_i + new_block_h, start_j : start_j + new_block_w] = 1
                    break

        # If we still couldn't place a block, return the grid as is
        return grid


# -------------------------------------------------------------------------------------------
class MazeMapParams(ObstaclesParams):
    """Parameters for data augmentation with maze structures of obstacle maps."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # Obstacles features
    # -----------------------------------------------------------------------------------
    corridor_width: int = Field(
        default=2,
        description="Width of corridors in maze-like structures",
    )


# -------------------------------------------------------------------------------------------
class MazeMapGen(BaseObstacleGenerator):
    """Generates grid maps with maze-like structures."""

    def __init__(self, params: Optional[MazeMapParams] = None):
        params = params or MazeMapParams()
        super().__init__(params)
        self.corridor_width = params.corridor_width

    # -----------------------------------------------------------------------------------
    def _add_obstacles(self, grid: ObstacleMap) -> ObstacleMap:
        """Add a maze-like segment to the grid."""
        return self._add_maze_segment(grid)

    # -----------------------------------------------------------------------------------
    def _add_maze_segment(self, grid: ObstacleMap) -> ObstacleMap:
        """Add a maze-like segment with corridors using vectorized operations."""
        h, w = grid.shape

        # Pick a starting point
        start_i, start_j = self._get_valid_position(grid, min_distance=2)

        # Choose a random direction and length
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction = random.choice(directions)
        length = random.randint(3, max(5, min(h, w) // 3))

        # Create the corridor path
        di, dj = direction
        path_is = [
            start_i + step * di
            for step in range(length)
            if 0 <= start_i + step * di < h and 0 <= start_j + step * dj < w
        ]
        path_js = [
            start_j + step * dj
            for step in range(length)
            if 0 <= start_i + step * di < h and 0 <= start_j + step * dj < w
        ]

        # If path is empty, return unchanged grid
        if not path_is:
            return grid

        # Calculate wall positions for left/right or top/bottom walls
        wall_positions = []
        for i, j in zip(path_is, path_js):
            for wall in range(1, self.corridor_width + 1):
                # Left/top wall
                wall_i = i + dj * wall
                wall_j = j + di * wall
                if 0 <= wall_i < h and 0 <= wall_j < w:
                    wall_positions.append((wall_i, wall_j))

                # Right/bottom wall
                wall_i = i - dj * wall
                wall_j = j - di * wall
                if 0 <= wall_i < h and 0 <= wall_j < w:
                    wall_positions.append((wall_i, wall_j))

        # Update grid with wall positions at once
        for wall_i, wall_j in wall_positions:
            grid[wall_i, wall_j] = 1

        # Add a turn at the end with probability 0.5
        if path_is and random.random() < 0.5:
            last_i, last_j = path_is[-1], path_js[-1]

            # Turn 90 degrees
            new_di, new_dj = dj, di  # perpendicular direction
            turn_length = random.randint(2, 4)

            # Calculate turn path
            turn_is = [
                last_i + step * new_di
                for step in range(turn_length)
                if 0 <= last_i + step * new_di < h and 0 <= last_j + step * new_dj < w
            ]
            turn_js = [
                last_j + step * new_dj
                for step in range(turn_length)
                if 0 <= last_i + step * new_di < h and 0 <= last_j + step * new_dj < w
            ]

            # Calculate wall positions for the turn
            turn_wall_positions = []
            for i, j in zip(turn_is, turn_js):
                for wall in range(1, self.corridor_width + 1):
                    # Left/top wall for turn
                    wall_i = i + new_dj * wall
                    wall_j = j + new_di * wall
                    if 0 <= wall_i < h and 0 <= wall_j < w:
                        turn_wall_positions.append((wall_i, wall_j))

                    # Right/bottom wall for turn
                    wall_i = i - new_dj * wall
                    wall_j = j - new_di * wall
                    if 0 <= wall_i < h and 0 <= wall_j < w:
                        turn_wall_positions.append((wall_i, wall_j))

            # Update grid with turn wall positions
            for wall_i, wall_j in turn_wall_positions:
                grid[wall_i, wall_j] = 1

        return grid


# -------------------------------------------------------------------------------------------
class PlotMapParams(BaseModel):
    """Parameters for plotting grid maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # -----------------------------------------------------------------------------------
    cmap: str = Field(
        default="binary",
        description="Colormap for the grid map",
    )
    # -----------------------------------------------------------------------------------
    cbar: bool = Field(
        default=False,
        description="Whether to show color bar",
    )
    # -----------------------------------------------------------------------------------
    annot: bool = Field(
        default=False,
        description="Whether to annotate cells",
    )
    # -----------------------------------------------------------------------------------
    title: Optional[str] = Field(
        default=None,
        description="Title for the plot",
    )

    # -----------------------------------------------------------------------------------
    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return params as a dictionary for seaborn heatmap."""
        return self.model_dump(exclude={"title"})


# -------------------------------------------------------------------------------------------
def plot(ax: Axes, grid: ObstacleMap, params: Optional[PlotMapParams] = None) -> None:
    """Adds a grid map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        grid: ObstacleMap containing obstacle map and goal position
        params: Grid map visualization params

    Returns:
        The matplotlib axis with the plot
    """
    params = PlotMapParams() if params is None else params

    # Extract obstacle map
    if isinstance(grid, Tensor):
        obstacles_data = grid.detach().cpu().numpy()
    else:
        obstacles_data = np.array(grid)

    # Plot the heatmap
    sns.heatmap(obstacles_data, ax=ax, **params.kwargs)

    # Set title if provided
    if params.title:
        ax.set_title(params.title)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add visual boundary to differentiate between subplots
    for spine in ax.spines.values():
        spine.set_visible(True)


# -------------------------------------------------------------------------------------------
# Example usage for BlockMapGen and MazeMapGen
if __name__ == "__main__":
    """
    This example demonstrates how to create and visualize obstacle maps using:
    1. BlockMapGen - Creates maps with rectangular block obstacles
    2. MazeMapGen - Creates maps with corridor-style maze obstacles
    3. The plotting utility to visualize the generated maps
    """
    import matplotlib.pyplot as plt

    # Create block map generator with custom parameters
    block_gen_params = BlockMapParams(
        grid_size=(20, 20),
        obstacle_density=0.25,
        min_obstacles=2,
        max_obstacles=6,
        min_block_size=(2, 2),
        max_block_size=(5, 5),
    )
    block_gen = BlockMapGen(block_gen_params)

    # Create maze map generator with custom parameters
    maze_gen_params = MazeMapParams(
        grid_size=(20, 20),
        obstacle_density=0.3,
        min_obstacles=2,
        max_obstacles=4,
        corridor_width=1,
    )
    maze_gen = MazeMapGen(maze_gen_params)

    # Generate samples from both generators
    num_samples = 3
    block_maps = [block_gen.__next__()[0] for _ in range(num_samples)]
    maze_maps = [maze_gen.__next__()[0] for _ in range(num_samples)]

    # Set up a figure to display the maps
    fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))
    plt.suptitle("Generated Obstacle Maps")

    # Plot block maps on the top row
    for i, block_map in enumerate(block_maps):
        plot_params = PlotMapParams(cmap="Blues", title=f"Block Map {i+1}")
        plot(axes[0, i], block_map, plot_params)

    # Plot maze maps on the bottom row
    for i, maze_map in enumerate(maze_maps):
        plot_params = PlotMapParams(cmap="Greens", title=f"Maze Map {i+1}")
        plot(axes[1, i], maze_map, plot_params)

    # Adjust layout and show
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.show()

    # Example of creating a DataModule with the generator
    print("\nCreating a DataModule with the BlockMapGen:")
    data_params = DataModuleParams(
        num_samples=1000,
        batch_size=16,
        val_split=0.2,
        test_split=0.1,
    )
    data_module = DataModule(generator=block_gen, params=data_params)

    # Get a batch from the train loader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Generated batch shape: {batch[0].shape}")
    print("DataModule setup successful!")
