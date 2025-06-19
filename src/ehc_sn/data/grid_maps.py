import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset

from ehc_sn.constants import Direction, GoalMap, GridMapSample, GridSize, ObstacleMap, Position


class GridMapParams(BaseModel):
    """Parameters for generating grid maps with obstacles and goal positions."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    grid_size: GridSize = Field(default=(16, 16), description="Size of the grid as (height, width)")
    obstacle_density: float = Field(default=0.2, description="Probability of a cell being an obstacle (0.0-1.0)")
    min_obstacles: int = Field(default=5, description="Minimum number of obstacles to generate")
    max_obstacles: int = Field(default=15, description="Maximum number of obstacles to generate")
    obstacle_size_range: Tuple[int, int] = Field(default=(1, 3), description="Range of obstacle sizes (min, max)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

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


class GridMapGenerator:
    """Generates grid maps with random obstacles and a goal position."""

    def __init__(self, parameters: Optional[GridMapParams] = None):
        parameters = parameters or GridMapParams()
        self.grid_size = parameters.grid_size
        self.obstacle_density = parameters.obstacle_density
        self.min_obstacles = parameters.min_obstacles
        self.max_obstacles = parameters.max_obstacles
        self.obstacle_size_range = parameters.obstacle_size_range

        if parameters.seed is not None:
            random.seed(parameters.seed)
            torch.random.seed(parameters.seed)

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

    def generate_map(self) -> GridMapSample:
        """
        Generate a grid map with obstacles and a goal position.

        Returns:
            A GridMapSample containing:
            - "dimensions": Size of the grid as (height, width)
            - "obstacles": ObstacleMap with obstacles (1) and free space (0)
            - "goal": GoalMap with one-hot encoding of the goal position
        """
        # Initialize empty grid
        grid = torch.zeros(self.grid_size, dtype=torch.uint8)

        # Add random obstacle clusters
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        for _ in range(num_obstacles):
            grid = self._add_random_obstacle_cluster(grid)

        # Ensure there's at least one free space for goal
        if torch.all(grid == 1):
            # If somehow all cells became obstacles, clear one
            i, j = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            grid[i, j] = 0

        # Set goal position in a free cell
        goal_pos: Position = self._get_valid_position(grid)

        # Create one-hot encoded goal map
        goal_map: GoalMap = torch.zeros_like(grid)
        goal_map[goal_pos] = 1

        # Convert to GridMapSample
        return GridMapSample(dimensions=self.grid_size, obstacles=grid, goal=goal_map)


class GridMapDataset(Dataset):
    """Dataset for grid maps with obstacles and goal positions."""

    def __init__(self, num_samples: int, parameters: Optional[GridMapParams] = None):
        """
        Initialize a dataset of grid maps with obstacles and goals.

        Args:
            num_samples: Number of grid map samples to generate
            parameters: Parameters for grid map generation
        """
        self.num_samples = num_samples
        self.parameters = parameters or GridMapParams()
        self.generator = GridMapGenerator(parameters=self.parameters)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> GridMapSample:
        return self.generator.generate_map()


class PlotGoalParams(BaseModel):
    """Parameters for plotting the goal marker."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    marker: str = Field(default="*", description="Marker for goal position")
    color: str = Field(default="red", description="Color for goal marker")
    s: int = Field(default=200, description="Size of goal marker")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for matplotlib scatter."""
        return dict(self.model_dump(exclude_unset=True))


class PlotMapParams(BaseModel):
    """Parameters for plotting grid maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    cmap: str = Field(default="binary", description="Colormap for the grid map")
    annot: bool = Field(default=False, description="Whether to annotate cells")
    title: Optional[str] = Field(default=None, description="Title for the plot")
    goal: Optional[PlotGoalParams] = Field(
        default_factory=PlotGoalParams, description="Parameters for plotting the goal marker"
    )

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for seaborn heatmap."""
        return dict(self.model_dump(exclude={"goal", "title"}))


def plot_goal(ax: Axes, goal: GoalMap, params: Optional[PlotGoalParams] = None) -> None:
    """Adds a goal marker to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        goal: GoalMap with one-hot encoding of the goal position
        params: Plot parameters for the goal marker
    """
    params = params or PlotGoalParams()

    if isinstance(goal, torch.Tensor):
        goal_data = goal.detach().cpu().numpy()
    else:
        goal_data = np.array(goal)

    # Find goal coordinates (y, x)
    goal_coords = np.argwhere(goal_data == 1)
    if len(goal_coords) > 0:
        goal_y, goal_x = goal_coords[0]
        # Use 0.5 offset to center the marker in the cell
        ax.scatter(goal_x + 0.5, goal_y + 0.5, **params.kwargs)


def plot_map(ax: Axes, sample: GridMapSample, params: Optional[PlotMapParams] = None) -> None:
    """Adds a grid map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        sample: GridMapSample containing obstacle map and goal position
        params: Grid map visualization parameters

    Returns:
        The matplotlib axis with the plot
    """
    params = params or PlotMapParams()

    # Extract obstacle map
    if isinstance(sample["obstacles"], torch.Tensor):
        obstacles_data = sample["obstacles"].detach().cpu().numpy()
    else:
        obstacles_data = np.array(sample["obstacles"])

    # Plot the heatmap
    sns.heatmap(obstacles_data, ax=ax, cbar=False, **params.kwargs)

    # Plot goal position if available
    if sample["goal"] is not None:
        plot_goal(ax, sample["goal"], params.goal)

    # Set title if provided
    if params.title:
        ax.set_title(params.title)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])


class PlotGoalParams(BaseModel):
    """Parameters for plotting grid maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    marker: str = Field(default="*", description="Marker for goal position")
    color: str = Field(default="red", description="Color for goal marker")
    s: int = Field(default=200, description="Size of goal marker")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for seaborn heatmap."""
        return self.model_dump(exclude_unset=True).items()


def plot_goal(ax: Axes, goal: GoalMap, params: Optional[PlotGoalParams] = None) -> None:
    """Adds a goal marker to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        goal: GoalMap with one-hot encoding of the goal position
        params: Plot parameters for the goal marker
    """
    params = params or PlotMapParams()

    if isinstance(goal, torch.Tensor):
        goal_data = goal.detach().cpu().numpy()
    else:
        goal_data = np.array(goal)

    # Find goal coordinates (y, x)
    goal_coords = np.argwhere(goal_data == 1)
    if len(goal_coords) > 0:
        goal_y, goal_x = goal_coords[0]
        ax.scatter(goal_x + 0.5, goal_y + 0.5, **params.kwargs)


# Example usage
if __name__ == "__main__":
    # Create a generator with custom parameters
    params = GridMapParams(grid_size=(10, 10), obstacle_density=0.3)
    dataset = GridMapDataset(num_samples=100, parameters=params)

    # Visualize the generated map
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_map(ax, dataset[0])
    plt.tight_layout()
    plt.show()
