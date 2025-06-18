import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset


class GridMapParameters(BaseModel):
    """Parameters for generating grid maps with obstacles and goal positions."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    grid_size: Tuple[int, int] = Field(default=(16, 16), description="Size of the grid as (height, width)")
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

    def __init__(self, parameters: Optional[GridMapParameters] = None):
        parameters = parameters or GridMapParameters()
        self.grid_size = parameters.grid_size
        self.obstacle_density = parameters.obstacle_density
        self.min_obstacles = parameters.min_obstacles
        self.max_obstacles = parameters.max_obstacles
        self.obstacle_size_range = parameters.obstacle_size_range

        if parameters.seed is not None:
            random.seed(parameters.seed)
            np.random.seed(parameters.seed)

    def _is_valid_position(self, grid: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)."""
        h, w = grid.shape
        i, j = pos
        return 0 <= i < h and 0 <= j < w and grid[i, j] == 0

    def _get_valid_position(self, grid: np.ndarray) -> Tuple[int, int]:
        """Get a random valid position (not occupied by an obstacle)."""
        h, w = grid.shape
        while True:
            i, j = random.randint(0, h - 1), random.randint(0, w - 1)
            if grid[i, j] == 0:
                return (i, j)

    def _add_random_obstacle_cluster(self, grid: np.ndarray) -> np.ndarray:
        """Add a cluster of obstacles with random size and shape."""
        h, w = grid.shape
        size = random.randint(self.obstacle_size_range[0], self.obstacle_size_range[1])

        # Start position for obstacle cluster
        start_i, start_j = random.randint(0, h - 1), random.randint(0, w - 1)
        grid[start_i, start_j] = 1  # Mark as obstacle

        # Grow obstacle cluster
        positions = [(start_i, start_j)]
        for _ in range(size - 1):
            if not positions:
                break

            # Pick a random position from existing obstacle positions
            pos_idx = random.randint(0, len(positions) - 1)
            i, j = positions[pos_idx]

            # Try to add an adjacent cell
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for di, dj in directions:
                new_i, new_j = i + di, j + dj
                if 0 <= new_i < h and 0 <= new_j < w and grid[new_i, new_j] == 0:
                    grid[new_i, new_j] = 1
                    positions.append((new_i, new_j))
                    break

        return grid

    def generate_map(self) -> Dict[str, Any]:
        """
        Generate a grid map with obstacles and a goal position.

        Returns:
            Dict containing:
                'map': Binary grid with 1s indicating obstacles, 0s indicating free space
                'goal': One-hot encoded goal position with 1 at the goal location
                'goal_position': Tuple of (row, col) coordinates of the goal
        """
        # Initialize empty grid
        grid = np.zeros(self.grid_size, dtype=np.uint8)

        # Add random obstacle clusters
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        for _ in range(num_obstacles):
            grid = self._add_random_obstacle_cluster(grid)

        # Ensure there's at least one free space for goal
        if np.all(grid == 1):
            # If somehow all cells became obstacles, clear one
            i, j = random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1)
            grid[i, j] = 0

        # Set goal position in a free cell
        goal_pos = self._get_valid_position(grid)

        # Create one-hot encoded goal map
        goal_map = np.zeros_like(grid)
        goal_map[goal_pos] = 1

        return {"map": grid, "goal": goal_map, "goal_position": goal_pos}


class GridMapDataset(Dataset):
    """Dataset for grid maps with obstacles and goal positions."""

    def __init__(self, num_samples: int, parameters: Optional[GridMapParameters] = None):
        """
        Initialize a dataset of grid maps with obstacles and goals.

        Args:
            num_samples: Number of grid map samples to generate
            parameters: Parameters for grid map generation
        """
        self.num_samples = num_samples
        self.parameters = parameters or GridMapParameters()
        self.generator = GridMapGenerator(parameters=self.parameters)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.generator.generate_map()

        # Convert numpy arrays to tensors
        return {
            "map": torch.tensor(sample["map"], dtype=torch.float32),
            "goal": torch.tensor(sample["goal"], dtype=torch.float32),
            "goal_position": torch.tensor(sample["goal_position"], dtype=torch.long),
        }


# Example usage
if __name__ == "__main__":
    # Create a generator with custom parameters
    params = GridMapParameters(grid_size=(10, 10), obstacle_density=0.3)

    generator = GridMapGenerator(params)
    grid_map = generator.generate_map()

    # Visualize the generated map
    print("Grid Map (0: free space, 1: obstacle):")
    print(grid_map["map"])

    print("\nGoal Position (One-hot encoded):")
    print(grid_map["goal"])

    print("\nGoal Coordinates:", grid_map["goal_position"])

    # Create a dataset and get a sample
    dataset = GridMapDataset(num_samples=100, parameters=params)
    sample = dataset[0]
    print("\nDataset Sample (tensors):")
    for key, value in sample.items():
        print(f"{key}: {value.shape}, {value.dtype}")
