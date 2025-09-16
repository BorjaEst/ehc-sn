# TODO: add docstring
"""2D obstacle map generation for spatial navigation and maze-like environments.

This module generates binary obstacle maps where 1 represents obstacles and 0 represents
free space. The maps create maze-like structures suitable for navigation testing.
"""

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ehc_sn.constants.types import ObstacleMap
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams


# -------------------------------------------------------------------------------------------
class DataParams(BaseModel):

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Data settings
    grid_size: Tuple[int, int] = Field(default=(16, 16), description="Size of the grid as (height, width)")
    obstacle_density: float = Field(default=0.3, ge=0.1, le=0.8, description="Fraction of grid to fill with obstacles")
    corridor_width: int = Field(default=1, ge=1, le=3, description="Width of corridors between obstacles")
    add_outer_walls: bool = Field(default=True, description="Add walls around the grid perimeter")

    seed: int = Field(default=42, ge=0, description="Random seed for reproducible generation")


# -------------------------------------------------------------------------------------------
class ObstacleMapDataset(Dataset):
    """Map-style Dataset that generates 2D obstacle maps on-the-fly.

    Creates maze-like binary obstacle maps where 1 represents obstacles and 0 represents
    free space. Follows PyTorch Lightning best practices with deterministic generation.
    """

    def __init__(self, n_samples: int, params: DataParams):
        """Initialize obstacle map dataset.

        Args:
            n_samples: Number of samples in the dataset
            params: Data generation parameters
        """
        self.n_samples = n_samples
        self.params = params
        self.height, self.width = params.grid_size

        # Precompute parameters for efficient generation
        self.wall_spacing = max(2, params.corridor_width + 1)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[ObstacleMap, ObstacleMap]:
        """Generate a single obstacle map sample deterministically.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input, target) obstacle maps for autoencoder training
        """
        # Create deterministic generator for this sample
        g = torch.Generator().manual_seed(self.params.seed + idx)

        # Generate base obstacle map
        obstacle_map = self._generate_maze_structure(g)

        # For autoencoder, target is the same as input
        return obstacle_map, obstacle_map

    def _generate_maze_structure(self, generator: torch.Generator) -> ObstacleMap:
        """Generate a maze-like obstacle structure.

        Args:
            generator: Random number generator for deterministic generation

        Returns:
            2D binary tensor with maze-like obstacle pattern
        """
        # Initialize empty grid (0 = free space, 1 = obstacle)
        grid = torch.zeros(self.height, self.width, dtype=torch.float32)

        # Add outer walls if requested
        if self.params.add_outer_walls:
            grid[0, :] = 1  # Top wall
            grid[-1, :] = 1  # Bottom wall
            grid[:, 0] = 1  # Left wall
            grid[:, -1] = 1  # Right wall

        # Create maze-like internal structure
        self._add_internal_walls(grid, generator)

        return grid

    def _add_internal_walls(self, grid: Tensor, generator: torch.Generator) -> None:
        """Add internal walls to create maze-like structure.

        Args:
            grid: Grid to modify (in-place)
            generator: Random number generator
        """
        # Create vertical corridors
        for x in range(self.wall_spacing, self.width - 1, self.wall_spacing):
            # Add vertical wall with random gaps
            wall_height = int(self.height * 0.7)  # Don't make walls too dense
            start_y = torch.randint(1, max(2, self.height - wall_height), (1,), generator=generator).item()

            for y in range(start_y, min(start_y + wall_height, self.height - 1)):
                grid[y, x] = 1

            # Add random gap in the wall
            gap_y = torch.randint(
                start_y, min(start_y + wall_height, self.height - 1), (1,), generator=generator
            ).item()
            gap_size = max(1, self.params.corridor_width)
            for gap in range(gap_size):
                if gap_y + gap < self.height - 1:
                    grid[gap_y + gap, x] = 0

        # Create horizontal corridors
        for y in range(self.wall_spacing, self.height - 1, self.wall_spacing):
            # Add horizontal wall with random gaps
            wall_width = int(self.width * 0.6)  # Slightly less dense than vertical
            start_x = torch.randint(1, max(2, self.width - wall_width), (1,), generator=generator).item()

            for x in range(start_x, min(start_x + wall_width, self.width - 1)):
                # Only add wall if it doesn't completely block a corridor
                if grid[y, x] == 0:  # Don't overwrite existing walls
                    grid[y, x] = 1

            # Add random gap in the wall
            gap_x = torch.randint(start_x, min(start_x + wall_width, self.width - 1), (1,), generator=generator).item()
            gap_size = max(1, self.params.corridor_width)
            for gap in range(gap_size):
                if gap_x + gap < self.width - 1:
                    grid[y, gap_x + gap] = 0

        # Add some random obstacles to increase complexity
        n_random_obstacles = int(self.height * self.width * (self.params.obstacle_density - 0.2))
        n_random_obstacles = max(0, n_random_obstacles)

        for _ in range(n_random_obstacles):
            y = torch.randint(1, self.height - 1, (1,), generator=generator).item()
            x = torch.randint(1, self.width - 1, (1,), generator=generator).item()

            # Only add obstacle if it doesn't create isolated regions
            if self._can_add_obstacle(grid, y, x):
                grid[y, x] = 1

    def _can_add_obstacle(self, grid: Tensor, y: int, x: int) -> bool:
        """Check if adding an obstacle at (y,x) would create isolated regions.

        Args:
            grid: Current grid state
            y, x: Position to check

        Returns:
            True if obstacle can be safely added
        """
        # Simple heuristic: don't add obstacle if it would surround a free space
        if grid[y, x] == 1:  # Already an obstacle
            return False

        # Count free neighbors
        free_neighbors = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if grid[ny, nx] == 0:
                    free_neighbors += 1

        # Allow obstacle if there are at least 2 free neighbors (maintains connectivity)
        return free_neighbors >= 2


# -------------------------------------------------------------------------------------------
class DataGenerator:

    def __init__(self, params: DataParams):
        self.params = params

    def __call__(self, n_samples: int) -> Dataset:
        return ObstacleMapDataset(n_samples, self.params)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test obstacle map generation
    print("=== Testing ObstacleMapDataset ===")

    # Create parameters and generator
    params = DataParams(grid_size=(20, 20), obstacle_density=0.3, corridor_width=1, add_outer_walls=True, seed=42)
    generator = DataGenerator(params)

    # Create dataset
    dataset = generator(n_samples=100)
    print(f"Dataset length: {len(dataset)}")

    # Test individual sample generation
    map1, target1 = dataset[0]
    map2, target2 = dataset[0]  # Should be identical (deterministic)
    print(f"Sample shapes: map={map1.shape}, target={target1.shape}")
    print(f"Deterministic: {torch.allclose(map1, map2)}")
    print(f"Obstacle density: {torch.mean(map1).item():.3f}")

    # Create DataLoader and test batching
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"Number of batches: {len(dataloader)}")

    # Get first batch and visualize
    batch_maps, batch_targets = next(iter(dataloader))
    print(f"Batch shapes: maps={batch_maps.shape}, targets={batch_targets.shape}")

    # Show a sample obstacle map
    sample_map = batch_maps[0].numpy()
    print(f"Sample obstacle map (1=obstacle, 0=free):")
    for row in sample_map:
        print("".join(["█" if cell > 0.5 else "·" for cell in row]))
