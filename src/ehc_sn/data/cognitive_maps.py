from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, NonNegativeInt, field_validator

from ehc_sn.constants import CognitiveMap, ObstacleMap, ValueMap
from ehc_sn.data import _base, obstacle_maps
from ehc_sn.data._base import DataModule, DataModuleParams


# -------------------------------------------------------------------------------------------
class BaseCMParameters(_base.GeneratorParams):
    """Parameters for generating maze-like cognitive maps with probability distributions."""

    # -----------------------------------------------------------------------------------
    diffusion_iterations: NonNegativeInt = Field(
        default=3,
        description="Number of diffusion iterations to apply to walls",
    )
    # -----------------------------------------------------------------------------------
    diffusion_strength: float = Field(
        default=0.2,
        description="Strength of the diffusion effect (0.0-1.0)",
    )
    # -----------------------------------------------------------------------------------
    noise_level: float = Field(
        default=0.05,
        description="Base noise level throughout the map (0.0-1.0)",
    )

    # -----------------------------------------------------------------------------------
    @field_validator("diffusion_strength", "noise_level")
    @classmethod
    def between_zero_and_one(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("level must be between 0.0 and 1.0")
        return v


# -------------------------------------------------------------------------------------------
class BaseGenerator(_base.Generator, ABC):
    """Generates cognitive maps with maze-like structures and diffused walls."""

    # -----------------------------------------------------------------------------------
    def __init__(self, params: Optional[BaseCMParameters] = None):
        params = params or BaseCMParameters()
        super().__init__(params=params)
        self.diffusion_iterations = params.diffusion_iterations
        self.diffusion_strength = params.diffusion_strength
        self.noise_level = params.noise_level

    # -----------------------------------------------------------------------------------
    def _apply_diffusion(self, grid: ValueMap) -> ValueMap:
        """
        Apply diffusion to the grid to smooth obstacle boundaries.

        Args:
            grid: ValueMap tensor

        Returns:
            Diffused ValueMap tensor
        """
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.float32)

        # Reshape grid to (1, 1, H, W) for convolution
        diffused_grid = grid.unsqueeze(0).unsqueeze(0)
        # Define convolution kernel to consider the four neighbors
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape: (1,1,3,3)

        for _ in range(self.diffusion_iterations):
            # Pad grid to preserve size
            padded_grid = F.pad(diffused_grid, (1, 1, 1, 1), mode="replicate")
            # Perform convolution to sum neighbor values
            conv_out = F.conv2d(padded_grid, kernel)
            # Compute the average of the neighbors
            neighbor_mean = conv_out / 4.0
            # Update grid with diffusion effect
            intermediate = diffused_grid + (neighbor_mean - diffused_grid) * self.diffusion_strength
            diffused_grid = intermediate.clamp(0.0, 1.0)

        # Remove extra dimensions
        return diffused_grid.squeeze(0).squeeze(0)

    # -----------------------------------------------------------------------------------
    def _add_noise(self, grid: ValueMap) -> ValueMap:
        """Add random noise to the cognitive map."""
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.float32)

        # Generate random noise
        noise = torch.randn_like(grid) * self.noise_level

        # Add noise and clamp values between 0 and 1
        noisy_grid = (grid + noise).clamp(0.0, 1.0)

        return noisy_grid

    # -----------------------------------------------------------------------------------
    def _preprocess_grid(self, obstacle_layer: ObstacleMap) -> ValueMap:
        """Convert binary obstacle map to probability distribution."""
        return 1.0 - obstacle_layer

    # -----------------------------------------------------------------------------------
    def _process_obstacle(self, obstacle_layer: ObstacleMap) -> ValueMap:
        """Process the raw obstacle map into a cognitive map layer.

        Args:
            obstacle_layer: Raw binary obstacle map

        Returns:
            Processed obstacle layer with diffusion and noise
        """
        processed = self._preprocess_grid(obstacle_layer)  # Convert to probability values
        processed = self._apply_diffusion(processed)  # Apply diffusion
        processed = self._add_noise(processed)  # Add noise
        return processed


class BlockMapParams(BaseCMParameters, obstacle_maps.BlockMapParams):
    """Parameters for generating block-based cognitive maps with diffused walls."""


class BlockMapGenerator(BaseGenerator):
    """Generates block-based cognitive maps with diffused walls."""

    # -----------------------------------------------------------------------------------
    def __init__(self, params: Optional[BlockMapParams] = None):
        params = params or BlockMapParams()
        super().__init__(params=params)
        self.block_generator = obstacle_maps.BlockMapGen(params)
        self.grid_size = params.grid_size
        self.obstacle_density = params.obstacle_density
        self.min_block_size = params.min_block_size
        self.max_block_size = params.max_block_size
        self.min_obstacles = params.min_obstacles
        self.max_obstacles = params.max_obstacles

    # -----------------------------------------------------------------------------------
    def __next__(self) -> Tuple[CognitiveMap]:
        """
        Generate a block-based cognitive map with diffused walls.

        Returns:
            A cognitive map tensor with probability values (0.0 to 1.0).
        """
        # Number of channels: obstacles, speed, trajectory probability
        num_channels = 1
        cognitive_map = torch.zeros([num_channels, *self.grid_size], dtype=torch.float32)

        # Generate base block structure
        (obstacle_layer,) = self.block_generator.__next__()

        # Process obstacle layer
        obstacle_layer = self._process_obstacle(obstacle_layer)
        # TODO: Create speed map, create trajectory map

        # Assign channels to cognitive map
        cognitive_map[0, :, :] = obstacle_layer  # First channel: obstacle layer
        return (cognitive_map,)


class MazeMapParams(BaseCMParameters, obstacle_maps.MazeMapParams):
    """Parameters for generating maze-based cognitive maps with diffused walls."""


class MazeMapGenerator(BaseGenerator):
    """Generates maze-based cognitive maps with diffused walls."""

    # -----------------------------------------------------------------------------------
    def __init__(self, params: Optional[MazeMapParams] = None):
        params = params or MazeMapParams()
        super().__init__(params=params)
        self.maze_generator = obstacle_maps.MazeMapGen(params)
        self.grid_size = params.grid_size
        self.obstacle_density = params.obstacle_density
        self.corridor_width = params.corridor_width
        self.min_obstacles = params.min_obstacles
        self.max_obstacles = params.max_obstacles

    # -----------------------------------------------------------------------------------
    def __next__(self) -> Tuple[CognitiveMap]:
        """
        Generate a maze-like cognitive map with diffused walls.

        Returns:
            A cognitive map tensor with probability values (0.0 to 1.0).
        """
        # Number of channels: obstacles, speed, trajectory probability, head direction
        num_channels = 1
        cognitive_map = torch.zeros([num_channels, *self.grid_size], dtype=torch.float32)

        # Generate base maze structure
        (obstacle_layer,) = self.maze_generator.__next__()

        # Process obstacle layer
        obstacle_layer = self._process_obstacle(obstacle_layer)
        # TODO: Create speed map, create trajectory map

        # Assign channels to cognitive map
        cognitive_map[0, :, :] = obstacle_layer  # First channel: obstacle layer
        return (cognitive_map,)


# -------------------------------------------------------------------------------------------
class PlotMapParams(obstacle_maps.PlotMapParams):
    """Parameters for plotting cognitive maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # -----------------------------------------------------------------------------------
    cmap: str = Field(
        default="gray",
        description="Colormap for the cognitive map",
    )
    # -----------------------------------------------------------------------------------
    cbar: bool = Field(
        default=True,
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
    vmin: float = Field(
        default=0.0,
        description="Minimum value for colormap scaling",
    )
    # -----------------------------------------------------------------------------------
    vmax: float = Field(
        default=1.0,
        description="Maximum value for colormap scaling",
    )

    # -----------------------------------------------------------------------------------
    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return params as a dictionary for seaborn heatmap."""
        return dict(self.model_dump(exclude={"title"}))


# -------------------------------------------------------------------------------------------
def plot(axs: List[Axes], grid: CognitiveMap, params: Optional[PlotMapParams] = None) -> None:
    """Adds a cognitive map visualization to the given axis.

    Args:
        axs: Matplotlib list of axis to plot on
        grid: CognitiveMap containing probability distribution
        params: Cognitive map visualization params
    """
    params = PlotMapParams() if params is None else params

    # Get the number of channels in the cognitive map
    num_channels = grid.shape[0]

    # Make sure we have enough axes
    if len(axs) < num_channels:
        raise ValueError(f"Need {num_channels} axes to plot all channels of the cognitive map")

    # Plot each channel
    for i in range(num_channels):
        obstacle_maps.plot(axs[i], grid[i, :, :], params)


# -------------------------------------------------------------------------------------------
# Example usage of the generators and plotting function
if __name__ == "__main__":
    """
    This example demonstrates:
    1. Creating both block-based and maze-based cognitive maps
    2. Visualizing the different channels of cognitive maps
    3. Setting up DataModules for training with these cognitive maps
    """
    import matplotlib.pyplot as plt

    # Create example parameters for both map types
    block_params = BlockMapParams(
        grid_size=(16, 16),
        obstacle_density=0.25,
        diffusion_iterations=4,
        diffusion_strength=0.3,
        noise_level=0.05,
        min_obstacles=2,
        max_obstacles=5,
    )

    maze_params = MazeMapParams(
        grid_size=(16, 16),
        obstacle_density=0.3,
        diffusion_iterations=3,
        diffusion_strength=0.2,
        noise_level=0.02,
        corridor_width=1,
        min_obstacles=2,
        max_obstacles=4,
    )

    # Create both types of generators
    block_generator = BlockMapGenerator(block_params)
    maze_generator = MazeMapGenerator(maze_params)

    # Generate a sample from each generator
    block_map = block_generator.__next__()[0]  # Returns (cognitive_map,)
    maze_map = maze_generator.__next__()[0]  # Returns (cognitive_map,)

    # Print shape information
    print(f"Block-based cognitive map shape: {block_map.shape}")
    print(f"Maze-based cognitive map shape: {maze_map.shape}")

    # Create a figure to display the cognitive maps
    num_channels = max(block_map.shape[0], maze_map.shape[0])
    fig, axs = plt.subplots(2, num_channels, figsize=(4 * num_channels, 8))

    # Plot both types of cognitive maps
    plt.suptitle("Cognitive Maps: Block vs Maze", fontsize=16)

    # Plot block-based cognitive map on top row
    axs_block_row = axs[0]
    if not isinstance(axs_block_row, list) and num_channels == 1:
        axs_block_row = [axs_block_row]  # Handle single-channel case
    plot(axs_block_row, block_map, PlotMapParams(title="Block-Based Cognitive Map"))

    # Plot maze-based cognitive map on bottom row
    axs_maze_row = axs[1]
    if not isinstance(axs_maze_row, list) and num_channels == 1:
        axs_maze_row = [axs_maze_row]  # Handle single-channel case
    plot(axs_maze_row, maze_map, PlotMapParams(title="Maze-Based Cognitive Map"))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make space for the title

    # Show the figure
    plt.show()

    # Example of creating a DataModule with a cognitive map generator
    print("\nCreating a DataModule with the MazeMapGenerator:")
    data_params = DataModuleParams(
        num_samples=1000,
        batch_size=16,
        val_split=0.2,
        test_split=0.1,
    )
    data_module = DataModule(generator=maze_generator, params=data_params)

    # Set up the data module and create loaders
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Generated batch shape: {batch[0].shape}")
    print("DataModule setup successful!")
