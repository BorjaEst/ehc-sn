from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, field_validator

from ehc_sn.constants import CognitiveMap, ObstacleMap, ValueMap
from ehc_sn.data import grid_maps
from ehc_sn.data._base import DataModule, DataModuleParams


class GeneratorParams(grid_maps.GeneratorParams):
    """Parameters for generating maze-like cognitive maps with probability distributions."""

    diffusion_iterations: int = Field(default=3, description="Number of diffusion iterations to apply to walls")
    diffusion_strength: float = Field(default=0.2, description="Strength of the diffusion effect (0.0-1.0)")
    noise_level: float = Field(default=0.05, description="Base noise level throughout the map (0.0-1.0)")

    @field_validator("diffusion_iterations")
    @classmethod
    def validate_diffusion_iterations(cls, v: int) -> int:
        if v < 0:
            raise ValueError("diffusion_iterations must be non-negative")
        return v

    @field_validator("diffusion_strength")
    @classmethod
    def validate_diffusion_strength(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("diffusion_strength must be between 0.0 and 1.0")
        return v

    @field_validator("noise_level")
    @classmethod
    def validate_noise_level(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0")
        return v


class Generator(grid_maps.Generator):
    """Generates cognitive maps with maze-like structures and diffused walls."""

    def __init__(self, params: Optional[GeneratorParams] = None):
        params = params or GeneratorParams()
        super().__init__(params=params)
        self.diffusion_iterations = params.diffusion_iterations
        self.diffusion_strength = params.diffusion_strength
        self.noise_level = params.noise_level

    def _apply_diffusion(self, grid: ValueMap) -> ValueMap:
        import torch.nn.functional as F

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

    def _add_noise(self, grid: ValueMap) -> ValueMap:
        """Add random noise to the cognitive map."""
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.float32)

        # Generate random noise
        noise = torch.randn_like(grid) * self.noise_level

        # Add noise and clamp values between 0 and 1
        noisy_grid = (grid + noise).clamp(0.0, 1.0)

        return noisy_grid

    def _preprocess_grid(self, obstacle_map: ObstacleMap) -> ValueMap:
        """Convert binary obstacle map to probability distribution."""
        return 1.0 - obstacle_map

    def __next__(self) -> Tuple[CognitiveMap]:
        """
        Generate a maze-like cognitive map with diffused walls.

        Returns:
            A cognitive map tensor with probability values (0.0 to 1.0).
        """
        cognitive_map = torch.zeros([1, *self.grid_size], dtype=torch.float32)

        # Generate base maze structure using the grid_maps generator
        (obstacle_layer, *_) = super().__next__()  # Take only the input grid
        obstacle_layer = self._preprocess_grid(obstacle_layer)  # Convert to probability values
        obstacle_layer = self._apply_diffusion(obstacle_layer)  # Apply diffusion
        obstacle_layer = self._add_noise(obstacle_layer)  # Add noise

        # TODO: Add more channels, for speed, trajectories, head direction, etc.
        pass

        # Reshape to cognitive map format and return
        cognitive_map[0, :, :] = obstacle_layer  # First channel is the obstacle layer
        return (cognitive_map,)


class PlotMapParams(BaseModel):
    """Parameters for plotting cognitive maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    cmap: str = Field(default="gray", description="Colormap for the cognitive map")
    annot: bool = Field(default=False, description="Whether to annotate cells")
    title: Optional[str] = Field(default=None, description="Title for the plot")
    vmin: float = Field(default=0.0, description="Minimum value for colormap scaling")
    vmax: float = Field(default=1.0, description="Maximum value for colormap scaling")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return params as a dictionary for seaborn heatmap."""
        return dict(self.model_dump(exclude={"title"}))


def plot(axs: List[Axes], grid: CognitiveMap, params: Optional[PlotMapParams] = None) -> None:
    """Adds a cognitive map visualization to the given axis.

    Args:
        axs: Matplotlib list of axis to plot on
        grid: CognitiveMap containing probability distribution
        params: Cognitive map visualization params
    """
    params = params or PlotMapParams()

    # Plot in the first available axis the obstacle map
    obstacles_params = grid_maps.PlotMapParams(cbar=True)
    grid_maps.plot(axs[0], grid[0, :, :], obstacles_params)

    # TODO: Add more channels, for speed, trajectories, head direction, etc.
    pass


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

    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for axs_, sample in zip(axs, data_module.train_dataset):
        plot(axs_, sample[0])  # sample[0] is the cognitive map

    plt.show()
