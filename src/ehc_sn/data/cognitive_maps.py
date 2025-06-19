from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel, Field, field_validator

import ehc_sn.data.grid_maps as base
from ehc_sn.constants import CognitiveMap, ObstacleMap
from ehc_sn.data.base import DataModule, DataModuleParams


class GeneratorParams(base.GeneratorParams):
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


class Generator(base.Generator):
    """Generates cognitive maps with maze-like structures and diffused walls."""

    def __init__(self, params: Optional[GeneratorParams] = None):
        params = params or GeneratorParams()
        super().__init__(params=params)
        self.diffusion_iterations = params.diffusion_iterations
        self.diffusion_strength = params.diffusion_strength
        self.noise_level = params.noise_level

    def _apply_diffusion(self, grid: CognitiveMap) -> CognitiveMap:
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

    def _add_noise(self, grid: CognitiveMap) -> CognitiveMap:
        """Add random noise to the cognitive map."""
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.float32)

        # Generate random noise
        noise = torch.randn_like(grid) * self.noise_level

        # Add noise and clamp values between 0 and 1
        noisy_grid = (grid + noise).clamp(0.0, 1.0)

        return noisy_grid

    def _preprocess_grid(self, obstacle_map: ObstacleMap) -> CognitiveMap:
        """Convert binary obstacle map to probability distribution."""
        return 1.0 - obstacle_map

    def __next__(self) -> CognitiveMap:
        """
        Generate a maze-like cognitive map with diffused walls.

        Returns:
            A cognitive map tensor with probability values (0.0 to 1.0).
        """
        # Generate base maze structure using the grid_maps generator
        obstacle_map = super().__next__()

        # Convert binary obstacle map to probability values
        cognitive_map = self._preprocess_grid(obstacle_map)

        # Apply diffusion to create blurred boundaries
        cognitive_map = self._apply_diffusion(cognitive_map)

        # Add noise for more realistic cognitive representation
        cognitive_map = self._add_noise(cognitive_map)

        return cognitive_map


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


def plot(ax: Axes, grid: CognitiveMap, params: Optional[PlotMapParams] = None) -> None:
    """Adds a cognitive map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        grid: CognitiveMap containing probability distribution
        params: Cognitive map visualization params
    """
    params = params or PlotMapParams()

    # Extract cognitive map data
    if isinstance(grid, torch.Tensor):
        cog_data = grid.detach().cpu().numpy()
    else:
        cog_data = np.array(grid)

    # Plot the heatmap
    sns.heatmap(cog_data, ax=ax, cbar=True, **params.kwargs)

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
