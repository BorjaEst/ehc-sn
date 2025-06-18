from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator

from ehc_sn.constants import (
    CognitiveMap,
    Direction,
    DistanceMap,
    Embedding,
    GoalMap,
    GridMap,
    GridMapSample,
    GridSize,
    NeuralActivity,
    ObstacleMap,
    Position,
    RegionalState,
    ValueMap,
)


class CognitiveMapParameters(BaseModel):
    """Parameters for cognitive map visualizations.

    Parameter names match those expected by seaborn.heatmap and matplotlib functions
    to enable direct unpacking into visualization functions.
    """

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # Figure parameters
    figsize: Tuple[float, float] = Field(default=(12, 10), description="Figure size as (width, height)")
    dpi: int = Field(default=100, description="DPI for figure output")

    # Heatmap parameters
    cmap: str = Field(default="viridis", description="Colormap for heatmaps")
    alpha: float = Field(default=0.8, description="Alpha transparency for grid plots")
    linewidths: float = Field(default=0.5, description="Line width for grid outlines")
    cbar: bool = Field(default=True, description="Whether to show colorbar")
    annot: bool = Field(default=False, description="Whether to show annotations")
    fmt: str = Field(default=".2f", description="Format string for annotations")

    # Text parameters
    fontsize: int = Field(default=12, description="Base font size")
    title_fontsize: int = Field(default=14, description="Title font size")
    suptitle_fontsize: int = Field(default=16, description="Super title font size")

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v


class CognitiveMapVisualizer:
    """Visualizes cognitive maps and their representations in the EHC model."""

    def __init__(self, parameters: Optional[CognitiveMapParameters] = None):
        """Initialize visualizer with parameters."""
        self.params = parameters or CognitiveMapParameters()

        # Set up the style for plots
        sns.set_theme(style="white")
        self._setup_custom_colors()

    def _setup_custom_colors(self) -> None:
        """Set up custom colormaps for different visualization modes."""
        # Create a custom diverging colormap for error maps
        self.error_cmap = LinearSegmentedColormap.from_list(
            "error_map", [(0, "darkblue"), (0.5, "white"), (1, "darkred")]
        )

        # Create a custom colormap for obstacle maps
        self.obstacle_cmap = LinearSegmentedColormap.from_list("obstacle_map", [(0, "white"), (1, "dimgrey")])

        # Create a custom colormap for goal positions
        self.goal_cmap = LinearSegmentedColormap.from_list("goal_map", [(0, "white"), (1, "green")])

        # Create a custom colormap for agent positions
        self.agent_cmap = LinearSegmentedColormap.from_list("agent_map", [(0, "white"), (1, "blue")])

    def _convert_to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert torch tensors to numpy arrays if needed."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        """Get parameters, overriding defaults with provided kwargs."""
        # Convert Pydantic model to dict and override with kwargs
        params_dict = {k: v for k, v in self.params.model_dump().items()}
        params_dict.update(kwargs)
        return params_dict

    def plot_grid_map(
        self, data: GridMap, title: str = "Grid Map", ax: Optional[plt.Axes] = None, **kwargs
    ) -> plt.Axes:
        """Plot a single grid map."""
        data = self._convert_to_numpy(data)

        # Create figure and axes if not provided
        if ax is None:
            fig_params = {k: v for k, v in self.params.model_dump().items() if k in ["figsize", "dpi"]}
            _, ax = plt.subplots(**fig_params)

        # Combine defaults with overrides for heatmap
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        # Plot the heatmap with all parameters forwarded
        sns.heatmap(data, ax=ax, **plot_kwargs)

        # Title setting is separate as it's for axes, not heatmap
        ax.set_title(title, fontsize=plot_kwargs.get("title_fontsize"))
        return ax

    def plot_environment(
        self,
        obstacle_map: ObstacleMap,
        goal_map: Optional[GoalMap] = None,
        agent_position: Optional[Position] = None,
        title: str = "Environment",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot an environment with obstacles, goal and agent position."""
        obstacle_map = self._convert_to_numpy(obstacle_map)
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        # Create figure if needed
        if ax is None:
            fig_params = {k: v for k, v in plot_kwargs.items() if k in ["figsize", "dpi"]}
            _, ax = plt.subplots(**fig_params)

        # Use a simplified parameter set for the heatmap
        heatmap_kwargs = {k: v for k, v in plot_kwargs.items() if k in ["alpha", "linewidths"]}

        # Plot obstacles with custom colormap
        sns.heatmap(obstacle_map, cmap=self.obstacle_cmap, cbar=False, ax=ax, **heatmap_kwargs)

        # Plot goal if provided
        if goal_map is not None:
            goal_map = self._convert_to_numpy(goal_map)
            goal_masked = np.ma.masked_where(goal_map == 0, goal_map)
            ax.imshow(goal_masked, cmap=self.goal_cmap, interpolation="none", alpha=0.7)

        # Add agent position if provided
        if agent_position is not None:
            ax.plot(
                agent_position[1] + 0.5,
                agent_position[0] + 0.5,
                "o",
                markersize=kwargs.get("markersize", 10),
                color=kwargs.get("agent_color", "blue"),
                markeredgecolor=kwargs.get("agent_edge_color", "black"),
            )

        ax.set_title(title, fontsize=plot_kwargs.get("title_fontsize"))

        # Grid alignment
        ax.set_yticks(np.arange(0, obstacle_map.shape[0], 1))
        ax.set_xticks(np.arange(0, obstacle_map.shape[1], 1))
        ax.grid(True, color="black", linewidth=plot_kwargs.get("linewidths"), alpha=0.3)

        return ax

    def plot_cognitive_map_comparison(
        self,
        original_map: CognitiveMap,
        reconstructed_map: CognitiveMap,
        error_map: Optional[CognitiveMap] = None,
        title: str = "Cognitive Map Comparison",
        include_error: bool = True,
        **kwargs,
    ) -> Figure:
        """Compare original and reconstructed cognitive maps side by side."""
        original_map = self._convert_to_numpy(original_map)
        reconstructed_map = self._convert_to_numpy(reconstructed_map)

        # Combine parameters
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        # Create figure layout based on whether to include error map
        fig_params = {k: v for k, v in plot_kwargs.items() if k in ["figsize", "dpi"]}

        if include_error:
            if error_map is None:
                error_map = original_map - reconstructed_map
            else:
                error_map = self._convert_to_numpy(error_map)

            fig, axes = plt.subplots(1, 3, **fig_params)
            ax_titles = ["Original", "Reconstructed", "Error"]
            maps = [original_map, reconstructed_map, error_map]
            cmaps = [plot_kwargs.get("cmap"), plot_kwargs.get("cmap"), self.error_cmap]
        else:
            fig, axes = plt.subplots(1, 2, **fig_params)
            ax_titles = ["Original", "Reconstructed"]
            maps = [original_map, reconstructed_map]
            cmaps = [plot_kwargs.get("cmap"), plot_kwargs.get("cmap")]

        # Extract just the heatmap parameters
        heatmap_kwargs = {k: v for k, v in plot_kwargs.items() if k in ["alpha", "linewidths", "cbar", "annot", "fmt"]}

        for i, (map_data, ax_title, cmap) in enumerate(zip(maps, ax_titles, cmaps)):
            sns.heatmap(map_data, cmap=cmap, ax=axes[i], **heatmap_kwargs)
            axes[i].set_title(ax_title, fontsize=plot_kwargs.get("title_fontsize"))

        plt.suptitle(title, fontsize=plot_kwargs.get("suptitle_fontsize"))
        plt.tight_layout()
        return fig

    def plot_cognitive_map_sequence(
        self,
        map_sequence: List[CognitiveMap],
        titles: Optional[List[str]] = None,
        title: str = "Cognitive Map Sequence",
        n_cols: int = 4,
        **kwargs,
    ) -> Figure:
        """Plot a sequence of cognitive maps in a grid layout."""
        n_maps = len(map_sequence)
        n_rows = (n_maps + n_cols - 1) // n_cols  # Ceiling division

        # Combine parameters
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        if titles is None:
            titles = [f"Map {i+1}" for i in range(n_maps)]

        # Scale figure size based on number of rows
        figsize = plot_kwargs.get("figsize")
        scaled_figsize = (figsize[0], figsize[1] * (n_rows / 2))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=scaled_figsize, dpi=plot_kwargs.get("dpi"))
        axes = np.array(axes).flatten()  # Ensure axes is a flattened array

        # Extract just the heatmap parameters
        heatmap_kwargs = {
            k: v for k, v in plot_kwargs.items() if k in ["cmap", "alpha", "linewidths", "cbar", "annot", "fmt"]
        }

        for i, (map_data, subtitle) in enumerate(zip(map_sequence, titles)):
            map_data = self._convert_to_numpy(map_data)
            sns.heatmap(map_data, ax=axes[i], **heatmap_kwargs)
            axes[i].set_title(subtitle, fontsize=plot_kwargs.get("title_fontsize"))

        # Hide unused axes
        for i in range(n_maps, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=plot_kwargs.get("suptitle_fontsize"))
        plt.tight_layout()
        return fig

    def plot_embedding_activations(
        self, embeddings: Embedding, title: str = "Neural Activations", sort: bool = True, **kwargs
    ) -> Figure:
        """Visualize embedding activations in a heatmap."""
        embeddings = self._convert_to_numpy(embeddings)

        # Combine parameters
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        # Handle 1D embeddings
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Create figure
        fig_params = {k: v for k, v in plot_kwargs.items() if k in ["figsize", "dpi"]}
        fig, ax = plt.subplots(**fig_params)

        # Sort neurons by activation if requested
        if sort and embeddings.shape[0] > 1:
            neuron_means = embeddings.mean(axis=0)
            sorted_indices = np.argsort(neuron_means)[::-1]  # Descending order
            embeddings = embeddings[:, sorted_indices]

        # Extract heatmap parameters
        heatmap_kwargs = {k: v for k, v in plot_kwargs.items() if k in ["cmap", "alpha", "cbar", "annot", "fmt"]}
        # Thinner lines for neuron grid
        heatmap_kwargs["linewidths"] = plot_kwargs.get("linewidths", 0.5) / 2

        sns.heatmap(embeddings, ax=ax, **heatmap_kwargs)

        ax.set_title(title, fontsize=plot_kwargs.get("title_fontsize"))
        ax.set_xlabel("Neuron Index", fontsize=plot_kwargs.get("fontsize"))
        ax.set_ylabel("Sample Index", fontsize=plot_kwargs.get("fontsize"))

        plt.tight_layout()
        return fig

    def plot_model_states(
        self,
        states: Dict[str, Union[RegionalState, NeuralActivity]],
        regions_order: Optional[List[str]] = None,
        title: str = "EHC Circuit States",
        **kwargs,
    ) -> Figure:
        """Visualize the activation states across different regions of the EHC model."""
        # If no specific order provided, use alphabetical sorting
        if regions_order is None:
            regions_order = sorted(states.keys())

        # Count how many regions to display
        n_regions = len(regions_order)

        # Combine parameters
        plot_kwargs = {**self.params.model_dump(), **kwargs}

        # Calculate layout
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols

        # Scale figure size based on number of rows
        figsize = plot_kwargs.get("figsize")
        scaled_figsize = (figsize[0], figsize[1] * (n_rows / 2))

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=scaled_figsize, dpi=plot_kwargs.get("dpi"))
        axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes]).flatten()

        # Extract heatmap parameters
        heatmap_kwargs = {k: v for k, v in plot_kwargs.items() if k in ["cmap", "cbar", "alpha", "annot", "fmt"]}
        # Thinner lines for neuron grid
        heatmap_kwargs["linewidths"] = plot_kwargs.get("linewidths", 0.5) / 2

        # Plot each region's state
        for i, region in enumerate(regions_order):
            if region in states:
                state_data = self._convert_to_numpy(states[region])

                # Handle different state shapes
                if state_data.ndim > 2:
                    # For 3D or higher tensors, flatten all but the last dimension
                    state_data = state_data.reshape(-1, state_data.shape[-1])

                sns.heatmap(state_data, ax=axes[i], **heatmap_kwargs)
                axes[i].set_title(region, fontsize=plot_kwargs.get("title_fontsize"))
            else:
                axes[i].axis("off")

        # Hide unused axes
        for i in range(n_regions, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=plot_kwargs.get("suptitle_fontsize"))
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    grid_size = (10, 10)
    obstacle_map = np.zeros(grid_size)
    obstacle_map[2:5, 3:6] = 1
    obstacle_map[7:9, 1:4] = 1
    obstacle_map[6:8, 7:9] = 1

    # Create goal position
    goal_map = np.zeros_like(obstacle_map)
    goal_map[8, 8] = 1

    # Sample cognitive map (e.g., distance to goal)
    from scipy.ndimage import distance_transform_edt

    grid_with_obstacles = obstacle_map.copy()
    grid_with_obstacles[goal_map == 1] = 0  # Ensure goal isn't considered an obstacle
    distance_map = distance_transform_edt(1 - goal_map)

    # Normalize distance map
    distance_map = distance_map / distance_map.max()

    # Create a "reconstructed" map with some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, grid_size)
    reconstructed_map = distance_map + noise
    reconstructed_map = np.clip(reconstructed_map, 0, 1)

    # Create a sample embedding matrix
    n_neurons = 64
    n_samples = 5
    embeddings = np.random.normal(0, 1, (n_samples, n_neurons))
    # Make it sparse
    embeddings *= np.random.random(embeddings.shape) > 0.7

    # Sample model states
    model_states = {
        "MEC_II": np.random.normal(0, 1, (8, 16)),
        "DG": np.random.normal(0, 1, (20, 10)) * (np.random.random((20, 10)) > 0.8),
        "CA3": np.random.normal(0, 1, (12, 12)) * (np.random.random((12, 12)) > 0.7),
        "CA1": np.random.normal(0, 1, (10, 14)),
    }

    # Create visualizer and plot examples
    params = CognitiveMapParameters(figsize=(14, 10), cmap="magma")
    visualizer = CognitiveMapVisualizer(params)

    # Environment plot
    fig1 = plt.figure(figsize=params.figsize, dpi=params.dpi)
    ax = fig1.add_subplot(111)
    visualizer.plot_environment(
        obstacle_map,
        goal_map,
        agent_position=(2, 2),
        title="Sample Environment",
        ax=ax,
    )

    # Cognitive map comparison
    fig2 = visualizer.plot_cognitive_map_comparison(
        distance_map, reconstructed_map, title="Distance Map Reconstruction"
    )

    # Embedding visualization
    fig3 = visualizer.plot_embedding_activations(embeddings, title="Sample Neural Embeddings")

    # Model states
    fig4 = visualizer.plot_model_states(
        model_states, regions_order=["MEC_II", "DG", "CA3", "CA1"], title="Sample EHC Circuit States"
    )

    plt.show()
