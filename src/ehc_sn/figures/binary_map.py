"""Binary map visualization for 2D spatial data.

This module provides visualization for binary 2D maps (walls, goals, obstacles).
Users extract specific channels from multi-channel tensors before plotting.
"""

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class BinaryMapParams(FigureParams):
    """Parameters for binary map visualization."""

    cmap: str = Field(default="Greys", description="Colormap for binary data")
    invert: bool = Field(default=True, description="Invert colors (1=dark, 0=light)")
    show_colorbar: bool = Field(default=False, description="Show colorbar")
    grid: bool = Field(default=False, description="Show grid lines")
    grid_color: str = Field(default="#cccccc", description="Grid line color")
    grid_linewidth: float = Field(default=0.5, ge=0.1, le=2.0, description="Grid line width")
    aspect: str = Field(default="equal", description="Aspect ratio")
    vmin: float = Field(default=0.0, description="Minimum value for colormap")
    vmax: float = Field(default=1.0, description="Maximum value for colormap")
    interpolation: str = Field(default="nearest", description="Image interpolation")
    frame: bool = Field(default=False, description="Show axes frame and ticks")
    annotate: bool = Field(default=False, description="Annotate cell values (small maps only)")
    annotate_threshold: int = Field(default=32, ge=8, le=128, description="Max size for annotation")


# -------------------------------------------------------------------------------------------
class BinaryMapFigure(BaseFigure):
    """Figure for visualizing single binary 2D maps.

    Designed for binary spatial data like walls, goals, or obstacles.
    Users must extract the desired channel from multi-channel tensors before plotting.

    Example:
        >>> from ehc_sn.figures import BinaryMapFigure, BinaryMapParams
        >>> from ehc_sn.data.minigrid_maps import generate
        >>>
        >>> # Generate MiniGrid map and extract walls channel
        >>> map_tensor = generate(seed=42, env_id="MiniGrid-Empty-8x8-v0")
        >>> walls = map_tensor[0]  # Extract walls channel (H, W)
        >>>
        >>> # Visualize
        >>> params = BinaryMapParams(title="Wall Layout", invert=True)
        >>> figure = BinaryMapFigure(params)
        >>> fig = figure.plot(walls)
        >>> fig.show()
    """

    def __init__(self, params: Optional[BinaryMapParams] = None) -> None:
        super().__init__(params or BinaryMapParams())

    @property
    def p(self) -> BinaryMapParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, data: Union[Tensor, np.ndarray], ax: Optional[Axes] = None) -> Figure:
        """Plot a single binary 2D map.

        Args:
            data: Binary map tensor of shape (H, W)
            ax: Optional axes to plot on (creates new figure if None)

        Returns:
            Matplotlib Figure object

        Raises:
            ValueError: If data is not 2D or contains invalid values
        """
        # Validate input
        if isinstance(data, Tensor):
            data_np = self.to_numpy(data)
        else:
            data_np = data

        if data_np.ndim != 2:
            raise ValueError(f"Expected 2D tensor (H, W), got shape {data_np.shape}")

        # Convert to float and clamp to [0, 1]
        data_np = data_np.astype(np.float32)
        if np.any(np.isnan(data_np)) or np.any(np.isinf(data_np)):
            raise ValueError("Data contains NaN or infinite values")

        data_np = np.clip(data_np, 0.0, 1.0)

        # Apply inversion if requested
        if self.p.invert:
            display_data = 1.0 - data_np
        else:
            display_data = data_np

        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.build_canvas(nrows=1, ncols=1)
            ax = axes[0, 0]
            fig_created = True
        else:
            fig = ax.get_figure()
            fig_created = False

        # Plot the binary map
        im = ax.imshow(
            display_data,
            cmap=self.p.cmap,
            vmin=self.p.vmin,
            vmax=self.p.vmax,
            interpolation=self.p.interpolation,
            aspect=self.p.aspect,
        )

        # Configure grid
        if self.p.grid:
            h, w = data_np.shape
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(
                True,
                which="minor",
                color=self.p.grid_color,
                linewidth=self.p.grid_linewidth,
            )

        # Configure axes visibility
        if not self.p.frame:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        # Add annotations for small maps
        if self.p.annotate and max(data_np.shape) <= self.p.annotate_threshold:
            self._add_annotations(ax, data_np)

        # Add colorbar if requested
        if self.p.show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Set title if figure was created here
        if fig_created and self.p.title:
            ax.set_title(self.p.title)

        return fig

    def _add_annotations(self, ax: Axes, data: np.ndarray) -> None:
        """Add text annotations showing cell values.

        Args:
            ax: Matplotlib axes
            data: 2D array of values to annotate
        """
        h, w = data.shape
        config = {"ha": "center", "va": "center", "fontsize": "small", "weight": "bold"}
        for y in range(h):
            for x in range(w):
                value = data[y, x]
                text_color = "white" if value < 0.5 else "black"
                ax.text(x, y, f"{value:.0f}", color=text_color, **config)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the binary map figure."""
    import torch

    # Create test binary map data
    torch.manual_seed(42)

    # Create a simple binary pattern
    data = torch.zeros(12, 12)
    # Add walls around perimeter
    data[0, :] = 1  # Top wall
    data[-1, :] = 1  # Bottom wall
    data[:, 0] = 1  # Left wall
    data[:, -1] = 1  # Right wall
    # Add some internal walls
    data[5:8, 3] = 1  # Vertical wall
    data[3, 5:8] = 1  # Horizontal wall
    # Add goal
    data[9, 9] = 1

    print(f"Test data shape: {data.shape}")
    print(f"Wall density: {torch.mean(data):.3f}")

    # Test basic visualization
    params = BinaryMapParams(
        title="Test Binary Map",
        invert=True,
        grid=True,
        annotate=True,
        frame=True,
        fig_width=8,
        fig_height=6,
    )

    figure = BinaryMapFigure(params)
    fig = figure.plot(data)

    print("✓ Binary map figure test completed!")

    # Uncomment to display
    # import matplotlib.pyplot as plt
    # plt.show()
