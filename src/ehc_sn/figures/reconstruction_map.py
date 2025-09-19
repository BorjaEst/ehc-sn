from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class ReconstructionMapParams(FigureParams):
    """Parameters specific to 2D map reconstruction plots."""

    cmap: str = Field(default="viridis", description="Colormap for images")
    add_colorbar: bool = Field(default=True, description="Add colorbar to plots")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap")
    interpolation: str = Field(default="nearest", description="Image interpolation method")
    show_mse: bool = Field(default=True, description="Show MSE annotation")
    show_axes: bool = Field(default=False, description="Show axis ticks and labels")


# -------------------------------------------------------------------------------------------
class ReconstructionMapFigure(BaseFigure):
    """Side-by-side 2D maps (images): input vs reconstruction for N samples.

    This figure displays 2D spatial data (cognitive maps, images) comparing
    input and reconstruction side by side. Supports both (N, H, W) and (N, C, H, W)
    tensor formats. For multi-channel data, uses the first channel.
    """

    def __init__(self, params: Optional[ReconstructionMapParams] = None) -> None:
        super().__init__(params or ReconstructionMapParams())

    @property
    def p(self) -> ReconstructionMapParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, inputs: Tensor, outputs: Tensor) -> Figure:
        """Plot input and reconstruction maps.

        Args:
            inputs: Input tensor of shape (N, H, W) or (N, C, H, W)
            outputs: Reconstruction tensor matching inputs shape

        Returns:
            Matplotlib Figure object

        Raises:
            AssertionError: If tensors don't have expected shapes or don't match
        """
        assert inputs.ndim in (3, 4) and outputs.ndim in (3, 4), "Expected (N,H,W) or (N,C,H,W) tensors"

        # Convert to (N, H, W) format, using first channel if needed
        def to_hw(t: Tensor) -> Tensor:
            return t if t.ndim == 3 else t[:, 0]  # (N, H, W)

        inputs_hw = to_hw(inputs)
        outputs_hw = to_hw(outputs)
        assert inputs_hw.shape == outputs_hw.shape, "Inputs and outputs must have matching shapes"

        n = self.select_n(inputs_hw.shape[0])
        x = self.to_numpy(inputs_hw[:n])
        y = self.to_numpy(outputs_hw[:n])

        fig, axes = self.build_canvas(nrows=n, ncols=2)

        for i in range(n):
            ax_in: Axes = axes[i, 0]
            ax_out: Axes = axes[i, 1]

            # Plot input map
            im1 = ax_in.imshow(
                x[i], cmap=self.p.cmap, vmin=self.p.vmin, vmax=self.p.vmax, interpolation=self.p.interpolation
            )
            ax_in.set_title(f"Input {i+1}")

            # Plot reconstruction map
            im2 = ax_out.imshow(
                y[i], cmap=self.p.cmap, vmin=self.p.vmin, vmax=self.p.vmax, interpolation=self.p.interpolation
            )
            ax_out.set_title("Reconstruction")

            # Configure axes
            if not self.p.show_axes:
                ax_in.set_xticks([])
                ax_in.set_yticks([])
                ax_out.set_xticks([])
                ax_out.set_yticks([])
            else:
                ax_in.set_xlabel("X")
                ax_in.set_ylabel("Y")
                ax_out.set_xlabel("X")

            # Add colorbars
            if self.p.add_colorbar:
                fig.colorbar(im1, ax=ax_in, fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=ax_out, fraction=0.046, pad=0.04)

            # Add MSE annotation
            if self.p.show_mse:
                mse = float(np.mean((x[i] - y[i]) ** 2))
                ax_out.text(
                    0.02,
                    0.98,
                    f"MSE={mse:.3e}",
                    transform=ax_out.transAxes,
                    fontsize="small",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the reconstruction map figure."""
    import torch

    # Create test 2D map data
    torch.manual_seed(42)
    n_samples, height, width = 4, 32, 32

    # Create spatial patterns (e.g., cognitive maps)
    inputs = torch.zeros(n_samples, height, width)
    for i in range(n_samples):
        # Create different spatial patterns for each sample
        x_center, y_center = torch.randint(8, 24, (2,))
        sigma = 3.0

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        gaussian = torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma**2))
        inputs[i] = gaussian

    # Add noise to create reconstruction
    outputs = inputs + 0.1 * torch.randn_like(inputs)
    outputs = torch.clamp(outputs, 0, 1)

    # Create and plot figure
    params = ReconstructionMapParams(
        n_samples=3, title="Cognitive Map Reconstruction Test", fig_width=10, fig_height=8, cmap="plasma"
    )

    figure = ReconstructionMapFigure(params)
    fig = figure.plot(inputs, outputs)

    import matplotlib.pyplot as plt

    plt.show()
