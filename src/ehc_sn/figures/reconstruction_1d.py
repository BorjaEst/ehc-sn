from typing import Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class ReconstructionTraceParams(FigureParams):
    """Parameters specific to 1D reconstruction trace plots."""

    line_width: float = Field(default=1.0, ge=0.1, le=5.0, description="Line width")
    alpha: float = Field(default=0.85, ge=0.1, le=1.0, description="Line alpha")
    show_grid: bool = Field(default=True, description="Show grid on plots")
    ylim: Optional[Tuple[float, float]] = Field(default=None, description="Y-axis limits")
    show_mse: bool = Field(default=True, description="Show MSE annotation")
    input_color: str = Field(default="tab:blue", description="Input line color")
    output_color: str = Field(default="tab:red", description="Reconstruction line color")


# -------------------------------------------------------------------------------------------
class ReconstructionTraceFigure(BaseFigure):
    """Side-by-side 1D traces: input vs reconstruction for N samples.

    This figure shows input and reconstruction traces for autoencoder evaluation.
    Each row shows one sample with input on the left and reconstruction on the right.
    MSE is calculated and displayed for each sample.
    """

    def __init__(self, params: Optional[ReconstructionTraceParams] = None) -> None:
        super().__init__(params or ReconstructionTraceParams())

    @property
    def p(self) -> ReconstructionTraceParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, inputs: Tensor, outputs: Tensor) -> Figure:
        """Plot input and reconstruction traces.

        Args:
            inputs: Input tensor of shape (N, D) where N is batch size, D is feature dimension
            outputs: Reconstruction tensor of shape (N, D)

        Returns:
            Matplotlib Figure object

        Raises:
            AssertionError: If tensors don't have expected shapes or don't match
        """
        assert inputs.ndim == 2 and outputs.ndim == 2, "Expected (N, D) tensors"
        assert inputs.shape == outputs.shape, "Inputs and outputs must have matching shapes"

        n = self.select_n(inputs.shape[0])
        x = self.to_numpy(inputs[:n])
        y = self.to_numpy(outputs[:n])

        fig, axes = self.build_canvas(nrows=n, ncols=2)

        for i in range(n):
            ax_in: Axes = axes[i, 0]
            ax_out: Axes = axes[i, 1]

            # Plot input trace
            ax_in.plot(x[i], color=self.p.input_color, linewidth=self.p.line_width, alpha=self.p.alpha, label="Input")

            # Plot reconstruction trace
            ax_out.plot(
                y[i], color=self.p.output_color, linewidth=self.p.line_width, alpha=self.p.alpha, label="Reconstruction"
            )

            # Configure grid
            if self.p.show_grid:
                ax_in.grid(True, alpha=0.3)
                ax_out.grid(True, alpha=0.3)

            # Set y-axis limits
            if self.p.ylim is not None:
                ax_in.set_ylim(*self.p.ylim)
                ax_out.set_ylim(*self.p.ylim)

            # Set titles and labels
            ax_in.set_title(f"Sample {i+1}")
            ax_in.set_xlabel("Feature Index")
            ax_in.set_ylabel("Amplitude")
            ax_out.set_title("Reconstruction")
            ax_out.set_xlabel("Feature Index")

            # Add MSE annotation
            if self.p.show_mse:
                mse = float(np.mean((x[i] - y[i]) ** 2))
                ax_out.text(
                    0.02,
                    0.92,
                    f"MSE={mse:.3e}",
                    transform=ax_out.transAxes,
                    fontsize="small",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )

            # Add legends to first row only
            if i == 0:
                ax_in.legend(loc="upper right", fontsize="small")
                ax_out.legend(loc="upper right", fontsize="small")

        return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the reconstruction trace figure."""
    import torch

    # Create test data
    torch.manual_seed(42)
    n_samples, n_features = 6, 64
    inputs = torch.rand(n_samples, n_features)
    outputs = inputs + 0.05 * torch.randn(n_samples, n_features)

    # Create and plot figure
    params = ReconstructionTraceParams(
        n_samples=4, title="Autoencoder Reconstruction Test", fig_width=12, fig_height=10
    )

    figure = ReconstructionTraceFigure(params)
    fig = figure.plot(inputs, outputs)

    import matplotlib.pyplot as plt

    plt.show()
