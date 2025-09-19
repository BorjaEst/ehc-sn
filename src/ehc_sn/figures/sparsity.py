from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class SparsityParams(FigureParams):
    """Parameters specific to sparsity analysis plots."""

    n_bins: int = Field(default=50, ge=10, le=100, description="Number of histogram bins")
    threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Threshold for considering activation as 'active'"
    )
    show_threshold: bool = Field(default=True, description="Show threshold line on histogram")
    alpha: float = Field(default=0.7, ge=0.1, le=1.0, description="Histogram alpha")
    color: str = Field(default="tab:blue", description="Histogram color")


# -------------------------------------------------------------------------------------------
class SparsityFigure(BaseFigure):
    """Analyze and visualize sparsity patterns in neural activations.

    This figure shows histograms of activation values and computes sparsity metrics
    for each sample or layer. Useful for understanding the sparsity properties of
    autoencoder latent representations or neural network activations.
    """

    def __init__(self, params: Optional[SparsityParams] = None) -> None:
        super().__init__(params or SparsityParams())

    @property
    def p(self) -> SparsityParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, activations: Tensor) -> Figure:
        """Plot sparsity analysis of activations.

        Args:
            activations: Activation tensor of shape (N, D) where N is batch size, D is features

        Returns:
            Matplotlib Figure object

        Raises:
            AssertionError: If tensor doesn't have expected shape
        """
        assert activations.ndim == 2, "Expected (N, D) tensor for activations"

        n = self.select_n(activations.shape[0])
        x = self.to_numpy(activations[:n])

        fig, axes = self.build_canvas(nrows=n, ncols=2)

        for i in range(n):
            ax_hist: Axes = axes[i, 0]
            ax_sparse: Axes = axes[i, 1]

            # Compute sparsity metrics
            sample_activations = x[i]
            active_count = np.sum(np.abs(sample_activations) > self.p.threshold)
            total_count = len(sample_activations)
            sparsity_rate = 1.0 - (active_count / total_count)
            mean_activation = np.mean(np.abs(sample_activations))

            # Histogram of activation values
            ax_hist.hist(
                sample_activations,
                bins=self.p.n_bins,
                alpha=self.p.alpha,
                color=self.p.color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_hist.set_title(f"Sample {i+1} - Activation Distribution")
            ax_hist.set_xlabel("Activation Value")
            ax_hist.set_ylabel("Count")
            ax_hist.grid(True, alpha=0.3)

            # Show threshold line
            if self.p.show_threshold:
                ax_hist.axvline(
                    self.p.threshold, color="red", linestyle="--", alpha=0.8, label=f"Threshold ({self.p.threshold})"
                )
                ax_hist.axvline(-self.p.threshold, color="red", linestyle="--", alpha=0.8)
                ax_hist.legend()

            # Sparsity visualization (activation pattern)
            sample_2d = sample_activations.reshape(-1, 1)  # Make it 2D for imshow
            im = ax_sparse.imshow(sample_2d.T, cmap="RdBu_r", aspect="auto", interpolation="nearest")
            ax_sparse.set_title("Activation Pattern")
            ax_sparse.set_xlabel("Feature Index")
            ax_sparse.set_ylabel("Activation")
            ax_sparse.set_yticks([])

            # Add colorbar
            fig.colorbar(im, ax=ax_sparse, fraction=0.046, pad=0.04)

            # Add sparsity metrics as text
            metrics_text = f"Sparsity: {sparsity_rate:.3f}\nActive: {active_count}/{total_count}\nMean |Act|: {mean_activation:.3e}"
            ax_sparse.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax_sparse.transAxes,
                fontsize="small",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the sparsity figure."""
    import torch

    # Create test sparse activations
    torch.manual_seed(42)
    n_samples, n_features = 4, 128

    # Generate sparse activations (most values near zero)
    activations = torch.zeros(n_samples, n_features)
    for i in range(n_samples):
        # Randomly activate 10-20% of features
        n_active = torch.randint(10, 25, (1,)).item()
        active_indices = torch.randperm(n_features)[:n_active]
        activations[i, active_indices] = torch.randn(n_active).abs() * 2.0

    # Add small noise to all features
    activations += 0.01 * torch.randn(n_samples, n_features)

    # Create and plot figure
    params = SparsityParams(n_samples=3, title="Sparsity Analysis Test", fig_width=12, fig_height=8, threshold=0.1)

    figure = SparsityFigure(params)
    fig = figure.plot(activations)

    import matplotlib.pyplot as plt

    plt.show()
