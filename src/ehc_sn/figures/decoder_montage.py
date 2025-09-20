"""Decoder montage visualization for 2D spatial data.

This module provides visualization for decoder weight montages showing the
reconstructed patterns when individual latent units are activated.
"""

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class DecoderMontageParams(FigureParams):
    """Parameters for decoder montage visualization."""

    cmap: str = Field(default="RdBu_r", description="Colormap for montage (diverging recommended)")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap")
    symmetric_clim: bool = Field(default=True, description="Force symmetric color limits")
    interpolation: str = Field(default="nearest", description="Image interpolation")
    add_colorbar: bool = Field(default=False, description="Add colorbar to montage")
    show_axes: bool = Field(default=False, description="Show axis ticks and labels")
    annotate_id: bool = Field(default=True, description="Show unit index in each tile")
    ncols: Optional[int] = Field(default=None, description="Number of columns (auto if None)")
    tile_padding: float = Field(default=0.02, ge=0.0, le=0.1, description="Subplot spacing")
    aspect: str = Field(default="equal", description="Aspect ratio")


# -------------------------------------------------------------------------------------------
class DecoderMontageFigure(BaseFigure):
    """Decoder weight montage showing reconstructed patterns from individual latent units.

    This figure displays a grid of 2D maps showing what each latent unit reconstructs
    when activated individually. Useful for understanding the learned basis functions
    or "atoms" of the decoder.

    Example:
        >>> from ehc_sn.figures import DecoderMontageFigure, DecoderMontageParams
        >>>
        >>> # Generate latent codes and their reconstructions
        >>> latents = torch.eye(64)  # One-hot encoding for each unit
        >>> outputs = decoder(latents)  # Shape: (64, H, W)
        >>>
        >>> # Visualize
        >>> params = DecoderMontageParams(title="Decoder Atoms", ncols=8)
        >>> figure = DecoderMontageFigure(params)
        >>> fig = figure.plot(latents, outputs)
        >>> fig.show()
    """

    def __init__(self, params: Optional[DecoderMontageParams] = None) -> None:
        super().__init__(params or DecoderMontageParams())

    @property
    def p(self) -> DecoderMontageParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, latents: Tensor, outputs: Tensor) -> Figure:
        """Plot decoder montage from latents and their reconstructions.

        Args:
            latents: Latent codes tensor of shape (K, D) where K is number of units
            outputs: Reconstructed maps tensor of shape (K, H, W)

        Returns:
            Matplotlib Figure object

        Raises:
            AssertionError: If tensors don't have expected shapes or don't match
        """
        assert latents.ndim == 2, f"Expected latents shape (K, D), got {latents.shape}"
        assert outputs.ndim == 3, f"Expected outputs shape (K, H, W), got {outputs.shape}"
        assert latents.shape[0] == outputs.shape[0], "Number of latents and outputs must match"

        k = latents.shape[0]
        maps = self.to_numpy(outputs)

        # Determine grid layout
        if self.p.ncols is None:
            ncols = int(np.ceil(np.sqrt(k)))
        else:
            ncols = self.p.ncols
        nrows = int(np.ceil(k / ncols))

        # Determine color limits
        vmin, vmax = self._compute_color_limits(maps)

        # Create figure and plot
        fig, axes = self.build_canvas(nrows=nrows, ncols=ncols)

        for i in range(k):
            row = i // ncols
            col = i % ncols
            ax: Axes = axes[row, col]

            # Plot the reconstructed map
            im = ax.imshow(
                maps[i],
                cmap=self.p.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=self.p.interpolation,
                aspect=self.p.aspect,
            )

            # Configure axes
            if not self.p.show_axes:
                ax.set_xticks([])
                ax.set_yticks([])

            # Add unit index annotation
            if self.p.annotate_id:
                ax.text(
                    0.02,
                    0.98,
                    f"{i}",
                    transform=ax.transAxes,
                    fontsize="small",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )

        # Hide unused subplots
        for i in range(k, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)

        # Add colorbar if requested
        if self.p.add_colorbar and k > 0:
            # Use the last plotted image for colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)

        # Adjust layout
        fig.subplots_adjust(
            wspace=self.p.tile_padding,
            hspace=self.p.tile_padding,
        )

        return fig

    def _compute_color_limits(self, maps: np.ndarray) -> tuple[float, float]:
        """Compute color limits for the montage.

        Args:
            maps: Array of shape (K, H, W)

        Returns:
            Tuple of (vmin, vmax)
        """
        if self.p.vmin is not None and self.p.vmax is not None:
            return self.p.vmin, self.p.vmax

        # Compute from data
        if self.p.symmetric_clim:
            max_abs = np.max(np.abs(maps))
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = np.min(maps), np.max(maps)

        # Override with explicit values if provided
        if self.p.vmin is not None:
            vmin = self.p.vmin
        if self.p.vmax is not None:
            vmax = self.p.vmax

        return vmin, vmax


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the decoder montage figure."""
    import torch

    # Create test decoder outputs
    torch.manual_seed(42)
    k, height, width = 16, 12, 12

    # Create one-hot latent codes
    latents = torch.eye(k)

    # Create synthetic decoder outputs (localized patterns)
    outputs = torch.zeros(k, height, width)
    for i in range(k):
        # Create different localized patterns
        x_center = (i % 4) * 3 + 2
        y_center = (i // 4) * 3 + 2
        sigma = 1.5

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        gaussian = torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma**2))

        # Add some polarity variation
        sign = 1 if i % 2 == 0 else -1
        outputs[i] = sign * gaussian

    print(f"Test data shapes: latents={latents.shape}, outputs={outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")

    # Create and plot figure
    params = DecoderMontageParams(
        title="Decoder Weight Montage Test",
        ncols=4,
        fig_width=10,
        fig_height=8,
        annotate_id=True,
        symmetric_clim=True,
    )

    figure = DecoderMontageFigure(params)
    fig = figure.plot(latents, outputs)

    print("✓ Decoder montage figure test completed!")

    # Uncomment to display
    # import matplotlib.pyplot as plt
    # plt.show()
