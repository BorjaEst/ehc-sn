from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
from torch import Tensor


# -------------------------------------------------------------------------------------------
class FigureParams(BaseModel):
    """Common configuration for figures."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    fig_width: float = Field(default=10.0, ge=4.0, le=24.0, description="Figure width (inches)")
    fig_height: float = Field(default=8.0, ge=3.0, le=24.0, description="Figure height (inches)")
    fig_dpi: int = Field(default=100, ge=50, le=300, description="Figure DPI")
    n_samples: int = Field(default=5, ge=1, le=64, description="Max samples to plot")
    title: Optional[str] = Field(default=None, description="Figure title")


# -------------------------------------------------------------------------------------------
class BaseFigure:
    """Base utilities for figure classes."""

    def __init__(self, params: Optional[FigureParams] = None) -> None:
        self.params = params or FigureParams()

    def build_canvas(self, nrows: int, ncols: int) -> Tuple[Figure, np.ndarray]:
        """Create a matplotlib canvas with standard settings."""
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(self.params.fig_width, self.params.fig_height),
            dpi=self.params.fig_dpi,
            squeeze=False,
            tight_layout=True,
        )
        if self.params.title:
            fig.suptitle(self.params.title)
        return fig, axes

    def select_n(self, available: int) -> int:
        """Select number of samples to draw based on params and availability."""
        return max(1, min(self.params.n_samples, available))

    @staticmethod
    def to_numpy(x: Tensor) -> np.ndarray:
        """Detach tensor and convert to NumPy."""
        return x.detach().cpu().numpy()
