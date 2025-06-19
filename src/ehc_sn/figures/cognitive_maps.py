from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator

from ehc_sn import data
from ehc_sn.constants import CognitiveMap, ObstacleMap

SUBPLOTS_PARAM = {"figsize", "dpi", "tight_layout", "constrained_layout"}


class CognitiveMapFigParam(BaseModel):
    """Base class for visualization parameters."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    # Parameters for figure alignment
    figsize: Tuple[int, int] = Field(
        default=(10, 8),
        description="Figure size in inches",
    )
    dpi: int = Field(
        default=100,
        description="Dots per inch for figure resolution",
    )
    tight_layout: bool = Field(
        default=True,
        description="Whether to use tight layout",
    )
    constrained_layout: bool = Field(
        default=False,
        description="Whether to use constrained layout",
    )
    # Parameters for title display
    title_fontsize: int = Field(
        default=14,
        description="Font size for title",
    )

    @field_validator("constrained_layout")
    def check_layout_conflict(cls, v, info):
        if v and info.data.get("tight_layout", True):
            raise ValueError("Cannot enable both tight_layout and constrained_layout")
        return v


class CognitiveMapFigure(ABC):
    """Base class for cognitive map figures."""

    def __init__(self, params: Optional[CognitiveMapFigParam] = None):
        self.params = params or CognitiveMapFigParam()
        self.fig = None
        self.axes = None

    @abstractmethod
    def title(self) -> str:
        raise NotImplementedError("Subclasses must implement the title method")

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> Tuple[Figure, Union[Axes, List[Axes]]]:
        raise NotImplementedError("Subclasses must implement the plot method")

    def show(self) -> None:
        plt.show()

    def subplot(self, nrows: int, ncols: int) -> None:
        kwargs = self.params.model_dump(include=SUBPLOTS_PARAM)
        self.fig, self.axes = plt.subplots(nrows, ncols, **kwargs)
        self.fig.suptitle(self.title(), fontsize=self.params.title_fontsize)

    def save(self, filepath: str) -> None:
        self.fig.savefig(filepath)


class CompareMapsFigParam(CognitiveMapFigParam):
    """Parameters for comparing multiple cognitive maps."""

    num_cols: int = Field(
        default=1,
        description="Number of columns in the grid layout",
    )


class CompareCognitiveMaps(CognitiveMapFigure):
    """Figure for comparing multiple cognitive maps."""

    def __init__(self, params: Optional[CompareMapsFigParam] = None):
        super().__init__(params or CompareMapsFigParam())

    def title(self) -> str:
        return "Comparison of Cognitive Maps"

    def plot(self, maps: List[Tuple[CognitiveMap, ObstacleMap]]) -> Tuple[Figure, List[Axes]]:
        num_maps = len(maps)
        num_cols = min(self.params.num_cols, num_maps)
        total_rows = (num_maps + num_cols - 1) // num_cols
        total_cols = num_cols * 2  # Each map has an obstacle and cognitive map

        # Create subplots with specified number of rows and columns
        self.subplot(total_rows, total_cols)
        self.axes = np.array([self.axes]) if total_rows == 1 else self.axes

        # Adjust the figure size based on the number of maps
        for i, (cognitive_map, obstacle_map) in enumerate(maps):
            row = i // num_cols
            col = (i % num_cols) * 2
            self._plot_single_map(row, col, i, cognitive_map, obstacle_map)

        self._hide_unused_axes(num_maps, total_rows, total_cols)
        return self.fig, self.axes

    def _plot_single_map(self, row, col, idx, cognitive_map, obstacle_map):
        # Plot the obstacle map
        obstacle_plot_params = data.grid_maps.PlotMapParams()
        obs_ax = self.axes[row, col]
        data.grid_maps.plot(obs_ax, obstacle_map, obstacle_plot_params)
        obs_ax.set_title(f"Obstacles {idx + 1}")

        # Plot the cognitive map
        cognitive_plot_params = data.cognitive_maps.PlotMapParams()
        cog_ax = self.axes[row, col + 1]
        data.cognitive_maps.plot(cog_ax, cognitive_map, cognitive_plot_params)
        cog_ax.set_title(f"Cognitive Map {idx + 1}")

    def _hide_unused_axes(self, num_maps: int, total_rows: int, total_cols: int) -> None:
        for i in range(total_rows):
            for j in range(min(num_maps * 2, total_cols), total_cols):
                self.axes[i, j].axis("off")


if __name__ == "__main__":
    # Example usage of the visualization classes
    cognitive_maps = [torch.rand((10, 10)) for _ in range(3)]
    obstacle_maps = [torch.zeros((10, 10)) for _ in range(3)]
    obstacle_maps[0][2:4, 3:7] = 1
    obstacle_maps[1][6:8, 2:8] = 1
    obstacle_maps[2][1:3, 1:5] = 1

    # Example cognitive map comparison with obstacles
    visualizer = CompareCognitiveMaps()
    visualizer.plot([x for x in zip(cognitive_maps, obstacle_maps)])
    visualizer.show()
