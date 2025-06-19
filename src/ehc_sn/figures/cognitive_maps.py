from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, Field, field_validator

from ehc_sn.constants import (
    CognitiveMap,
    Direction,
    GridMap,
    GridMapSample,
    NeuralActivity,
    ObstacleMap,
    Position,
    ValueMap,
)


class GridMapParam(BaseModel):
    """Parameters for plotting grid maps on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    cmap: str = Field(default="viridis", description="Colormap for the grid map")
    annot: bool = Field(default=False, description="Whether to annotate cells")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap")
    show_colorbar: bool = Field(default=True, description="Whether to show the colorbar")
    cbar_label: str = Field(default="", description="Label for the colorbar")
    fmt: str = Field(default=".2f", description="Format for annotations")
    linewidths: float = Field(default=0.5, description="Width of grid lines")
    linecolor: str = Field(default="white", description="Color of grid lines")
    square: bool = Field(default=True, description="Whether cells should be square")
    mask_zeros: bool = Field(default=False, description="Whether to mask zero values")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for seaborn heatmap."""
        base_kwargs = {
            k: v
            for k, v in self.model_dump().items()
            if k not in ["show_colorbar", "cbar_label", "mask_zeros", "model_config"]
        }
        # Explicitly add cbar parameter based on show_colorbar
        base_kwargs["cbar"] = self.show_colorbar
        return base_kwargs


def plot_gridmap(ax: Axes, grid_map: GridMap, param: Optional[GridMapParam] = None) -> None:
    """Adds a grid map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        grid_map: 2D tensor representing the map to visualize
        param: Grid map visualization param
    """
    params = param or GridMapParam()

    # Convert to numpy if it's a tensor
    if isinstance(grid_map, torch.Tensor):
        grid_data = grid_map.detach().cpu().numpy()
    else:
        grid_data = np.array(grid_map)

    # Create mask for zeros if requested
    mask = None
    if params.mask_zeros:
        mask = grid_data == 0

    # Plot the heatmap with the explicit cbar parameter
    heatmap = sns.heatmap(grid_data, ax=ax, mask=mask, **params.kwargs)

    # Add colorbar label if a colorbar exists
    if params.show_colorbar and hasattr(heatmap, "collections") and heatmap.collections:
        cbar = getattr(heatmap.collections[0], "colorbar", None)
        if cbar is not None and params.cbar_label:
            cbar.set_label(params.cbar_label)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])


class ObstaclesParam(BaseModel):
    """Parameters for plotting obstacle overlays on an axis."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    overlay_obstacles: bool = Field(default=True, description="Whether to overlay obstacles")
    obstacle_color: str = Field(default="#333333", description="Color for obstacles")
    obstacle_alpha: float = Field(default=0.7, description="Opacity of obstacle overlay")
    obstacle_hatch: str = Field(default="///", description="Hatch pattern for obstacles")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for plotting."""
        return {k: v for k, v in self.model_dump().items() if k not in ["overlay_obstacles", "model_config"]}


def plot_obstacles(ax: Axes, obstacles: ObstacleMap, param: Optional[ObstaclesParam] = None) -> None:
    """Adds obstacle overlays to a map plot.

    Args:
        ax: Matplotlib axis to plot on
        obstacles: Binary tensor marking obstacle positions
        param: Visualization param
    """
    params = param or ObstaclesParam()

    if not params.overlay_obstacles:
        return

    # Convert to numpy if it's a tensor
    if isinstance(obstacles, torch.Tensor):
        obstacles_data = obstacles.detach().cpu().numpy()
    else:
        obstacles_data = np.array(obstacles)

    # Get the indices of obstacle cells
    obstacle_positions = np.where(obstacles_data > 0)

    # Draw rectangles for each obstacle
    for i, j in zip(obstacle_positions[0], obstacle_positions[1]):
        rect = plt.Rectangle(
            (j, i),  # Bottom left corner (x, y)
            1,
            1,  # Width, Height
            color=params.obstacle_color,
            alpha=params.obstacle_alpha,
            hatch=params.obstacle_hatch,
            fill=True,
        )
        ax.add_patch(rect)


class GoalParam(BaseModel):
    """Parameters for plotting goal overlays on maps."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    overlay_goal: bool = Field(default=True, description="Whether to overlay goal")
    goal_color: str = Field(default="#ff5733", description="Color for goal")
    goal_marker: str = Field(default="*", description="Marker for goal")
    goal_size: int = Field(default=200, description="Size of goal marker")
    goal_alpha: float = Field(default=1.0, description="Opacity of goal marker")
    goal_zorder: int = Field(default=10, description="Z-order of goal marker")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary."""
        return {k: v for k, v in self.model_dump().items() if k not in ["overlay_goal", "model_config"]}


def plot_goal(ax: Axes, position: Position, param: Optional[GoalParam] = None) -> None:
    """Adds goal overlays to a map plot.

    Args:
        ax: Matplotlib axis to plot on
        position: Tuple indicating goal position as (row, col)
        param: Visualization param
    """
    params = param or GoalParam()

    if not params.overlay_goal:
        return

    # Plot the goal marker
    # Note: Position is (row, col) but scatter expects (x, y)
    ax.scatter(
        position[1] + 0.5,  # x coordinate (centered in cell)
        position[0] + 0.5,  # y coordinate (centered in cell)
        marker=params.goal_marker,
        color=params.goal_color,
        s=params.goal_size,
        alpha=params.goal_alpha,
        zorder=params.goal_zorder,
    )


class PathParam(BaseModel):
    """Parameters for plotting a path on a grid map."""

    model_config = {"extra": "forbid"}

    path_color: str = Field(default="#3477eb", description="Color for path line")
    linewidth: float = Field(default=2.5, description="Width of path line")  # Changed from path_width to linewidth
    path_alpha: float = Field(default=0.8, description="Opacity of path line")
    path_style: str = Field(default="-", description="Line style for path")
    path_zorder: int = Field(default=5, description="Z-order of path")
    show_markers: bool = Field(default=True, description="Whether to show markers at path points")
    marker_style: str = Field(default="o", description="Style for path markers")
    marker_size: int = Field(default=30, description="Size of path markers")
    marker_color: str = Field(default="#3477eb", description="Color of path markers")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for path plotting."""
        return {
            "color": self.path_color,
            "linewidth": self.linewidth,  # Correctly mapped to matplotlib's parameter name
            "alpha": self.path_alpha,
            "linestyle": self.path_style,
            "zorder": self.path_zorder,
        }

    @property
    def marker_kwargs(self) -> Dict[str, Any]:
        """Return parameters for marker plotting."""
        return {
            "marker": self.marker_style,
            "s": self.marker_size,
            "color": self.marker_color,
            "zorder": self.path_zorder + 1,
        }


def plot_path(ax: Axes, path: List[Position], param: Optional[PathParam] = None) -> None:
    """Adds a path visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        path: List of (row, col) positions representing the path
        param: Path visualization parameters
    """
    params = param or PathParam()

    if not path:
        return

    # Convert path positions to (x, y) coordinates for plotting
    # Offset by 0.5 to center in grid cells
    xs = [pos[1] + 0.5 for pos in path]
    ys = [pos[0] + 0.5 for pos in path]

    # Plot the path line
    ax.plot(xs, ys, **params.kwargs)

    # Add path markers if requested
    if params.show_markers:
        ax.scatter(xs, ys, **params.marker_kwargs)


class AgentParam(BaseModel):
    """Parameters for plotting an agent on a grid map."""

    model_config = {"extra": "forbid"}

    agent_color: str = Field(default="#32a852", description="Color for agent marker")
    agent_marker: str = Field(default="o", description="Marker for agent")
    agent_size: int = Field(default=250, description="Size of agent marker")
    agent_alpha: float = Field(default=0.9, description="Opacity of agent marker")
    agent_zorder: int = Field(default=15, description="Z-order of agent marker")
    show_heading: bool = Field(default=True, description="Whether to show agent heading")
    heading_length: float = Field(default=0.4, description="Length of heading arrow")
    heading_width: float = Field(default=0.15, description="Width of heading arrow")
    heading_color: str = Field(default="#246b3c", description="Color of heading arrow")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in ["show_heading", "heading_length", "heading_width", "heading_color", "model_config"]
        }


def plot_agent(
    ax: Axes, position: Position, direction: Optional[Direction] = None, param: Optional[AgentParam] = None
) -> None:
    """Adds an agent visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        position: (row, col) position of the agent
        direction: Optional (dy, dx) direction of agent heading
        param: Agent visualization parameters
    """
    params = param or AgentParam()

    # Plot the agent marker
    # Note: Position is (row, col) but scatter expects (x, y)
    agent_x = position[1] + 0.5  # Center in cell
    agent_y = position[0] + 0.5  # Center in cell

    ax.scatter(
        agent_x,
        agent_y,
        marker=params.agent_marker,
        color=params.agent_color,
        s=params.agent_size,
        alpha=params.agent_alpha,
        zorder=params.agent_zorder,
    )

    # Add heading direction if provided
    if params.show_heading and direction is not None:
        # Normalize direction vector for display
        dir_y, dir_x = direction
        mag = np.sqrt(dir_x**2 + dir_y**2)
        if mag > 0:  # Avoid division by zero
            dir_x, dir_y = dir_x / mag, dir_y / mag

            # In grid system, positive y is downward
            # We need to convert to matplotlib's coordinate system where positive y is upward
            ax.arrow(
                agent_x,
                agent_y,
                dir_x * params.heading_length,
                -dir_y * params.heading_length,  # Negate y for matplotlib coordinates
                width=params.heading_width,
                head_width=params.heading_width * 3,
                head_length=params.heading_length * 0.3,
                color=params.heading_color,
                zorder=params.agent_zorder + 1,
            )


class ValueMapParam(BaseModel):
    """Parameters for plotting value maps."""

    model_config = {"extra": "forbid"}

    cmap: str = Field(default="viridis", description="Colormap for value maps")
    show_colorbar: bool = Field(default=True, description="Whether to show the colorbar")
    cbar_label: str = Field(default="Value", description="Label for the colorbar")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap")
    alpha: float = Field(default=0.7, description="Opacity of the value map overlay")

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary for imshow."""
        return {
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "alpha": self.alpha,
        }


def plot_valuemap(ax: Axes, value_map: ValueMap, param: Optional[ValueMapParam] = None) -> None:
    """Adds a value map visualization to the given axis.

    Args:
        ax: Matplotlib axis to plot on
        value_map: 2D tensor representing values
        param: Value map visualization parameters
    """
    params = param or ValueMapParam()

    # Convert to numpy if it's a tensor
    if isinstance(value_map, torch.Tensor):
        map_data = value_map.detach().cpu().numpy()
    else:
        map_data = np.array(value_map)

    # Plot the value map using imshow
    im = ax.imshow(map_data, origin="upper", **params.kwargs)

    # Add colorbar if specified
    if params.show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if params.cbar_label:
            cbar.set_label(params.cbar_label)


class CognitiveMapFigParam(BaseModel):
    """Base class for visualization parameters."""

    model_config = {"extra": "forbid"}  # Pydantic v2 way to forbid extra fields

    figsize: Tuple[int, int] = Field(default=(10, 8), description="Figure size in inches")
    dpi: int = Field(default=100, description="Dots per inch for figure resolution")
    title_fontsize: int = Field(default=14, description="Font size for title")
    tight_layout: bool = Field(default=True, description="Whether to use tight layout")
    constrained_layout: bool = Field(default=False, description="Whether to use constrained layout")

    @field_validator("constrained_layout")
    def check_layout_conflict(cls, v, info):
        if v and info.data.get("tight_layout", True):
            raise ValueError("Cannot enable both tight_layout and constrained_layout")
        return v

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Return parameters as a dictionary."""
        return {
            "figsize": self.figsize,
            "dpi": self.dpi,
            "tight_layout": self.tight_layout,
            "constrained_layout": self.constrained_layout,
        }


class CognitiveMapFigure:
    """Base class for cognitive map figures."""

    def __init__(self, param: Optional[CognitiveMapFigParam] = None):
        self.param = param or CognitiveMapFigParam()
        self.title = "Cognitive Map Visualization"
        self.fig, self.ax = plt.subplots(**self.param.kwargs)

    def plot(self, map_sample: GridMapSample) -> Tuple[Figure, Axes]:
        """Plots the figure on the internal figure and axis.

        Args:
            map_sample: Sample containing grid map, goal, and goal position

        Returns:
            Figure and Axes objects
        """
        # Plot the base grid map
        plot_gridmap(self.ax, map_sample["map"], GridMapParam())

        # Add obstacles (grid map itself represents obstacles)
        obstacles = map_sample["map"]
        plot_obstacles(self.ax, obstacles, ObstaclesParam())

        # Add goal if provided
        goal_position = map_sample["goal_position"]
        plot_goal(self.ax, goal_position, GoalParam())

        # Use custom title if provided, otherwise use default
        title = self.title
        self.ax.set_title(title, fontsize=self.param.title_fontsize)

        return self.fig, self.ax

    def show(self) -> None:
        """Display the figure."""
        plt.show()

    def save(self, filepath: str, **kwargs) -> None:
        """Save the figure to a file.

        Args:
            filepath: Path where to save the figure
            **kwargs: Additional arguments for plt.savefig
        """
        self.fig.savefig(filepath, **kwargs)


class CompareMapsFigParam(CognitiveMapFigParam):
    """Parameters for comparing multiple cognitive maps."""

    n_cols: int = Field(default=2, description="Number of columns in the grid")
    titles: List[str] = Field(default=["Map 1", "Map 2"], description="Titles for each subplot")
    shared_colorbar: bool = Field(default=False, description="Whether to use a shared colorbar")
    colorbar_label: str = Field(default="Value", description="Label for the shared colorbar")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap (shared)")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap (shared)")


class CompareCognitiveMaps(CognitiveMapFigure):
    """Figure for comparing multiple cognitive maps."""

    def __init__(self, param: Optional[CompareMapsFigParam] = None):
        self.param = param or CompareMapsFigParam()
        self.title = "Cognitive Map Comparison"

        # Override the default figure/axes creation
        self.fig = plt.figure(**self.param.kwargs)
        self.axes = []

    def plot(
        self,
        maps: List[CognitiveMap],
        obstacles: Optional[ObstacleMap] = None,
        goal_position: Optional[Position] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot multiple maps for comparison.

        Args:
            maps: List of cognitive maps to compare
            obstacles: Optional obstacle map to overlay
            goal_position: Optional goal position to mark

        Returns:
            Figure and list of Axes objects
        """
        if not maps:
            return self.fig, []

        n_maps = len(maps)
        n_cols = min(self.param.n_cols, n_maps)
        n_rows = (n_maps + n_cols - 1) // n_cols  # Ceiling division

        # Create gridspec for the subplots
        self.axes = [self.fig.add_subplot(n_rows, n_cols, i + 1) for i in range(n_maps)]

        # Common parameters for all subplots
        grid_param = GridMapParam(
            cmap="viridis",
            vmin=self.param.vmin,
            vmax=self.param.vmax,
            show_colorbar=False,  # Never show individual colorbars in comparison view
        )

        # Plot each map
        for i, (ax, map_data) in enumerate(zip(self.axes, maps)):
            # Plot the map
            plot_gridmap(ax, map_data, grid_param)

            # Add obstacles if provided
            if obstacles is not None:
                plot_obstacles(ax, obstacles, ObstaclesParam())

            # Add goal if provided
            if goal_position is not None:
                plot_goal(ax, goal_position, GoalParam())

            # Set subplot title
            if i < len(self.param.titles):
                ax.set_title(self.param.titles[i])

        # Add a single shared colorbar
        norm = plt.Normalize(
            vmin=self.param.vmin if self.param.vmin is not None else maps[0].min(),
            vmax=self.param.vmax if self.param.vmax is not None else maps[0].max(),
        )
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.axes)
        if self.param.colorbar_label:
            cbar.set_label(self.param.colorbar_label)

        # Set the overall figure title
        self.fig.suptitle(self.title, fontsize=self.param.title_fontsize)

        # Apply tight layout if specified
        if self.param.tight_layout:
            self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

        return self.fig, self.axes


class ActivityVisualizationParam(CognitiveMapFigParam):
    """Parameters for neural activity visualization."""

    activity_cmap: str = Field(default="hot", description="Colormap for neural activity")
    activity_vmin: Optional[float] = Field(default=0.0, description="Minimum value for activity colormap")
    activity_vmax: Optional[float] = Field(default=None, description="Maximum value for activity colormap")
    show_colorbar: bool = Field(default=True, description="Whether to show colorbar")
    colorbar_label: str = Field(default="Activity", description="Label for the colorbar")


class NeuralActivityFigure(CognitiveMapFigure):
    """Figure for visualizing neural activity patterns."""

    def __init__(self, param: Optional[ActivityVisualizationParam] = None):
        self.param = param or ActivityVisualizationParam()
        self.title = "Neural Activity Pattern"
        self.fig, self.ax = plt.subplots(**self.param.kwargs)

    def plot(self, activity: NeuralActivity) -> Tuple[Figure, Axes]:
        """Plot neural activity pattern.

        Args:
            activity: Neural activity pattern to visualize

        Returns:
            Figure and Axes objects
        """
        # Convert to numpy if it's a tensor
        if isinstance(activity, torch.Tensor):
            activity_data = activity.detach().cpu().numpy()
        else:
            activity_data = np.array(activity)

        # Reshape if needed for better visualization
        if len(activity_data.shape) == 1:
            # Find a reasonable 2D shape for 1D activity
            n = len(activity_data)
            side = int(np.sqrt(n))
            # Pad with zeros if needed to make a square
            padded = np.zeros(side * side)
            padded[:n] = activity_data
            activity_data = padded.reshape(side, side)

        # Plot the activity
        im = self.ax.imshow(
            activity_data, cmap=self.param.activity_cmap, vmin=self.param.activity_vmin, vmax=self.param.activity_vmax
        )

        # Add colorbar if requested
        if self.param.show_colorbar:
            cbar = self.fig.colorbar(im, ax=self.ax)
            if self.param.colorbar_label:
                cbar.set_label(self.param.colorbar_label)

        # Use custom title if provided, otherwise use default
        title = self.title
        self.ax.set_title(title, fontsize=self.param.title_fontsize)

        # Remove ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        return self.fig, self.ax


if __name__ == "__main__":
    # Example usage of the visualization classes
    # Create a sample grid map
    grid_size = (10, 10)
    grid_map = torch.zeros(grid_size)

    # Add some obstacles
    grid_map[2:4, 3:7] = 1
    grid_map[6:8, 2:8] = 1

    # Create goal position
    goal_position = (9, 9)

    # Create a sample map
    map_sample = GridMapSample(
        map=grid_map,
        goal=torch.zeros(grid_size),  # Create a goal map with a single 1 at goal position
        goal_position=goal_position,
    )
    map_sample["goal"][goal_position[0], goal_position[1]] = 1

    # Basic cognitive map visualization
    basic_visualizer = CognitiveMapFigure()
    basic_visualizer.plot(map_sample)
    basic_visualizer.show()

    # Create two sample cognitive maps for comparison
    map1 = torch.zeros(grid_size)
    map1[0:5, 0:5] = torch.linspace(0, 1, 25).reshape(5, 5)

    map2 = torch.zeros(grid_size)
    map2[5:10, 5:10] = torch.linspace(0, 1, 25).reshape(5, 5)

    # Compare cognitive maps
    compare_param = CompareMapsFigParam(
        figsize=(12, 6),
        titles=["Map Region 1", "Map Region 2"],
        shared_colorbar=True,
        colorbar_label="Activation",
    )
    compare_visualizer = CompareCognitiveMaps(compare_param)
    compare_visualizer.plot([map1, map2], obstacles=grid_map, goal_position=goal_position)
    compare_visualizer.show()

    # Neural activity visualization
    activity = torch.rand(100)  # Random 1D activity pattern
    activity_visualizer = NeuralActivityFigure()
    activity_visualizer.plot(activity)
    activity_visualizer.show()
