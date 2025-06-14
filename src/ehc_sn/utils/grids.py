from typing import Optional, Tuple

import numpy as np
import torch


def arrange_2d(activity: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Arrange 1D cell activities into a 2D grid for visualization.

    Args:
        activity: 1D tensor of cell activations
        grid_size: Optional (height, width) for the grid. If None, tries to create a square grid.

    Returns:
        2D numpy array of cell activities arranged in a grid
    """
    n_cells = activity.shape[-1] if len(activity.shape) > 1 else activity.shape[0]

    if grid_size is None:
        # Try to create a roughly square grid
        grid_width = int(np.ceil(np.sqrt(n_cells)))
        grid_height = int(np.ceil(n_cells / grid_width))
        grid_size = (grid_height, grid_width)

    # Create grid and fill with activities
    grid = np.zeros(grid_size)
    activity_np = activity.detach().cpu().numpy().flatten()

    for i in range(min(n_cells, grid_size[0] * grid_size[1])):
        row = i // grid_size[1]
        col = i % grid_size[1]
        grid[row, col] = activity_np[i]

    return grid


def optimal_grid_shape(n_items: int) -> Tuple[int, int]:
    """Calculate an optimal grid shape for displaying n items.

    Args:
        n_items: Number of items to arrange

    Returns:
        Tuple of (rows, columns)
    """
    # Try to get a square-ish grid
    cols = int(np.ceil(np.sqrt(n_items)))
    rows = int(np.ceil(n_items / cols))
    return rows, cols
