from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ehc_sn.figures import settings
from ehc_sn.models.item_memory import ItemMemory
from ehc_sn.utils import grids


def visualize_item_memory_state(
    hpc_activity: torch.Tensor,
    mec_activity: List[torch.Tensor],
    config: Optional[settings.ItemMemoryStateConfig] = None,
) -> Figure:
    """Visualize the current state of an ItemMemory model.

    Args:
        hpc_activity: Tensor representing hippocampal place cell activity
        mec_activity: List of tensors representing MEC grid cell activity for each module
        config: Optional configuration object with colormap and figure size

    Returns:
        Matplotlib figure with subplots showing hippocampal and grid cell activity
    """
    # Use default config if none provided
    config = config or settings.ItemMemoryStateConfig()
    n_mec_modules = len(mec_activity)  # Number of subplots needed

    # Determine grid layout based on whether item units are present
    fig = plt.figure(figsize=config.figsize)
    rows = 2
    gs = GridSpec(rows, n_mec_modules + 1, figure=fig)

    # Plot hippocampal place cell activity
    ax_hpc = fig.add_subplot(gs[0, :])
    activity_np = hpc_activity.detach().cpu().numpy()
    im_hpc = ax_hpc.imshow(activity_np.reshape(1, -1), aspect="auto", cmap=config.cmap_hpc)
    plt.colorbar(im_hpc, ax=ax_hpc, label="Activation")
    ax_hpc.set_title("Hippocampal Place Cell Activity")
    ax_hpc.set_xlabel("Cell Index")
    ax_hpc.set_yticks([])

    # Plot each MEC module's grid cell activity
    for i, grid_activity in enumerate(mec_activity):
        ax_mec = fig.add_subplot(gs[1, i])
        grid = grids.arrange_2d(grid_activity, None)
        im_mec = ax_mec.imshow(grid, interpolation="nearest", cmap=config.cmap_mec)
        plt.colorbar(im_mec, ax=ax_mec, label="Activation")
        ax_mec.set_title(f"MEC Module {i+1}")
        ax_mec.set_xlabel("X position")
        ax_mec.set_ylabel("Y position")

    # Add a text box with summary statistics
    stats_row = 1
    ax_stats = fig.add_subplot(gs[stats_row, -1])
    ax_stats.axis("off")

    hpc_stats = f"HPC: mean={hpc_activity.mean().item():.3f}, std={hpc_activity.std().item():.3f}"
    mec_stats = ""
    for i, grid_activity in enumerate(mec_activity):
        mec_stats += f"MEC {i+1}: mean={grid_activity.mean().item():.3f}, std={grid_activity.std().item():.3f}\n"

    ax_stats.text(0.05, 0.95, "Activity Statistics:", fontsize=12, fontweight="bold", va="top")
    ax_stats.text(0.05, 0.85, hpc_stats, fontsize=10, va="top")
    ax_stats.text(0.05, 0.75, mec_stats, fontsize=10, va="top")

    plt.tight_layout()
    return fig
