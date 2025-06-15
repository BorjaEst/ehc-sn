from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ehc_sn import models, simulations
from ehc_sn.figures import settings
from ehc_sn.utils import grids


def visualize_neuron_actications(
    model: models.CANModule,
    config: Optional[settings.NeuronActivationConfig] = None,
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
    config = config or settings.NeuronActivationConfig()
    n_mec_modules = len(model.mec)  # Number of subplots needed
    hpc_activity = model.hpc.activations  # HPC place cell activity
    mec_activity = [grid.activations for grid in model.mec]

    # Determine grid layout based on whether item units are present
    fig = plt.figure(figsize=config.figsize)
    gs = GridSpec(nrows=2, ncols=n_mec_modules + 1, figure=fig)

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


def visualize_connectivity_matrices(
    model: models.CANModule,
    config: Optional[settings.ConnectivityMatricesConfig] = None,
) -> Figure:
    """Visualize the connection weights between components.

    Args:
        model: An ItemMemory model instance
        config: Optional configuration object with colormap, figure size, etc.

    Returns:
        Matplotlib figure with subplots weight matrix visualizations
    """
    # Use default config if none provided
    config = config or settings.ConnectivityMatricesConfig()
    n_mec_modules = len(model.mec)

    # Calculate layout
    n_rows = 2  # HPC->MEC and MEC->HPC
    n_cols = n_mec_modules

    # Create figure
    fig = plt.figure(figsize=config.figsize)
    gs = GridSpec(n_rows, n_cols + 1, figure=fig)

    # 1. Visualize MEC->HPC synapses (one for each MEC module)
    for i in range(n_mec_modules):
        ax = fig.add_subplot(gs[0, i])
        weights = model.hpc.synapses_mec[i].weight.detach().cpu().numpy()
        im = ax.imshow(weights, cmap=config.cmap_hpc, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"MEC{i+1}->HPC")
        ax.set_xlabel("MEC Grid Cell")
        ax.set_ylabel("HPC Place Cell")

    # 2. Visualize HPC->MEC synapses (one for each MEC module)
    for i in range(n_mec_modules):
        ax = fig.add_subplot(gs[1, i])
        weights = model.mec[i].synapses_hpc.weight.detach().cpu().numpy()
        im = ax.imshow(weights, cmap=config.cmap_mec, aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"HPC->MEC{i+1}")
        ax.set_xlabel("HPC Place Cell")
        ax.set_ylabel("MEC Grid Cell")

    # 3. Visualize EC->HPC synapses in the first row, last column
    ax_feat = fig.add_subplot(gs[0, -1])
    feat_weights = model.hpc.synapses_ec.weight.detach().cpu().numpy()
    im_feat = ax_feat.imshow(feat_weights, cmap=config.cmap_hpc, aspect="auto")
    plt.colorbar(im_feat, ax=ax_feat)
    ax_feat.set_title("EC->HPC")
    ax_feat.set_xlabel("EC Feature Cell")
    ax_feat.set_ylabel("HPC Place Cell")

    # 4. Show statistics for MEC recurrent connections in the last cell
    ax_stats = fig.add_subplot(gs[1, -1])
    ax_stats.axis("off")

    stats_text = "MEC Recurrent Connection Stats:\n\n"
    for i in range(n_mec_modules):
        weights = model.mec[i].synapses_rcc.weight.detach().cpu().numpy()
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        stats_text += f"MEC {i+1}:\n"
        stats_text += f"  Mean: {mean_w:.4f}\n"
        stats_text += f"  Std: {std_w:.4f}\n\n"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 verticalalignment='top', fontsize=10)  # fmt: skip

    plt.tight_layout()
    return fig


def visualize_memory_retrieval(
    simulation: simulations.SimulationResults,
    config: Optional[settings.MemoryRetrievalConfig] = None,
) -> Figure:
    """Visualize the dynamics of memory retrieval over iterations.

    Args:
        simulation: Simulation results containing HPC and MEC states
        config: Optional configuration object with visualization parameters

    Returns:
        Matplotlib figure with subplots showing the dynamics of retrieval
    """
    # Use default config if none provided
    config = config or settings.MemoryRetrievalConfig()
    hpc_states = simulation.hpc_states
    mec_states = simulation.mec_states

    # Determine number of MEC modules from the simulation data
    if mec_states and len(mec_states) > 0:
        n_mec_modules = len(mec_states[0])
    else:
        n_mec_modules = 0

    # Select iterations to visualize
    iter_indices = list(np.linspace(0, len(hpc_states) - 1, config.n_frames, dtype=int))

    # Create figure
    n_frames = len(iter_indices)
    n_cols = n_mec_modules + 1  # HPC + each MEC module
    fig, axs = plt.subplots(n_frames, n_cols, figsize=config.figsize)

    # Handle single row case
    if n_frames == 1:
        axs = axs.reshape(1, -1)

    # Plot each selected iteration
    for frame, iter_idx in enumerate(iter_indices):
        # Plot HPC activity
        ax = axs[frame, 0]
        hpc_data = hpc_states[iter_idx].detach().cpu().numpy().reshape(1, -1)
        sns.heatmap(hpc_data, ax=ax, cmap=config.cmap_hpc, cbar_kws={"label": "Act"})
        ax.set_title(f"Iteration {iter_idx+1}: HPC")
        ax.set_yticks([])

        # Plot each MEC module's grid cell activity
        for i in range(n_mec_modules):
            ax = axs[frame, i + 1]
            grid = grids.arrange_2d(mec_states[iter_idx][i])
            sns.heatmap(grid, ax=ax, cmap=config.cmap_mec, cbar_kws={"label": "Act"})
            ax.set_title(f"MEC Module {i+1}")

    plt.tight_layout()
    return fig
