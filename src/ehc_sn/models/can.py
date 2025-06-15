from typing import Optional

import torch
from torch import jit, nn

from ehc_sn.core import neurons, synapses
from ehc_sn.utils import methods


class GridCells(nn.Module):
    """Medial Entorhinal Cortex Grid Cells with attractor dynamics"""

    def __init__(self, g: torch.Tensor, h: torch.Tensor):
        super().__init__()

        # Register a persistent tensor to hold the activations
        self.neurons = neurons.ReLU(g.shape[1])  # Initialize grid cell neurons
        self.synapses_hpc = synapses.Silent(methods.w_gh(g, h))  # (3)
        self.synapses_rcc = synapses.Silent(methods.w_gg(g))  # (3*)

    def forward(self, place_cells: neurons.Layer) -> torch.Tensor:
        # Forward pass through the MEC grid cells
        currents = []  # List to collect currents from synapses
        for task in [
            jit.fork(self.synapses_hpc, place_cells.activations),  # Current from HPC to MEC
            jit.fork(self.synapses_rcc, self.activations),  # Recurrent current
        ]:
            currents.append(jit.wait(task))  # Wait for all tasks to complete

        # Run the grid cell neurons with the summed currents and return the activations
        return self.neurons(sum(currents))

    @property
    def activations(self) -> torch.Tensor:
        """Get the current activations of the grid cells."""
        return self.neurons.activations


class PlaceCells(nn.Module):
    """Hippocampal Place Cells with recurrent connections"""

    def __init__(self, n: int, mec_dims: list[int], ec_dim: int):
        super().__init__()

        # Linear layer to compute place cell activations
        self.neurons = neurons.ReLU(n)  # Initialize place cell neurons
        self.synapses_mec = nn.ModuleList([synapses.Silent.normal(m, n, bias=True) for m in mec_dims])
        self.synapses_ec = synapses.Hybrid.normal(ec_dim, n)  # Synapses from EC to HPC

    def forward(self, ec_activations: torch.Tensor, mec: list[neurons.Layer]) -> torch.Tensor:
        # Compute place cell activations from features and MEC grid cells
        currents = []  # List to collect currents from synapses
        for taks in [
            jit.fork(self.synapses_ec, ec_activations),  # Current from EC to HPC
            *[jit.fork(synp, grid.activations) for synp, grid in zip(self.synapses_mec, mec)],
        ]:
            currents.append(jit.wait(taks))

        # Run the place cell neurons with the summed currents and return the activations
        return self.neurons(sum(currents))

    @property
    def activations(self) -> torch.Tensor:
        """Get the current activations of the grid cells."""
        return self.neurons.activations


class CANModule(nn.Module):
    """Cortical Attractor Network (CAN) for item memory"""

    def __init__(self, h: torch.Tensor, gx: list[torch.Tensor], ec_dim: int):
        super().__init__()
        mec_dims = [g.shape[1] for g in gx]  # (n_attractors, n_grid_cells)
        hpc_dim = h.shape[1]  # (n_attractors, n_hpc_cells)

        # Initialize hippocampal and MEC modules
        self.mec = nn.ModuleList([GridCells(g, h) for g in gx])
        self.hpc = PlaceCells(hpc_dim, mec_dims, ec_dim)

    def forward(self, ec_activations: torch.Tensor) -> torch.Tensor:
        # Forward pass through the hippocampal module
        for task in [
            jit.fork(self.hpc, ec_activations, self.mec),  # (2)
            *[jit.fork(grid, self.hpc) for grid in self.mec],  # (1 and 4)
        ]:
            jit.wait(task)

        # Return hippocampal activations
        return self.hpc.activations
