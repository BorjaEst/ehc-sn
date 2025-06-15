from typing import Optional

import torch
from torch import jit, nn

from ehc_sn.core import synapses
from ehc_sn.utils import methods

# TODO: Add bias from methods


class MECGrid(nn.Module):
    """Medial Entorhinal Cortex Grid Cells with attractor dynamics"""

    def __init__(self, g: torch.Tensor, h: torch.Tensor):
        super().__init__()

        # Register a persistent tensor to hold the activations
        self.register_buffer("grid_cells", torch.zeros(1, g.shape[1]))
        self.synapses_hpc = synapses.FixedNonRandom(methods.w_gh(g, h))  # (3)
        self.synapses_rcc = synapses.FixedNonRandom(methods.w_gg(g))  # (3*)

        # Apply additional nonlinearity and normalization for grid cells
        self.nonlinearity = nn.ReLU()

    def forward(self, hpc_activations: torch.Tensor) -> torch.Tensor:
        # Forward pass through the MEC grid cells
        currents = []  # List to collect currents from synapses
        for task in [
            jit.fork(self.synapses_hpc, hpc_activations),  # Current from HPC to MEC
            jit.fork(self.synapses_rcc, self.grid_cells),  # Recurrent current
        ]:
            currents.append(jit.wait(task))  # Wait for all tasks to complete

        # Apply attractor dynamics with gain parameter
        self.grid_cells = self.grid_cells + sum(currents)
        self.grid_cells = self.nonlinearity(self.grid_cells)  # Bounded activation

        # Return the updated grid cell activations
        return self.grid_cells


class HPCGrid(nn.Module):
    """Hippocampal Place Cells with recurrent connections"""

    def __init__(self, hpc_dim: int, mec_dims: list[int], ec_dim: int):
        super().__init__()

        # Linear layer to compute place cell activations
        self.register_buffer("place_cells", torch.zeros(1, hpc_dim))
        self.synapses_mec = nn.ModuleList([synapses.FixedRandom(n, hpc_dim) for n in mec_dims])
        self.synapses_ec = synapses.Plastic(ec_dim, hpc_dim)  # Synapses from EC to HPC

        # Apply additional nonlinearity and normalization for place fields
        self.nonlinearity = nn.ReLU()

    def forward(self, ec_activations: torch.Tensor, mec_activations: list[torch.Tensor]) -> torch.Tensor:
        # Compute place cell activations from features and MEC grid cells
        currents = []  # List to collect currents from synapses
        for taks in [
            jit.fork(self.synapses_ec, ec_activations),  # Current from EC to HPC
            *[jit.fork(*x) for x in zip(self.synapses_mec, mec_activations)],
        ]:
            currents.append(jit.wait(taks))

        # Update place cell activations with recurrent connections
        self.place_cells = self.place_cells + sum(currents)
        self.place_cells = self.nonlinearity(self.place_cells)

        # Return the updated place cell activations
        return self.place_cells


class CANModule(nn.Module):
    """Cortical Attractor Network (CAN) for item memory"""

    def __init__(self, h: torch.Tensor, gx: list[torch.Tensor], ec_dim: int):
        super().__init__()
        hpc_dim = h.shape[1]  # (n_attractors, n_hpc_cells)
        mec_dims = [g.shape[1] for g in gx]  # (n_attractors, n_grid_cells)

        # Initialize hippocampal and MEC modules
        self.mec = nn.ModuleList([MECGrid(g, h) for g in gx])
        self.hpc = HPCGrid(hpc_dim, mec_dims, ec_dim)

    def forward(self, ec_activations: torch.Tensor) -> torch.Tensor:
        # Forward pass through the hippocampal module
        mec_activations = [mec.grid_cells for mec in self.mec]  # Collect MEC activations
        for task in [
            jit.fork(self.hpc, ec_activations, mec_activations),  # (2)
            *[jit.fork(grid, self.hpc.place_cells) for grid in self.mec],  # (1 and 4)
        ]:
            jit.wait(task)

        # Return hippocampal activations
        return self.hpc.place_cells
