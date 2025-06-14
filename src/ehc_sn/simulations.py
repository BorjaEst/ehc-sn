from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

import ehc_sn.models


@dataclass
class SimulationResults:
    hpc_states: List[torch.Tensor] = field(default_factory=list)
    mec_states: List[List[torch.Tensor]] = field(default_factory=list)


class Simulation(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> SimulationResults:
        pass


class ItemMemoryRetrieval(Simulation):
    """
    Simulate memory retrieval for a given item using the ItemMemory model.
    This class encapsulates the retrieval process, iterating over the model
    for a specified number of iterations to collect HPC and MEC states.
    Attributes:
        model (ehc_sn.models.ItemMemory): The item memory model.
        item (torch.Tensor): The item to retrieve.
        iterations (int): Number of iterations for the simulation.
    """

    def __init__(
        self: "ItemMemoryRetrieval",
        model: ehc_sn.models.ItemMemory,
        item: torch.Tensor,
        iterations: int = 10,
    ):
        self.model = model
        self.item = item
        self.iterations = iterations

    def __call__(self, *args: Any, **kwds: Any) -> SimulationResults:
        #  Initialize model and results
        self.model.eval()
        results = SimulationResults()

        # Simulate retrieval process
        with torch.no_grad():
            for _ in range(self.iterations):
                self.model(self.item)  # Forward pass to get HPC and MEC states
                results.hpc_states.append(self.model.hpc.place_cells)
                results.mec_states.append([mec.grid_cells for mec in self.model.mec])

        return results
