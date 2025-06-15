from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

import ehc_sn.models


@dataclass
class SimulationResults:
    hpc_states: List[torch.Tensor] = field(default_factory=list)
    mec_states: List[List[torch.Tensor]] = field(default_factory=list)

    def append(self, model: ehc_sn.models.CANModule) -> None:
        self.hpc_states.append(model.hpc.activations.clone())
        self.mec_states.append([grid.activations.clone() for grid in model.mec])


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
        model: ehc_sn.models.CANModule,
        ec_activations: torch.Tensor,
        iterations: int = 10,
    ):
        self.model = model
        self.ec_activations = ec_activations
        self.iterations = iterations

    def __call__(self, *args: Any, **kwds: Any) -> SimulationResults:
        #  Initialize model and results
        self.model.eval()
        results = SimulationResults()

        # Simulate retrieval process
        with torch.no_grad():
            for _ in range(self.iterations):
                self.model(self.ec_activations)  # Forward pass
                results.append(self.model)  # Store the states

        return results
