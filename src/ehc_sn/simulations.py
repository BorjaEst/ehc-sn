from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

import torch

import ehc_sn.models


@dataclass
class SimulationResults:
    """
    Container for holding the simulation results.

    Attributes:
        hpc_states (List[torch.Tensor]): List storing the HPC activations per iteration.
        mec_states (List[List[torch.Tensor]]): List of lists storing the MEC grid activations per iteration.
    """

    hpc_states: List[torch.Tensor] = field(default_factory=list)
    mec_states: List[List[torch.Tensor]] = field(default_factory=list)

    def append(self, model: ehc_sn.models.CANModule) -> None:
        """
        Append the current state of the model to the results.

        Args:
            model (ehc_sn.models.CANModule): The model instance containing the current states.
        """
        self.hpc_states.append(model.hpc.activations.clone())
        self.mec_states.append([grid.activations.clone() for grid in model.mec])


class Simulation(ABC):
    """
    Abstract base class for simulations.

    Subclasses should implement the __call__ method to define the simulation behavior.
    """

    @abstractmethod
    def __call__(self, n_iterations: int, **kwds: Any) -> SimulationResults:
        """
        Run the simulation for a specified number of iterations.

        Args:
            n_iterations (int): Number of simulation iterations.
            **kwds (Any): Additional keyword arguments for simulation parameters.

        Returns:
            SimulationResults: The aggregated simulation results containing HPC and MEC states.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Attractor(Simulation):
    """
    Simulation implementation based on attractor dynamics.

    This simulation class uses an input activation (e.g., from the entorhinal cortex)
    to drive the CAN model and record the states over multiple iterations.
    """

    def __init__(
        self,
        model: ehc_sn.models.CANModule,
        ec_activations: torch.Tensor,
    ) -> None:
        """
        Initialize the attractor simulation.

        Args:
            model (ehc_sn.models.CANModule): The CAN model instance to simulate.
            ec_activations (torch.Tensor): The input activations for the entorhinal cortex.
        """
        self.model = model
        self.ec_activations = ec_activations

    def __call__(self, n_iterations: int, **kwds: Any) -> SimulationResults:
        """
        Run the attractor simulation for a given number of iterations.

        The simulation sets the model to evaluation mode and processes the input
        activations over each iteration, collecting neural states along the way.

        Args:
            n_iterations (int): The number of iterations to run the simulation.
            **kwds (Any): Additional keyword arguments (currently unused).

        Returns:
            SimulationResults: A container with HPC and MEC states recorded over
            the simulation iterations.
        """
        self.model.eval()
        results = SimulationResults()

        # Run simulation without computing gradients for efficiency.
        with torch.no_grad():
            for _ in range(n_iterations):
                self.model(self.ec_activations)  # Forward pass of the model.
                results.append(self.model)  # Record the current state.

        return results
