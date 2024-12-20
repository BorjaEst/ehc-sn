"""Sequential Navigation (SN) module for the Entorhinal–Hippocampal circuit (EHC)"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

# Type alias for integer space arrays
Observation = npt.NDArray[np.float32]  # Observation
Velocity = npt.NDArray[np.float32]  # Velocity
Item = npt.NDArray[np.float32]  # Navigation Item
Trajectory = npt.NDArray[np.float32]  # Navigation Trajectory
Map = npt.NDArray[np.float32]  # Cognitive Map
MapSet = dict[Map, float]  # Set of cognitive maps


@dataclass
class HGModelParams:
    """The param set class example for model configuration."""

    δ: float = 0.7  # Discount factor for trajectory
    τ: float = 0.9  # Exponential decay for mixing categorical distribution
    c: float = 0.4  # Velocity rate for item code update

    def validate(self):
        """Validate the parameters."""
        if not isinstance(self.δ, float) or not 0 <= self.δ <= 1:
            raise ValueError(f"δ: {self.δ}. Must be a float between 0 and 1.")
        if not isinstance(self.τ, float) or not 0 <= self.τ <= 1:
            raise ValueError(f"τ: {self.τ}. Must be a float between 0 and 1.")
        if not isinstance(self.c, float) or not 0 <= self.c <= 1:
            raise ValueError(f"c: {self.c}. Must be a float between 0 and 1.")


class BaseModel:
    """Base class for sequential navigation models."""

    def __init__(self, parameters: HGModelParams):
        if not isinstance(parameters, HGModelParams):
            raise TypeError("Expecting instance of HGModelParams.")
        parameters.validate()  # Validate the parameters
        self.parameters = parameters

    def get_trajectory(self, items: List[Item]) -> Trajectory:
        """Return the hidden code for trajectory."""
        T, δ = len(items), self.parameters.δ  # Extract parameters
        discounted = [x * δ ** (T - t) for t, x in enumerate(items, 1)]
        return np.array(discounted).sum(axis=0)  # Eq. (2)

    def p_trajectory(self, y: Trajectory, Θ: dict[Map, float]) -> float:
        """Return the probability of a trajectory."""
        p_dist = [self.likelihood(y, θ) * p for θ, p in Θ.items()]
        return np.array(p_dist).sum(axis=0)  # Eq. (3)

    def likelihood(self, y: Trajectory, θ: Map) -> float:
        """Calculate the likelihood of trajectory given a map."""
        lnpΘ = y @ np.log(θ)  # Eq. (5)
        # Note lnp(y|Θ_k) actually proportional to y·ln(θ_k)
        return np.exp(lnpΘ)


class HierarchicalGenerativeModel(BaseModel):
    """Model for representing hierarchical spatiotemporal data."""

    def __init__(self, shape: Tuple[int], parameters: HGModelParams):
        super().__init__(parameters)
        self.ξ: Observation = np.zeros(shape, dtype=np.float32)
        self.v: Velocity = np.zeros(shape, dtype=np.float32)

    def inference(
        self, x: Item, y: Trajectory, Θ: MapSet
    ) -> Tuple[Item, Trajectory, MapSet]:
        """Inference function."""
        Θ, θ = self.f_mixing(y, Θ)  # Update mixing distributions
        x, ξ = self.f_item(x, y, θ)  # Predict the item code and observation
        self.v, self.ξ = ξ - self.ξ, ξ  # Update v(t) and ξ(t)
        y = self.f_trajectory(y, θ, ξ)  # Update the trajectory
        return x, y, Θ

    def f_mixing(self, y: Trajectory, Θ: MapSet) -> Tuple[MapSet, Map]:
        """Estimate the posterior mixing probability distribution."""
        τ = self.parameters.τ  # Extract parameters
        Θ = {θ: z**τ * self.likelihood(y, θ) ** (1 - τ) for θ, z in Θ.items()}
        return Θ, max(Θ, key=Θ.__getitem__)  # Eq. (6) and Eq. (7)

    def f_item(self, x: Item, y: Trajectory, θ: Map) -> Tuple[Item, Observation]:
        """Estimate the posterior hidden item code."""
        c = self.parameters.c  # Extract parameters
        μ, Σ = self.ξ + c * self.v, self.v  # Using ξ(t-1) and v(t-1)
        x = np.random.normal(μ, Σ) * θ - y
        return x, np.argmax(x, axis=0)  # Eq. (8) and Eq. (9)

    def f_trajectory(self, y: Trajectory, θ: Map, ξ: Observation) -> Trajectory:
        """Estimate the posterior sequence code."""
        δ = self.parameters.δ  # Extract parameters
        return δ * y + (1 - δ) * θ + ξ  # Eq. (10)
