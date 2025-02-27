"""Sequential Navigation (SN) for the Entorhinal–Hippocampal circuit (EHC)"""

from typing import Any, Optional, Tuple, TypeAlias

import numpy as np
from ehc_sn import equations as eq
from ehc_sn.config import GenSettings, HGMSettings, LearningSettings
from ehc_sn.equations import Item, Map, Mixing, Observation, Sequence
from ehc_sn.equations import get_item as item  # noqa: F401
from ehc_sn.equations import get_sequence as sequence  # noqa: F401
from numpy.typing import NDArray
from pydantic import InstanceOf, PositiveInt, validate_call

# pylint: disable=non-ascii-name
# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods


Episode: TypeAlias = list[Observation]  # List of observations
Episodes: TypeAlias = list[Episode]  # List of episodes
Experiment: TypeAlias = list[Episodes]  # List of list of episodes


class HierarchicalGenerativeModel:
    """Hierarchical generative model for sequential navigation."""

    @validate_call
    def __init__(
        self,
        α: list[float],  # Mixing hyperparameters
        N: PositiveInt,  # Number of items in the maps
        settings: Optional[HGMSettings] = None,
    ):
        self.settings = settings or HGMSettings()  # type: ignore
        ρh, size = np.ones(N), len(α)
        self.π = np.random.dirichlet(α)  # Mixing hyperparameter
        self.ρ = [np.random.dirichlet(ρh) for _ in range(size)]
        self.v = np.zeros(N)  # Velocity for the model

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the model (k, N)."""
        return len(self.π), len(self.ρ[0])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model for inference."""
        return inference(self, *args, **kwargs)

    def sample_maps(self) -> list[Map]:
        """Return a list of generated cognitive maps."""
        return [ρ / ρ.sum() for ρ in self.ρ]  # Cat(ρ)


@validate_call()
def inference(  # pylint: disable=too-many-arguments
    model: InstanceOf[HierarchicalGenerativeModel],
    ξ: Observation,
    y: Sequence,
    Θ: list[Map],
    z: Optional[Mixing] = None,
) -> Tuple[Item, Sequence, NDArray[np.float64], np.int64]:
    """Inference function, returns predicted next item and sequence."""
    δ, τ, c = model.settings.δ, model.settings.τ, model.settings.c

    # Get the mixing probabilities and the best map
    z = model.π if z is None else eq.mixing(y, Θ, z, τ)
    k = z.argmax()  # Convert to list and get best

    # Inference item and sequence
    x = eq.item(ξ, y, Θ[k], model.v, c)  # Predict item code
    model.v = eq.observation(x) - ξ  # Update the velocity
    y = eq.sequence(ξ, y, Θ[k], δ)  # Update the sequence

    # Return the predicted item, sequence, and mixing probabilities
    return x, y, z / z.sum(), k  # z ~ Cat(π)


@validate_call
def learning(  # pylint: disable=too-many-arguments
    model: InstanceOf[HierarchicalGenerativeModel],
    episodes: list[Episode],
    settings: Optional[LearningSettings] = None,
) -> list[Map]:
    """Learning function for the model."""
    settings = settings or LearningSettings()  # type: ignore
    γ, λ = settings.γ, settings.λ  # Get the learning settings
    n_clusters, N = model.shape  # Get the number of items

    # Train the model on the episode data
    p_maps = [np.zeros(N) for _ in range(n_clusters)]
    for episode in episodes:
        _train(model, episode, γ, λ, p_maps)  # Train the model

    # Return the updated priority maps
    return p_maps


def _train(model, episode, γ, λ, p_maps):
    Θ, N = model.sample_maps(), model.shape[1]
    x, y = np.zeros(N), np.zeros(N)  # First item and sequence
    model.v = np.zeros(N)  # Reset the velocity
    for ξ in episode:
        x, y, z, k = inference(model, ξ, y, Θ)
        model.π[k] = eq.π_update(model.π[k], z[k], γ)
        model.ρ[k] = eq.ρ_update(model.ρ[k], z[k], y)
        p_maps[k] = eq.pmap_update(p_maps[k], ξ, x, λ)


@validate_call
def baseline(
    experiment: Experiment,
    α: list[float],
    N: int,
    settings: Optional[GenSettings] = None,
) -> InstanceOf[HierarchicalGenerativeModel]:
    """Baseline procedure for the model."""
    settings = settings or GenSettings()  # type: ignore
    model = HierarchicalGenerativeModel(α, N, settings)

    # Train the model on the experiment data
    for episode in experiment:
        _ = learning(model, episode, settings)

    # Return the trained model
    return model
