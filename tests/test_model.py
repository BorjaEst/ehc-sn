"""Tests for the HierarchicalGenerativeModel class."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from ehc_sn import HGModelParams, HierarchicalGenerativeModel
from ehc_sn.utils import CognitiveMap


@pytest.fixture(scope="module")
def alpha(request):
    """Return the Dirichlet hyperparameters for mixing initialization."""
    if hasattr(request, "param"):
        return request.param
    k = 3  # Number of clusters for the Dirichlet distribution
    return np.random.rand(k)


@pytest.fixture(scope="module")
def parameters():
    """Return the parameters for the HierarchicalGenerativeModel."""
    return HGModelParams(δ=0.5, τ=0.5, c=0.5)


@pytest.fixture(scope="function")
def model(alpha, parameters, size):
    """Return the HierarchicalGenerativeModel instance."""
    return HierarchicalGenerativeModel(alpha, size, parameters)


@pytest.fixture(scope="function", name="ξ")
def observation(request, size):
    """Return the observation input."""
    if hasattr(request, "param"):
        return request.param
    default = np.zeros(size, dtype=np.float32)  # Initialize with zeros
    default[np.random.randint(size)] = 1.0  # Set random position to 1.0
    return default


@pytest.fixture(scope="function", name="x")
def item(request, size):
    """Return the item input."""
    if hasattr(request, "param"):
        return request.param
    return np.random.rand(size)  # Random item code


@pytest.fixture(scope="function", name="y")
def sequence(request, size):
    """Return the sequence input."""
    if hasattr(request, "param"):
        return request.param
    return np.random.rand(size)  # Random sequence


@pytest.fixture(scope="function", name="Θ")
def cognitive_maps(request, alpha, size):
    """Return the cognitive maps."""
    if hasattr(request, "param"):
        return request.param
    k, N = len(alpha), size
    return [CognitiveMap(θ=np.random.rand(N)) for _ in range(k)]


@pytest.mark.parametrize("size", [10])
def test_instantiation(model):
    """Test instantiation of the HierarchicalGenerativeModel class."""
    assert model is not None


@pytest.mark.parametrize("size", [10])
def test_inference(model, size, ξ, x, y, Θ):
    """Test inference using the HierarchicalGenerativeModel class."""
    x, y, z, k = model.inference(ξ, x, y, Θ, z=None)
    assert isinstance(x, np.ndarray) and x.shape == (size,)
    assert isinstance(y, np.ndarray) and y.shape == (size,)
    assert isinstance(z, np.ndarray) and z.shape == (len(Θ),)
    assert isinstance(k, np.int64) and 0 <= k < len(Θ)


@pytest.mark.parametrize("episode", [np.random.rand(2, 6, 10)])
@pytest.mark.parametrize("size", [10])
def test_learning(model, episode, size, alpha):
    """Test learning using the HierarchicalGenerativeModel class."""
    Θ = model.learning(episode)
    assert isinstance(model.π, np.ndarray) and model.π.shape == (len(alpha),)
    assert isinstance(model.ρ, list)
    assert all(isinstance(ρ_k, np.ndarray) for ρ_k in model.ρ)
    assert all(p_k.shape == (size,) for p_k in model.ρ)
    assert isinstance(Θ, list)
    assert all(isinstance(Θ_k, CognitiveMap) for Θ_k in Θ)
