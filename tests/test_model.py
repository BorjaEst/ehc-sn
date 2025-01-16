"""Tests for the HierarchicalGenerativeModel class."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from ehc_sn import HGModelParams, HierarchicalGenerativeModel


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
def trajectory(request, size):
    """Return the trajectory input."""
    if hasattr(request, "param"):
        return request.param
    return np.random.rand(size)  # Random trajectory


@pytest.mark.parametrize("size", [10])
def test_instantiation(model):
    """Test instantiation of the HierarchicalGenerativeModel class."""
    assert model is not None


@pytest.mark.parametrize("size", [10])
def test_inference(model, ξ, x, y):
    """Test inference using the HierarchicalGenerativeModel class."""
    x, y = model.inference(ξ, x, y)
    assert x is not None
    assert y is not None
