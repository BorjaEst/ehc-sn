"""Tests for the HierarchicalGenerativeModel class."""

# pylint: disable=redefined-outer-name

import pytest

from ehc_sn import HGModelParams, HierarchicalGenerativeModel


@pytest.fixture(scope="module")
def alpha(request):
    """Return the Dirichlet hyperparameters for mixing initialization."""
    return request.param if hasattr(request, "param") else [0.1, 0.1]


@pytest.fixture(scope="module")
def shape(request):
    """Return the shape for the HierarchicalGenerativeModel."""
    return request.param if hasattr(request, "param") else (10,)


@pytest.fixture(scope="module")
def parameters():
    """Return the parameters for the HierarchicalGenerativeModel."""
    return HGModelParams(δ=0.5, τ=0.5, c=0.5)


@pytest.fixture(scope="module")
def model(alpha, shape, parameters):
    """Return the HierarchicalGenerativeModel instance."""
    return HierarchicalGenerativeModel(alpha, shape, parameters)


@pytest.mark.parametrize("shape", [(10,), (20,)], indirect=True)
class Test1DInstance:
    """Test the HierarchicalGenerativeModel class for 1D data"""

    def test_instantiation(self, model):
        """Test instantiation of the HierarchicalGenerativeModel class."""
        assert model is not None

    def test_inference(self, model):
        """Test inference using the HierarchicalGenerativeModel class."""
        x, y = model.inference(0, [0, 1, 2, 3, 4])
        assert x is not None
        assert y is not None


@pytest.mark.parametrize("shape", [(10, 6)], indirect=True)
class Test2DInstance:
    """Test the HierarchicalGenerativeModel class for 1D data"""

    def test_instantiation(self, model):
        """Test instantiation of the HierarchicalGenerativeModel class."""
        assert model is not None

    def test_inference(self, model):
        """Test inference using the HierarchicalGenerativeModel class."""
        x, y = model.inference(0, [0, 1, 2, 3, 4])
        assert x is not None
        assert y is not None
