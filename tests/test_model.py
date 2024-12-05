"""Tests for the HierarchicalGenerativeModel class."""

# pylint: disable=redefined-outer-name

import pytest

from ehc_sn import HGModelParams, HierarchicalGenerativeModel


@pytest.fixture(scope="module")
def shape(request):
    """Return the shape for the HierarchicalGenerativeModel."""
    return request.param if hasattr(request, "param") else (10,)


@pytest.fixture(scope="module")
def parameters():
    """Return the parameters for the HierarchicalGenerativeModel."""
    return HGModelParams(δ=0.5, τ=0.5, c=0.5)


@pytest.mark.parametrize("shape", [(10,), (20,)], indirect=True)
def test_1d_instantiation(shape, parameters):
    """Test instantiation of the HierarchicalGenerativeModel class."""
    model = HierarchicalGenerativeModel(shape, parameters)
    assert model is not None


@pytest.mark.parametrize("shape", [(10, 6)], indirect=True)
def test_2d_instantiation(shape, parameters):
    """Test instantiation of the HierarchicalGenerativeModel class."""
    model = HierarchicalGenerativeModel(shape, parameters)
    assert model is not None
