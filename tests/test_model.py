"""Tests for the HierarchicalGenerativeModel class."""

from ehc_sn import HierarchicalGenerativeModel


def test_instantiation():
    """Test instantiation of the HierarchicalGenerativeModel class."""
    model = HierarchicalGenerativeModel(observation_size=32)
    assert model is not None
