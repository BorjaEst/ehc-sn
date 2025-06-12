import pytest
import torch

import ehc_sn.nn
from ehc_sn.settings import config


class TestNetworkInitialization:
    """Test suite for the Network initialization and basic functionality."""

    def test_instance(self, model):
        """Test that the Network initializes correctly with given parameters."""
        assert isinstance(model, ehc_sn.nn.Network)

    def test_layers(self, model, model_parameters):
        """Test that the layers are initialized correctly."""
        assert len(model.layers) == len(model_parameters.layers)
        for layer_name, layer_params in model_parameters.layers.items():
            assert layer_name in model.layers
            assert isinstance(model.layers[layer_name], ehc_sn.nn.Layer)
            assert model.layers[layer_name].neurons.size == layer_params.neurons.size


def test_network_forward(model):
    """Test the forward pass through a simple network."""
    # Create sample input
    input_tensor = torch.ones((1, 3), device=config.device)
    # Forward pass through network
    output = model(input_tensor)
    assert output.shape == (1, model.layers["layer_2"].neurons.size)
