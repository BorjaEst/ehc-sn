"""
Tests for the neural network models in the ehc-sn package.

This module contains tests for model initialization, layer creation, connection setup,
and forward passes of neural networks for spatial navigation.
"""

from collections import namedtuple

import pytest
import torch

import ehc_sn.models
from ehc_sn.models import Layer, Network
from ehc_sn.settings import config

# Define test data structures for better readability
ModelConfig = namedtuple("ModelConfig", ["name", "path"])
LayerTest = namedtuple("LayerTest", ["name", "type"])
ConnectionTest = namedtuple("ConnectionTest", ["source", "target", "synapse"])

# Define test configurations
TEST_MODELS = [ModelConfig(name="model_1", path="tests/config/model_1.toml")]
TEST_NEURAL_LAYERS = [LayerTest("hpc-place_cells", "neural"), LayerTest("mec-grid_1", "neural")]
TEST_INPUT_LAYERS = [LayerTest("visual_stimulus", "input"), LayerTest("velocity", "input")]
TEST_CONNECTIONS = [
    ConnectionTest("hpc-place_cells", "mec-grid_1", "hybrid"),
    ConnectionTest("mec-grid_1", "hpc-place_cells", "silent"),
    ConnectionTest("velocity", "mec-grid_1", "ampa"),
]


@pytest.fixture(scope="module")
def model_config():
    """Load the model configuration from TOML file.

    This fixture provides the parsed configuration data for the test model,
    which can be reused across multiple tests.
    """
    return ehc_sn.models.load_model(TEST_MODELS[0].path)


@pytest.fixture(scope="module")
def model(model_config):
    """Create a neural network model instance.

    This fixture instantiates the model once per test module
    for efficiency, since model creation is expensive.
    """
    return ehc_sn.models.Network(model_config)


class TestModelStructure:
    """Test suite verifying the correct structure of loaded models."""

    def test_model_instance(self, model):
        """Verify the model is instantiated as the correct type."""
        assert isinstance(model, Network), "Model should be an instance of Network class"

    @pytest.mark.parametrize("layer", TEST_NEURAL_LAYERS)
    def test_neural_layers(self, model, layer):
        """Verify neural layers are properly created."""
        assert layer.name in model.layers, f"Expected neural layer '{layer.name}' not found in model"
        assert isinstance(model.layers[layer.name], Layer), f"Layer '{layer.name}' is not a Layer instance"

    @pytest.mark.parametrize("layer", TEST_INPUT_LAYERS)
    def test_input_layers(self, model, layer):
        """Verify input layers are properly created."""
        assert layer.name in model.layers, f"Expected input layer '{layer.name}' not found in model"
        assert isinstance(model.layers[layer.name], Layer), f"Layer '{layer.name}' is not a Layer instance"

    @pytest.mark.parametrize("conn", TEST_CONNECTIONS)
    def test_connections(self, model, conn):
        """Verify synaptic connections are properly configured."""
        # Check that target layer has links
        assert conn.target in model.links, f"Target layer '{conn.target}' has no links"

        # Check that synapse type exists
        link = model.links[conn.target]
        assert hasattr(link, conn.synapse), f"Synapse type '{conn.synapse}' not found in target '{conn.target}'"

        # Check that source is connected
        synapse = getattr(link, conn.synapse)
        assert hasattr(
            synapse, conn.source
        ), f"Source '{conn.source}' not connected to '{conn.target}' via '{conn.synapse}'"


class TestModelFunctionality:
    """Test suite verifying the functional behavior of the model."""

    def test_forward_pass(self, model):
        """Test the forward pass with sample inputs."""
        # Create test inputs
        inputs = {
            "visual_stimulus": torch.ones((1, 4), device=config.device),
            "velocity": torch.ones((1, 2), device=config.device),
        }

        # Run forward pass
        output = model(inputs)

        # Verify output structure
        assert isinstance(output, dict), "Model output should be a dictionary"

        # Verify expected output layers
        expected_outputs = ["hpc-place_cells", "mec-grid_1", "mec-grid_2", "mec-grid_3"]
        for layer_name in expected_outputs:
            assert layer_name in output, f"Expected output layer '{layer_name}' not found in model results"
            assert isinstance(output[layer_name], torch.Tensor), f"Output for '{layer_name}' should be a tensor"
            assert output[layer_name].shape[0] == 1, f"Output batch size for '{layer_name}' should be 1"

    def test_output_shapes(self, model):
        """Verify output tensor shapes match expected layer sizes."""
        # Create test inputs
        inputs = {
            "visual_stimulus": torch.ones((1, 4), device=config.device),
            "velocity": torch.ones((1, 2), device=config.device),
        }

        # Run forward pass
        output = model(inputs)

        # Check that shapes match the expected neuron counts
        for layer_name, layer_output in output.items():
            expected_size = model.layers[layer_name].neurons.size
            assert layer_output.shape[1] == expected_size, f"Output size for '{layer_name}' does not match layer size"
