import pytest
import torch

from ehc_sn.models import Layer, Network
from ehc_sn.settings import config


@pytest.mark.parametrize("model_data", ["model_1"], indirect=True)
class TestModelInitialization:

    def test_instance(self, model):
        assert isinstance(model, Network)

    @pytest.mark.parametrize("layer_name", ["hpc-place_cells", "mec-grid_1"])
    def test_layer(self, model, layer_name):
        assert layer_name in model.layers
        assert isinstance(model.layers[layer_name], Layer)

    @pytest.mark.parametrize("layer_name", ["visual_stimulus", "velocity"])
    def test_inputs(self, model, layer_name):
        assert layer_name in model.layers
        assert isinstance(model.layers[layer_name], Layer)

    @pytest.mark.parametrize("connection", [
        {"source": "hpc-place_cells", "target": "mec-grid_1", "synapse": "hybrid"},
        {"source": "mec-grid_1", "target": "hpc-place_cells", "synapse": "silent"},
        {"source": "velocity", "target": "mec-grid_1", "synapse": "ampa"},
    ])  # fmt: skip
    def test_connections(self, model, connection):
        assert connection["target"] in model.links
        link = model.links[connection["target"]]
        assert hasattr(link, connection["synapse"])
        synapse = getattr(link, connection["synapse"])
        assert any(connection["source"] == c.source for c in synapse.connections)


@pytest.mark.parametrize("model_data", ["model_1"], indirect=True)
def test_model_forward(model):
    """Test the forward pass of the model with sample inputs."""
    output = model(
        {
            "visual_stimulus": torch.ones((1, 4), device=config.device),
            "velocity": torch.ones((1, 2), device=config.device),
        }
    )
    assert isinstance(output, dict)
    assert "hpc-place_cells" in output
    assert "mec-grid_1" in output
    assert "mec-grid_2" in output
    assert "mec-grid_3" in output
