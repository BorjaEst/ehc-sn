import torch

import ehc_sn.nn
from ehc_sn.settings import config


class TestModelInitialization:

    def test_instance(self, model):
        assert isinstance(model, ehc_sn.EHC_SN)

    def test_networks(self, model):
        assert isinstance(model.hpc, ehc_sn.nn.Network)
        assert isinstance(model.mec, ehc_sn.nn.Network)

    def test_hpc_layers(self, model):
        assert "place_cells" in model.hpc.layers
        assert isinstance(model.hpc.layers["place_cells"], ehc_sn.nn.Layer)

    def test_mec_layers(self, model):
        assert "grid_1" in model.mec.layers
        assert isinstance(model.mec.layers["grid_1"], ehc_sn.nn.Layer)
        assert "grid_2" in model.mec.layers
        assert isinstance(model.mec.layers["grid_2"], ehc_sn.nn.Layer)
        assert "grid_3" in model.mec.layers
        assert isinstance(model.mec.layers["grid_3"], ehc_sn.nn.Layer)


def test_model_forward(model):
    # Create sample input
    hpc_input = torch.ones((1, 3), device=config.device)
    mec_input = torch.ones((1, 3), device=config.device)
    # Forward pass through network
    output = model(hpc_input, mec_input)
