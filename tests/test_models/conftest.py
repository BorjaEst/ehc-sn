import pytest
import tomli

import ehc_sn.models


@pytest.fixture(scope="module", params=["model_1"])
def model_data(request):
    """Fixture providing model parameters from a TOML file."""
    return ehc_sn.models.load_model(f"tests/config/{request.param}.toml")


@pytest.fixture()
def model(model_data):
    """Fixture providing a simple network instance."""
    return ehc_sn.models.Network(model_data)
