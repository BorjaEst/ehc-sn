import numpy as np
import pytest
import tomli

from ehc_sn import hpc, parameters


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a random seed for reproducibility in tests."""
    np.random.seed(42)


@pytest.fixture(scope="module", params=["model_1"])
def model_parameters(request):
    """Fixture providing model parameters from a TOML file."""
    with open(f"tests/config/{request.param}.toml", "rb") as f:
        data = tomli.load(f)
    return parameters.Network.model_validate(data)


@pytest.fixture()
def model(model_parameters):
    """Fixture providing a simple network instance."""
    return hpc.Network(model_parameters)
