import pytest
import tomli

import ehc_sn


@pytest.fixture(scope="module", params=["config_1"])
def parameters(request):
    """Fixture providing model parameters from a TOML file."""
    with open(f"tests/test_models/{request.param}.toml", "rb") as f:
        data = tomli.load(f)
    return ehc_sn.parameters.Model.model_validate(data)


@pytest.fixture()
def model(parameters):
    """Fixture providing a simple network instance."""
    return ehc_sn.EHC_SN(parameters)
