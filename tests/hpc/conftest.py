import numpy as np
import pytest

from ehc_sn.hpc import HPCNetwork, PlaceCellsLayer


@pytest.fixture(scope="module")
def layer_gen():
    """Fixture providing a function to generate place cell layers."""

    def _generate_layer(n_cells=100, field_size=0.3, environment_size=(10.0, 10.0)):
        """Generate a place cell layer with given parameters."""
        return PlaceCellsLayer(n_cells, field_size, environment_size)

    return _generate_layer


@pytest.fixture(scope="module")
def network_gen(layer_gen):
    """Fixture providing a function to generate HPC test networks."""

    def _generate_network(place_cells=None, noise_level=0.1, memory_decay=0.05):
        """Generate a HPC network with place cells."""
        place_cells = place_cells or layer_gen()
        return HPCNetwork(place_cells, noise_level, memory_decay)

    return _generate_network
