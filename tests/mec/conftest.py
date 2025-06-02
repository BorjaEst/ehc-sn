import numpy as np
import pytest

from ehc_sn.mec import GridCellsLayer, MECNetwork


@pytest.fixture(scope="module")
def layer_gen():
    """Fixture providing a function to generate grid cell layers with a given scale."""

    def _generate_layer(width=10, height=10, spacing=0.1, orientation=0.0):
        """Fixture providing a default grid cell layer."""
        return GridCellsLayer(width, height, spacing, orientation)

    return _generate_layer


@pytest.fixture(scope="module")
def network_gen(layer_gen):
    """Fixture providing a function to generate MEC test networks."""
    layers = [layer_gen(spacing=x) for x in [0.2, 0.4, 0.8]]

    def _generate_network(layers=layers, noise_level=0.1):
        """Generate a MEC network with multiple grid cell layers."""
        return MECNetwork(layers, noise_level)

    return _generate_network


@pytest.fixture(scope="module")
def position_gen():
    """Fixture providing a function to generate test positions."""

    def _generate_positions(start, end, num_points=10):
        """Generate evenly spaced positions from start to end."""
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        return np.column_stack((x, y))

    return _generate_positions
