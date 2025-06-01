import numpy as np
import pytest

from ehc_sn.mec import GridCellsLayer, MECNetwork


@pytest.fixture(scope="module", params=[0.2])
def scale(request):
    """Fixture providing different grid cell scales."""
    return request.param


@pytest.fixture
def layer(scale):
    """Fixture providing a default grid cell layer."""
    return GridCellsLayer(width=10, height=10, spacing=scale, orientation=0.0)


@pytest.fixture
def network():
    """Fixture providing an MEC network with different grid cell scales."""
    return MECNetwork(
        [
            GridCellsLayer(width=10, height=10, spacing=scale, orientation=0.0)
            for scale in [0.2, 0.5, 1.0]
        ]
    )


@pytest.fixture
def position_gen():
    """Fixture providing a function to generate test positions."""

    def _generate_positions(start, end, num_points):
        """Generate evenly spaced positions from start to end."""
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        return np.column_stack((x, y))

    return _generate_positions
