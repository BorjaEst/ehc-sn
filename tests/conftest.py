import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a random seed for reproducibility in tests."""
    np.random.seed(42)


@pytest.fixture(scope="module")
def position_gen():
    """Fixture providing a function to generate test positions."""

    def _generate_positions(start, end, num_points=10):
        """Generate evenly spaced positions from start to end."""
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        return np.column_stack((x, y))

    return _generate_positions
