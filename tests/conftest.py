import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a random seed for reproducibility in tests."""
    np.random.seed(42)
