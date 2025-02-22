"""Tests for the utility functions."""

import numpy as np
import pytest
from ehc_sn.utils import kronecker_delta
from numpy.testing import assert_allclose

# pylint: disable=non-ascii-name


@pytest.mark.parametrize(
    "ξ_index, x, desired",
    [
        (np.array([1, 0, 0], dtype=bool), [1, 2, np.inf], [1, 0, 0]),
        (np.array([1, 1, 0], dtype=bool), [1, 2, np.inf], [1, 2, 0]),
        (np.array([1, 0, 1], dtype=bool), [1, 2, np.inf], [1, 0, np.inf]),
    ],
)
def test_kronecker_delta(ξ_index, x, desired):
    """Test the Kronecker delta calculation."""
    result = kronecker_delta(ξ_index, x)
    assert_allclose(result, desired, 1e-3)
