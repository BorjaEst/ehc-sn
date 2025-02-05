import numpy as np
from ehc_sn.utils import kron_delta
import pytest


@pytest.mark.parametrize(
    "ξ, x, expected",
    [
        (0, np.array([1, 2, np.inf]), [1, 0, 0]),
        (1, np.array([1, 2, np.inf]), [0, 2, 0]),
        (2, np.array([1, 2, np.inf]), [0, 0, np.inf]),
    ],
)
def test_kron_delta(ξ, x, expected):
    assert all(kron_delta(ξ, x) == expected)
