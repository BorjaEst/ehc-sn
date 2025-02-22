"""Utility functions and classes for EHC-SN"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# pylint: disable=non-ascii-name


def kronecker_delta(
    ix: npt.NDArray[np.bool], x: Optional[npt.NDArray[Any]] = None
) -> npt.NDArray[Any]:
    """Return the Kronecker delta calculation."""
    x = np.ones_like(ix, dtype=float) if x is None else x
    return np.where(~ix, 0, x)
