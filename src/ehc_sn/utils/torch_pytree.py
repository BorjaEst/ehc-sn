"""Torch pytree wrapper to isolate private APIs and provide stable helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from torch.utils import _pytree as pytree


# =================================================================================================
@dataclass(frozen=True)
class FlattenedPyTree:
    """Flattened pytree payload with aligned leaf paths."""

    leaves: list[Any]
    spec: Any
    paths: list[tuple[str, ...]]


# =================================================================================================
def tree_flatten(  # ------------------------------------------------------------------------------
    value: Any,
) -> tuple[list[Any], Any]:  # fmt: skip
    """Flatten a pytree into leaves and a TreeSpec."""

    leaves, spec = pytree.tree_flatten(value)
    return list(leaves), spec


# =================================================================================================
def tree_unflatten(  # -----------------------------------------------------------------------------
    leaves: list[Any], spec: Any,
) -> Any:  # fmt: skip
    """Rebuild a pytree from leaves and a TreeSpec."""

    return pytree.tree_unflatten(leaves, spec)


# =================================================================================================
def spec_equal(  # ---------------------------------------------------------------------------------
    left: Any, right: Any,
) -> bool:  # fmt: skip
    """Return True if two TreeSpecs are equal."""

    return left == right


# =================================================================================================
def flatten_with_paths(  # ------------------------------------------------------------------------
    value: Any,
) -> tuple[list[Any], Any, list[tuple[str, ...]]]:  # fmt: skip
    """Flatten a pytree and return leaves, spec, and leaf paths."""

    leaves, spec = tree_flatten(value)
    paths = _paths_from_spec(spec, len(leaves))

    if len(paths) != len(leaves):
        raise ValueError("Leaf path enumeration mismatch")

    return leaves, spec, paths


# =================================================================================================
def _paths_from_spec(  # --------------------------------------------------------------------------
    spec: Any, count: int,
) -> list[tuple[str, ...]]:  # fmt: skip
    """Reconstruct leaf paths from a TreeSpec using index unflattening."""

    if count == 0:
        return []
    index_tree = tree_unflatten(list(range(count)), spec)
    paths: list[Optional[tuple[str, ...]]] = [None] * count

    def _walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(child, path + (str(key),))
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                _walk(child, path + (str(idx),))
            return
        if not isinstance(value, int):
            raise ValueError("Index tree contains non-integer leaf")
        if value < 0 or value >= count:
            raise ValueError("Index tree leaf out of range")
        if paths[value] is not None:
            raise ValueError("Duplicate index in leaf path enumeration")
        paths[value] = path

    _walk(index_tree, ())
    if any(path is None for path in paths):
        raise ValueError("Leaf path enumeration incomplete")

    return [path for path in paths if path is not None]


# =================================================================================================
def to_numeric_array(  # --------------------------------------------------------------------------
    value: Any,
) -> Optional[np.ndarray]:  # fmt: skip
    """Convert a value to a numeric NumPy array if possible."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().numpy()  # type: ignore
        except Exception:
            return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None

    return arr
