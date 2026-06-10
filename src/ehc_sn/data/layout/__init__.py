"""Layout source package.

Public exports:

- :class:`SpatialLayout` — graph-indexed world record protocol.
- :class:`ActionSpace` — action space descriptor.
- :func:`validate_spatial_layout` — layout contract validator.
- :func:`write_layout_dataset` — generic layout dataset writer.
- :func:`load_layout_dataset` — generic layout dataset reader.
- :data:`DEFAULT_GRID_ACTION_SPACE` — 4-direction + stay action space.
- :data:`HEX_ACTION_SPACE` — 6-direction + stay action space (future).
"""

from ehc_sn.data.layout._protocol import (
    DEFAULT_GRID_ACTION_SPACE,
    HEX_ACTION_SPACE,
    ActionSpace,
    SpatialLayout,
    validate_spatial_layout,
)
from ehc_sn.data.layout.io import load_layout_dataset, write_layout_dataset

__all__ = [
    "ActionSpace",
    "DEFAULT_GRID_ACTION_SPACE",
    "HEX_ACTION_SPACE",
    "SpatialLayout",
    "load_layout_dataset",
    "validate_spatial_layout",
    "write_layout_dataset",
]
