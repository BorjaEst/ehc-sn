"""Shared-substrate namespace for processed dataset source families.

Public sub-namespaces:

- :mod:`~ehp_sn.data.substrate.reader` — substrate loading helpers.
- :mod:`~ehp_sn.data.substrate.grid2d` — grid2d channel constants and validator.
- :mod:`~ehp_sn.data.substrate.dungeongen` — dungeongen family builder.
- :mod:`~ehp_sn.data.substrate.maze_nd` — maze-nd family builder.
"""

from ehc_sn.data.substrate import dungeongen, grid2d, maze_nd, reader

__all__ = ["reader", "grid2d", "dungeongen", "maze_nd"]
