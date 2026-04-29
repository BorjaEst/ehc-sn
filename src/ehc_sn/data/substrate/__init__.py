"""Shared-substrate namespace for maze data source families.

Public sub-namespaces:

- :mod:`~ehc_sn.data.substrate.reader` — substrate loading helpers.
- :mod:`~ehc_sn.data.substrate.dungeongen` — dungeongen family builder.
- :mod:`~ehc_sn.data.substrate.maze_nd` — maze-nd family builder.
"""

from ehc_sn.data.substrate import dungeongen, maze_nd, reader

__all__ = ["reader", "dungeongen", "maze_nd"]
