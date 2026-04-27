"""MazeHard task execution-binding capabilities.

Sub-modules:

- :mod:`ehc_sn.tasks.mazehard.capabilities.deliberation` — deliberation actor-critic
  capability (:class:`~ehc_sn.tasks.mazehard.capabilities.deliberation.MazeHardDeliberationCapability`)
  and config (:class:`~ehc_sn.tasks.mazehard.capabilities.deliberation.MazeHardDeliberationConfig`).
"""

from ehc_sn.tasks.mazehard.capabilities import deliberation

__all__ = ["deliberation"]
