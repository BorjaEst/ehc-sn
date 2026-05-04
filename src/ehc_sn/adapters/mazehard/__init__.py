"""MazeHard adapter namespace.

This package is a namespace barrel only. Canonical public MazeHard adapter
symbols live in ``ehc_sn.adapters.mazehard.hrm`` and
``ehc_sn.adapters.mazehard.ehc``.
"""

from . import ehc, hrm

__all__ = ["ehc", "hrm"]
