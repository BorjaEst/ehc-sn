"""Shared MazeHard+HRM bridge namespace.

Re-exports the public symbols shared across both HRM v1 (ACT) and HRM v2 (RL)
Lightning runtimes.  Version-specific bridge classes live in :mod:`hrm_v1` and
:mod:`hrm_v2` respectively and are not re-exported here.
"""

from ehc_sn.adapters.mazehard.bridges.hrm.objectives import MazeHardHRMACTTaskBinding, MazeHardHRMHybridRLTaskBinding
from ehc_sn.adapters.mazehard.bridges.hrm.traces import MAZE_HARD_HRM_ACT_TRACE_FIELDS, MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY

__all__ = [
    "MazeHardHRMACTTaskBinding",
    "MazeHardHRMHybridRLTaskBinding",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
]
