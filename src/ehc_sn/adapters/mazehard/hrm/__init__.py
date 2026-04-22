"""MazeHard+HRM bridge family public surface.

Import stable adapter settings, versioned bridge adapters, task bindings, and
trace helpers from this barrel.
"""

from .core import MazeHardHRMAdapterSettings
from .hrm_v1 import MazeHardHRMV1BridgeAdapter
from .hrm_v2 import MazeHardHRMV2BridgeAdapter
from .objectives import MazeHardHRMV1ACTTaskBinding, MazeHardHRMV2HybridTaskBinding
from .traces import MAZE_HARD_HRM_ACT_TRACE_FIELDS, MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS, MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY

__all__ = [
    "MazeHardHRMAdapterSettings",
    "MazeHardHRMV1BridgeAdapter",
    "MazeHardHRMV1ACTTaskBinding",
    "MazeHardHRMV2BridgeAdapter",
    "MazeHardHRMV2HybridTaskBinding",
    "MAZE_HARD_HRM_ACTOR_CRITIC_TRACE_FIELDS",
    "MAZE_HARD_HRM_ACT_TRACE_FIELDS",
    "MAZE_HARD_HRM_TRACE_SOLUTION_OVERLAY",
]
